from __future__ import annotations
import copy, json, math, os, re, sys, time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Load .env from repo root if present (same pattern as test.py)
_ENV_FILE = Path(__file__).resolve().parent / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

ELEMENT_BYTES = 1

# ── Strategy Configuration (set by _select_strategy before main loop) ──

_STRATEGY = {
    "name": "balanced",
    "max_len": 6,        # max ops per fusion candidate
    "gran_top_k": 10,    # granularity candidates to evaluate
    "retention_pct": 0.70,  # fraction of fast memory for retention
    "beam_window": 8,    # beam partition lookahead window
    "beam_width": 6,     # beam partition width
}

# ── Data Structures ──

@dataclass(frozen=True)
class Problem:
    widths: List[int]; heights: List[int]
    inputs: List[List[int]]; outputs: List[List[int]]
    base_costs: List[float]; op_types: List[str]
    fast_memory_capacity: float; slow_memory_bandwidth: float
    native_granularity: Tuple[int, int]

@dataclass(frozen=True)
class Op:
    idx: int; op_type: str; inputs: List[int]; outputs: List[int]; base_cost: float

def load_problem(path: str) -> Tuple[Problem, List[Op]]:
    with open(path) as f:
        d = json.load(f)
    prob = Problem(widths=d["widths"], heights=d["heights"], inputs=d["inputs"],
                   outputs=d["outputs"], base_costs=d["base_costs"], op_types=d["op_types"],
                   fast_memory_capacity=d["fast_memory_capacity"],
                   slow_memory_bandwidth=d["slow_memory_bandwidth"],
                   native_granularity=tuple(d["native_granularity"]))
    ops = [Op(idx=i, op_type=d["op_types"][i], inputs=d["inputs"][i],
              outputs=d["outputs"][i], base_cost=d["base_costs"][i])
           for i in range(len(d["op_types"]))]
    return prob, ops

# ── Graph Utilities ──

def topo_sort(ops: List[Op]) -> List[int]:
    producer = {}
    for op in ops:
        for t in op.outputs: producer[t] = op.idx
    in_deg = [0]*len(ops); adj = [[] for _ in range(len(ops))]
    for op in ops:
        for t in op.inputs:
            if t in producer:
                adj[producer[t]].append(op.idx); in_deg[op.idx] += 1
    q = sorted(i for i in range(len(ops)) if in_deg[i] == 0); order = []
    while q:
        n = q.pop(0); order.append(n)
        for m in adj[n]:
            in_deg[m] -= 1
            if in_deg[m] == 0: q.append(m); q.sort()
    return order

def find_boundary_tensors(ops_in_step, all_ops):
    produced, consumed = set(), set()
    for op in ops_in_step: produced.update(op.outputs); consumed.update(op.inputs)
    b_in = consumed - produced
    consumed_in = set()
    for op in ops_in_step: consumed_in.update(op.inputs)
    b_out = {t for t in produced if t not in consumed_in}
    return b_in, b_out

def _get_orphan_promotions(step_op_ids, ops, remaining):
    """Return set of ephemeral tensors that need to be promoted to boundary
    outputs because they have remaining consumers outside the step."""
    step_set = set(step_op_ids)
    step_ops = [ops[i] for i in step_op_ids]
    produced = set()
    consumed_in = set()
    for op in step_ops:
        produced.update(op.outputs)
        consumed_in.update(op.inputs)
    ephemeral = produced & consumed_in
    if not ephemeral:
        return set()
    promotions = set()
    for op in ops:
        if op.idx in remaining and op.idx not in step_set:
            for t in op.inputs:
                if t in ephemeral:
                    promotions.add(t)
    return promotions

# ── Latency Model (verified against PROBLEM.md) ──

@lru_cache(maxsize=4096)
def tile_count(dim, tile): return max(1, math.ceil(dim / max(1, tile)))

_mm_extent_memo: dict = {}  # (op.op_type, inputs_tuple, prob_widths_key) -> int

def matmul_reduction_extent(op, prob):
    k = (op.op_type, tuple(op.inputs), prob.widths[op.inputs[0]] if op.inputs else 0)
    if k not in _mm_extent_memo:
        _mm_extent_memo[k] = prob.widths[op.inputs[0]] if op.op_type == "MatMul" else 1
    return _mm_extent_memo[k]

_slice_memo: dict = {}  # (op.op_type, inputs_tuple, outputs_tuple, tid, w, h, k_step) -> int

def tensor_slice_elems_for_role(op, tid, w, h, k_step, prob):
    key = (op.op_type, tuple(op.inputs), tuple(op.outputs), tid, w, h, k_step)
    if key not in _slice_memo:
        if op.op_type == "MatMul":
            if tid in op.outputs:
                v = w * h * ELEMENT_BYTES
            elif tid == op.inputs[0]:
                v = h * k_step * ELEMENT_BYTES
            elif tid == op.inputs[1]:
                v = w * k_step * ELEMENT_BYTES
            else:
                v = w * h * ELEMENT_BYTES
        else:
            v = w * h * ELEMENT_BYTES
        _slice_memo[key] = v
    return _slice_memo[key]

def working_set_bytes_for_subgraph(ops_in_step, b_in, b_out, w, h, k_step, prob):
    ws = 0; seen_in = set(); seen_out = set()
    for op in ops_in_step:
        for t in op.inputs:
            if t in b_in and t not in seen_in:
                seen_in.add(t)
                if op.op_type == "MatMul" and t == op.inputs[0]:
                    K = matmul_reduction_extent(op, prob)
                    if max(1, math.ceil(K / max(1, k_step))) > 1:
                        ws += h * prob.widths[t] * ELEMENT_BYTES
                    else:
                        ws += tensor_slice_elems_for_role(op, t, w, h, k_step, prob)
                else:
                    ws += tensor_slice_elems_for_role(op, t, w, h, k_step, prob)
        for t in op.outputs:
            if t in b_out and t not in seen_out:
                seen_out.add(t)
                ws += tensor_slice_elems_for_role(op, t, w, h, k_step, prob)
    return ws

def _precompute_slices(ops_in_step, b_in, b_out, w, h, k_step, retain_after, prob,
                        retained_from_prior):
    """Precompute slice bytes shared across all traversal evaluations.
    Returns a context dict with all precomputed values.
    """
    max_K = max((matmul_reduction_extent(op, prob) for op in ops_in_step), default=1)
    split_k = max(1, math.ceil(max_K / max(1, k_step)))
    out_w = max((prob.widths[t] for t in b_out), default=w)
    out_h = max((prob.heights[t] for t in b_out), default=h)
    tw, th = tile_count(out_w, w), tile_count(out_h, h)

    lhs_slice = {}; rhs_slice = {}; pw_slice = {}; out_slice = {}; lhs_full = {}
    seen_in = set(); seen_out = set()
    comp_mm_k = 0.0; comp_pw = 0.0
    for op in ops_in_step:
        if op.op_type == "MatMul":
            comp_mm_k += op.base_cost / split_k
            t_lhs, t_rhs = op.inputs[0], op.inputs[1]
            if t_lhs not in seen_in:
                seen_in.add(t_lhs)
                if t_lhs in b_in and t_lhs not in retained_from_prior:
                    lhs_slice[t_lhs] = tensor_slice_elems_for_role(op, t_lhs, w, h, k_step, prob)
                    lhs_full[t_lhs]  = prob.heights[t_lhs] * prob.widths[t_lhs] * ELEMENT_BYTES
            if t_rhs not in seen_in:
                seen_in.add(t_rhs)
                if t_rhs in b_in and t_rhs not in retained_from_prior:
                    rhs_slice[t_rhs] = tensor_slice_elems_for_role(op, t_rhs, w, h, k_step, prob)
        else:
            comp_pw += op.base_cost
            for t in op.inputs:
                if t not in seen_in:
                    seen_in.add(t)
                    if t in b_in and t not in retained_from_prior:
                        pw_slice[t] = tensor_slice_elems_for_role(op, t, w, h, k_step, prob)
        for t in op.outputs:
            if t not in seen_out:
                seen_out.add(t)
                if t in b_out and t not in retain_after:
                    out_slice[t] = tensor_slice_elems_for_role(op, t, w, h, k_step, prob)
    return {
        "split_k": split_k, "tw": tw, "th": th,
        "comp_mm_k": comp_mm_k, "comp_pw": comp_pw,
        "total_lhs_slice": sum(lhs_slice.values()),
        "total_rhs_slice": sum(rhs_slice.values()),
        "total_pw_slice":  sum(pw_slice.values()),
        "total_out_slice": sum(out_slice.values()),
        "total_lhs_full":  sum(lhs_full.get(t, 0) for t in lhs_slice),
        "has_lhs": bool(lhs_slice), "has_rhs": bool(rhs_slice),
        "has_pw":  bool(pw_slice),  "has_out": bool(out_slice),
    }


def _eval_traversal_from_slices(sc, tile_seq, use_lhs_row_reuse, B):
    """Run the roofline tile loop given precomputed slice context `sc`."""
    split_k = sc["split_k"]; tw = sc["tw"]
    comp_mm_k = sc["comp_mm_k"]; comp_pw = sc["comp_pw"]
    total_lhs_slice = sc["total_lhs_slice"]; total_rhs_slice = sc["total_rhs_slice"]
    total_pw_slice  = sc["total_pw_slice"];  total_out_slice = sc["total_out_slice"]
    total_lhs_full  = sc["total_lhs_full"]
    has_lhs = sc["has_lhs"]; has_rhs = sc["has_rhs"]
    has_pw  = sc["has_pw"];  has_out = sc["has_out"]
    first_tile = tile_seq[0] if tile_seq else 0
    total = 0.0
    for ki in range(split_k):
        fk = (ki == 0); lk = (ki == split_k - 1)
        comp_ki = comp_mm_k + (comp_pw if fk else 0.0)
        pr = pc = -1
        for ti in tile_seq:
            r = ti // tw; c = ti % tw
            mb = 0.0
            if has_lhs:
                if split_k > 1:
                    if fk and ti == first_tile: mb += total_lhs_full
                else:
                    if (not use_lhs_row_reuse) or r != pr: mb += total_lhs_slice
            if has_rhs and c != pc: mb += total_rhs_slice
            if has_pw  and fk:      mb += total_pw_slice
            if has_out and lk:      mb += total_out_slice
            total += max(comp_ki, mb / B)
            pr = r; pc = c
    return total


def estimate_latency_step(ops_in_step, b_in, b_out, w, h, k_step,
                           retain_after, prob, retained_from_prior=None,
                           traversal_order=None):
    """Per-tile roofline latency."""
    B = max(prob.slow_memory_bandwidth, 1e-9)
    if retained_from_prior is None: retained_from_prior = set()
    if retain_after is None: retain_after = set()
    sc = _precompute_slices(ops_in_step, b_in, b_out, w, h, k_step,
                             retain_after, prob, retained_from_prior)
    tw, th = sc["tw"], sc["th"]
    tile_seq = traversal_order if traversal_order is not None else list(range(tw * th))
    use_lhs_row_reuse = (traversal_order is not None)
    return _eval_traversal_from_slices(sc, tile_seq, use_lhs_row_reuse, B)

# ── Agent State ──

@dataclass
class Candidate:
    cid: int; ops: List[int]; kind: str

def _build_next_consumer(ops: List[Op]) -> Dict[int, int]:
    """Returns next_consumer[t] = op.idx of the topologically-first op that consumes tensor t."""
    producer = {}
    for op in ops:
        for t in op.outputs: producer[t] = op.idx
    consumer_lists: Dict[int, List[int]] = {}
    for op in ops:
        for t in op.inputs:
            consumer_lists.setdefault(t, []).append(op.idx)
    topo_pos = {op.idx: i for i, op in enumerate(ops)}
    nc: Dict[int, int] = {}
    for t, cons in consumer_lists.items():
        first = min(cons, key=lambda oid: topo_pos.get(oid, 999999))
        nc[t] = first
    return nc


@dataclass
class AgentState:
    prob: Problem; ops: List[Op]; topo: List[int]; remaining: Set[int]
    schedule_subgraphs: List[List[int]] = field(default_factory=list)
    schedule_gran: List[List[int]] = field(default_factory=list)
    schedule_retain: List[List[int]] = field(default_factory=list)
    schedule_trav: List[Optional[List[int]]] = field(default_factory=list)
    schedule_lat: List[float] = field(default_factory=list)
    candidates: Dict[int, Candidate] = field(default_factory=dict)
    next_cid: int = 0
    next_consumer: Dict[int, int] = field(default_factory=dict)
    def is_complete(self): return len(self.remaining) == 0
    def compact_summary(self):
        return {"remaining_ops": sorted(self.remaining), "n_remaining": len(self.remaining),
                "n_steps": len(self.schedule_subgraphs),
                "total_lat": round(sum(self.schedule_lat), 1)}
    def _register(self, ops, kind):
        cid = self.next_cid; self.next_cid += 1
        self.candidates[cid] = Candidate(cid=cid, ops=ops, kind=kind); return cid

# ── Tools ──

def tool_propose_fusion_candidates(state, k=40, max_len=6):
    producer = {}
    for op in state.ops:
        for t in op.outputs: producer[t] = op.idx

    resident = _compute_retained_from_prior(state)
    def _resident_bytes(oid):
        total = 0
        for t in state.ops[oid].inputs:
            if t in resident:
                total += state.prob.widths[t] * state.prob.heights[t]
        return total

    frontier = []
    for oid in state.topo:
        if oid not in state.remaining: continue
        if all(producer.get(t, -1) not in state.remaining for t in state.ops[oid].inputs if t in producer):
            frontier.append(oid)

    # Sort frontier: prefer ops whose inputs are already resident (minimize incremental bytes-in)
    frontier.sort(key=lambda oid: -_resident_bytes(oid))

    state.candidates.clear(); cids = []
    for oid in frontier[:k]:
        cids.append(state._register([oid], "single"))
    for seed in frontier[:20]:
        cur, cs = [seed], {seed}
        for _ in range(max_len - 1):
            bn, bs = None, -1.0
            for oid in state.topo:
                if oid not in state.remaining or oid in cs: continue
                ok, sc = True, 0.0
                for t in state.ops[oid].inputs:
                    if t in producer:
                        p = producer[t]
                        if p in state.remaining and p not in cs: ok = False; break
                        if p in cs: sc += 1.0
                if not ok: continue
                # Boost for shared inputs (horizontal fusion) and resident tensors
                for t in state.ops[oid].inputs:
                    for co in cur:
                        if t in state.ops[co].inputs: sc += 0.6
                    if t in resident: sc += 0.4  # cache-aware bonus
                if sc > bs: bs, bn = sc, oid
            if bn is None or bs < 0.1: break
            cur.append(bn); cs.add(bn)
            ordered = [x for x in state.topo if x in cs]
            cids.append(state._register(ordered, "fusion"))
    return {"candidate_ids": cids}

def _working_set_fast(ops_in, b_in, b_out, ww, hh, kk, prob,
                       _slice_cache):
    """Working set bytes using precomputed slice-elem cache keyed by (t, role, ww, hh, kk).
    _slice_cache is a dict mutated in place: maps (t, role, ww, hh, kk) -> elems.
    'role' is 'lhs','rhs','pw','out'.
    """
    ws = 0
    seen_in = set(); seen_out = set()
    for op in ops_in:
        for t in op.inputs:
            if t in seen_in: continue
            seen_in.add(t)
            if t not in b_in: continue
            if op.op_type == "MatMul" and t == op.inputs[0]:
                # LHS: if split-k active, must hold the full LHS in fast memory
                K = matmul_reduction_extent(op, prob)
                if max(1, math.ceil(K / max(1, kk))) > 1:
                    key = (t, 'lhs_full', hh, id(prob))
                    if key not in _slice_cache:
                        _slice_cache[key] = hh * prob.widths[t] * ELEMENT_BYTES
                else:
                    key = (t, 'lhs', ww, hh, kk)
                    if key not in _slice_cache:
                        _slice_cache[key] = tensor_slice_elems_for_role(op, t, ww, hh, kk, prob)
            else:
                role = 'rhs' if (op.op_type == "MatMul" and t == op.inputs[1]) else 'pw'
                key = (t, role, ww, hh, kk)
                if key not in _slice_cache:
                    _slice_cache[key] = tensor_slice_elems_for_role(op, t, ww, hh, kk, prob)
            ws += _slice_cache[key]
        for t in op.outputs:
            if t in seen_out: continue
            seen_out.add(t)
            if t not in b_out: continue
            key = (t, 'out', ww, hh, kk)
            if key not in _slice_cache:
                _slice_cache[key] = tensor_slice_elems_for_role(op, t, ww, hh, kk, prob)
            ws += _slice_cache[key]
    return ws

def _dim_divisors(n, native):
    """Return sorted-descending divisors and near-divisors of n, plus native and powers-of-two."""
    s = {n, native, 1}
    # All divisors of n up to sqrt(n)
    i = 1
    while i * i <= n:
        if n % i == 0:
            s.add(i); s.add(n // i)
        i += 1
    # Power-of-two halvings of max(n, native)
    d = max(n, native)
    while d >= 1:
        s.add(d); d //= 2
    # Near-divisors: n // p for small p (reduces ragged boundary tiles)
    for p in range(1, 9):
        v = n // p
        if v > 0: s.add(v)
    return sorted(s, reverse=True)


def _padding_penalty(w, h, nw, nh):
    """Penalty factor for sub-native granularity: ceil(nw/w) * ceil(nh/h)."""
    return math.ceil(nw / max(1, w)) * math.ceil(nh / max(1, h))


def _gen_grans(ops_in, b_in, b_out, prob, top_k=10):
    """Generate granularity candidates ([w,h,k], ws) sorted by effective score."""
    nw, nh = prob.native_granularity
    ow = max((prob.widths[t] for t in b_out), default=nw)
    oh = max((prob.heights[t] for t in b_out), default=nh)
    cap = prob.fast_memory_capacity
    _slice_cache = {}
    has_mm = any(op.op_type == "MatMul" for op in ops_in)

    # ── k candidates: full K, then halvings + intermediate divisors + small absolutes ──
    ks = {1}
    for op in ops_in:
        if op.op_type == "MatMul":
            K = matmul_reduction_extent(op, prob)
            for d in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128]: ks.add(max(1, K // d))
            ks.add(K)
            # Small absolute k_step values to fill gap between 1 and K//128
            for small_k in [2, 3, 4, 6, 8, 12, 16, 24]:
                if small_k < K: ks.add(small_k)

    # ── w/h candidates: divisors of output dims + native + powers-of-two ──
    ws_cands = _dim_divisors(ow, nw)
    hs_cands = _dim_divisors(oh, nh)

    # Rectangular tiling: also try w×nh and nw×h orientations
    extra_combos = set()
    for ww in ws_cands[:6]:
        extra_combos.add((ww, nh))   # w-dominant: wide tiles, native height
        extra_combos.add((ww, oh))
    for hh in hs_cands[:6]:
        extra_combos.add((nw, hh))   # h-dominant: native width, tall tiles
        extra_combos.add((ow, hh))
    extra_combos.update([(nw, nh), (ow, oh), (ow, nh), (nw, oh)])

    seen = set()
    res = []  # ([w,h,k], ws)

    # ── Primary strategy: Split-K as lever — prefer (w,h)>=native, reduce k to fit ──
    # For each (h,k) in descending order, find max feasible w (linear scan over divisors).
    for kk in sorted(ks, reverse=True):
        for hh in hs_cands:
            for ww in ws_cands:
                key = (ww, hh, kk)
                if key in seen: continue
                seen.add(key)
                ws = _working_set_fast(ops_in, b_in, b_out, ww, hh, kk, prob, _slice_cache)
                if ws <= cap:
                    res.append(([ww, hh, kk], ws))
                    break  # first feasible ww (descending) is the largest for this (hh, kk)

        # Also check rectangular combos for this k
        for ww, hh in extra_combos:
            key = (ww, hh, kk)
            if key in seen: continue
            seen.add(key)
            ws = _working_set_fast(ops_in, b_in, b_out, ww, hh, kk, prob, _slice_cache)
            if ws <= cap:
                res.append(([ww, hh, kk], ws))

    if not res:
        return []

    # ── Sort: penalize sub-native tiles, prefer large area + large k ──
    # score = area / padding_penalty — native-or-above tiles score at full area;
    # sub-native tiles are down-ranked proportionally to their padding overhead.
    def _score(g):
        w, h, k = g
        pad = _padding_penalty(w, h, nw, nh)
        return (-(w * h) // pad, -k)

    res.sort(key=lambda x: _score(x[0]))

    # Deduplicate
    seen2 = set(); deduped = []
    for g, ws in res:
        key = tuple(g)
        if key not in seen2:
            seen2.add(key); deduped.append((g, ws))
    return deduped[:top_k]


def _compute_retained_from_prior(state):
    """Tensors currently retained in fast memory from the previous step."""
    if not state.schedule_retain: return set()
    return set(state.schedule_retain[-1])

def _compute_retain_set_full_schedule(state, b_out, cand_ops):
    """Value-density retention with evict-late policy."""
    B = max(state.prob.slow_memory_bandwidth, 1e-9)
    consumers = {}
    for op in state.ops:
        for t in op.inputs: consumers.setdefault(t, []).append(op.idx)

    # Topo position map for proximity computation
    topo_pos = {oid: i for i, oid in enumerate(state.topo)}
    # Current frontier position = min topo_pos of remaining ops
    if state.remaining:
        frontier_pos = min(topo_pos.get(oid, 0) for oid in state.remaining)
    else:
        frontier_pos = 0

    candidates = []
    for t in b_out:
        future = [c for c in consumers.get(t, []) if c in state.remaining and c not in cand_ops]
        if not future:
            continue
        t_bytes = state.prob.widths[t] * state.prob.heights[t] * ELEMENT_BYTES
        reload_cost_next = t_bytes
        evict_cost_now   = t_bytes / B
        base_value = reload_cost_next - evict_cost_now
        next_cons_pos = min(topo_pos.get(c, frontier_pos) for c in future)
        topo_dist = max(0, next_cons_pos - frontier_pos)
        proximity_bonus = 1.0 / (1.0 + topo_dist)
        value = base_value * (1.0 + proximity_bonus)
        n_future = len(future)
        candidates.append((t, t_bytes, value, n_future))

    if not candidates:
        return set()

    budget = int(state.prob.fast_memory_capacity * _STRATEGY["retention_pct"])
    candidates = [c for c in candidates if c[2] > 0]
    candidates.sort(key=lambda x: (-x[2], -x[3], x[1]))
    retain = set(); used = 0
    for t, t_bytes, _, _ in candidates:
        if used + t_bytes <= budget:
            retain.add(t); used += t_bytes
    return retain

def tool_try_recomputation(state):
    """Generalized fanout/join solver: enumerate recompute/residency/fusion strategies."""
    op_map = {op.idx: op for op in state.ops}
    producer = {}; consumers = {}
    for op in state.ops:
        for t in op.outputs: producer[t] = op.idx
        for t in op.inputs: consumers.setdefault(t, []).append(op.idx)
    topo_map = {oid: i for i, oid in enumerate(state.topo)}

    subgraph_set = set()  # deduplicate by frozenset of ops
    proposed_cids = []; diamonds = []

    def _all_deps_satisfied(ops_list):
        """True iff every op's external inputs are from committed (non-remaining) ops."""
        cand_set = set(ops_list)
        for oid in ops_list:
            for t in op_map[oid].inputs:
                p = producer.get(t)
                if p is None: continue          # initial input tensor, always available
                if p in state.remaining and p not in cand_set:
                    return False                # uncommitted external dependency
        return True

    def _register_if_new(ops_list, kind):
        key = frozenset(ops_list)
        if key in subgraph_set: return
        if not _all_deps_satisfied(ops_list): return  # skip topologically invalid candidates
        subgraph_set.add(key)
        sorted_ops = sorted(ops_list, key=lambda x: topo_map.get(x, 0))
        proposed_cids.append(state._register(sorted_ops, kind))

    for t_id, cons_list in consumers.items():
        prod_op = producer.get(t_id)
        if prod_op is None: continue
        rem_cons = [c for c in cons_list if c in state.remaining]
        if len(rem_cons) < 2 or prod_op not in state.remaining: continue

        diamonds.append({"tensor": t_id, "producer": prod_op, "consumers": rem_cons})

        # (a) Recompute: fuse producer into each consumer separately
        for ci in rem_cons:
            _register_if_new([prod_op, ci], "recomputation")

        # (b) Selective residency: run all consumers together (producer separate, t_id retained)
        if len(rem_cons) <= 6:
            _register_if_new(rem_cons, "recomputation")

        # (c) Mega-fusion: producer + all consumers in one subgraph
        mega = list(set([prod_op] + rem_cons))
        if len(mega) <= 6:
            _register_if_new(mega, "recomputation")

        # (d) Hybrid: for each pair (ci, cj), fuse producer+ci in one step, retain t_id for cj
        if len(rem_cons) == 2:
            ci, cj = rem_cons[0], rem_cons[1]
            _register_if_new([prod_op, ci], "recomputation")  # recompute for ci, keep for cj
            _register_if_new([prod_op, cj], "recomputation")  # recompute for cj, keep for ci

        # (e) Extended chain: producer -> consumer -> downstream consumer
        for ci in rem_cons:
            for out_t in op_map[ci].outputs:
                ds = [c for c in consumers.get(out_t, []) if c in state.remaining and c != ci]
                for di in ds[:2]:
                    chain = list(set([prod_op, ci, di]))
                    if len(chain) <= 6:
                        _register_if_new(chain, "recomputation")

    return {"diamonds_found": len(diamonds), "diamonds": diamonds, "candidate_ids": proposed_cids}


def tool_evaluate_step_bundle(state, candidate_ids, granularity_k=15):
    """Two-stage scoring + dominance pruning over candidate granularities."""
    eval_start = time.time()
    retained_from_prior = _compute_retained_from_prior(state)

    # Stage A: for each candidate, generate feasible grans (WS already computed in _gen_grans)
    # Stage B cap: limit total latency evals across all candidates
    STAGE_A_TOP = max(3, granularity_k)
    MAX_TOTAL_EVALS = 200
    total_evals = 0
    ranked = []

    recomp_cids = [c for c in candidate_ids
                   if c in state.candidates and state.candidates[c].kind == "recomputation"]
    normal_cids = [c for c in candidate_ids
                   if c in state.candidates and state.candidates[c].kind != "recomputation"]
    recomp_cids = sorted(recomp_cids,
                         key=lambda c: len(state.candidates[c].ops), reverse=True)[:8]
    candidate_ids_filtered = normal_cids + recomp_cids

    # Pre-compute individual op latencies only for ops that appear in current candidates.
    # Previously computed for all ops in the graph (range(len(state.ops))), which wasted
    # O(n_ops × n_steps) latency evaluations on ops not relevant to the current decision.
    ind_lat_cache = {}
    cand_ops_needed = set()
    for cid in candidate_ids_filtered:
        if cid in state.candidates:
            cand_ops_needed.update(state.candidates[cid].ops)
    for oid in cand_ops_needed:
        op = state.ops[oid]
        bi_s, bo_s = find_boundary_tensors([op], state.ops)
        grans_ws = _gen_grans([op], bi_s, bo_s, state.prob, top_k=1)
        if grans_ws:
            w, h, kk = grans_ws[0][0]
            ind_lat_cache[oid] = estimate_latency_step(
                [op], bi_s, bo_s, w, h, kk, set(), state.prob,
                retained_from_prior=retained_from_prior)
        else:
            ind_lat_cache[oid] = op.base_cost

    for cid in candidate_ids_filtered:
        if cid not in state.candidates: continue
        if total_evals >= MAX_TOTAL_EVALS: break
        cand = state.candidates[cid]
        ois = [state.ops[i] for i in cand.ops]
        bi, bo = find_boundary_tensors(ois, state.ops)
        # Promote orphaned ephemeral tensors to boundary outputs for accurate scoring
        promotions = _get_orphan_promotions(cand.ops, state.ops, state.remaining)
        bo = bo | promotions
        retain = _compute_retain_set_full_schedule(state, bo, set(cand.ops))
        sum_ind = sum(ind_lat_cache.get(oid, state.ops[oid].base_cost) for oid in cand.ops)

        # Stage A: get feasible (gran, ws) pairs — WS already computed, no recompute needed
        grans_ws = _gen_grans(ois, bi, bo, state.prob, top_k=STAGE_A_TOP)

        B = max(state.prob.slow_memory_bandwidth, 1e-9)
        bytes_in  = sum(state.prob.widths[t] * state.prob.heights[t] * ELEMENT_BYTES for t in bi)
        bytes_out = sum(state.prob.widths[t] * state.prob.heights[t] * ELEMENT_BYTES for t in bo)

        for g, ws in grans_ws:
            if total_evals >= MAX_TOTAL_EVALS: break
            w, h, kk = g
            # Apply best traversal (raster vs row-snake vs col-snake) for all candidates
            _, lat = _best_traversal(ois, bi, bo, w, h, kk, state.prob,
                                     retain_after=retain,
                                     retained_from_prior=retained_from_prior)
            total_evals += 1
            fusion_ratio = lat / max(sum_ind, 1e-9)
            ranked.append({"cid": cid, "ops": cand.ops, "kind": cand.kind,
                           "gran": [w, h, kk], "ws_bytes": ws, "lat": round(lat, 2),
                           "fusion_ratio": round(fusion_ratio, 4),
                           "n_ops": len(cand.ops), "ok": True,
                           "bytes_in": bytes_in, "bytes_out": bytes_out})

    # Sort by (fusion_ratio, lat)
    ranked.sort(key=lambda x: (x["fusion_ratio"], x["lat"]))

    # Dominance pruning: keep entry X only if no entry Y has Y.lat <= X.lat and Y.n_ops >= X.n_ops
    # (a more aggressive fusion with lower/equal lat dominates)
    pruned = []
    for x in ranked:
        dominated = False
        for y in pruned:
            if y["lat"] <= x["lat"] and y["n_ops"] >= x["n_ops"] and y["cid"] != x["cid"]:
                dominated = True; break
        if not dominated:
            pruned.append(x)
        if len(pruned) >= 30: break

    return {"ranked": pruned}

def _build_col_major_seq(tw, th):
    """Column-major tile sequence: visit all rows in column 0, then column 1, etc.
    Minimises RHS reloads for high split_k: the RHS strip (w×k_step) is reused
    across all th rows in the same column, so only tw column-changes occur per
    k-iteration instead of tw*th row-changes in raster order.
    Within each column, alternate direction (col-snake) to amortise LHS loads.
    """
    seq = []
    for c in range(tw):
        col = [r * tw + c for r in range(th)]
        if c % 2 == 1: col.reverse()
        seq.extend(col)
    return seq


def _best_traversal(ops_in_step, b_in, b_out, w, h, kk, prob,
                     retain_after=None, retained_from_prior=None):
    """Analytic traversal selection: evaluates raster, row-snake, col-snake, and col-major."""
    ra = retain_after or set()
    rp = retained_from_prior or set()
    B  = max(prob.slow_memory_bandwidth, 1e-9)
    has_mm = any(op.op_type == "MatMul" for op in ops_in_step)

    sc = _precompute_slices(ops_in_step, b_in, b_out, w, h, kk, ra, prob, rp)
    tw, th = sc["tw"], sc["th"]
    raster_seq = list(range(tw * th))
    lat_raster = _eval_traversal_from_slices(sc, raster_seq, False, B)

    if not has_mm or tw * th <= 1:
        return None, lat_raster

    # Build traversal orders once (O(tw*th))
    row_snake = []
    for r in range(th):
        row = list(range(r * tw, (r + 1) * tw))
        if r % 2 == 1: row.reverse()
        row_snake.extend(row)
    col_snake = []
    for c in range(tw):
        col = [r * tw + c for r in range(th)]
        if c % 2 == 1: col.reverse()
        col_snake.extend(col)

    # ── Shared tile loops — slice bytes NOT recomputed ──
    lat_row = _eval_traversal_from_slices(sc, row_snake, True,  B)
    lat_col = _eval_traversal_from_slices(sc, col_snake, False, B)

    best_lat, best_order = lat_raster, None
    if lat_row < best_lat: best_lat, best_order = lat_row, row_snake
    if lat_col < best_lat: best_lat, best_order = lat_col, col_snake

    return best_order, best_lat

def tool_propose_traversal_order(state, cid, gran):
    """Evaluate raster vs row-snake vs col-snake for a candidate+gran. Returns best traversal info."""
    if cid not in state.candidates:
        return {"error": f"Unknown cid {cid}"}
    cand = state.candidates[cid]
    ois = [state.ops[i] for i in cand.ops]
    bi, bo = find_boundary_tensors(ois, state.ops)
    w, h, kk = int(gran[0]), int(gran[1]), max(1, int(gran[2]))
    retain = _compute_retain_set_full_schedule(state, bo, set(cand.ops))
    retained_from_prior = _compute_retained_from_prior(state)
    has_mm = any(op.op_type == "MatMul" for op in ois)
    if not has_mm:
        lat = estimate_latency_step(ois, bi, bo, w, h, kk, retain, state.prob,
                                     retained_from_prior=retained_from_prior)
        return {"traversal_order": None, "latency_default": round(lat, 2),
                "latency_snake": round(lat, 2), "improvement_pct": 0.0, "best_latency": round(lat, 2)}
    best_order, best_lat = _best_traversal(ois, bi, bo, w, h, kk, state.prob,
                                            retain_after=retain, retained_from_prior=retained_from_prior)
    lat_default = estimate_latency_step(ois, bi, bo, w, h, kk, retain, state.prob,
                                         retained_from_prior=retained_from_prior)
    improvement_pct = (lat_default - best_lat) / max(lat_default, 1e-9) * 100
    return {"traversal_order": best_order, "latency_default": round(lat_default, 2),
            "latency_snake": round(best_lat, 2),
            "improvement_pct": round(improvement_pct, 2),
            "best_latency": round(best_lat, 2)}

def tool_apply_step(state, cid, gran):
    if cid not in state.candidates: raise ValueError(f"Unknown cid {cid}")
    cand = state.candidates[cid]; step_ops = cand.ops[:]
    ois = [state.ops[i] for i in step_ops]
    bi, bo = find_boundary_tensors(ois, state.ops)
    # Promote orphaned ephemeral tensors to boundary outputs
    promotions = _get_orphan_promotions(step_ops, state.ops, state.remaining)
    bo = bo | promotions
    w, h, kk = int(gran[0]), int(gran[1]), max(1, int(gran[2]))
    ws = working_set_bytes_for_subgraph(ois, bi, bo, w, h, kk, state.prob)
    if ws > state.prob.fast_memory_capacity:
        raise ValueError(f"OOM: ws={ws} > cap={state.prob.fast_memory_capacity}")
    retain = _compute_retain_set_full_schedule(state, bo, set(step_ops))
    retained_from_prior = _compute_retained_from_prior(state)
    trav, lat = _best_traversal(ois, bi, bo, w, h, kk, state.prob,
                                 retain_after=retain, retained_from_prior=retained_from_prior)
    state.schedule_subgraphs.append(step_ops)
    state.schedule_gran.append([w,h,kk])
    state.schedule_retain.append(sorted(retain))
    state.schedule_trav.append(trav)
    state.schedule_lat.append(float(lat))
    for oid in step_ops: state.remaining.discard(oid)
    return {"applied": {"cid": cid, "ops": step_ops, "gran": [w,h,kk],
                         "lat": round(float(lat),2), "remaining": len(state.remaining),
                         "traversal": "snake" if trav else "raster",
                         "retained": len(retain)}}

# ── LLM Interaction ──

def _repair_json(text):
    text = re.sub(r"```json\s*", "", text); text = re.sub(r"```\s*", "", text)
    text = re.sub(r"//[^\n]*", "", text); text = re.sub(r",\s*([\]}])", r"\1", text)
    text = text.replace("'", '"').strip()
    # Fix truncated JSON: balance brackets
    opens = sum(1 for c in text if c in '{[') - sum(1 for c in text if c in '}]')
    if opens > 0:
        stack = []
        for c in text:
            if c == '{': stack.append('}')
            elif c == '[': stack.append(']')
            elif c in '}]' and stack and stack[-1] == c: stack.pop()
        text += ''.join(reversed(stack))
    return text

def parse_tool_calls(text):
    text = _repair_json(text)
    try: obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try: obj = json.loads(_repair_json(m.group()))
            except json.JSONDecodeError as e: raise ValueError(f"JSON parse: {e}\n{text[:500]}")
        else: raise ValueError(f"No JSON: {text[:500]}")
    calls = []
    if isinstance(obj, dict):
        raw = obj.get("calls") or obj.get("tool_calls") or [obj]
        if not isinstance(raw, list): raw = [raw]
        for c in raw:
            if not isinstance(c, dict): continue
            t = c.get("tool") or c.get("tool_name") or c.get("name") or ""
            a = c.get("args") or c.get("arguments") or c.get("parameters") or {}
            if t: calls.append({"tool": t, "args": a})
    elif isinstance(obj, list):
        for c in obj:
            if not isinstance(c, dict): continue
            t = c.get("tool") or c.get("tool_name") or c.get("name") or ""
            a = c.get("args") or c.get("arguments") or c.get("parameters") or {}
            if t: calls.append({"tool": t, "args": a})
    if not calls: raise ValueError(f"No tool calls: {text[:500]}")
    return {"calls": calls}

def gemini_plan(prompt_obj, max_tokens=4096):
    if genai is None: raise RuntimeError("google-genai not installed")
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key: raise RuntimeError("GOOGLE_API_KEY not set")
    client = genai.Client(api_key=api_key)
    cfg = (types.GenerateContentConfig(temperature=0.3, max_output_tokens=max_tokens,
           response_mime_type="application/json") if types
           else {"temperature": 0.3, "max_output_tokens": max_tokens,
                 "response_mime_type": "application/json"})
    contents = prompt_obj if isinstance(prompt_obj, str) else json.dumps(prompt_obj)
    resp = client.models.generate_content(model="gemini-3.1-flash-lite-preview",
                                           contents=contents, config=cfg)
    text = (resp.text or "").strip()
    if not text: raise RuntimeError("Empty Gemini response")
    return text

def _cut_cost_score(ops_list, prob):
    """Score a candidate subgraph by avoided cut-bytes.

    avoided_cut_bytes = sum of bytes on edges that become INTERNAL (ephemeral)
                        if these ops are fused into one subgraph.
    An edge (producer→consumer) becomes internal iff both endpoints are in ops_list.
    No _gen_grans call — pure O(ops * inputs) arithmetic.

    Higher score = better partition candidate (more bandwidth saved by fusing).
    """
    producers_in = {}
    for op in ops_list:
        for t in op.outputs:
            producers_in[t] = op.idx
    internal = set()
    for op in ops_list:
        for t in op.inputs:
            if t in producers_in:
                internal.add(t)
    return sum(prob.widths[t] * prob.heights[t] * ELEMENT_BYTES for t in internal)


def _beam_partition(state, window=8, beam_width=6):
    """Beam search over a topological window to find near-global min-cut partition.

    Considers the next `window` ready ops and scores candidate fusions by
    avoided cut-bytes (_cut_cost_score — pure arithmetic, no _gen_grans).
    Returns a list of op-lists (partition) or None if window < 2.
    """
    producer = {}
    for op in state.ops:
        for t in op.outputs:
            producer[t] = op.idx

    # Precompute topo position map — O(1) lookup instead of O(n) list.index
    topo_pos = {oid: i for i, oid in enumerate(state.topo)}

    # Build ready frontier
    frontier = []
    for oid in state.topo:
        if oid not in state.remaining: continue
        if all(producer.get(t, -1) not in state.remaining
               for t in state.ops[oid].inputs if t in producer):
            frontier.append(oid)

    window_ops = frontier[:window]
    if len(window_ops) < 2:
        return None

    first = window_ops[0]
    beam = [(0, [frozenset([first])])]  # (total_score, [frozenset, ...])
    # Seed 2: fuse first two if beneficial
    if len(window_ops) >= 2:
        second = window_ops[1]
        ops2 = [state.ops[first], state.ops[second]]
        sc = _cut_cost_score(ops2, state.prob)
        if sc > 0:
            beam.append((sc, [frozenset([first, second])]))

    for step_idx in range(1, len(window_ops)):
        oid = window_ops[step_idx]
        next_beam = []
        for total_sc, partition in beam:
            # Option A: new subgraph
            next_beam.append((total_sc, partition + [frozenset([oid])]))
            # Option B: extend last subgraph (only if topo-safe)
            last = partition[-1]
            cand_set = last | {oid}
            feasible = True
            for coid in cand_set:
                for t in state.ops[coid].inputs:
                    prod = producer.get(t, -1)
                    if prod in state.remaining and prod not in cand_set:
                        feasible = False; break
                if not feasible: break
            if feasible:
                ops_objs = [state.ops[x] for x in cand_set]
                sc = _cut_cost_score(ops_objs, state.prob)
                if sc > 0:
                    next_beam.append((total_sc + sc, partition[:-1] + [cand_set]))
        next_beam.sort(key=lambda x: -x[0])
        beam = next_beam[:beam_width]

    if not beam:
        return None
    best_partition = beam[0][1]
    return [sorted(list(s), key=lambda x: topo_pos.get(x, 0))
            for s in best_partition]


def _cluster_consumers(state):
    """Cluster consecutive consumers of large activations.

    For each tensor > 20% of fast_memory_capacity with 2+ remaining consumers whose
    producer is already scheduled, propose fusing all those consumers together.
    Pure graph-structure check — no _gen_grans call (feasibility checked later in
    tool_evaluate_step_bundle via Stage-A WS filter).

    Returns list of candidate ids registered in state.
    """
    producer = {}
    consumers = {}
    for op in state.ops:
        for t in op.outputs: producer[t] = op.idx
        for t in op.inputs: consumers.setdefault(t, []).append(op.idx)

    threshold = state.prob.fast_memory_capacity * 0.20
    topo_pos = {oid: i for i, oid in enumerate(state.topo)}
    proposed_cids = []
    seen = set()

    for t, cons_list in consumers.items():
        t_bytes = state.prob.widths[t] * state.prob.heights[t] * ELEMENT_BYTES
        if t_bytes < threshold:
            continue
        rem_cons = [c for c in cons_list if c in state.remaining]
        if len(rem_cons) < 2:
            continue
        # Only cluster if producer is NOT in remaining (t already computed)
        prod_op = producer.get(t)
        if prod_op is not None and prod_op in state.remaining:
            continue
        sorted_cons = sorted(rem_cons, key=lambda x: topo_pos.get(x, 0))
        key = frozenset(sorted_cons)
        if key in seen: continue
        seen.add(key)
        # Verify all consumers are ready (their non-t inputs resolved)
        cons_set = set(sorted_cons)
        all_ready = True
        for coid in sorted_cons:
            for inp in state.ops[coid].inputs:
                p = producer.get(inp, -1)
                if p in state.remaining and p not in cons_set:
                    all_ready = False; break
            if not all_ready: break
        if not all_ready: continue
        proposed_cids.append(state._register(sorted_cons, "cluster"))

    return proposed_cids


def _fanout_join_solver(state):
    """Generalised fanout/join solver.

    Scans the remaining ops for intermediate tensors that are produced by one op
    and consumed by 2+ ops (fanout pattern), or tensors that feed into a join op
    (all producers scheduled, one consumer remaining — join pattern).

    For each such tensor, evaluates three strategies over a 2-4 step local
    neighbourhood and proposes the best as a candidate:
      - keep:       retain the tensor in fast memory across all consumers (no reload)
      - spill:      write to slow memory after first use; reload before each consumer
      - recompute:  fuse the producer into each consumer step (recomputation)

    Returns list of candidate ids registered in state (may be empty).
    Pure graph arithmetic — no _gen_grans calls; WS feasibility deferred to eval.
    """
    producer = {}
    consumers_map = {}
    for op in state.ops:
        for t in op.outputs:
            producer[t] = op.idx
        for t in op.inputs:
            consumers_map.setdefault(t, []).append(op.idx)

    topo_pos = {oid: i for i, oid in enumerate(state.topo)}
    B_slow = max(state.prob.slow_memory_bandwidth, 1e-9)

    proposed_cids = []
    seen = set()

    for t, cons_list in consumers_map.items():
        prod_oid = producer.get(t)
        if prod_oid is None:
            continue  # model input — not recomputable
        rem_cons = [c for c in cons_list if c in state.remaining]
        if not rem_cons:
            continue

        t_bytes = state.prob.widths[t] * state.prob.heights[t] * ELEMENT_BYTES

        # ── Fanout pattern: producer not yet scheduled, 2+ remaining consumers ──
        if prod_oid in state.remaining and len(rem_cons) >= 2:
            sorted_cons = sorted(rem_cons, key=lambda x: topo_pos.get(x, 0))
            key = ("fanout", prod_oid, frozenset(sorted_cons))
            if key in seen:
                continue
            seen.add(key)

            # Strategy A — keep: fuse producer with ALL consumers into one step
            all_ops = [prod_oid] + sorted_cons
            all_ops_sorted = sorted(all_ops, key=lambda x: topo_pos.get(x, 0))
            # Feasibility: check no external dep blocks the fusion
            all_set = set(all_ops_sorted)
            feasible = True
            for oid in all_ops_sorted:
                for inp in state.ops[oid].inputs:
                    p = producer.get(inp, -1)
                    if p in state.remaining and p not in all_set:
                        feasible = False; break
                if not feasible: break
            if feasible:
                proposed_cids.append(state._register(all_ops_sorted, "fanout_keep"))

            # Strategy B — recompute: fuse producer into EACH consumer separately
            # Each consumer gets its own copy of the producer (recompute)
            prod_ready = all(
                producer.get(inp, -1) not in state.remaining
                for inp in state.ops[prod_oid].inputs
                if inp in producer
            )
            if prod_ready:
                for c in sorted_cons:
                    pair = sorted([prod_oid, c], key=lambda x: topo_pos.get(x, 0))
                    pair_set = frozenset(pair)
                    if pair_set not in seen:
                        seen.add(pair_set)
                        # Only register if consumer has no other unresolved deps
                        c_ready = all(
                            producer.get(inp, -1) not in state.remaining
                            or producer.get(inp, -1) == prod_oid
                            for inp in state.ops[c].inputs
                            if inp in producer
                        )
                        if c_ready:
                            proposed_cids.append(state._register(pair, "fanout_recompute"))

        # ── Join pattern: producer already scheduled, 2+ remaining consumers ──
        elif prod_oid not in state.remaining and len(rem_cons) >= 2:
            sorted_cons = sorted(rem_cons, key=lambda x: topo_pos.get(x, 0))
            key = ("join", t, frozenset(sorted_cons))
            if key in seen:
                continue
            seen.add(key)

            # Estimate spill cost: write once + reload per consumer
            spill_cost = t_bytes / B_slow * len(rem_cons)

            if spill_cost > 0:
                # Prefer clustering consumers — already handled by _cluster_consumers
                # but add as explicit join candidate if they're topo-consecutive
                cons_set = set(sorted_cons)
                feasible = True
                for c in sorted_cons:
                    for inp in state.ops[c].inputs:
                        p = producer.get(inp, -1)
                        if p in state.remaining and p not in cons_set:
                            feasible = False; break
                    if not feasible: break
                if feasible:
                    key2 = frozenset(sorted_cons)
                    if key2 not in seen:
                        seen.add(key2)
                        proposed_cids.append(state._register(sorted_cons, "join_keep"))

    return proposed_cids


def tool_structural_local_search(state, time_budget_s=45.0):
    """Structural local search moves on the committed schedule.

    Attempts four classes of structural moves in order of cheapness:
      1. Swap independent consecutive steps (no data dependency between them) —
         reorders two adjacent steps to improve residency (minimise reload churn).
      2. Merge adjacent subgraphs — fuse two consecutive steps into one if feasible.
      3. Split at boundary tensor — split a multi-op step at its largest boundary
         output tensor to reduce peak WS and enable better tiling.
      4. Flip spill↔recompute — for a step that uses a tensor retained from prior,
         try dropping the retention (force reload) and see if the prior step benefits
         from a lower-WS granularity that reduces its own latency.

    After each accepted structural move, re-optimises granularity + traversal for
    affected steps only (not the whole schedule).

    Returns dict with improvement count and total latency.
    """
    if not state.schedule_subgraphs:
        return {"improvements": 0, "total_latency": 0}

    ls_start = time.time()
    improvements = 0
    n = len(state.schedule_subgraphs)

    # Build dependency sets for quick independence checks
    # produced_by_step[s] = set of tensor ids produced (boundary output) by step s
    # consumed_by_step[s] = set of tensor ids consumed (boundary input) by step s
    def _step_boundary(s):
        so = [state.ops[x] for x in state.schedule_subgraphs[s]]
        b_in, b_out = find_boundary_tensors(so, state.ops)
        return b_in, b_out

    def _reopt_step(s):
        """Re-optimise gran+traversal for a single step in-place."""
        so = [state.ops[x] for x in state.schedule_subgraphs[s]]
        b_in, b_out = find_boundary_tensors(so, state.ops)
        retain = set(state.schedule_retain[s])
        retained_prior = set(state.schedule_retain[s-1]) if s > 0 else set()
        grans_ws = _gen_grans(so, b_in, b_out, state.prob, top_k=8)
        best_lat = state.schedule_lat[s]
        best_gran = state.schedule_gran[s]
        best_trav = state.schedule_trav[s]
        for g, ws in grans_ws:
            w, h, kk = g
            trav, lat = _best_traversal(so, b_in, b_out, w, h, kk, state.prob,
                                         retain_after=retain, retained_from_prior=retained_prior)
            if lat < best_lat:
                best_lat = lat; best_gran = g; best_trav = trav
        state.schedule_lat[s] = best_lat
        state.schedule_gran[s] = list(best_gran)
        state.schedule_trav[s] = best_trav

    # ── Move 1: Swap independent consecutive steps ──
    s = 0
    while s < len(state.schedule_subgraphs) - 1:
        if time.time() - ls_start > time_budget_s: break
        b_in_s,  b_out_s  = _step_boundary(s)
        b_in_s1, b_out_s1 = _step_boundary(s + 1)
        # Also include orphan-promoted tensors: ephemeral within a step but consumed by the other.
        # find_boundary_tensors omits these; they are written to slow memory implicitly.
        ops_s_objs  = [state.ops[x] for x in state.schedule_subgraphs[s]]
        ops_s1_objs = [state.ops[x] for x in state.schedule_subgraphs[s + 1]]
        eph_s  = {t for op in ops_s_objs  for t in op.outputs} & {t for op in ops_s_objs  for t in op.inputs}
        eph_s1 = {t for op in ops_s1_objs for t in op.outputs} & {t for op in ops_s1_objs for t in op.inputs}
        # Independent if neither produces tensors the other needs
        dep_s_uses_s1 = bool(b_out_s1 & b_in_s)  or bool(eph_s1 & b_in_s)   # s+1 produces something s needs
        dep_s1_uses_s = bool(b_out_s  & b_in_s1) or bool(eph_s  & b_in_s1)   # s produces something s+1 needs
        if not dep_s_uses_s1 and not dep_s1_uses_s:
            # Try swap: compute latency gain from better residency ordering
            # After swap: step s+1's inputs may be resident from the step before s
            prior_resident = set(state.schedule_retain[s-1]) if s > 0 else set()
            # Latency of (s+1) running first with prior_resident, then s running after
            ops_s1 = [state.ops[x] for x in state.schedule_subgraphs[s+1]]
            ops_s  = [state.ops[x] for x in state.schedule_subgraphs[s]]
            bi_s1, bo_s1 = b_in_s1, b_out_s1
            bi_s,  bo_s  = b_in_s,  b_out_s
            ret_s1 = set(state.schedule_retain[s+1])
            ret_s  = set(state.schedule_retain[s])
            # Current latency
            cur_lat = state.schedule_lat[s] + state.schedule_lat[s+1]
            # Swapped latency: s+1 first (with prior_resident), then s (with bo_s1 as retained)
            _, lat_s1_first = _best_traversal(ops_s1, bi_s1, bo_s1,
                                               *state.schedule_gran[s+1][:3], state.prob,
                                               retain_after=ret_s1, retained_from_prior=prior_resident)
            new_retained = prior_resident | (bo_s1 & ret_s1)
            _, lat_s_after = _best_traversal(ops_s, bi_s, bo_s,
                                              *state.schedule_gran[s][:3], state.prob,
                                              retain_after=ret_s, retained_from_prior=new_retained)
            if lat_s1_first + lat_s_after < cur_lat - 0.5:
                # Accept swap
                for attr in ("schedule_subgraphs", "schedule_gran", "schedule_retain",
                             "schedule_trav", "schedule_lat"):
                    lst = getattr(state, attr)
                    lst[s], lst[s+1] = lst[s+1], lst[s]
                state.schedule_lat[s]   = lat_s1_first
                state.schedule_lat[s+1] = lat_s_after
                improvements += 1
        s += 1

    # ── Move 2: Merge adjacent subgraphs ──
    s = 0
    while s < len(state.schedule_subgraphs) - 1:
        if time.time() - ls_start > time_budget_s: break
        merged_ops = state.schedule_subgraphs[s] + state.schedule_subgraphs[s+1]
        # Topo-sort the merged ops
        merged_ops_sorted = [x for x in state.topo if x in set(merged_ops)]
        ops_objs = [state.ops[x] for x in merged_ops_sorted]
        b_in, b_out = find_boundary_tensors(ops_objs, state.ops)
        # Promote ephemeral tensors needed by later steps to boundary outputs
        merged_set = set(merged_ops_sorted)
        produced_m = set(); consumed_m = set()
        for op in ops_objs: produced_m.update(op.outputs); consumed_m.update(op.inputs)
        ephemeral_m = produced_m & consumed_m
        if ephemeral_m:
            for later_s in range(s+2, len(state.schedule_subgraphs)):
                for oid in state.schedule_subgraphs[later_s]:
                    for t in state.ops[oid].inputs:
                        if t in ephemeral_m:
                            b_out.add(t)
        grans_ws = _gen_grans(ops_objs, b_in, b_out, state.prob, top_k=4)
        if not grans_ws:
            s += 1; continue
        cur_lat = state.schedule_lat[s] + state.schedule_lat[s+1]
        retained_prior = set(state.schedule_retain[s-1]) if s > 0 else set()
        # Compute best latency for merged step
        best_lat = float('inf'); best_gran = None; best_trav = None
        for g, ws in grans_ws:
            w, h, kk = g
            trav, lat = _best_traversal(ops_objs, b_in, b_out, w, h, kk, state.prob,
                                         retain_after=set(state.schedule_retain[s+1]),
                                         retained_from_prior=retained_prior)
            if lat < best_lat:
                best_lat = lat; best_gran = g; best_trav = trav
        if best_lat < cur_lat - 0.5:
            # Accept merge: replace steps s and s+1 with merged step
            retain_merged = list(state.schedule_retain[s+1])  # keep downstream retention
            state.schedule_subgraphs[s] = merged_ops_sorted
            state.schedule_gran[s]      = list(best_gran)
            state.schedule_retain[s]    = retain_merged
            state.schedule_trav[s]      = best_trav
            state.schedule_lat[s]       = best_lat
            del state.schedule_subgraphs[s+1]
            del state.schedule_gran[s+1]
            del state.schedule_retain[s+1]
            del state.schedule_trav[s+1]
            del state.schedule_lat[s+1]
            improvements += 1
            # Don't advance s — try merging the newly merged step with s+1
        else:
            s += 1

    # ── Move 3: Split a high-WS step at its largest boundary output tensor ──
    for s in range(len(state.schedule_subgraphs)):
        if time.time() - ls_start > time_budget_s: break
        step_ops_ids = state.schedule_subgraphs[s]
        if len(step_ops_ids) < 2: continue
        so = [state.ops[x] for x in step_ops_ids]
        b_in, b_out = find_boundary_tensors(so, state.ops)
        w, h, kk = state.schedule_gran[s]
        ws = working_set_bytes_for_subgraph(so, b_in, b_out, w, h, kk, state.prob)
        if ws <= state.prob.fast_memory_capacity * 0.75: continue  # only split if high WS

        # Find a split point: largest boundary output tensor whose producer splits the step
        producer_map = {}
        for op in so:
            for t in op.outputs: producer_map[t] = op.idx
        best_split_t = None; best_split_bytes = 0
        for t in b_out:
            tb = state.prob.widths[t] * state.prob.heights[t] * ELEMENT_BYTES
            if tb > best_split_bytes:
                best_split_bytes = tb; best_split_t = t

        if best_split_t is None: continue
        prod_oid = producer_map.get(best_split_t)
        if prod_oid is None: continue

        # Split: step A = ops up to and including prod_oid's topo position
        # step B = remaining ops
        topo_pos = {x: i for i, x in enumerate(state.topo)}
        split_point = topo_pos.get(prod_oid, 0)
        ops_a = [x for x in step_ops_ids if topo_pos.get(x, 0) <= split_point]
        ops_b = [x for x in step_ops_ids if topo_pos.get(x, 0) >  split_point]
        if not ops_a or not ops_b: continue

        ops_a_objs = [state.ops[x] for x in ops_a]
        ops_b_objs = [state.ops[x] for x in ops_b]
        bi_a, bo_a = find_boundary_tensors(ops_a_objs, state.ops)
        bi_b, bo_b = find_boundary_tensors(ops_b_objs, state.ops)
        # Promote ephemeral tensors from ops_a that are consumed by ops_b
        eph_a = {t for op in ops_a_objs for t in op.outputs} & {t for op in ops_a_objs for t in op.inputs}
        cons_b = {t for op in ops_b_objs for t in op.inputs}
        bo_a = bo_a | (eph_a & cons_b)

        grans_a = _gen_grans(ops_a_objs, bi_a, bo_a, state.prob, top_k=3)
        grans_b = _gen_grans(ops_b_objs, bi_b, bo_b, state.prob, top_k=3)
        if not grans_a or not grans_b: continue

        retained_prior = set(state.schedule_retain[s-1]) if s > 0 else set()
        # Best latency for each half
        lat_a = float('inf'); gran_a = None; trav_a = None
        for g, _ in grans_a:
            trav, lat = _best_traversal(ops_a_objs, bi_a, bo_a, *g, state.prob,
                                         retain_after={best_split_t},
                                         retained_from_prior=retained_prior)
            if lat < lat_a: lat_a = lat; gran_a = g; trav_a = trav

        lat_b = float('inf'); gran_b = None; trav_b = None
        for g, _ in grans_b:
            trav, lat = _best_traversal(ops_b_objs, bi_b, bo_b, *g, state.prob,
                                         retain_after=set(state.schedule_retain[s]),
                                         retained_from_prior={best_split_t})
            if lat < lat_b: lat_b = lat; gran_b = g; trav_b = trav

        if lat_a + lat_b < state.schedule_lat[s] - 0.5:
            # Accept split: insert new step at s+1
            orig_retain = list(state.schedule_retain[s])
            state.schedule_subgraphs[s] = ops_a
            state.schedule_gran[s]      = list(gran_a)
            state.schedule_retain[s]    = [best_split_t]
            state.schedule_trav[s]      = trav_a
            state.schedule_lat[s]       = lat_a
            state.schedule_subgraphs.insert(s+1, ops_b)
            state.schedule_gran.insert(s+1, list(gran_b))
            state.schedule_retain.insert(s+1, orig_retain)
            state.schedule_trav.insert(s+1, trav_b)
            state.schedule_lat.insert(s+1, lat_b)
            improvements += 1

    # ── Move 4: Flip spill↔recompute ──
    # For each step that retains tensors, try dropping the retention
    # and re-optimise the prior step without retention overhead.
    for s in range(1, len(state.schedule_subgraphs)):
        if time.time() - ls_start > time_budget_s: break
        retained = set(state.schedule_retain[s-1])
        if not retained: continue
        # Current combined latency of step s-1 and s
        cur_lat = state.schedule_lat[s-1] + state.schedule_lat[s]
        # Try: drop retention from step s-1, let step s reload
        so_prev = [state.ops[x] for x in state.schedule_subgraphs[s-1]]
        so_curr = [state.ops[x] for x in state.schedule_subgraphs[s]]
        bi_prev, bo_prev = find_boundary_tensors(so_prev, state.ops)
        bi_curr, bo_curr = find_boundary_tensors(so_curr, state.ops)
        retained_prior_prev = set(state.schedule_retain[s-2]) if s > 1 else set()

        # Re-eval step s-1 with empty retention
        grans_prev = _gen_grans(so_prev, bi_prev, bo_prev, state.prob, top_k=4)
        lat_prev_no_retain = float('inf'); gran_prev_nr = None; trav_prev_nr = None
        for g, _ in grans_prev:
            trav, lat = _best_traversal(so_prev, bi_prev, bo_prev, *g, state.prob,
                                         retain_after=set(),
                                         retained_from_prior=retained_prior_prev)
            if lat < lat_prev_no_retain:
                lat_prev_no_retain = lat; gran_prev_nr = g; trav_prev_nr = trav

        # Re-eval step s with no retained_from_prior (forced reload)
        grans_curr = _gen_grans(so_curr, bi_curr, bo_curr, state.prob, top_k=4)
        lat_curr_reload = float('inf'); gran_curr_rl = None; trav_curr_rl = None
        for g, _ in grans_curr:
            trav, lat = _best_traversal(so_curr, bi_curr, bo_curr, *g, state.prob,
                                         retain_after=set(state.schedule_retain[s]),
                                         retained_from_prior=set())
            if lat < lat_curr_reload:
                lat_curr_reload = lat; gran_curr_rl = g; trav_curr_rl = trav

        if lat_prev_no_retain + lat_curr_reload < cur_lat - 0.5:
            # Flip accepted: drop retention, gain from better gran on step s-1
            state.schedule_retain[s-1]  = []
            state.schedule_gran[s-1]    = list(gran_prev_nr)
            state.schedule_trav[s-1]    = trav_prev_nr
            state.schedule_lat[s-1]     = lat_prev_no_retain
            state.schedule_gran[s]      = list(gran_curr_rl)
            state.schedule_trav[s]      = trav_curr_rl
            state.schedule_lat[s]       = lat_curr_reload
            improvements += 1

    return {"improvements": improvements, "total_latency": round(sum(state.schedule_lat), 2)}


def tool_local_search(state, time_budget_s=30.0):
    """Post-schedule local search: try alternative granularities + traversals on highest-latency steps."""
    if not state.schedule_subgraphs:
        return {"improvements": 0, "total_latency": 0}
    ls_start = time.time(); improvements = 0
    n_steps = len(state.schedule_subgraphs)
    per_step_budget = max(1.0, time_budget_s / max(n_steps, 1))
    step_order = sorted(range(n_steps),
                        key=lambda i: state.schedule_lat[i], reverse=True)

    # Phase 1: Gran/traversal search on each step (highest-latency first)
    for s in step_order:
        if time.time() - ls_start > time_budget_s: break
        step_start = time.time()
        step_ops = [state.ops[x] for x in state.schedule_subgraphs[s]]
        b_in, b_out = find_boundary_tensors(step_ops, state.ops)
        current_lat = state.schedule_lat[s]
        current_retain = set(state.schedule_retain[s])
        retained_prior = set(state.schedule_retain[s-1]) if s > 0 else set()
        best_lat, best_gran, best_trav = current_lat, state.schedule_gran[s], state.schedule_trav[s]
        grans_ws = _gen_grans(step_ops, b_in, b_out, state.prob, top_k=40)
        for (g, ws) in grans_ws:
            now = time.time()
            if now - ls_start > time_budget_s or now - step_start > per_step_budget: break
            tw, th, tk = g
            trav, lat = _best_traversal(step_ops, b_in, b_out, tw, th, tk, state.prob,
                                         retain_after=current_retain, retained_from_prior=retained_prior)
            if lat < best_lat:
                best_lat = lat; best_gran = [tw, th, tk]; best_trav = trav
        if best_lat < current_lat:
            state.schedule_gran[s] = list(best_gran)
            state.schedule_lat[s] = best_lat
            state.schedule_trav[s] = best_trav
            improvements += 1

    # Phase 2: Retention re-evaluation — for adjacent step pairs, try adding
    # or removing retention to see if the combined latency improves.
    for s in range(len(state.schedule_subgraphs) - 1):
        if time.time() - ls_start > time_budget_s: break
        cur_retain = set(state.schedule_retain[s])
        so_s = [state.ops[x] for x in state.schedule_subgraphs[s]]
        so_s1 = [state.ops[x] for x in state.schedule_subgraphs[s + 1]]
        bi_s, bo_s = find_boundary_tensors(so_s, state.ops)
        bi_s1, bo_s1 = find_boundary_tensors(so_s1, state.ops)
        retained_prior_s = set(state.schedule_retain[s - 1]) if s > 0 else set()
        cur_pair_lat = state.schedule_lat[s] + state.schedule_lat[s + 1]

        # Try toggling: for each boundary output of step s that's also a boundary
        # input of step s+1, try retaining it (if not retained) or dropping it.
        toggle_candidates = bo_s & bi_s1
        if not toggle_candidates:
            continue

        for t in toggle_candidates:
            if time.time() - ls_start > time_budget_s: break
            new_retain = set(cur_retain)
            if t in cur_retain:
                new_retain.discard(t)
            else:
                new_retain.add(t)
            # Re-eval step s with new retention
            gs = state.schedule_gran[s]
            trav_s, lat_s = _best_traversal(so_s, bi_s, bo_s, *gs, state.prob,
                                             retain_after=new_retain,
                                             retained_from_prior=retained_prior_s)
            # Re-eval step s+1 with new retained_from_prior
            gs1 = state.schedule_gran[s + 1]
            retain_s1 = set(state.schedule_retain[s + 1])
            trav_s1, lat_s1 = _best_traversal(so_s1, bi_s1, bo_s1, *gs1, state.prob,
                                               retain_after=retain_s1,
                                               retained_from_prior=new_retain)
            if lat_s + lat_s1 < cur_pair_lat - 0.5:
                state.schedule_retain[s] = list(new_retain)
                state.schedule_lat[s] = lat_s
                state.schedule_trav[s] = trav_s
                state.schedule_lat[s + 1] = lat_s1
                state.schedule_trav[s + 1] = trav_s1
                cur_retain = new_retain
                cur_pair_lat = lat_s + lat_s1
                improvements += 1

    return {"improvements": improvements, "total_latency": sum(state.schedule_lat)}

# ── Solution Output ──

def materialize_solution(state):
    return {"subgraphs": state.schedule_subgraphs, "granularities": state.schedule_gran,
            "tensors_to_retain": state.schedule_retain, "traversal_orders": state.schedule_trav,
            "subgraph_latencies": state.schedule_lat}

def validate_solution(sol, prob, ops):
    n = len(sol["subgraphs"])
    assert len(sol["granularities"]) == n
    assert len(sol["tensors_to_retain"]) == n
    assert len(sol["subgraph_latencies"]) == n
    covered = set()
    for step in sol["subgraphs"]:
        for oid in step: assert 0 <= oid < len(ops); covered.add(oid)
    assert len(covered) == len(ops), "Must cover every op"
    # Tensor availability check: every boundary input must be in slow or fast memory
    # Account for promoted ephemeral tensors (produced+consumed internally but also
    # consumed by later steps — promoted to boundary outputs so they're written to slow)
    producer_of = {}
    for op in ops:
        for t in op.outputs: producer_of[t] = op.idx
    graph_inputs = set(range(len(prob.widths))) - set(producer_of.keys())
    # Build set of tensors consumed by each step for promotion detection
    step_consumed = []
    for s in range(n):
        sc = set()
        for oid in sol["subgraphs"][s]: sc.update(ops[oid].inputs)
        step_consumed.append(sc)
    in_slow = set(graph_inputs); in_fast = set()
    for s in range(n):
        so = [ops[x] for x in sol["subgraphs"][s]]
        w, h, kk = sol["granularities"][s]
        bi, bo = find_boundary_tensors(so, ops)
        # Promote ephemeral tensors consumed by later steps
        produced_s = set(); consumed_s = set()
        for op in so: produced_s.update(op.outputs); consumed_s.update(op.inputs)
        ephemeral_s = produced_s & consumed_s
        if ephemeral_s:
            for later in range(s+1, n):
                for t in ephemeral_s:
                    if t in step_consumed[later]:
                        bo.add(t)
        for t in bi:
            assert t in in_slow or t in in_fast, \
                f"Step {s}: tensor {t} not available (ephemeral in prior step)"
        ws = working_set_bytes_for_subgraph(so, bi, bo, int(w), int(h), max(1,int(kk)), prob)
        assert ws <= prob.fast_memory_capacity, f"OOM step {s}: ws={ws}"
        retain = set(sol["tensors_to_retain"][s]) if s < len(sol["tensors_to_retain"]) else set()
        for t in bo:
            in_slow.add(t)
        new_fast = set()
        for t in retain:
            if t in bo or t in in_fast:
                new_fast.add(t)
        in_fast = new_fast

def compute_step_metrics(state):
    total_lat = sum(state.schedule_lat) or 1e-9; metrics = []
    for s in range(len(state.schedule_subgraphs)):
        so = [state.ops[x] for x in state.schedule_subgraphs[s]]
        bi, bo = find_boundary_tensors(so, state.ops)
        w, h, kk = state.schedule_gran[s]; lat = state.schedule_lat[s]
        ws = working_set_bytes_for_subgraph(so, bi, bo, w, h, kk, state.prob)
        ws_pct = ws / max(state.prob.fast_memory_capacity, 1) * 100
        ot = {}
        for op in so: ot[op.op_type] = ot.get(op.op_type, 0) + 1
        ow = max((state.prob.widths[t] for t in bo), default=w)
        oh = max((state.prob.heights[t] for t in bo), default=h)
        tw2, th2 = tile_count(ow, w), tile_count(oh, h)
        sk = 1
        for op in so:
            if op.op_type == "MatMul": sk = max(sk, math.ceil(matmul_reduction_extent(op, state.prob)/max(1,kk)))
        B = state.prob.slow_memory_bandwidth
        ce = sum(op.base_cost for op in so) * tw2 * th2
        mb = 0
        for t in bi:
            for op in so:
                if t in op.inputs: mb += tensor_slice_elems_for_role(op,t,w,h,kk,state.prob)*tw2*th2; break
        for t in bo:
            for op in so:
                if t in op.outputs: mb += tensor_slice_elems_for_role(op,t,w,h,kk,state.prob)*tw2*th2; break
        me = mb / max(B, 1e-9)
        bn = "compute" if ce >= me else "memory"
        metrics.append({"step": s, "ops": list(state.schedule_subgraphs[s]), "op_types": ot,
            "n_ops": len(so), "gran": [w,h,kk], "latency": round(lat,2),
            "pct_of_total": round(lat/total_lat*100,1), "cumulative_pct": 0.0,
            "bottleneck": bn, "compute_time_est": round(ce,1), "memory_time_est": round(me,1),
            "mem_bytes_in": 0, "mem_bytes_out": 0, "tiles": f"{tw2}x{th2}={tw2*th2}",
            "split_k": sk, "total_iterations": tw2*th2*sk, "ws_bytes": ws,
            "ws_pct": round(ws_pct,1), "retained_in": 0, "retained_out": 0,
            "has_traversal": state.schedule_trav[s] is not None})
    cum = 0.0
    si = sorted(range(len(metrics)), key=lambda i: metrics[i]["latency"], reverse=True)
    for rank, idx in enumerate(si):
        cum += metrics[idx]["pct_of_total"]
        metrics[idx]["cumulative_pct"] = round(cum,1); metrics[idx]["latency_rank"] = rank+1
    return metrics

# ── Traces ──

def _trace_dir():
    d = Path(__file__).resolve().parent / "traces"; d.mkdir(exist_ok=True); return d

def _wt(td, name, obj):
    if td:
        try: (td/name).write_text(json.dumps(obj, indent=2, default=str))
        except Exception: pass

def _wtt(td, name, text):
    if td:
        try: (td/name).write_text(text)
        except Exception: pass

# ── Main Agent Loop ──

def _load_system_prompt():
    p = Path(__file__).resolve().parent / "prompts" / "system_prompt.txt"
    if p.exists(): return p.read_text().strip()
    return (
        "You are a scheduler for computational graph fusion on an accelerator with limited fast memory.\n"
        "You receive pre-evaluated candidate subgraphs ranked by estimated latency.\n"
        "Each option shows: cid, ops, gran [w,h,k], lat, ws_bytes.\n"
        "Pick the best candidate and call apply_step.\n"
        "Key: fusion eliminates intermediate IO; larger tiles = fewer iterations but bigger WS; "
        "for MatMul, smaller k = more split-k steps but fits memory.\n"
        'Respond: {"calls": [{"tool": "apply_step", "args": {"cid": <int>, "gran": [w,h,k]}}]}'
    )

SYSTEM_PROMPT = _load_system_prompt()

def _llm_available():
    """Return True only when Gemini API key is present and genai is importable."""
    return (genai is not None) and bool(os.environ.get("GOOGLE_API_KEY", ""))


def _is_confident(ranked, time_remaining, remaining_ops):
    """Decide if we can skip the LLM and auto-apply the top candidate.
    Always auto-applies when LLM is unavailable (no API key).
    """
    if not ranked: return False
    # No API key — always auto-apply, never hang waiting for LLM
    if not _llm_available(): return True
    # Only 1 feasible option
    if len(ranked) == 1: return True
    # Time pressure
    if time_remaining < 90: return True
    # Top candidate dominates by >30% — confident enough to skip LLM
    if ranked[0]["lat"] <= ranked[1]["lat"] * 0.70: return True
    # Estimate LLM cost: ~12s per step; only use LLM when genuinely affordable
    steps_left = remaining_ops  # rough upper bound
    if (steps_left * 12) >= (time_remaining * 0.6): return True
    return False

# ── Strategy Selection System ──

STRATEGIES = {
    "aggressive_fusion": {
        "description": "Maximize fusion: large subgraphs (up to 8 ops), fewer steps. "
                       "Best when internal edges dominate IO and memory allows large fusions. "
                       "Reduces boundary transfers but may force smaller tiles.",
        "max_len": 8, "gran_top_k": 8, "retention_pct": 0.70,
        "beam_window": 10, "beam_width": 8,
    },
    "conservative": {
        "description": "Fine-grained: small subgraphs (1-2 ops per step), many steps. "
                       "Best when memory is very tight (large tensors vs cap) and optimal "
                       "granularity per op matters more than fusion savings.",
        "max_len": 2, "gran_top_k": 20, "retention_pct": 0.50,
        "beam_window": 4, "beam_width": 4,
    },
    "balanced": {
        "description": "Balanced approach: moderate fusion (up to 6 ops), moderate granularity search. "
                       "Good default for most graphs.",
        "max_len": 6, "gran_top_k": 10, "retention_pct": 0.70,
        "beam_window": 8, "beam_width": 6,
    },
    "memory_optimized": {
        "description": "Maximize tensor retention: keep 90% of fast memory for retained tensors. "
                       "Best when there are many diamonds/fanouts and reload cost dominates. "
                       "Sacrifices some WS budget for granularity to keep more tensors resident.",
        "max_len": 6, "gran_top_k": 10, "retention_pct": 0.90,
        "beam_window": 8, "beam_width": 6,
    },
    "compute_optimized": {
        "description": "Minimize padding/split-K overhead: prefer native-or-larger tiles, "
                       "moderate fusion. Best for compute-bound graphs where padding waste "
                       "from sub-native tiles is the main latency contributor.",
        "max_len": 4, "gran_top_k": 15, "retention_pct": 0.60,
        "beam_window": 6, "beam_width": 6,
    },
}


def _compute_graph_metrics(state):
    """Analyze the graph structure and problem characteristics (fast, O(ops+edges))."""
    prob, ops = state.prob, state.ops
    B = max(prob.slow_memory_bandwidth, 1e-9)

    n_ops = len(ops)
    n_mm = sum(1 for op in ops if op.op_type == "MatMul")

    # Build producer/consumer maps
    producer = {}
    consumers_map = {}
    for op in ops:
        for t in op.outputs: producer[t] = op.idx
        for t in op.inputs: consumers_map.setdefault(t, []).append(op.idx)

    # External IO (unavoidable)
    ext_inputs = set()
    for op in ops:
        for t in op.inputs:
            if t not in producer: ext_inputs.add(t)
    all_consumed = set()
    for op in ops: all_consumed.update(op.inputs)
    ext_outputs = {t for op in ops for t in op.outputs if t not in all_consumed}

    ext_in_bytes = sum(prob.widths[t] * prob.heights[t] * ELEMENT_BYTES for t in ext_inputs)
    ext_out_bytes = sum(prob.widths[t] * prob.heights[t] * ELEMENT_BYTES for t in ext_outputs)

    # Internal edges (fusion potential)
    internal_edges = set()
    for op in ops:
        for t in op.inputs:
            if t in producer: internal_edges.add(t)
    int_bytes = sum(prob.widths[t] * prob.heights[t] * ELEMENT_BYTES for t in internal_edges)

    # Graph structure
    max_fanout = max((len(consumers_map.get(t, [])) for t in producer), default=1)
    n_diamonds = sum(1 for t in producer if len(consumers_map.get(t, [])) >= 2)

    # Chain depth via topo
    depths = {}
    for oid in state.topo:
        d = 0
        for t in ops[oid].inputs:
            p = producer.get(t)
            if p is not None and p in depths:
                d = max(d, depths[p] + 1)
        depths[oid] = d
    chain_depth = max(depths.values()) if depths else 0

    # Memory pressure
    all_tensors = set()
    for op in ops: all_tensors.update(op.inputs); all_tensors.update(op.outputs)
    tensor_sizes = sorted([prob.widths[t] * prob.heights[t] * ELEMENT_BYTES for t in all_tensors],
                          reverse=True)
    largest_tensor = tensor_sizes[0] if tensor_sizes else 0
    avg_tensor = sum(tensor_sizes) / max(len(tensor_sizes), 1)

    # Compute intensity
    total_compute = sum(op.base_cost for op in ops)
    total_io_bytes = ext_in_bytes + ext_out_bytes + int_bytes * 2
    arith_intensity = total_compute / max(total_io_bytes / B, 1e-9)

    mem_pressure = largest_tensor / max(prob.fast_memory_capacity, 1)
    fusion_potential = int_bytes / max(int_bytes + ext_in_bytes + ext_out_bytes, 1) * 100

    return {
        "n_ops": n_ops, "n_matmul": n_mm, "n_pointwise": n_ops - n_mm,
        "chain_depth": chain_depth, "max_fanout": max_fanout, "n_diamonds": n_diamonds,
        "ext_in_bytes": ext_in_bytes, "ext_out_bytes": ext_out_bytes,
        "internal_edge_bytes": int_bytes,
        "fusion_potential_pct": round(fusion_potential, 1),
        "total_compute": total_compute,
        "arithmetic_intensity": round(arith_intensity, 3),
        "largest_tensor": largest_tensor,
        "mem_pressure": round(mem_pressure, 2),
        "avg_tensor_vs_cap": round(avg_tensor / max(prob.fast_memory_capacity, 1), 3),
        "top5_tensor_sizes": tensor_sizes[:5],
        "fast_memory_capacity": prob.fast_memory_capacity,
        "slow_memory_bandwidth": prob.slow_memory_bandwidth,
        "native_granularity": list(prob.native_granularity),
    }


def _probe_schedule(state):
    """Evaluate the first batch of candidates WITHOUT modifying state.

    Proposes and evaluates candidates for the first scheduling step to measure
    performance characteristics: fusion ratio, WS pressure, bottleneck type.
    Candidates are registered in state.candidates but that's fine — the main
    loop clears them immediately via tool_propose_fusion_candidates.
    """
    prop = tool_propose_fusion_candidates(state, k=20, max_len=6)
    cids = prop["candidate_ids"]
    if not cids:
        return {"probe_steps": 0}

    recomp = tool_try_recomputation(state)
    if recomp["candidate_ids"]:
        cids = cids + recomp["candidate_ids"]

    evals = tool_evaluate_step_bundle(state, candidate_ids=cids, granularity_k=8)
    ranked = evals["ranked"]
    if not ranked:
        return {"probe_steps": 0}

    top = ranked[:min(10, len(ranked))]
    best = ranked[0]
    B = max(state.prob.slow_memory_bandwidth, 1e-9)

    avg_lat = sum(r["lat"] for r in top) / len(top)
    avg_fusion = sum(r["fusion_ratio"] for r in top) / len(top)
    avg_nops = sum(r["n_ops"] for r in top) / len(top)
    avg_ws = sum(r["ws_bytes"] for r in top) / len(top)
    ws_pressure = avg_ws / max(state.prob.fast_memory_capacity, 1)

    mem_time = (best["bytes_in"] + best["bytes_out"]) / B
    comp_time = sum(state.ops[oid].base_cost for oid in best["ops"])
    bottleneck = "compute" if comp_time > mem_time else "memory"

    spread = ranked[-1]["lat"] / max(ranked[0]["lat"], 1e-9) if len(ranked) >= 2 else 1.0

    return {
        "probe_steps": 1,
        "n_candidates_evaluated": len(ranked),
        "best_latency": best["lat"],
        "best_n_ops": best["n_ops"],
        "best_gran": best["gran"],
        "best_fusion_ratio": best["fusion_ratio"],
        "avg_latency_top10": round(avg_lat, 1),
        "avg_fusion_ratio": round(avg_fusion, 3),
        "avg_ops_per_step": round(avg_nops, 1),
        "ws_pressure": round(ws_pressure, 3),
        "bottleneck": bottleneck,
        "candidate_spread": round(spread, 2),
    }


def _select_strategy(metrics, probe, time_budget_s):
    """Ask the LLM to pick a scheduling strategy based on graph metrics and probe results.
    Falls back to heuristics when LLM is unavailable."""
    global _STRATEGY

    # ── Heuristic fallback (no LLM) ──
    # Key insight: probe fusion_ratio is the single most reliable signal.
    # If fusing ops saves >50% latency (ratio < 0.5), aggressive fusion almost
    # always wins because it eliminates expensive boundary IO between steps.
    # Conservative is ONLY good when memory is truly maxed AND fusion doesn't help.
    def _heuristic():
        if metrics["n_ops"] <= 3:
            return "aggressive_fusion"  # small graph — always fuse
        probe_fusion = probe.get("best_fusion_ratio", 1.0)
        probe_ws = probe.get("ws_pressure", 0.5)
        # Fusion saves >30% latency → aggressive (was 50%, lowered for more coverage)
        if probe_fusion < 0.7:
            return "aggressive_fusion"
        # Very tight WS AND fusion doesn't help much → conservative for better gran
        if probe_ws > 0.95 and probe_fusion > 0.85:
            return "conservative"
        # Many fanout diamonds → retention across steps matters
        if metrics["n_diamonds"] > metrics["n_ops"] * 0.3:
            return "memory_optimized"
        # Compute-bound graph → minimize padding/split-K overhead
        if probe.get("bottleneck") == "compute" and probe_fusion > 0.8:
            return "compute_optimized"
        # Default to aggressive_fusion — it wins on the vast majority of benchmarks
        return "aggressive_fusion"

    chosen = None

    if _llm_available():
        strat_names = list(STRATEGIES.keys())
        strat_desc_lines = "\n".join(f'  "{k}": {v["description"]}' for k, v in STRATEGIES.items())

        # ── Prompt with few-shot examples for reliable strategy selection ──
        prompt = (
            "You are a scheduling strategy advisor for a computational graph on an accelerator. "
            "Pick the SINGLE best strategy name to minimize total latency.\n\n"
            "RULES:\n"
            "- best_fusion_ratio = fused_latency / sum_individual_latencies. "
            "LOWER = fusion saves MORE. If < 0.7, aggressive_fusion almost always wins.\n"
            "- Use conservative ONLY when ws_pressure > 0.95 AND fusion_ratio > 0.85.\n"
            "- High mem_pressure does NOT mean conservative — fusion REDUCES memory traffic.\n"
            "- Default to aggressive_fusion when uncertain.\n\n"
            f"STRATEGIES:\n{strat_desc_lines}\n\n"
            "FEW-SHOT EXAMPLES:\n"
            'Example 1: 5 ops, fusion_ratio=0.334, ws=0.96, bottleneck=memory → {"strategy": "aggressive_fusion", "reasoning": "Low fusion ratio means fusion saves 67% latency"}\n'
            'Example 2: 103 ops, fusion_ratio=0.85, ws=0.99, bottleneck=compute → {"strategy": "compute_optimized", "reasoning": "Fusion does not help much and graph is compute-bound"}\n'
            'Example 3: 63 ops, fusion_ratio=0.056, ws=0.94, bottleneck=memory → {"strategy": "aggressive_fusion", "reasoning": "Extremely low fusion ratio means fusion saves 94% latency"}\n'
            'Example 4: 10 ops, fusion_ratio=0.9, ws=0.98, bottleneck=memory, many diamonds → {"strategy": "memory_optimized", "reasoning": "High WS pressure with many diamonds needs careful retention"}\n\n'
            f"YOUR TASK:\n"
            f"GRAPH: {metrics['n_ops']} ops ({metrics['n_matmul']} MatMul), "
            f"mem_pressure={metrics['mem_pressure']:.2f}x, "
            f"fusion_potential={metrics['fusion_potential_pct']:.0f}%, "
            f"n_diamonds={metrics['n_diamonds']}, "
            f"fast_memory={metrics['fast_memory_capacity']}\n"
            f"PROBE: best_fusion_ratio={probe.get('best_fusion_ratio', 1.0):.3f}, "
            f"ws_pressure={probe.get('ws_pressure', 0.5):.3f}, "
            f"bottleneck={probe.get('bottleneck', 'unknown')}, "
            f"time_budget={round(time_budget_s)}s\n"
            f"STRUCTURAL_HINT: mega_fusion_lat={probe.get('mega_fusion_lat', 'N/A')}, "
            f"mega_vs_greedy_ratio={probe.get('mega_vs_greedy_ratio', 'N/A')} "
            f"(ratio<1.0 means mega-fusion beats greedy probe — consider aggressive fusion)\n\n"
            f"Pick one of: {strat_names}\n"
            'Respond with JSON: {"strategy": "<name>", "reasoning": "<1 sentence>"}'
        )

        # ── Use response_json_schema with enum to constrain strategy field ──
        strategy_schema = {
            "type": "object",
            "properties": {
                "strategy": {
                    "type": "string",
                    "enum": strat_names,
                    "description": "The exact strategy name to use"
                },
                "reasoning": {
                    "type": "string",
                    "description": "One sentence explaining the choice"
                }
            },
            "required": ["strategy", "reasoning"],
            "propertyOrdering": ["strategy", "reasoning"]
        }

        try:
            # Direct API call with schema constraint for reliable output
            api_key = os.environ.get("GOOGLE_API_KEY", "")
            client = genai.Client(api_key=api_key)
            schema_cfg = types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=256,
                response_mime_type="application/json",
                response_json_schema=strategy_schema,
            ) if types else {
                "temperature": 0.2, "max_output_tokens": 256,
                "response_mime_type": "application/json",
                "response_json_schema": strategy_schema,
            }
            model_name = "gemini-3.1-flash-lite-preview"
            for attempt in range(3):
                try:
                    resp = client.models.generate_content(
                        model=model_name, contents=prompt, config=schema_cfg)
                    raw = (resp.text or "").strip()
                    if not raw:
                        print(f"  [Strategy] attempt {attempt+1}: empty response, retrying", file=sys.stderr)
                        continue
                    parsed = None
                    try:
                        parsed = json.loads(_repair_json(raw))
                    except json.JSONDecodeError:
                        parsed = {}
                        m = re.search(r'"strategy"\s*:\s*"([^"]+)"', raw)
                        if m: parsed["strategy"] = m.group(1)
                        m2 = re.search(r'"reasoning"\s*:\s*"([^"]*)"', raw)
                        if m2: parsed["reasoning"] = m2.group(1)
                    name = parsed.get("strategy", "").strip().lower().replace(" ", "_").replace("-", "_")
                    reason = parsed.get("reasoning", "")
                    name_map = {
                        "aggressive": "aggressive_fusion",
                        "aggressivefusion": "aggressive_fusion",
                        "conservative_fusion": "conservative",
                        "memory": "memory_optimized",
                        "compute": "compute_optimized",
                    }
                    name = name_map.get(name, name)
                    if name in STRATEGIES:
                        chosen = name
                        print(f"  [Strategy] LLM picked: {name} — {reason}", file=sys.stderr)
                        break
                    else:
                        print(f"  [Strategy] attempt {attempt+1}: unknown '{name}', retrying", file=sys.stderr)
                except Exception as me:
                    print(f"  [Strategy] attempt {attempt+1} failed: {me}, retrying", file=sys.stderr)
            if chosen is None:
                print(f"  [Strategy] All retries failed, using heuristic", file=sys.stderr)
        except Exception as e:
            print(f"  [Strategy] LLM failed: {e}, using heuristic", file=sys.stderr)

    if chosen is None:
        chosen = _heuristic()
        print(f"  [Strategy] Heuristic picked: {chosen}", file=sys.stderr)

    # Apply chosen strategy to module-level config
    cfg = STRATEGIES[chosen]
    _STRATEGY["name"] = chosen
    for k in ("max_len", "gran_top_k", "retention_pct", "beam_window", "beam_width"):
        _STRATEGY[k] = cfg[k]

    return chosen


# ── DP for Linear Chains ──

def _find_linear_chains(state):
    """Find maximal linear chains (≥3 ops) in the remaining DAG.

    A chain is a maximal run of ops where every op has at most one remaining
    producer AND at most one remaining consumer. This ensures any consecutive
    window [j:i] of the chain can be evaluated as a standalone subgraph.

    Returns list of chains (each a list of op indices in topological order).
    """
    remaining = state.remaining
    ops = state.ops

    producer = {}
    consumers = {}
    for op in ops:
        for t in op.outputs:
            producer[t] = op.idx
        for t in op.inputs:
            consumers.setdefault(t, []).append(op.idx)

    def _rem_consumer_count(oid):
        seen = set()
        for t in ops[oid].outputs:
            for c in consumers.get(t, []):
                if c in remaining:
                    seen.add(c)
        return len(seen)

    def _rem_producer_count(oid):
        seen = set()
        for t in ops[oid].inputs:
            p = producer.get(t)
            if p is not None and p in remaining:
                seen.add(p)
        return len(seen)

    # An op is chain-eligible if it has ≤1 remaining consumer and ≤1 remaining producer
    eligible = {oid for oid in remaining
                if _rem_consumer_count(oid) <= 1 and _rem_producer_count(oid) <= 1}

    # Build forward links: nxt[oid] = unique remaining eligible consumer
    nxt = {}
    for oid in eligible:
        rem_cons = [c for t in ops[oid].outputs
                    for c in consumers.get(t, [])
                    if c in remaining and c in eligible]
        if len(rem_cons) == 1:
            nxt[oid] = rem_cons[0]

    # Chain starts: eligible ops with no eligible predecessor pointing to them
    # AND no remaining producers outside the chain (all inputs already committed).
    has_pred = set(nxt.values())

    chains = []
    visited = set()
    for oid in state.topo:
        if oid not in eligible or oid in visited or oid in has_pred:
            continue
        # A valid chain start must be a true frontier op: every remaining producer
        # of its inputs must NOT be in remaining (i.e., already scheduled).
        # Without this check, an op whose sole remaining producer is a non-eligible
        # "branching" op (e.g. fan-out node) would incorrectly become a chain start,
        # causing it to be scheduled before its actual producer.
        if _rem_producer_count(oid) > 0:
            continue
        chain = [oid]
        visited.add(oid)
        cur = oid
        while cur in nxt:
            nx = nxt[cur]
            if nx in visited:
                break
            chain.append(nx)
            visited.add(nx)
            cur = nx
        if len(chain) >= 3:
            chains.append(chain)

    return chains


def _dp_eval_subgraph(op_idxs, state, gran_top_k=10):
    """Evaluate best (latency, gran) for a subgraph. Returns (None, None) if infeasible."""
    ois = [state.ops[i] for i in op_idxs]
    bi, bo = find_boundary_tensors(ois, state.ops)
    promotions = _get_orphan_promotions(list(op_idxs), state.ops, state.remaining)
    bo = bo | promotions
    grans_ws = _gen_grans(ois, bi, bo, state.prob, top_k=gran_top_k)
    if not grans_ws:
        return None, None
    best_lat, best_gran = float("inf"), None
    for g, _ in grans_ws[:gran_top_k]:
        w, h, kk = g
        _, lat = _best_traversal(ois, bi, bo, w, h, kk, state.prob)
        if lat < best_lat:
            best_lat, best_gran = lat, g
    return best_lat, best_gran


def _dp_partition_chain(chain_idxs, state, max_len=6, gran_top_k=10):
    """Find optimal consecutive partition of a linear chain via DP.

    dp[i] = min total latency to schedule chain_idxs[0:i].
    For each position i, try all windows [j:i] with len ≤ max_len.

    Returns list of (op_idx_list, gran) tuples, or None if no feasible partition.
    """
    n = len(chain_idxs)
    INF = float("inf")

    # Precompute costs for all feasible [j:i] windows
    cost = {}
    for j in range(n):
        for i in range(j + 1, min(j + max_len + 1, n + 1)):
            lat, gran = _dp_eval_subgraph(chain_idxs[j:i], state, gran_top_k)
            if lat is not None:
                cost[(j, i)] = (lat, gran)

    dp = [INF] * (n + 1)
    prev = [None] * (n + 1)
    dp[0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(0, i - max_len), i):
            if dp[j] == INF or (j, i) not in cost:
                continue
            lat, gran = cost[(j, i)]
            total = dp[j] + lat
            if total < dp[i]:
                dp[i] = total
                prev[i] = (j, gran)

    if dp[n] == INF:
        return None

    steps = []
    i = n
    while i > 0:
        j, gran = prev[i]
        steps.append((list(chain_idxs[j:i]), gran))
        i = j
    steps.reverse()
    return steps


def _run_mega_fusion(state):
    """Try fusing ALL committed ops into one step. Strictly better only.

    Ported from Track A TryMegaFusion. Covers cases like small Pointwise chains
    (mlsys-2026-5) where a single mega-step outperforms any partition.
    """
    if not state.is_complete():
        return 0.0
    op_seq = [oid for sg in state.schedule_subgraphs for oid in sg]
    if len(op_seq) < 2:
        return 0.0
    current_lat = sum(state.schedule_lat)
    ois = [state.ops[i] for i in op_seq]
    bi, bo = find_boundary_tensors(ois, state.ops)
    bo = bo | _get_orphan_promotions(op_seq, state.ops, set())
    grans_ws = _gen_grans(ois, bi, bo, state.prob, top_k=20)
    if not grans_ws:
        return 0.0
    best_lat, best_gran = float('inf'), None
    for g, _ in grans_ws:
        _, lat = _best_traversal(ois, bi, bo, *g, state.prob)
        if lat < best_lat:
            best_lat, best_gran = lat, g
    if best_lat >= current_lat - 1.0:
        return 0.0
    backup = dict(subgraphs=[list(sg) for sg in state.schedule_subgraphs],
                  gran=[list(g) for g in state.schedule_gran],
                  retain=[(list(r) if r is not None else None) for r in state.schedule_retain],
                  trav=[(list(t) if t is not None else None) for t in state.schedule_trav],
                  lat=list(state.schedule_lat), remaining=set(state.remaining))
    state.remaining = set(op_seq)
    state.schedule_subgraphs.clear(); state.schedule_gran.clear()
    state.schedule_retain.clear(); state.schedule_trav.clear(); state.schedule_lat.clear()
    cid = state._register(op_seq, "fusion")
    try:
        tool_apply_step(state, cid=cid, gran=best_gran)
    except Exception as e:
        state.remaining = backup["remaining"]
        state.schedule_subgraphs[:] = backup["subgraphs"]; state.schedule_gran[:] = backup["gran"]
        state.schedule_retain[:] = backup["retain"]; state.schedule_trav[:] = backup["trav"]
        state.schedule_lat[:] = backup["lat"]
        return 0.0
    improvement = current_lat - best_lat
    print(f"  [MegaFusion] {len(op_seq)} ops, lat {current_lat:.1f} → {best_lat:.1f} "
          f"({improvement/max(current_lat,1e-9)*100:+.1f}%)", file=sys.stderr)
    return improvement


def _run_nstep_partition(state, N=2):
    """Try all N-step partitions of committed ops, keep the globally best.

    Ported from Track A TryNStepPartition(2). Finds the optimal 2-step split
    of the committed op sequence (enumerates all n-1 split points).
    Especially effective on benchmarks like mlsys-2026-9 where the 2-step
    optimum is far from what greedy or DP repartition finds.
    """
    if not state.is_complete() or N != 2:
        return 0.0
    op_seq = [oid for sg in state.schedule_subgraphs for oid in sg]
    n = len(op_seq)
    if n < 3:
        return 0.0
    current_lat = sum(state.schedule_lat)
    best_total = current_lat
    best_sp = -1
    best_g1 = best_g2 = None
    for sp in range(1, n):
        left, right = op_seq[:sp], op_seq[sp:]
        ois_l = [state.ops[i] for i in left]; ois_r = [state.ops[i] for i in right]
        bi_l, bo_l = find_boundary_tensors(ois_l, state.ops)
        bi_r, bo_r = find_boundary_tensors(ois_r, state.ops)
        gl = _gen_grans(ois_l, bi_l, bo_l, state.prob, top_k=8)
        gr = _gen_grans(ois_r, bi_r, bo_r, state.prob, top_k=8)
        if not gl or not gr:
            continue
        ll, lg = min((_best_traversal(ois_l, bi_l, bo_l, *g, state.prob)[1], g) for g, _ in gl[:8])
        rl, rg = min((_best_traversal(ois_r, bi_r, bo_r, *g, state.prob)[1], g) for g, _ in gr[:8])
        if ll + rl < best_total - 0.5:
            best_total = ll + rl; best_sp = sp; best_g1 = lg; best_g2 = rg
    if best_sp < 0:
        return 0.0
    left, right = op_seq[:best_sp], op_seq[best_sp:]
    backup = dict(subgraphs=[list(sg) for sg in state.schedule_subgraphs],
                  gran=[list(g) for g in state.schedule_gran],
                  retain=[(list(r) if r is not None else None) for r in state.schedule_retain],
                  trav=[(list(t) if t is not None else None) for t in state.schedule_trav],
                  lat=list(state.schedule_lat), remaining=set(state.remaining))
    state.remaining = set(op_seq)
    state.schedule_subgraphs.clear(); state.schedule_gran.clear()
    state.schedule_retain.clear(); state.schedule_trav.clear(); state.schedule_lat.clear()
    ok = True
    for ops_part, gran in [(left, best_g1), (right, best_g2)]:
        cid = state._register(ops_part, "fusion" if len(ops_part) > 1 else "single")
        try:
            tool_apply_step(state, cid=cid, gran=gran)
        except Exception:
            ok = False; break
    if not ok or not state.is_complete():
        state.remaining = backup["remaining"]
        state.schedule_subgraphs[:] = backup["subgraphs"]; state.schedule_gran[:] = backup["gran"]
        state.schedule_retain[:] = backup["retain"]; state.schedule_trav[:] = backup["trav"]
        state.schedule_lat[:] = backup["lat"]
        return 0.0
    improvement = current_lat - best_total
    print(f"  [2StepPart] split@{best_sp}/{n}, lat {current_lat:.1f} → {best_total:.1f} "
          f"({improvement/max(current_lat,1e-9)*100:+.1f}%)", file=sys.stderr)
    return improvement


def _run_dp_repartition(state, time_budget_s=30.0):
    """Globally repartition the complete schedule via DP to escape greedy local optima.

    After the greedy+local-search pipeline, the committed op order is fixed but the
    partition into steps may be suboptimal (e.g., over-merged into one large step that
    forces tiny k, or under-merged with many steps that miss fusion bandwidth savings).

    This re-runs DP on the FULL op sequence (treating the topo-sorted committed order as
    a flexible partition target) and replaces the schedule if DP finds a better total
    latency.

    Works for any graph shape: a topo-sorted sequence guarantees that any contiguous
    block is a valid fusion (all deps either internal or from outside/earlier ops).
    """
    if not state.is_complete():
        return 0.0

    t0 = time.time()
    op_sequence = [oid for sg in state.schedule_subgraphs for oid in sg]
    n = len(op_sequence)
    if n < 3:
        return 0.0

    current_lat = sum(state.schedule_lat)
    max_len = _STRATEGY["max_len"]
    gran_top_k = _STRATEGY["gran_top_k"]

    # Create evaluation state with ALL ops in remaining so _dp_eval_subgraph
    # correctly identifies boundary tensors for any contiguous block.
    # (Topo order guarantees ops before block j can't consume block's outputs.)
    eval_state = AgentState(
        prob=state.prob, ops=state.ops, topo=state.topo,
        remaining=set(op_sequence), next_consumer=state.next_consumer,
    )

    steps = _dp_partition_chain(op_sequence, eval_state, max_len=max_len, gran_top_k=gran_top_k)
    if steps is None or time.time() - t0 > time_budget_s:
        return 0.0

    # Evaluate DP partition total latency (conservative: no retention between steps)
    eval_state2 = AgentState(
        prob=state.prob, ops=state.ops, topo=state.topo,
        remaining=set(op_sequence), next_consumer=state.next_consumer,
    )
    new_total = 0.0
    for op_idxs, gran in steps:
        lat, _ = _dp_eval_subgraph(op_idxs, eval_state2, gran_top_k=gran_top_k)
        if lat is None:
            return 0.0
        new_total += lat

    if new_total >= current_lat - 1.0:
        return 0.0

    improvement = current_lat - new_total
    pct = improvement / max(current_lat, 1e-9) * 100
    print(f"  [DP-Repart] {len(state.schedule_subgraphs)}→{len(steps)} steps, "
          f"lat {current_lat:.1f}→{new_total:.1f} ({pct:+.1f}%), wall={time.time()-t0:.2f}s",
          file=sys.stderr)

    # Save current schedule for rollback
    backup = dict(
        subgraphs=[list(sg) for sg in state.schedule_subgraphs],
        gran=[list(g) for g in state.schedule_gran],
        retain=[(list(r) if r is not None else None) for r in state.schedule_retain],
        trav=[(list(t) if t is not None else None) for t in state.schedule_trav],
        lat=list(state.schedule_lat),
        remaining=set(state.remaining),
    )

    # Clear and recommit the DP partition
    state.remaining = set(op_sequence)
    state.schedule_subgraphs.clear()
    state.schedule_gran.clear()
    state.schedule_retain.clear()
    state.schedule_trav.clear()
    state.schedule_lat.clear()

    ok = True
    for op_idxs, gran in steps:
        cid = state._register(op_idxs, "fusion" if len(op_idxs) > 1 else "single")
        try:
            tool_apply_step(state, cid=cid, gran=gran)
        except Exception as e:
            print(f"  [DP-Repart] Apply failed ops={op_idxs[:3]}...: {e}; rolling back",
                  file=sys.stderr)
            ok = False
            break

    if not ok or not state.is_complete():
        state.remaining = backup["remaining"]
        state.schedule_subgraphs[:] = backup["subgraphs"]
        state.schedule_gran[:] = backup["gran"]
        state.schedule_retain[:] = backup["retain"]
        state.schedule_trav[:] = backup["trav"]
        state.schedule_lat[:] = backup["lat"]
        return 0.0

    return improvement


def _chain_terminal_feeds_junction(chain, state):
    """Return True if pre-committing this chain would block a larger optimal fusion.

    A chain's terminal op feeds a "junction" if its downstream consumer has other
    remaining producers outside the chain.  In that case the greedy can propose a
    larger fusion (chain + non-chain input + junction op) that is cheaper than the
    chain committed in isolation.  Skipping such chains lets the greedy find those
    larger fusions (e.g., multi-head attention where each head's QK matmuls connect
    through a shared attention-score op).
    """
    chain_set = set(chain)
    terminal  = chain[-1]
    producer_map = {t: op.idx for op in state.ops for t in op.outputs}
    consumers_map: Dict[int, List[int]] = {}
    for op in state.ops:
        for t in op.inputs:
            consumers_map.setdefault(t, []).append(op.idx)

    for t in state.ops[terminal].outputs:
        for c in consumers_map.get(t, []):
            if c not in state.remaining:
                continue
            for tin in state.ops[c].inputs:
                p = producer_map.get(tin)
                if p is not None and p in state.remaining and p not in chain_set:
                    return True   # c has a remaining producer outside this chain
    return False


def _run_dp_chains(state, time_budget_s=30.0):
    """Find linear chains, run DP on each, commit optimal steps to state.

    Chains whose terminal op feeds a "junction" consumer (one with remaining
    producers outside the chain) are skipped.  Pre-committing those chains would
    prevent the greedy from discovering larger fusions that span the junction.
    For example, in multi-head attention the optimal fusion combines a chain's
    ops with the non-chain key/value input and the attention-score junction op;
    the DP would split this into two smaller, more expensive steps.

    Returns (n_chains_found, n_steps_committed, n_ops_committed).
    """
    t0 = time.time()
    chains = _find_linear_chains(state)
    if not chains:
        return 0, 0, 0

    max_len   = _STRATEGY["max_len"]
    gran_top_k = _STRATEGY["gran_top_k"]
    total_steps = 0
    total_ops   = 0

    chain_lengths = [len(c) for c in chains]
    print(f"  [DP] {len(chains)} linear chains, lengths={chain_lengths}", file=sys.stderr)

    for chain in chains:
        if time.time() - t0 > time_budget_s:
            print(f"  [DP] Budget exhausted, stopping early", file=sys.stderr)
            break

        # Filter to ops still remaining (earlier chains may have consumed some)
        chain = [oid for oid in chain if oid in state.remaining]
        if len(chain) < 3:
            continue

        # Skip chains whose terminal feeds a junction — the greedy can find
        # a better fusion that includes the junction op and its other inputs.
        if _chain_terminal_feeds_junction(chain, state):
            print(f"  [DP] Chain len={len(chain)}: terminal feeds junction, deferring to greedy",
                  file=sys.stderr)
            continue

        steps = _dp_partition_chain(chain, state, max_len=max_len, gran_top_k=gran_top_k)
        if steps is None:
            print(f"  [DP] Chain len={len(chain)}: no feasible partition, skipping",
                  file=sys.stderr)
            continue

        pre_lat = sum(state.schedule_lat)
        ok = True
        for op_idxs, gran in steps:
            cid = state._register(op_idxs, "fusion" if len(op_idxs) > 1 else "single")
            try:
                tool_apply_step(state, cid=cid, gran=gran)
                total_steps += 1
                total_ops   += len(op_idxs)
            except Exception as e:
                print(f"  [DP] Apply failed ops={op_idxs} gran={gran}: {e}", file=sys.stderr)
                ok = False
                break

        if ok:
            dp_lat = sum(state.schedule_lat) - pre_lat
            print(f"  [DP] Chain len={len(chain)}: {len(steps)} steps, lat={dp_lat:.1f}",
                  file=sys.stderr)

    print(f"  [DP] Done: {total_ops} ops in {total_steps} steps, "
          f"wall={time.time()-t0:.1f}s", file=sys.stderr)
    return len(chains), total_steps, total_ops


def _copy_post_dp_state(state):
    """Copy state including already-committed schedule (DP steps + any prior steps).

    Used by multi-restart so all restarts begin from the same post-DP baseline.
    """
    s = AgentState(
        prob=state.prob,
        ops=state.ops,
        topo=state.topo,
        remaining=set(state.remaining),
        next_consumer=state.next_consumer,
    )
    s.schedule_subgraphs = [list(sg) for sg in state.schedule_subgraphs]
    s.schedule_gran     = [list(g) for g in state.schedule_gran]
    s.schedule_retain   = [(list(r) if r is not None else None) for r in state.schedule_retain]
    s.schedule_trav     = [(list(t) if t is not None else None) for t in state.schedule_trav]
    s.schedule_lat      = list(state.schedule_lat)
    return s


# ── Multi-Restart Portfolio Helpers ──

def _copy_initial_state(state):
    """Create a fresh scheduling state from the same problem graph (for multi-restart).

    Shares immutable references (prob, ops, topo, next_consumer).
    All scheduling state (schedule_*, candidates, next_cid) is reset to empty.
    """
    return AgentState(
        prob=state.prob,
        ops=state.ops,
        topo=state.topo,
        remaining=set(state.remaining),
        next_consumer=state.next_consumer,
    )


# ── 1-Step Lookahead ──

def _lookahead_score(state, cid, gran, gran_top_k=5):
    """Simulate applying (cid, gran), then evaluate the best next step.

    Returns this_step_lat + best_next_lat — a 2-step latency score.
    Returns None if the simulation fails (infeasible apply or evaluation error).
    Uses small k values to stay cheap: ~0.1-0.5s per call vs ~12s for LLM.
    """
    s2 = _copy_post_dp_state(state)
    try:
        tool_apply_step(s2, cid=cid, gran=gran)
    except Exception:
        return None

    step_lat = s2.schedule_lat[-1] if s2.schedule_lat else 0.0

    if s2.is_complete():
        return step_lat  # no next step to evaluate

    try:
        prop = tool_propose_fusion_candidates(s2, k=15, max_len=_STRATEGY["max_len"])
        next_cids = prop["candidate_ids"]
        if not next_cids:
            return step_lat
        evals = tool_evaluate_step_bundle(s2, candidate_ids=next_cids, granularity_k=gran_top_k)
        next_ranked = evals["ranked"]
        if not next_ranked:
            return step_lat
        return step_lat + next_ranked[0]["lat"]
    except Exception:
        return step_lat


def _pick_by_lookahead(state, ranked, time_remaining):
    """Run 1-step lookahead on top-5 distinct candidates and return the clear winner.

    Collects the best-gran entry for each distinct cid in ranked (up to 5),
    scores each via _lookahead_score, and returns (cid, gran) if the best score
    is ≤97% of the second-best (a >3% advantage). Returns None otherwise.

    Only active when time_remaining > 120s so the simulation budget is affordable.
    """
    if time_remaining <= 120:
        return None

    # Top-5 distinct fusion candidates (best gran per cid)
    seen: set = set()
    la_cands = []
    for r in ranked:
        if r["cid"] not in seen:
            la_cands.append(r)
            seen.add(r["cid"])
        if len(la_cands) >= 5:
            break

    if len(la_cands) < 2:
        return None

    la_scores: dict = {}
    for r in la_cands:
        score = _lookahead_score(state, r["cid"], r["gran"])
        if score is not None:
            la_scores[r["cid"]] = (score, r["gran"])

    if len(la_scores) < 2:
        return None

    sorted_la = sorted(la_scores.items(), key=lambda x: x[1][0])
    best_cid, (best_score, best_g) = sorted_la[0]
    _, (second_score, _) = sorted_la[1]

    if best_score <= second_score * 0.97:
        return best_cid, best_g
    return None


_ALT_STRATEGY_MAP = {
    "aggressive_fusion": ["balanced", "memory_optimized"],
    "balanced":          ["aggressive_fusion", "conservative"],
    "conservative":      ["balanced", "compute_optimized"],
    "memory_optimized":  ["aggressive_fusion", "balanced"],
    "compute_optimized": ["balanced", "aggressive_fusion"],
}

def _alt_strategy(primary, idx):
    """Return an alternative strategy for restart idx (0-based), different from primary."""
    opts = _ALT_STRATEGY_MAP.get(primary, ["balanced", "aggressive_fusion"])
    return opts[idx % len(opts)]


def _run_single_schedule(state, strategy_name, deadline):
    """Run the greedy scheduling loop with a given strategy, stopping at deadline.

    Applies the strategy config to _STRATEGY, then runs the main propose→evaluate→apply
    loop until state.is_complete() or deadline is exceeded.

    Returns (llm_calls, auto_calls, la_calls). Raises TimeoutError on deadline, RuntimeError on failure.
    """
    cfg = STRATEGIES[strategy_name]
    _STRATEGY["name"] = strategy_name
    for k in ("max_len", "gran_top_k", "retention_pct", "beam_window", "beam_width"):
        _STRATEGY[k] = cfg[k]

    td = _trace_dir()
    llm_calls = 0; auto_calls = 0; la_calls = 0

    while not state.is_complete():
        now = time.time()
        if now >= deadline: raise TimeoutError("Schedule deadline exceeded")
        time_remaining = deadline - now

        prop = tool_propose_fusion_candidates(state, k=40, max_len=_STRATEGY["max_len"])
        cids = prop["candidate_ids"]
        if not cids: raise RuntimeError("No candidates")
        recomp = tool_try_recomputation(state)
        if recomp["candidate_ids"]:
            cids = cids + recomp["candidate_ids"]
        cluster_cids = _cluster_consumers(state)
        if cluster_cids:
            cids = cids + cluster_cids
        fanout_cids = _fanout_join_solver(state)
        if fanout_cids:
            cids = cids + fanout_cids
        beam_part = _beam_partition(state, window=_STRATEGY["beam_window"], beam_width=_STRATEGY["beam_width"])
        if beam_part:
            for bp_ops in beam_part[:2]:
                if len(bp_ops) > 1:
                    key = frozenset(bp_ops)
                    existing = {frozenset(state.candidates[c].ops) for c in cids if c in state.candidates}
                    if key not in existing:
                        cids.append(state._register(bp_ops, "beam_partition"))
        evals = tool_evaluate_step_bundle(state, candidate_ids=cids, granularity_k=_STRATEGY["gran_top_k"])
        ranked = evals["ranked"]
        if not ranked: raise RuntimeError("No feasible candidates")
        si = len(state.schedule_subgraphs)

        # ── Confidence-based LLM skip + batch auto-apply ──
        if _is_confident(ranked, time_remaining, len(state.remaining)):
            while ranked and not state.is_complete():
                best = ranked[0]
                cid, g = best["cid"], best["gran"]
                try:
                    tool_apply_step(state, cid=cid, gran=g)
                    bsi = len(state.schedule_subgraphs) - 1
                    print(f"  [Step {bsi}] AUTO cid={cid} gran={g}", file=sys.stderr)
                    auto_calls += 1
                except Exception as e:
                    print(f"  [AutoApply] Failed: {e}, falling back to LLM", file=sys.stderr)
                    break

                if state.is_complete(): break
                now = time.time()
                if now >= deadline: raise TimeoutError("Schedule deadline exceeded")
                time_remaining = deadline - now

                prop2 = tool_propose_fusion_candidates(state, k=40, max_len=_STRATEGY["max_len"])
                cids2 = prop2["candidate_ids"]
                if not cids2: break
                recomp2 = tool_try_recomputation(state)
                if recomp2["candidate_ids"]:
                    cids2 = cids2 + recomp2["candidate_ids"]
                evals2 = tool_evaluate_step_bundle(state, candidate_ids=cids2, granularity_k=_STRATEGY["gran_top_k"])
                ranked = evals2["ranked"]
                if not ranked: break
                if not _is_confident(ranked, time_remaining, len(state.remaining)):
                    break
            continue

        # ── 1-Step Lookahead batch path ──
        # Mirrors the batch confidence loop: run _pick_by_lookahead; if it returns
        # a clear winner, apply it and continue lookahead until ambiguous or until
        # confidence kicks in (which is cheaper than lookahead).
        # Cost per step: ~0.5-2s (5 simulations) vs ~12s LLM.
        la_pick = _pick_by_lookahead(state, ranked, time_remaining)
        if la_pick is not None:
            while la_pick is not None and not state.is_complete():
                cid_la, g_la = la_pick
                try:
                    tool_apply_step(state, cid=cid_la, gran=g_la)
                    bsi = len(state.schedule_subgraphs) - 1
                    print(f"  [Step {bsi}] LOOKAHEAD cid={cid_la} gran={g_la}", file=sys.stderr)
                    la_calls += 1
                except Exception as e:
                    print(f"  [Lookahead] Apply failed: {e}, falling back to LLM", file=sys.stderr)
                    break

                if state.is_complete(): break
                now = time.time()
                if now >= deadline: raise TimeoutError("Schedule deadline exceeded")
                time_remaining = deadline - now

                # Re-propose and re-evaluate for the next iteration
                prop3 = tool_propose_fusion_candidates(state, k=40, max_len=_STRATEGY["max_len"])
                cids3 = prop3["candidate_ids"]
                if not cids3: break
                recomp3 = tool_try_recomputation(state)
                if recomp3["candidate_ids"]:
                    cids3 = cids3 + recomp3["candidate_ids"]
                evals3 = tool_evaluate_step_bundle(state, candidate_ids=cids3,
                                                   granularity_k=_STRATEGY["gran_top_k"])
                ranked = evals3["ranked"]
                if not ranked: break
                # Yield to confidence batch if applicable (cheaper than lookahead)
                if _is_confident(ranked, time_remaining, len(state.remaining)):
                    break
                la_pick = _pick_by_lookahead(state, ranked, time_remaining)
            continue

        # ── LLM path: ask Gemini to pick ──
        traversal_opps = []
        for r in ranked[:5]:
            ops_in = [state.ops[i] for i in r["ops"]]
            if any(op.op_type == "MatMul" for op in ops_in):
                trav_info = tool_propose_traversal_order(state, r["cid"], r["gran"])
                if trav_info.get("improvement_pct", 0) > 1.0:
                    traversal_opps.append({"cid": r["cid"], "gran": r["gran"],
                                           "improvement_pct": trav_info["improvement_pct"],
                                           "latency_snake": trav_info["latency_snake"]})
        recomp_summary = recomp.get("diamonds", [])[:4]
        prompt = {"system": SYSTEM_PROMPT, "state": state.compact_summary(),
                  "ranked_options": ranked[:15],
                  "traversal_opportunities": traversal_opps if traversal_opps else "none",
                  "recomputation_diamonds": recomp_summary if recomp_summary else "none",
                  "time_remaining_s": round(time_remaining),
                  "instruction": "Pick best option, call apply_step. JSON only."}
        _wt(td, f"iter_{si:04d}_prompt.json", prompt)
        plan = None; last_err = None
        for attempt in range(3):
            try:
                raw = gemini_plan(prompt, max_tokens=2048)
                _wtt(td, f"iter_{si:04d}_raw_a{attempt}.txt", raw)
            except Exception as e:
                last_err = e; print(f"  [Retry {attempt+1}/3] Gemini: {e}", file=sys.stderr); continue
            try: plan = parse_tool_calls(raw)
            except Exception as e:
                last_err = e; print(f"  [Retry {attempt+1}/3] Parse: {e}", file=sys.stderr); continue
            ac = [c for c in plan["calls"] if c["tool"] == "apply_step"]
            if not ac:
                last_err = RuntimeError("No apply_step"); plan = None; continue
            a = ac[0]["args"]
            if "cid" not in a or "gran" not in a:
                last_err = RuntimeError(f"Missing cid/gran: {a}"); plan = None; continue
            break
        if plan is None: raise RuntimeError(f"All retries failed: {last_err}")
        _wt(td, f"iter_{si:04d}_parsed.json", plan)
        for c in plan["calls"]:
            if c["tool"] == "apply_step":
                a = c["args"]; cid = int(a["cid"]); g = [int(x) for x in a["gran"]]
                print(f"  [Step {si}] apply_step cid={cid} gran={g}", file=sys.stderr)
                tool_apply_step(state, cid=cid, gran=g); llm_calls += 1; break

    return llm_calls, auto_calls, la_calls


def run_agent(state, time_budget_s=9.0*60):
    start = time.time(); td = _trace_dir()

    # ── Phase 0: Strategy Selection ──
    metrics = _compute_graph_metrics(state)
    _wt(td, "graph_metrics.json", metrics)
    print(f"  [Metrics] {metrics['n_ops']} ops ({metrics['n_matmul']} MM), "
          f"mem_pressure={metrics['mem_pressure']:.1f}x, "
          f"fusion_potential={metrics['fusion_potential_pct']:.0f}%, "
          f"arith_intensity={metrics['arithmetic_intensity']:.3f}", file=sys.stderr)

    probe = _probe_schedule(state)
    _wt(td, "probe_results.json", probe)
    if probe.get("probe_steps", 0):
        print(f"  [Probe] best_lat={probe['best_latency']:.0f}, "
              f"fusion_ratio={probe['best_fusion_ratio']:.3f}, "
              f"ws_pressure={probe['ws_pressure']:.3f}, "
              f"bottleneck={probe['bottleneck']}", file=sys.stderr)

    # ── Structural opportunity probe ──
    # Compute what mega-fusion and 2-step partition would give BEFORE scheduling.
    # This informs the LLM about structural optima it should aim for or avoid.
    try:
        all_op_ids = list(range(len(state.ops)))
        ois_all = state.ops
        bi_all, bo_all = find_boundary_tensors(ois_all, state.ops)
        grans_all = _gen_grans(ois_all, bi_all, bo_all, state.prob, top_k=8)
        mega_lat = float('inf')
        if grans_all:
            mega_lat = min(_best_traversal(ois_all, bi_all, bo_all, *g, state.prob)[1]
                          for g, _ in grans_all[:8])
        struct_probe = {"mega_fusion_lat": round(mega_lat, 1),
                        "n_ops": len(state.ops),
                        "best_greedy_lat": probe.get("best_latency", float('inf'))}
        mega_ratio = mega_lat / max(probe.get("best_latency", mega_lat), 1e-9)
        print(f"  [StructProbe] mega_fusion_lat={mega_lat:.0f} "
              f"({mega_ratio*100:.0f}% of greedy probe)", file=sys.stderr)
        # Attach to probe dict so _select_strategy can use it
        probe["mega_fusion_lat"] = mega_lat
        probe["mega_vs_greedy_ratio"] = round(mega_ratio, 3)
        _wt(td, "struct_probe.json", struct_probe)
    except Exception as e:
        print(f"  [StructProbe] failed: {e}", file=sys.stderr)

    chosen = _select_strategy(metrics, probe, time_budget_s - (time.time() - start))
    _wt(td, "strategy_chosen.json", {"strategy": chosen, "config": dict(_STRATEGY)})

    # ── Phase 0.5: DP chain optimization ──
    # Commit globally-optimal partitions for all linear chains before the greedy
    # loop. Pure Python computation (no LLM) — uses O(n*max_len) latency evaluations.
    dp_budget = min(30.0, max(5.0, time_budget_s - (time.time() - start) - 120.0))
    _run_dp_chains(state, time_budget_s=dp_budget)

    # ── Phase 1: Multi-Restart Portfolio ──
    # Save post-DP state so all restarts begin from the same chain-committed baseline.
    initial_state = _copy_post_dp_state(state)

    # Reserve 90s for post-schedule local search; split the rest across restarts.
    phase0_elapsed = time.time() - start
    sched_budget = max(60.0, time_budget_s - phase0_elapsed - 90.0)

    # Restart configs: (strategy_name, fraction_of_sched_budget)
    restart_configs = [
        (chosen,                   0.45),
        (_alt_strategy(chosen, 0), 0.35),
        (_alt_strategy(chosen, 1), 0.20),
    ]

    best_state = None
    best_lat = float("inf")
    total_llm = 0; total_auto = 0; total_la = 0

    # If DP already completed the whole schedule, skip restarts entirely.
    if state.is_complete():
        best_state = state
        best_lat = sum(state.schedule_lat)

    # Ops left after DP: if very few remain, a tiny time budget is enough.
    n_dp_remaining = len(state.remaining)
    _restart_min_secs = 5.0 if n_dp_remaining <= 3 else 20.0

    for i, (strategy, frac) in enumerate(restart_configs):
        if best_state is not None and best_state.is_complete() and n_dp_remaining == 0:
            break  # DP solved it, no restarts needed
        # How much wall time is genuinely left for scheduling?
        now = time.time()
        remaining_overall = (start + time_budget_s) - now - 90.0
        if remaining_overall < _restart_min_secs:
            print(f"  [Restart {i+1}/3] Skipping — only {remaining_overall:.0f}s left", file=sys.stderr)
            break

        budget = min(sched_budget * frac, remaining_overall)
        deadline = now + budget
        s = state if i == 0 else _copy_post_dp_state(initial_state)
        print(f"  [Restart {i+1}/3] strategy={strategy}, budget={budget:.0f}s", file=sys.stderr)

        try:
            lc, ac, lac = _run_single_schedule(s, strategy, deadline)
            total_llm += lc; total_auto += ac; total_la += lac
            lat = sum(s.schedule_lat)
            print(f"  [Restart {i+1}/3] Done: lat={lat:.1f}, steps={len(s.schedule_subgraphs)}", file=sys.stderr)
            # ── Structural polish per restart ──
            # Run 2-step and mega-fusion immediately after each restart so the
            # winner selection benefits from these improvements, not just the final pass.
            if s.is_complete():
                lat_before_struct = lat
                nstep_imp = _run_nstep_partition(s, N=2)
                mega_imp  = _run_mega_fusion(s)
                lat = sum(s.schedule_lat)
                if nstep_imp > 0 or mega_imp > 0:
                    print(f"  [Restart {i+1}/3] Structural: "
                          f"2step={nstep_imp:.0f} mega={mega_imp:.0f} "
                          f"→ lat={lat:.1f} ({(lat-lat_before_struct)/lat_before_struct*100:+.1f}%)",
                          file=sys.stderr)
                else:
                    print(f"  [Restart {i+1}/3] Structural: no improvement", file=sys.stderr)
            if lat < best_lat:
                best_lat = lat; best_state = s
        except TimeoutError as e:
            print(f"  [Restart {i+1}/3] Timed out: {e}", file=sys.stderr)
            if s.is_complete():
                lat = sum(s.schedule_lat)
                if lat < best_lat:
                    best_lat = lat; best_state = s
        except Exception as e:
            print(f"  [Restart {i+1}/3] Failed: {e}", file=sys.stderr)

        # If restart 1 already found a single-step solution, no point trying others.
        if i == 0 and best_state is not None and len(best_state.schedule_subgraphs) <= 1:
            print(f"  [Restart] Single/trivial solution found, skipping further restarts", file=sys.stderr)
            break

    if best_state is None:
        raise RuntimeError("All restarts failed to produce a complete schedule")

    state = best_state
    snapshots = [copy.deepcopy(materialize_solution(state))]  # later phases can break memory limits
    # ── Post-schedule structural + granularity/traversal local search ──
    remaining_time = time_budget_s - (time.time() - start)
    pre_lat = sum(state.schedule_lat)

    # ── Multi-pass iterative local search ──
    # Alternate structural moves and gran/traversal search until no more
    # improvements or time runs out. Each pass can unlock improvements for
    # the next: structural changes alter step boundaries, which changes
    # optimal granularity; granularity changes alter WS, which affects
    # which structural merges/splits are profitable.
    ls_pass = 0
    total_improvements = 0
    while True:
        remaining_time = time_budget_s - (time.time() - start)
        if remaining_time < 15: break  # need at least 15s for a useful pass

        pass_lat_before = sum(state.schedule_lat)
        pass_improvements = 0

        # Structural moves (merge/split/swap/flip)
        sls_budget = min(45, max(5, remaining_time * 0.35))
        sls = tool_structural_local_search(state, time_budget_s=sls_budget)
        pass_improvements += sls["improvements"]
        if sls["improvements"] > 0:
            mid_lat = sum(state.schedule_lat)
            pct = (pass_lat_before - mid_lat) / max(pass_lat_before, 1e-9) * 100
            print(f"  [StructuralLS pass {ls_pass}] {sls['improvements']} improvements, "
                  f"lat {pass_lat_before:.1f} → {mid_lat:.1f} ({pct:+.1f}%)", file=sys.stderr)

        # Gran/traversal polish
        remaining_time2 = time_budget_s - (time.time() - start)
        if remaining_time2 < 10: break
        ls_budget = max(5, remaining_time2 - 10)
        mid_lat2 = sum(state.schedule_lat)
        ls = tool_local_search(state, time_budget_s=ls_budget)
        pass_improvements += ls["improvements"]
        if ls["improvements"] > 0:
            post_lat = sum(state.schedule_lat)
            pct = (mid_lat2 - post_lat) / max(mid_lat2, 1e-9) * 100
            print(f"  [LocalSearch pass {ls_pass}] {ls['improvements']} improvements, "
                  f"lat {mid_lat2:.1f} → {post_lat:.1f} ({pct:+.1f}%)", file=sys.stderr)

        total_improvements += pass_improvements
        ls_pass += 1
        snapshots.append(copy.deepcopy(materialize_solution(state)))

        # Stop if no improvements this pass (converged)
        if pass_improvements == 0:
            break

    # ── Global DP repartition: escape greedy/StructuralLS local optima ──
    # After local search converges, the committed op ORDER is fixed but the
    # partition into steps may be suboptimal.  Re-run DP on the full topo
    # sequence to find the globally optimal step boundaries.
    dp_repart_budget = time_budget_s - (time.time() - start) - 10.0
    if dp_repart_budget > 5.0:
        dp_imp = _run_dp_repartition(state, time_budget_s=dp_repart_budget)
        if dp_imp > 0:
            # Re-tune granularities after repartition
            ls_remaining = time_budget_s - (time.time() - start) - 5.0
            if ls_remaining > 5.0:
                tool_local_search(state, time_budget_s=min(30.0, ls_remaining))
            snapshots.append(copy.deepcopy(materialize_solution(state)))

    # ── Structural polish (ported from Track A) ──
    # These only accept strict improvements so they never regress prior quality.
    remaining = time_budget_s - (time.time() - start) - 3.0
    if remaining > 2.0:
        lat_pre = sum(state.schedule_lat)
        # 2-step partition: globally optimal 2-partition
        imp2 = _run_nstep_partition(state, N=2)
        # 3-step partition: only for small graphs (O(n²) cost)
        imp3 = 0.0
        if len(state.ops) <= 20:
            imp3 = _run_nstep_partition(state, N=3)
        lat_post = sum(state.schedule_lat)
        print(f"  [FinalStruct] 2step={imp2:.0f} 3step={imp3:.0f} "
              f"total={lat_pre-lat_post:.0f} ({(lat_pre-lat_post)/max(lat_pre,1)*100:+.1f}%)",
              file=sys.stderr)
    # Mega-fusion: checks if all ops as one step beats current partition.
    remaining = time_budget_s - (time.time() - start) - 2.0
    if remaining > 1.0:
        imp_mega = _run_mega_fusion(state)
        print(f"  [FinalMega] improvement={imp_mega:.0f}", file=sys.stderr)

    sol = _best_valid([copy.deepcopy(materialize_solution(state))] + snapshots[::-1], state)
    tl = sum(sol["subgraph_latencies"])
    ns = len(sol['subgraphs'])
    print(f"  [Done] {ns} steps, lat={tl:.1f}, wall={time.time()-start:.1f}s "
          f"(LLM={total_llm}, AUTO={total_auto}, LA={total_la}, strategy={_STRATEGY['name']})", file=sys.stderr)
    return sol

def _is_valid(sol, state):
    try:
        validate_solution(sol, state.prob, state.ops)
        return True
    except Exception:
        return False


def _one_op_per_step(state):
    """Always-valid fallback: each op alone at its best tile size, nothing retained."""
    sol = {"subgraphs": [], "granularities": [], "tensors_to_retain": [],
           "traversal_orders": [], "subgraph_latencies": []}
    for oid in state.topo:
        step = [state.ops[oid]]
        b_in, b_out = find_boundary_tensors(step, state.ops)
        best = None
        for g, _ in _gen_grans(step, b_in, b_out, state.prob, top_k=20):
            trav, lat = _best_traversal(step, b_in, b_out, *g, state.prob)
            if best is None or lat < best[2]:
                best = (list(g), trav, lat)
        sol["subgraphs"].append([oid]); sol["granularities"].append(best[0])
        sol["tensors_to_retain"].append([]); sol["traversal_orders"].append(best[1])
        sol["subgraph_latencies"].append(best[2])
    return sol


def _best_valid(candidates, state):
    """Lowest-latency candidate that passes validation, else the one-op-per-step schedule."""
    valid = [c for c in candidates if _is_valid(c, state)]
    if valid:
        return min(valid, key=lambda c: sum(c["subgraph_latencies"]))
    print("  [Fallback] no phase produced a valid schedule; using one op per step", file=sys.stderr)
    return _one_op_per_step(state)


# ── Entry Point ──

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 agent.py <input.json> <output.json>", file=sys.stderr); sys.exit(2)
    prob, ops = load_problem(sys.argv[1]); topo = topo_sort(ops)
    state = AgentState(prob=prob, ops=ops, topo=topo, remaining=set(range(len(ops))),
                       next_consumer=_build_next_consumer(ops))
    sol = run_agent(state)
    # Write output FIRST so the evaluator always has a file, even if validation fails
    with open(sys.argv[2], "w") as f: json.dump(sol, f)
    print(f"Wrote {sys.argv[2]} (steps={len(sol['subgraphs'])})", file=sys.stderr)
    try:
        validate_solution(sol, prob, ops)
    except Exception as e:
        print(f"  [WARNING] Validation failed: {e}", file=sys.stderr)

if __name__ == "__main__": main()
