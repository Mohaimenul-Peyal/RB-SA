#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HP_ssa_manybody_ising_cpu_numba_saw.py

Numba-JIT (CPU) simulated annealing (Metropolis) for k-local Ising (1-4 body)
instances exported by your HP QUBO/Ising exporter.

Main RB-full features:
  1) In residue-block mode, each trial starts from a random self-avoiding walk (SAW),
     i.e., chain continuity + collision-free are satisfied at initialization.
  2) Residue moves are chain-preserving and collision-free by construction:
       - endpoints move to an unoccupied neighbor of their single neighbor residue
       - interior residues move to an unoccupied site that is adjacent to BOTH neighbors
     This keeps the Markov chain on the feasible SAW manifold (when starting feasible).
  3) Pivot moves can rotate either chain side and include 180-degree rotations,
     which gives compact folds more legal non-local escape routes.
  4) Sequence-aware feasible initialization, an in-worker seed archive,
     pull-like proposals, and endpoint segment regrowth proposals.
  5) Diversity-aware archive replacement and internal fragment bridge-regrowth
     moves let the solver keep multiple high-quality basins and rebuild trapped
     middle-chain segments.
  6) The saved best feasible state can be selected by contact count first and
     QUBO energy second.
  7) Optional contact-guided acceptance during the exploration
     portion of each anneal, contact-band paving, and a final pure-QUBO polish
     fraction. This keeps proposals on the feasible SAW manifold while making
     the early acceptance rule less blind to the HP contact objective.

Performance:
  - Uses incremental dE updates via per-variable incidence lists (no full energy recompute per step).
  - No per-step temporary array allocations in the Numba hot loop (fixed-size candidate handling).

Tested for compatibility with Python 3.6 + numba 0.53.x style APIs.
"""

import argparse
import json
import math
import time
from typing import Dict, Any, Tuple

import numpy as np
from numba import njit


# -------------------------
# RNG helpers (fast, Numba-friendly)
# -------------------------
@njit(cache=True)
def _splitmix64(x):
    z = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return z ^ (z >> np.uint64(31))


@njit(cache=True)
def _rand_uint64(rng):
    rng = _splitmix64(rng)
    return rng, rng


@njit(cache=True)
def _rand_float01(rng):
    rng, u = _rand_uint64(rng)
    # 53-bit mantissa style scaling (enough for Metropolis)
    return rng, (u >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True)
def _rand_int(rng, n):
    # n must be >0
    rng, u = _rand_uint64(rng)
    return rng, int(u % np.uint64(n))


# -------------------------
# Load JSON + term prep
# -------------------------
def load_ising(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _decode_contacts_from_saved_spins(spins: np.ndarray, data: Dict[str, Any]) -> int:
    coords = data.get("coords") or []
    seq = data.get("seq") or ""
    if not coords or not seq:
        return -1

    S = len(coords)
    N = len(seq)
    if spins.shape[0] != N * S:
        return -1

    pos = []
    for r in range(N):
        base = r * S
        active = [j for j in range(S) if spins[base + j] > 0.0]
        if len(active) != 1:
            return -1
        pos.append(active[0])

    if len(set(pos)) != len(pos):
        return -1

    def manh(a, b):
        ax, ay = coords[a]
        bx, by = coords[b]
        return abs(ax - bx) + abs(ay - by)

    for r in range(N - 1):
        if manh(pos[r], pos[r + 1]) != 1:
            return -1

    contacts = 0
    for i in range(N):
        if seq[i] != "H":
            continue
        for j in range(i + 2, N):
            if seq[j] != "H":
                continue
            if manh(pos[i], pos[j]) == 1:
                contacts += 1
    return int(contacts)


def _load_warm_start_seed(path: str, V: int, dtype, data: Dict[str, Any]):
    seed_spins = np.zeros((V,), dtype=dtype)
    if not path:
        return seed_spins, 0, -1

    with open(path, "r") as f:
        raw = json.load(f)

    obj = raw.get("best", raw)
    spins_raw = obj.get("spins")
    if not isinstance(spins_raw, list) or len(spins_raw) != V:
        raise ValueError("warm_start_file does not contain a valid spins vector for this Ising instance")

    seed_spins = np.array(spins_raw, dtype=dtype)
    contacts = obj.get("contacts")
    if not isinstance(contacts, int):
        contacts = _decode_contacts_from_saved_spins(seed_spins, data)

    return seed_spins, 1, int(contacts)


def _prepare_coords_and_seq(data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = data.get("coords") or []
    seq = data.get("seq") or ""
    if not coords or not seq:
        return np.zeros((0,), np.int32), np.zeros((0,), np.int32), np.zeros((0,), np.int8)

    coords_x = np.array([int(c[0]) for c in coords], dtype=np.int32)
    coords_y = np.array([int(c[1]) for c in coords], dtype=np.int32)
    seq_is_H = np.array([1 if ch == "H" else 0 for ch in seq], dtype=np.int8)
    return coords_x, coords_y, seq_is_H


def _prepare_neighbors(coords_x: np.ndarray, coords_y: np.ndarray) -> np.ndarray:
    """
    Build neighbor table nbrs[S,4] from coords arrays.
    nbrs[i,k] gives neighbor site index or -1 for missing.
    Order: (x+1,y),(x-1,y),(x,y+1),(x,y-1).
    """
    S = int(coords_x.shape[0])
    nbrs = np.full((S, 4), -1, dtype=np.int32)
    if S == 0:
        return nbrs
    mapping = {(int(coords_x[i]), int(coords_y[i])): i for i in range(S)}
    for i in range(S):
        x = int(coords_x[i]); y = int(coords_y[i])
        nbrs[i, 0] = mapping.get((x + 1, y), -1)
        nbrs[i, 1] = mapping.get((x - 1, y), -1)
        nbrs[i, 2] = mapping.get((x, y + 1), -1)
        nbrs[i, 3] = mapping.get((x, y - 1), -1)
    return nbrs


def _prepare_xy2s(coords_x: np.ndarray, coords_y: np.ndarray) -> np.ndarray:
    """
    Build a dense lookup table xy2s[x,y] -> site index (or -1).
    Assumes coords are on a small integer grid (true for LxL HP lattices).
    """
    if coords_x.size == 0:
        return np.zeros((0, 0), dtype=np.int32)
    max_x = int(coords_x.max())
    max_y = int(coords_y.max())
    xy2s = np.full((max_x + 1, max_y + 1), -1, dtype=np.int32)
    for i in range(coords_x.shape[0]):
        xy2s[int(coords_x[i]), int(coords_y[i])] = i
    return xy2s


def prepare_terms_cpu(data: Dict[str, Any], dtype=np.float32) -> Dict[str, Any]:
    ising = data.get("ising", {})
    h = ising.get("h", {})
    J2 = ising.get("J2", []) or []
    J3 = ising.get("J3", []) or []
    J4 = ising.get("J4", []) or []

    V = int(data.get("V") or 0)
    if V <= 0:
        # fall back: infer from h keys
        if isinstance(h, dict) and len(h) > 0:
            V = max(int(k) for k in h.keys()) + 1
        else:
            raise ValueError("Could not determine V")

    # Dense h
    h_dense = np.zeros((V,), dtype=dtype)
    if isinstance(h, dict) and len(h) > 0:
        for k, v in h.items():
            h_dense[int(k)] = dtype(v)

    # J2 arrays
    J2_a = np.empty((len(J2),), dtype=np.int32)
    J2_b = np.empty((len(J2),), dtype=np.int32)
    J2_c = np.empty((len(J2),), dtype=dtype)
    for t, e in enumerate(J2):
        i, j, c = int(e[0]), int(e[1]), float(e[2])
        J2_a[t] = i; J2_b[t] = j; J2_c[t] = dtype(c)

    # J3 arrays: each entry [i,j,k,c]
    J3_a = np.empty((len(J3),), dtype=np.int32)
    J3_b = np.empty((len(J3),), dtype=np.int32)
    J3_k = np.empty((len(J3),), dtype=np.int32)
    J3_c = np.empty((len(J3),), dtype=dtype)
    for t, e in enumerate(J3):
        i, j, k, c = int(e[0]), int(e[1]), int(e[2]), float(e[3])
        J3_a[t] = i; J3_b[t] = j; J3_k[t] = k; J3_c[t] = dtype(c)

    # J4 arrays: each entry [i,j,k,l,c]
    J4_a = np.empty((len(J4),), dtype=np.int32)
    J4_b = np.empty((len(J4),), dtype=np.int32)
    J4_k = np.empty((len(J4),), dtype=np.int32)
    J4_l = np.empty((len(J4),), dtype=np.int32)
    J4_c = np.empty((len(J4),), dtype=dtype)
    for t, e in enumerate(J4):
        i, j, k, l, c = int(e[0]), int(e[1]), int(e[2]), int(e[3]), float(e[4])
        J4_a[t] = i; J4_b[t] = j; J4_k[t] = k; J4_l[t] = l; J4_c[t] = dtype(c)

    # Build incidence lists for Î”E:
    # For each i, store all terms containing i as:
    # 2-body: (other, coeff)
    # 3-body: (o1,o2, coeff)
    # 4-body: (o1,o2,o3, coeff)
    inc2_idx = [[] for _ in range(V)]
    inc2_c = [[] for _ in range(V)]
    for t in range(len(J2)):
        i = int(J2_a[t]); j = int(J2_b[t]); c = float(J2_c[t])
        inc2_idx[i].append(j); inc2_c[i].append(c)
        inc2_idx[j].append(i); inc2_c[j].append(c)

    inc3_o1 = [[] for _ in range(V)]
    inc3_o2 = [[] for _ in range(V)]
    inc3_c = [[] for _ in range(V)]
    for t in range(len(J3)):
        i = int(J3_a[t]); j = int(J3_b[t]); k = int(J3_k[t]); c = float(J3_c[t])
        inc3_o1[i].append(j); inc3_o2[i].append(k); inc3_c[i].append(c)
        inc3_o1[j].append(i); inc3_o2[j].append(k); inc3_c[j].append(c)
        inc3_o1[k].append(i); inc3_o2[k].append(j); inc3_c[k].append(c)

    inc4_o1 = [[] for _ in range(V)]
    inc4_o2 = [[] for _ in range(V)]
    inc4_o3 = [[] for _ in range(V)]
    inc4_c = [[] for _ in range(V)]
    for t in range(len(J4)):
        i = int(J4_a[t]); j = int(J4_b[t]); k = int(J4_k[t]); l = int(J4_l[t]); c = float(J4_c[t])
        # for i: others j,k,l
        inc4_o1[i].append(j); inc4_o2[i].append(k); inc4_o3[i].append(l); inc4_c[i].append(c)
        inc4_o1[j].append(i); inc4_o2[j].append(k); inc4_o3[j].append(l); inc4_c[j].append(c)
        inc4_o1[k].append(i); inc4_o2[k].append(j); inc4_o3[k].append(l); inc4_c[k].append(c)
        inc4_o1[l].append(i); inc4_o2[l].append(j); inc4_o3[l].append(k); inc4_c[l].append(c)

    def flatten_ptr(lst_of_lst):
        ptr = np.zeros((V + 1,), dtype=np.int32)
        total = 0
        for i in range(V):
            ptr[i] = total
            total += len(lst_of_lst[i])
        ptr[V] = total
        return ptr, total

    # flatten 2-body
    ptr2, nnz2 = flatten_ptr(inc2_idx)
    j2_idx = np.empty((nnz2,), dtype=np.int32)
    j2_c = np.empty((nnz2,), dtype=dtype)
    cur = ptr2[:-1].copy()
    for i in range(V):
        for k in range(len(inc2_idx[i])):
            p = cur[i]; cur[i] += 1
            j2_idx[p] = int(inc2_idx[i][k])
            j2_c[p] = dtype(inc2_c[i][k])

    # flatten 3-body
    ptr3, nnz3 = flatten_ptr(inc3_o1)
    j3_o1 = np.empty((nnz3,), dtype=np.int32)
    j3_o2 = np.empty((nnz3,), dtype=np.int32)
    j3_c = np.empty((nnz3,), dtype=dtype)
    cur = ptr3[:-1].copy()
    for i in range(V):
        for k in range(len(inc3_o1[i])):
            p = cur[i]; cur[i] += 1
            j3_o1[p] = int(inc3_o1[i][k])
            j3_o2[p] = int(inc3_o2[i][k])
            j3_c[p] = dtype(inc3_c[i][k])

    # flatten 4-body
    ptr4, nnz4 = flatten_ptr(inc4_o1)
    j4_o1 = np.empty((nnz4,), dtype=np.int32)
    j4_o2 = np.empty((nnz4,), dtype=np.int32)
    j4_o3 = np.empty((nnz4,), dtype=np.int32)
    j4_c = np.empty((nnz4,), dtype=dtype)
    cur = ptr4[:-1].copy()
    for i in range(V):
        for k in range(len(inc4_o1[i])):
            p = cur[i]; cur[i] += 1
            j4_o1[p] = int(inc4_o1[i][k])
            j4_o2[p] = int(inc4_o2[i][k])
            j4_o3[p] = int(inc4_o3[i][k])
            j4_c[p] = dtype(inc4_c[i][k])

    return dict(
        V=V,
        h_dense=h_dense,
        J2_a=J2_a, J2_b=J2_b, J2_c=J2_c,
        J3_a=J3_a, J3_b=J3_b, J3_k=J3_k, J3_c=J3_c,
        J4_a=J4_a, J4_b=J4_b, J4_k=J4_k, J4_l=J4_l, J4_c=J4_c,
        ptr2=ptr2, j2_idx=j2_idx, j2_c=j2_c,
        ptr3=ptr3, j3_o1=j3_o1, j3_o2=j3_o2, j3_c=j3_c,
        ptr4=ptr4, j4_o1=j4_o1, j4_o2=j4_o2, j4_o3=j4_o3, j4_c=j4_c,
    )


# -------------------------
# Numba energy + Î”E
# -------------------------
@njit(cache=True, fastmath=True)
def _energy_full(s, h_dense, J2_a, J2_b, J2_c, J3_a, J3_b, J3_k, J3_c, J4_a, J4_b, J4_k, J4_l, J4_c):
    e = 0.0
    V = h_dense.shape[0]
    for i in range(V):
        e += float(h_dense[i]) * float(s[i])
    for t in range(J2_a.shape[0]):
        i = J2_a[t]; j = J2_b[t]
        e += float(J2_c[t]) * float(s[i]) * float(s[j])
    for t in range(J3_a.shape[0]):
        i = J3_a[t]; j = J3_b[t]; k = J3_k[t]
        e += float(J3_c[t]) * float(s[i]) * float(s[j]) * float(s[k])
    for t in range(J4_a.shape[0]):
        i = J4_a[t]; j = J4_b[t]; k = J4_k[t]; l = J4_l[t]
        e += float(J4_c[t]) * float(s[i]) * float(s[j]) * float(s[k]) * float(s[l])
    return e


@njit(cache=True, fastmath=True)
def _deltaE_flip(i, s, h_dense, ptr2, j2_idx, j2_c, ptr3, j3_o1, j3_o2, j3_c, ptr4, j4_o1, j4_o2, j4_o3, j4_c):
    si = float(s[i])
    field = float(h_dense[i])

    b = int(ptr2[i]); e = int(ptr2[i + 1])
    for p in range(b, e):
        j = int(j2_idx[p])
        field += float(j2_c[p]) * float(s[j])

    b = int(ptr3[i]); e = int(ptr3[i + 1])
    for p in range(b, e):
        o1 = int(j3_o1[p]); o2 = int(j3_o2[p])
        field += float(j3_c[p]) * float(s[o1]) * float(s[o2])

    b = int(ptr4[i]); e = int(ptr4[i + 1])
    for p in range(b, e):
        o1 = int(j4_o1[p]); o2 = int(j4_o2[p]); o3 = int(j4_o3[p])
        field += float(j4_c[p]) * float(s[o1]) * float(s[o2]) * float(s[o3])

    return -2.0 * si * field


# -------------------------
# Strict HP contact decode (only meaningful for feasible SAWs)
# -------------------------
@njit(cache=True)
def _decode_contacts_onehot_strict(s_best, seq_is_H, coords_x, coords_y, N, S, pos_tmp, x_tmp, y_tmp, occ_tmp):
    # occupancy reset
    for j in range(S):
        occ_tmp[j] = 0

    # decode one-hot + collisions
    for r in range(N):
        base = r * S
        active = -1
        cnt = 0
        for j in range(S):
            if s_best[base + j] > 0.0:
                active = j
                cnt += 1
                if cnt > 1:
                    break
        if cnt != 1:
            return -1, 0
        if occ_tmp[active] != 0:
            return -1, 0
        occ_tmp[active] = 1
        pos_tmp[r] = active
        x_tmp[r] = coords_x[active]
        y_tmp[r] = coords_y[active]

    # chain adjacency
    for r in range(N - 1):
        dx = x_tmp[r] - x_tmp[r + 1]
        if dx < 0:
            dx = -dx
        dy = y_tmp[r] - y_tmp[r + 1]
        if dy < 0:
            dy = -dy
        if dx + dy != 1:
            return -1, 0

    # H-H contacts (non-consecutive)
    contacts = 0
    for i in range(N):
        if seq_is_H[i] == 0:
            continue
        xi = x_tmp[i]
        yi = y_tmp[i]
        for j in range(i + 2, N):
            if seq_is_H[j] == 0:
                continue
            dx = xi - x_tmp[j]
            if dx < 0:
                dx = -dx
            dy = yi - y_tmp[j]
            if dy < 0:
                dy = -dy
            if dx + dy == 1:
                contacts += 1
    return contacts, 1


@njit(cache=True)
def _contacts_from_pos(pos, N, nbrs, seq_is_H):
    contacts = 0
    for i in range(N):
        if seq_is_H[i] == 0:
            continue
        pi = int(pos[i])
        for j in range(i + 2, N):
            if seq_is_H[j] == 0:
                continue
            pj = int(pos[j])
            adjacent = 0
            for k in range(4):
                if int(nbrs[pi, k]) == pj:
                    adjacent = 1
                    break
            if adjacent == 1:
                contacts += 1
    return contacts


@njit(cache=True)
def _contact_guided_delta(dE, c_old, c_new, step, steps, contact_bias, contact_bias_final_frac, contact_paving_weight, qubo_polish_frac, contact_hist):
    if c_old < 0 or c_new < 0:
        return dE
    phase = float(step) / float(max(1, steps - 1))
    if qubo_polish_frac > 0.0 and phase >= 1.0 - qubo_polish_frac:
        return dE
    ramp = 1.0 - phase
    if ramp < contact_bias_final_frac:
        ramp = contact_bias_final_frac
    bias = contact_bias * ramp
    dc = float(c_new - c_old)
    hist_old = 0.0
    hist_new = 0.0
    if c_old >= 0 and c_old < contact_hist.shape[0]:
        hist_old = float(contact_hist[c_old])
    if c_new >= 0 and c_new < contact_hist.shape[0]:
        hist_new = float(contact_hist[c_new])
    return dE - bias * dc + contact_paving_weight * (hist_new - hist_old)


# -------------------------
# Sequence-aware initialization helpers
# -------------------------
@njit(cache=True)
def _manhattan_sites(a, b, coords_x, coords_y):
    dx = int(coords_x[a]) - int(coords_x[b])
    if dx < 0:
        dx = -dx
    dy = int(coords_y[a]) - int(coords_y[b])
    if dy < 0:
        dy = -dy
    return dx + dy


@njit(cache=True)
def _site_free_degree(site, nbrs, occ):
    cnt = 0
    for k in range(4):
        nb = int(nbrs[site, k])
        if nb >= 0 and occ[nb] == 0:
            cnt += 1
    return cnt


@njit(cache=True)
def _site_h_contacts(site, residue, pos, seq_is_H, coords_x, coords_y):
    if seq_is_H[residue] == 0:
        return 0
    cnt = 0
    for j in range(residue):
        if seq_is_H[j] == 0:
            continue
        if j == residue - 1:
            continue
        if _manhattan_sites(site, int(pos[j]), coords_x, coords_y) == 1:
            cnt += 1
    return cnt


@njit(cache=True)
def _score_growth_site(site, residue, pos, seq_is_H, coords_x, coords_y, nbrs, occ, center_x, center_y, rng):
    rng, noise = _rand_float01(rng)
    contacts = _site_h_contacts(site, residue, pos, seq_is_H, coords_x, coords_y)
    free_deg = _site_free_degree(site, nbrs, occ)
    dx = float(coords_x[site]) - center_x
    if dx < 0.0:
        dx = -dx
    dy = float(coords_y[site]) - center_y
    if dy < 0.0:
        dy = -dy
    center_penalty = dx + dy
    score = noise
    if seq_is_H[residue] == 1:
        score += 12.0 * float(contacts)
        score -= 0.20 * center_penalty
    else:
        score += 0.20 * float(free_deg)
        score -= 0.04 * center_penalty
    # Keep growth alive. A zero-free-degree H can still be useful at the end,
    # but early dead ends are expensive for long chains.
    if free_deg == 0 and residue < pos.shape[0] - 1:
        score -= 8.0
    return rng, score


@njit(cache=True)
def _pick_growth_neighbor(prev_site, residue, pos, seq_is_H, coords_x, coords_y, nbrs, occ, center_x, center_y, init_mode, rng):
    cand0 = -1
    cand1 = -1
    cand2 = -1
    cand3 = -1
    cnt = 0
    for k in range(4):
        nb = int(nbrs[prev_site, k])
        if nb >= 0 and occ[nb] == 0:
            if cnt == 0:
                cand0 = nb
            elif cnt == 1:
                cand1 = nb
            elif cnt == 2:
                cand2 = nb
            else:
                cand3 = nb
            cnt += 1
    if cnt == 0:
        return rng, -1
    if init_mode == 0:
        rng, pick = _rand_int(rng, cnt)
        if pick == 0:
            return rng, cand0
        if pick == 1:
            return rng, cand1
        if pick == 2:
            return rng, cand2
        return rng, cand3

    best_site = cand0
    best_score = -1.0e100
    for idx in range(cnt):
        site = cand0
        if idx == 1:
            site = cand1
        elif idx == 2:
            site = cand2
        elif idx == 3:
            site = cand3
        rng, score = _score_growth_site(site, residue, pos, seq_is_H, coords_x, coords_y, nbrs, occ, center_x, center_y, rng)
        if score > best_score:
            best_score = score
            best_site = site
    return rng, best_site


@njit(cache=True)
def _validate_pos(pos_candidate, N, S, nbrs, occ_tmp):
    for j in range(S):
        occ_tmp[j] = 0
    for r in range(N):
        site = int(pos_candidate[r])
        if site < 0 or site >= S:
            return 0
        if occ_tmp[site] != 0:
            return 0
        occ_tmp[site] = 1
        if r > 0:
            prev = int(pos_candidate[r - 1])
            ok_adj = 0
            for k in range(4):
                if int(nbrs[prev, k]) == site:
                    ok_adj = 1
                    break
            if ok_adj == 0:
                return 0
    return 1


@njit(cache=True)
def _score_bridge_site(site, residue, right_anchor, remaining_edges, pos, seq_is_H, coords_x, coords_y, nbrs, occ, center_x, center_y, rng):
    rng, noise = _rand_float01(rng)
    contacts = _site_h_contacts(site, residue, pos, seq_is_H, coords_x, coords_y)
    free_deg = _site_free_degree(site, nbrs, occ)
    dist = _manhattan_sites(site, right_anchor, coords_x, coords_y)
    slack = remaining_edges - dist
    score = noise
    if seq_is_H[residue] == 1:
        score += 14.0 * float(contacts)
    else:
        score += 0.15 * float(free_deg)
    score += 0.45 * float(free_deg)
    score -= 0.35 * float(dist)
    if slack >= 0:
        score += 0.20 * float(slack)
    dx = float(coords_x[site]) - center_x
    if dx < 0.0:
        dx = -dx
    dy = float(coords_y[site]) - center_y
    if dy < 0.0:
        dy = -dy
    score -= 0.04 * (dx + dy)
    return rng, score


@njit(cache=True)
def _pick_bridge_neighbor(prev_site, residue, right_anchor, remaining_edges, pos, seq_is_H, coords_x, coords_y, nbrs, occ, center_x, center_y, rng):
    cand0 = -1
    cand1 = -1
    cand2 = -1
    cand3 = -1
    cnt = 0
    for k in range(4):
        nb = int(nbrs[prev_site, k])
        if nb < 0 or occ[nb] != 0:
            continue
        dist = _manhattan_sites(nb, right_anchor, coords_x, coords_y)
        slack = remaining_edges - dist
        if slack < 0:
            continue
        # A bridge path on a square lattice can only close if the parity matches.
        if slack % 2 != 0:
            continue
        if cnt == 0:
            cand0 = nb
        elif cnt == 1:
            cand1 = nb
        elif cnt == 2:
            cand2 = nb
        else:
            cand3 = nb
        cnt += 1
    if cnt == 0:
        return rng, -1

    best_site = cand0
    best_score = -1.0e100
    for idx in range(cnt):
        site = cand0
        if idx == 1:
            site = cand1
        elif idx == 2:
            site = cand2
        elif idx == 3:
            site = cand3
        rng, score = _score_bridge_site(
            site, residue, right_anchor, remaining_edges,
            pos, seq_is_H, coords_x, coords_y, nbrs, occ, center_x, center_y, rng
        )
        if score > best_score:
            best_score = score
            best_site = site
    return rng, best_site


@njit(cache=True)
def _propose_internal_regrow(pos, pos_tmp, occ, occ_tmp, N, S, nbrs, coords_x, coords_y, seq_is_H, regrow_max_len, rng):
    for r in range(N):
        pos_tmp[r] = pos[r]
    if N < 8:
        return rng, 0, 0, 0

    max_len = regrow_max_len
    if max_len < 2:
        max_len = 2
    if max_len > N - 3:
        max_len = N - 3

    center_x = 0.5 * float(coords_x.max())
    center_y = 0.5 * float(coords_y.max())
    attempts = 12
    for _attempt in range(attempts):
        rng, seg_extra = _rand_int(rng, max_len - 1)
        seg_len = 2 + seg_extra
        if N - seg_len - 1 <= 1:
            continue
        rng, start_extra = _rand_int(rng, N - seg_len - 1)
        start = 1 + start_extra
        end = start + seg_len
        if end >= N:
            continue

        left_anchor = int(pos[start - 1])
        right_anchor = int(pos[end])
        bridge_edges = seg_len + 1
        anchor_dist = _manhattan_sites(left_anchor, right_anchor, coords_x, coords_y)
        slack0 = bridge_edges - anchor_dist
        if slack0 < 0 or slack0 % 2 != 0:
            continue

        for r in range(N):
            pos_tmp[r] = pos[r]
        for j in range(S):
            occ_tmp[j] = occ[j]
        for r in range(start, end):
            occ_tmp[int(pos[r])] = 0

        prev = left_anchor
        ok = 1
        for r in range(start, end):
            remaining_edges = end - r
            rng, site = _pick_bridge_neighbor(
                prev, r, right_anchor, remaining_edges,
                pos_tmp, seq_is_H, coords_x, coords_y, nbrs, occ_tmp,
                center_x, center_y, rng
            )
            if site < 0:
                ok = 0
                break
            pos_tmp[r] = site
            occ_tmp[site] = 1
            prev = site

        if ok == 1 and _manhattan_sites(int(pos_tmp[end - 1]), right_anchor, coords_x, coords_y) == 1:
            if _validate_pos(pos_tmp, N, S, nbrs, occ_tmp) == 1:
                return rng, 1, start, end

    return rng, 0, 0, 0


@njit(cache=True)
def _spin_distance_frac(archive_spins, slot, candidate_spins, V, distance_den):
    diff = 0
    for i in range(V):
        if archive_spins[slot, i] != candidate_spins[i]:
            diff += 1
    return float(diff) / float(max(1, distance_den))


@njit(cache=True)
def _archive_slot_min_distance(archive_spins, slot, archive_count, V, distance_den):
    if archive_count <= 1:
        return 1.0
    best = 1.0e100
    for a in range(archive_count):
        if a == slot:
            continue
        diff = 0
        for i in range(V):
            if archive_spins[slot, i] != archive_spins[a, i]:
                diff += 1
        d = float(diff) / float(max(1, distance_den))
        if d < best:
            best = d
    return best


@njit(cache=True)
def _archive_update_diverse(archive_spins, archive_e, archive_contacts, archive_count, archive_size, candidate_spins, candidate_e, candidate_contacts, V, distance_den, min_hamming_frac, contact_slack):
    if archive_size <= 0:
        return archive_count
    if archive_count < archive_size:
        slot = archive_count
        archive_count += 1
        archive_e[slot] = candidate_e
        archive_contacts[slot] = int(candidate_contacts)
        for i in range(V):
            archive_spins[slot, i] = candidate_spins[i]
        return archive_count

    best_contacts = archive_contacts[0]
    for a in range(1, archive_count):
        if archive_contacts[a] > best_contacts:
            best_contacts = archive_contacts[a]

    if int(candidate_contacts) < best_contacts - contact_slack:
        return archive_count

    min_dist = 1.0e100
    similar = 0
    for a in range(archive_count):
        d = _spin_distance_frac(archive_spins, a, candidate_spins, V, distance_den)
        if d < min_dist:
            min_dist = d
            similar = a

    if min_dist < min_hamming_frac:
        if int(candidate_contacts) > archive_contacts[similar] or (
            int(candidate_contacts) == archive_contacts[similar] and candidate_e < archive_e[similar]
        ):
            archive_e[similar] = candidate_e
            archive_contacts[similar] = int(candidate_contacts)
            for i in range(V):
                archive_spins[similar, i] = candidate_spins[i]
        return archive_count

    worst = 0
    worst_div = _archive_slot_min_distance(archive_spins, 0, archive_count, V, distance_den)
    for a in range(1, archive_count):
        div_a = _archive_slot_min_distance(archive_spins, a, archive_count, V, distance_den)
        if archive_contacts[a] < archive_contacts[worst]:
            worst = a
            worst_div = div_a
        elif archive_contacts[a] == archive_contacts[worst]:
            if div_a < worst_div:
                worst = a
                worst_div = div_a
            elif div_a == worst_div and archive_e[a] > archive_e[worst]:
                worst = a
                worst_div = div_a

    if int(candidate_contacts) > archive_contacts[worst] or int(candidate_contacts) >= best_contacts - contact_slack or min_dist > worst_div:
        archive_e[worst] = candidate_e
        archive_contacts[worst] = int(candidate_contacts)
        for i in range(V):
            archive_spins[worst, i] = candidate_spins[i]

    return archive_count


@njit(cache=True)
def _propose_pull(pos, pos_tmp, occ, occ_tmp, N, S, nbrs, xy2s, coords_x, coords_y, rng):
    for r in range(N):
        pos_tmp[r] = pos[r]
    if N < 4:
        return rng, 0, 0, 0
    rng, i = _rand_int(rng, N - 2)
    i = i + 1
    rng, backward = _rand_int(rng, 2)
    rng, side = _rand_int(rng, 2)

    ax = int(coords_x[pos[i]])
    ay = int(coords_y[pos[i]])
    bx = 0
    by = 0
    if backward == 1:
        bx = int(coords_x[pos[i + 1]])
        by = int(coords_y[pos[i + 1]])
    else:
        bx = int(coords_x[pos[i - 1]])
        by = int(coords_y[pos[i - 1]])
    vx = bx - ax
    vy = by - ay
    wx = -vy
    wy = vx
    if side == 1:
        wx = vy
        wy = -vx

    lx = bx + wx
    ly = by + wy
    cx = ax + wx
    cy = ay + wy
    if lx < 0 or ly < 0 or cx < 0 or cy < 0:
        return rng, 0, 0, 0
    if lx >= xy2s.shape[0] or ly >= xy2s.shape[1] or cx >= xy2s.shape[0] or cy >= xy2s.shape[1]:
        return rng, 0, 0, 0
    lsite = int(xy2s[lx, ly])
    csite = int(xy2s[cx, cy])
    if lsite < 0 or csite < 0:
        return rng, 0, 0, 0
    if occ[lsite] != 0:
        return rng, 0, 0, 0

    move_start = i
    move_end = i + 1
    if backward == 1:
        allowed = int(pos[i - 1])
        if occ[csite] != 0 and csite != allowed:
            return rng, 0, 0, 0
        pos_tmp[i] = lsite
        if csite != allowed:
            pos_tmp[i - 1] = csite
            prev = csite
            move_start = i - 1
            for j in range(i - 2, -1, -1):
                if _manhattan_sites(int(pos[j]), prev, coords_x, coords_y) == 1:
                    break
                pos_tmp[j] = int(pos[j + 2])
                prev = int(pos_tmp[j])
                move_start = j
        else:
            move_start = i
        move_end = i + 1
    else:
        allowed = int(pos[i + 1])
        if occ[csite] != 0 and csite != allowed:
            return rng, 0, 0, 0
        pos_tmp[i] = lsite
        if csite != allowed:
            pos_tmp[i + 1] = csite
            prev = csite
            move_end = i + 2
            for j in range(i + 2, N):
                if _manhattan_sites(int(pos[j]), prev, coords_x, coords_y) == 1:
                    break
                pos_tmp[j] = int(pos[j - 2])
                prev = int(pos_tmp[j])
                move_end = j + 1
        else:
            move_end = i + 1
        move_start = i

    if _validate_pos(pos_tmp, N, S, nbrs, occ_tmp) == 0:
        return rng, 0, 0, 0
    return rng, 1, move_start, move_end


@njit(cache=True)
def _propose_end_regrow(pos, pos_tmp, occ, occ_tmp, N, S, nbrs, coords_x, coords_y, seq_is_H, regrow_max_len, init_mode, rng):
    for r in range(N):
        pos_tmp[r] = pos[r]
    if N < 6:
        return rng, 0, 0, 0
    max_len = regrow_max_len
    if max_len < 2:
        max_len = 2
    if max_len > N - 2:
        max_len = N - 2
    rng, seg_extra = _rand_int(rng, max_len - 1)
    seg_len = 2 + seg_extra
    rng, side = _rand_int(rng, 2)
    center_x = 0.5 * float(coords_x.max())
    center_y = 0.5 * float(coords_y.max())

    for j in range(S):
        occ_tmp[j] = occ[j]

    if side == 1:
        start = N - seg_len
        if start <= 0:
            return rng, 0, 0, 0
        for r in range(start, N):
            occ_tmp[int(pos[r])] = 0
        prev = int(pos[start - 1])
        for r in range(start, N):
            rng, site = _pick_growth_neighbor(prev, r, pos_tmp, seq_is_H, coords_x, coords_y, nbrs, occ_tmp, center_x, center_y, init_mode, rng)
            if site < 0:
                return rng, 0, 0, 0
            pos_tmp[r] = site
            occ_tmp[site] = 1
            prev = site
        if _validate_pos(pos_tmp, N, S, nbrs, occ_tmp) == 0:
            return rng, 0, 0, 0
        return rng, 1, start, N

    end = seg_len
    if end >= N:
        return rng, 0, 0, 0
    for r in range(0, end):
        occ_tmp[int(pos[r])] = 0
    prev = int(pos[end])
    for rr in range(end - 1, -1, -1):
        rng, site = _pick_growth_neighbor(prev, rr, pos_tmp, seq_is_H, coords_x, coords_y, nbrs, occ_tmp, center_x, center_y, init_mode, rng)
        if site < 0:
            return rng, 0, 0, 0
        pos_tmp[rr] = site
        occ_tmp[site] = 1
        prev = site
    if _validate_pos(pos_tmp, N, S, nbrs, occ_tmp) == 0:
        return rng, 0, 0, 0
    return rng, 1, 0, end


# -------------------------
# SAW init + chain-preserving residue SA (Numba hot loop)
# -------------------------
@njit(cache=True, fastmath=True)
def _anneal_trials_saw_chain(
    trials, steps, t_init, t_final, seed, reseed_each_trial, warm_start_best, best_t_scale,
    warm_start_prob, warm_start_min_contacts,
    pivot_prob, pivot_max_tail,
    pull_prob, reptation_prob, regrow_prob, frag_regrow_prob, regrow_max_len, init_mode,
    archive_size, archive_min_hamming_frac, archive_contact_slack,
    contact_priority_best, contact_check_every,
    target_contacts, stop_on_target,
    contact_guided_accept, contact_bias, contact_bias_final_frac, contact_paving_weight, qubo_polish_frac,
    move_mode, block_size, reheat_every, reheat_factor,
    seed_best_spins, seed_best_has, seed_best_contacts,
    h_dense,
    J2_a, J2_b, J2_c,
    J3_a, J3_b, J3_k, J3_c_full,
    J4_a, J4_b, J4_k, J4_l, J4_c_full,
    ptr2, j2_idx, j2_c,
    ptr3, j3_o1, j3_o2, j3_c,
    ptr4, j4_o1, j4_o2, j4_o3, j4_c,
    nbrs, xy2s, coords_x, coords_y, seq_is_H,
):
    V = h_dense.shape[0]
    S = block_size
    N = V // S if S > 0 else 0
    archive_distance_den = 2 * N if N > 0 else V

    # output arrays
    trial_best_e = np.empty((trials,), dtype=np.float64)
    trial_best_contacts = np.empty((trials,), dtype=np.int32)
    trial_best_feasible = np.empty((trials,), dtype=np.int8)
    trial_accept_rate = np.empty((trials,), dtype=np.float64)
    best_so_far = np.empty((trials,), dtype=np.float64)
    for tr in range(trials):
        trial_best_e[tr] = 1.0e300
        trial_best_contacts[tr] = -1
        trial_best_feasible[tr] = 0
        trial_accept_rate[tr] = 0.0
        best_so_far[tr] = 1.0e300

    # temps
    beta = (t_final / t_init) ** (1.0 / float(steps - 1)) if steps > 1 else 1.0

    # state buffers
    s = np.empty((V,), dtype=h_dense.dtype)
    s_best_trial = np.empty((V,), dtype=h_dense.dtype)
    s_best_global = np.empty((V,), dtype=h_dense.dtype)
    archive_spins = np.empty((max(1, archive_size), V), dtype=h_dense.dtype)
    archive_e = np.empty((max(1, archive_size),), dtype=np.float64)
    archive_contacts = np.empty((max(1, archive_size),), dtype=np.int32)
    contact_hist = np.zeros((256,), dtype=np.int32)
    archive_count = 0
    for a in range(max(1, archive_size)):
        archive_e[a] = 1.0e300
        archive_contacts[a] = -1

    # residue-mode helpers
    pos = np.empty((N,), dtype=np.int32)
    occ = np.empty((S,), dtype=np.int8)

    # decode temps
    pos_tmp = np.empty((N,), dtype=np.int32)
    x_tmp = np.empty((N,), dtype=np.int32)
    y_tmp = np.empty((N,), dtype=np.int32)
    occ_tmp = np.empty((S,), dtype=np.int8)

    if seed_best_has == 1:
        for i in range(V):
            s_best_global[i] = seed_best_spins[i]
            archive_spins[0, i] = seed_best_spins[i]
        global_best_e = _energy_full(s_best_global, h_dense, J2_a, J2_b, J2_c, J3_a, J3_b, J3_k, J3_c_full, J4_a, J4_b, J4_k, J4_l, J4_c_full)
        global_has_best = 1
        global_best_contacts = int(seed_best_contacts)
        archive_e[0] = global_best_e
        archive_contacts[0] = int(seed_best_contacts)
        archive_count = 1
    else:
        global_best_e = 1.0e300
        global_has_best = 0
        global_best_contacts = -1

    can_decode = 1 if (coords_x.shape[0] == S and seq_is_H.shape[0] == N and N > 0) else 0

    # SAW attempts (constant; fast and safe)
    MAX_SAW_ATTEMPTS = 2048

    for tr in range(trials):
        # per-trial RNG
        if reseed_each_trial == 1:
            rng = _splitmix64(np.uint64(seed + tr + 1234567))
        else:
            rng = _splitmix64(np.uint64(seed))

        # -------- init --------
        # Warm-start is helpful for polishing, but can hurt exploration.
        # We therefore gate it by (a) contact threshold and (b) probability.
        use_warm = 0
        if warm_start_best == 1 and archive_count > 0 and move_mode == 1 and global_best_contacts >= warm_start_min_contacts:
            rng, uu = _rand_float01(rng)
            if uu < warm_start_prob:
                use_warm = 1

        if use_warm:
            # Try warm-start from a diverse in-worker seed archive. We sample
            # among near-best contact states instead of replaying one basin.
            archive_pick_i = np.int64(0)
            eligible_count = 0
            min_archive_contacts = global_best_contacts - archive_contact_slack
            for a in range(archive_count):
                if archive_contacts[a] >= min_archive_contacts:
                    eligible_count += 1
                    rng, take = _rand_int(rng, eligible_count)
                    if take == 0:
                        archive_pick_i = np.int64(a)
            if eligible_count == 0 and archive_count > 1:
                rng, archive_pick = _rand_int(rng, archive_count)
                archive_pick_i = np.int64(archive_pick)
            for i in range(V):
                s[i] = archive_spins[archive_pick_i, i]
            ok = 1
            for j in range(S):
                occ[j] = 0
            for r in range(N):
                base = r * S
                active = -1
                cnt = 0
                for j in range(S):
                    if s[base + j] > 0.0:
                        active = j
                        cnt += 1
                        if cnt > 1:
                            break
                if cnt != 1 or occ[active] != 0:
                    ok = 0
                    break
                occ[active] = 1
                pos[r] = active
            # If warm start isn't strictly chain-valid, fall back to SAW init
            if ok == 1 and can_decode == 1:
                c, feas = _decode_contacts_onehot_strict(s, seq_is_H, coords_x, coords_y, N, S, pos_tmp, x_tmp, y_tmp, occ_tmp)
                if feas == 0:
                    ok = 0
            if ok == 0:
                use_warm = 0

        if not use_warm:
            if move_mode == 1:
                # random or sequence-aware SAW init: chain-valid + collision-free
                ok = 0
                center_x = 0.5 * float(coords_x.max()) if coords_x.shape[0] > 0 else 0.0
                center_y = 0.5 * float(coords_y.max()) if coords_y.shape[0] > 0 else 0.0
                for attempt in range(MAX_SAW_ATTEMPTS):
                    for j in range(S):
                        occ[j] = 0
                    rng, start_site = _rand_int(rng, S)
                    if init_mode > 0 and coords_x.shape[0] > 0:
                        # Prefer central starts for H-rich chains but keep randomness.
                        best_start = start_site
                        best_start_score = -1.0e100
                        for _samp in range(8):
                            rng, candidate_start = _rand_int(rng, S)
                            dx0 = float(coords_x[candidate_start]) - center_x
                            if dx0 < 0.0:
                                dx0 = -dx0
                            dy0 = float(coords_y[candidate_start]) - center_y
                            if dy0 < 0.0:
                                dy0 = -dy0
                            rng, noise0 = _rand_float01(rng)
                            score0 = noise0 - 0.15 * (dx0 + dy0)
                            if score0 > best_start_score:
                                best_start_score = score0
                                best_start = candidate_start
                        start_site = best_start
                    pos[0] = start_site
                    occ[start_site] = 1
                    ok = 1
                    for r in range(1, N):
                        prev = pos[r - 1]
                        rng, site = _pick_growth_neighbor(prev, r, pos, seq_is_H, coords_x, coords_y, nbrs, occ, center_x, center_y, init_mode, rng)
                        if site < 0:
                            ok = 0
                            break
                        pos[r] = site
                        occ[site] = 1
                    if ok == 1:
                        break

                # build spins from pos
                for i in range(V):
                    s[i] = -1.0
                for r in range(N):
                    s[r * S + pos[r]] = 1.0
            else:
                # single-bit: random +/-1
                for i in range(V):
                    rng, u = _rand_float01(rng)
                    s[i] = 1.0 if u < 0.5 else -1.0

        # energy init
        e = _energy_full(s, h_dense, J2_a, J2_b, J2_c, J3_a, J3_b, J3_k, J3_c_full, J4_a, J4_b, J4_k, J4_l, J4_c_full)
        if use_warm == 1:
            T = float(t_init) * float(best_t_scale)
        else:
            T = float(t_init)

        # per-trial best
        e_best = e
        c_best_trial = -1
        cur_contacts = -1
        if can_decode == 1:
            cur_contacts = _contacts_from_pos(pos, N, nbrs, seq_is_H)
            if cur_contacts >= 0 and cur_contacts < contact_hist.shape[0]:
                contact_hist[cur_contacts] += 1
        if contact_priority_best == 1 and can_decode == 1:
            c_best_trial = cur_contacts
        for i in range(V):
            s_best_trial[i] = s[i]
        last_improve = 0
        accepted = 0

        # -------- main loop --------
        for step in range(steps):
            did_move = 0

            if move_mode == 1:
                # Optional slithering-snake/reptation move: remove one end and
                # grow one legal site at the other end, shifting residue
                # identities along the chain. This is useful for escaping
                # compact 35-contact basins without destroying SAW feasibility.
                rng, up_rept = _rand_float01(rng)
                if did_move == 0 and reptation_prob > 0.0 and up_rept < reptation_prob and N > 3:
                    rng, direction = _rand_int(rng, 2)
                    ok_rept = 0
                    cnt_rept = 0
                    cand0_rept = -1
                    cand1_rept = -1
                    cand2_rept = -1
                    cand3_rept = -1
                    for j in range(S):
                        occ_tmp[j] = occ[j]

                    if direction == 0:
                        # Grow a new N-terminal site adjacent to residue 0 and
                        # drop the current C-terminal site.
                        occ_tmp[int(pos[N - 1])] = 0
                        ref = int(pos[0])
                        for k in range(4):
                            nb = int(nbrs[ref, k])
                            if nb >= 0 and occ_tmp[nb] == 0:
                                if cnt_rept == 0:
                                    cand0_rept = nb
                                elif cnt_rept == 1:
                                    cand1_rept = nb
                                elif cnt_rept == 2:
                                    cand2_rept = nb
                                else:
                                    cand3_rept = nb
                                cnt_rept += 1
                        if cnt_rept > 0:
                            rng, pick_rept = _rand_int(rng, cnt_rept)
                            new_site_rept = cand0_rept
                            if pick_rept == 1:
                                new_site_rept = cand1_rept
                            elif pick_rept == 2:
                                new_site_rept = cand2_rept
                            elif pick_rept == 3:
                                new_site_rept = cand3_rept
                            pos_tmp[0] = new_site_rept
                            for r2 in range(1, N):
                                pos_tmp[r2] = pos[r2 - 1]
                            ok_rept = 1
                    else:
                        # Grow a new C-terminal site adjacent to residue N-1
                        # and drop the current N-terminal site.
                        occ_tmp[int(pos[0])] = 0
                        ref = int(pos[N - 1])
                        for k in range(4):
                            nb = int(nbrs[ref, k])
                            if nb >= 0 and occ_tmp[nb] == 0:
                                if cnt_rept == 0:
                                    cand0_rept = nb
                                elif cnt_rept == 1:
                                    cand1_rept = nb
                                elif cnt_rept == 2:
                                    cand2_rept = nb
                                else:
                                    cand3_rept = nb
                                cnt_rept += 1
                        if cnt_rept > 0:
                            rng, pick_rept = _rand_int(rng, cnt_rept)
                            new_site_rept = cand0_rept
                            if pick_rept == 1:
                                new_site_rept = cand1_rept
                            elif pick_rept == 2:
                                new_site_rept = cand2_rept
                            elif pick_rept == 3:
                                new_site_rept = cand3_rept
                            for r2 in range(N - 1):
                                pos_tmp[r2] = pos[r2 + 1]
                            pos_tmp[N - 1] = new_site_rept
                            ok_rept = 1

                    if ok_rept == 1:
                        dE = 0.0
                        for r2 in range(N):
                            old_site = int(pos[r2])
                            new_site = int(pos_tmp[r2])
                            if old_site == new_site:
                                continue
                            old_idx = r2 * S + old_site
                            new_idx = r2 * S + new_site
                            dE += _deltaE_flip(old_idx, s, h_dense, ptr2, j2_idx, j2_c,
                                               ptr3, j3_o1, j3_o2, j3_c,
                                               ptr4, j4_o1, j4_o2, j4_o3, j4_c)
                            s[old_idx] = -s[old_idx]
                            dE += _deltaE_flip(new_idx, s, h_dense, ptr2, j2_idx, j2_c,
                                               ptr3, j3_o1, j3_o2, j3_c,
                                               ptr4, j4_o1, j4_o2, j4_o3, j4_c)
                            s[new_idx] = -s[new_idx]

                        c_candidate = cur_contacts
                        dE_test = dE
                        if contact_guided_accept == 1 and can_decode == 1:
                            c_candidate = _contacts_from_pos(pos_tmp, N, nbrs, seq_is_H)
                            dE_test = _contact_guided_delta(
                                dE, cur_contacts, c_candidate, step, steps,
                                contact_bias, contact_bias_final_frac, contact_paving_weight,
                                qubo_polish_frac, contact_hist
                            )
                        accept_move = 0
                        if dE_test <= 0.0:
                            accept_move = 1
                        else:
                            rng, u = _rand_float01(rng)
                            if u < math.exp(-dE_test / max(1e-12, T)):
                                accept_move = 1
                        if accept_move == 1:
                            accepted += 1
                            e += dE
                            for j in range(S):
                                occ[j] = 0
                            for r2 in range(N):
                                pos[r2] = int(pos_tmp[r2])
                                occ[int(pos[r2])] = 1
                            cur_contacts = c_candidate
                            if cur_contacts >= 0 and cur_contacts < contact_hist.shape[0]:
                                contact_hist[cur_contacts] += 1
                            did_move = 1
                        else:
                            for r2 in range(N - 1, -1, -1):
                                old_site = int(pos[r2])
                                new_site = int(pos_tmp[r2])
                                if old_site == new_site:
                                    continue
                                old_idx = r2 * S + old_site
                                new_idx = r2 * S + new_site
                                s[new_idx] = -s[new_idx]
                                s[old_idx] = -s[old_idx]

                # Optional pull-like non-local move before pivot/local moves.
                rng, up_pull = _rand_float01(rng)
                if did_move == 0 and pull_prob > 0.0 and up_pull < pull_prob and N > 3:
                    rng, ok_pull, move_start_pull, move_end_pull = _propose_pull(
                        pos, pos_tmp, occ, occ_tmp, N, S, nbrs, xy2s, coords_x, coords_y, rng
                    )
                    if ok_pull == 1:
                        dE = 0.0
                        for r2 in range(move_start_pull, move_end_pull):
                            old_site = int(pos[r2])
                            new_site = int(pos_tmp[r2])
                            if old_site == new_site:
                                continue
                            old_idx = r2 * S + old_site
                            new_idx = r2 * S + new_site
                            dE += _deltaE_flip(old_idx, s, h_dense, ptr2, j2_idx, j2_c,
                                               ptr3, j3_o1, j3_o2, j3_c,
                                               ptr4, j4_o1, j4_o2, j4_o3, j4_c)
                            s[old_idx] = -s[old_idx]
                            dE += _deltaE_flip(new_idx, s, h_dense, ptr2, j2_idx, j2_c,
                                               ptr3, j3_o1, j3_o2, j3_c,
                                               ptr4, j4_o1, j4_o2, j4_o3, j4_c)
                            s[new_idx] = -s[new_idx]

                        c_candidate = cur_contacts
                        dE_test = dE
                        if contact_guided_accept == 1 and can_decode == 1:
                            c_candidate = _contacts_from_pos(pos_tmp, N, nbrs, seq_is_H)
                            dE_test = _contact_guided_delta(
                                dE, cur_contacts, c_candidate, step, steps,
                                contact_bias, contact_bias_final_frac, contact_paving_weight,
                                qubo_polish_frac, contact_hist
                            )
                        accept_move = 0
                        if dE_test <= 0.0:
                            accept_move = 1
                        else:
                            rng, u = _rand_float01(rng)
                            if u < math.exp(-dE_test / max(1e-12, T)):
                                accept_move = 1
                        if accept_move == 1:
                            accepted += 1
                            e += dE
                            for r2 in range(move_start_pull, move_end_pull):
                                old_site = int(pos[r2])
                                new_site = int(pos_tmp[r2])
                                if old_site != new_site:
                                    occ[old_site] = 0
                                    occ[new_site] = 1
                                    pos[r2] = new_site
                            cur_contacts = c_candidate
                            if cur_contacts >= 0 and cur_contacts < contact_hist.shape[0]:
                                contact_hist[cur_contacts] += 1
                            did_move = 1
                        else:
                            for r2 in range(move_end_pull - 1, move_start_pull - 1, -1):
                                old_site = int(pos[r2])
                                new_site = int(pos_tmp[r2])
                                if old_site == new_site:
                                    continue
                                old_idx = r2 * S + old_site
                                new_idx = r2 * S + new_site
                                s[new_idx] = -s[new_idx]
                                s[old_idx] = -s[old_idx]

                # Optional internal bridge-regrowth. This rebuilds a
                # middle fragment while keeping both anchors fixed.
                rng, up_frag = _rand_float01(rng)
                if did_move == 0 and frag_regrow_prob > 0.0 and up_frag < frag_regrow_prob and N > 7:
                    rng, ok_frag, move_start_frag, move_end_frag = _propose_internal_regrow(
                        pos, pos_tmp, occ, occ_tmp, N, S, nbrs, coords_x, coords_y,
                        seq_is_H, regrow_max_len, rng
                    )
                    if ok_frag == 1:
                        dE = 0.0
                        for r2 in range(move_start_frag, move_end_frag):
                            old_site = int(pos[r2])
                            new_site = int(pos_tmp[r2])
                            if old_site == new_site:
                                continue
                            old_idx = r2 * S + old_site
                            new_idx = r2 * S + new_site
                            dE += _deltaE_flip(old_idx, s, h_dense, ptr2, j2_idx, j2_c,
                                               ptr3, j3_o1, j3_o2, j3_c,
                                               ptr4, j4_o1, j4_o2, j4_o3, j4_c)
                            s[old_idx] = -s[old_idx]
                            dE += _deltaE_flip(new_idx, s, h_dense, ptr2, j2_idx, j2_c,
                                               ptr3, j3_o1, j3_o2, j3_c,
                                               ptr4, j4_o1, j4_o2, j4_o3, j4_c)
                            s[new_idx] = -s[new_idx]

                        c_candidate = cur_contacts
                        dE_test = dE
                        if contact_guided_accept == 1 and can_decode == 1:
                            c_candidate = _contacts_from_pos(pos_tmp, N, nbrs, seq_is_H)
                            dE_test = _contact_guided_delta(
                                dE, cur_contacts, c_candidate, step, steps,
                                contact_bias, contact_bias_final_frac, contact_paving_weight,
                                qubo_polish_frac, contact_hist
                            )
                        accept_move = 0
                        if dE_test <= 0.0:
                            accept_move = 1
                        else:
                            rng, u = _rand_float01(rng)
                            if u < math.exp(-dE_test / max(1e-12, T)):
                                accept_move = 1
                        if accept_move == 1:
                            accepted += 1
                            e += dE
                            for r2 in range(move_start_frag, move_end_frag):
                                old_site = int(pos[r2])
                                new_site = int(pos_tmp[r2])
                                if old_site != new_site:
                                    occ[old_site] = 0
                                    occ[new_site] = 1
                                    pos[r2] = new_site
                            cur_contacts = c_candidate
                            if cur_contacts >= 0 and cur_contacts < contact_hist.shape[0]:
                                contact_hist[cur_contacts] += 1
                            did_move = 1
                        else:
                            for r2 in range(move_end_frag - 1, move_start_frag - 1, -1):
                                old_site = int(pos[r2])
                                new_site = int(pos_tmp[r2])
                                if old_site == new_site:
                                    continue
                                old_idx = r2 * S + old_site
                                new_idx = r2 * S + new_site
                                s[new_idx] = -s[new_idx]
                                s[old_idx] = -s[old_idx]

                # Optional sequence-aware endpoint segment regrowth.
                rng, up_regrow = _rand_float01(rng)
                if did_move == 0 and regrow_prob > 0.0 and up_regrow < regrow_prob and N > 5:
                    rng, ok_regrow, move_start_regrow, move_end_regrow = _propose_end_regrow(
                        pos, pos_tmp, occ, occ_tmp, N, S, nbrs, coords_x, coords_y,
                        seq_is_H, regrow_max_len, init_mode, rng
                    )
                    if ok_regrow == 1:
                        dE = 0.0
                        for r2 in range(move_start_regrow, move_end_regrow):
                            old_site = int(pos[r2])
                            new_site = int(pos_tmp[r2])
                            if old_site == new_site:
                                continue
                            old_idx = r2 * S + old_site
                            new_idx = r2 * S + new_site
                            dE += _deltaE_flip(old_idx, s, h_dense, ptr2, j2_idx, j2_c,
                                               ptr3, j3_o1, j3_o2, j3_c,
                                               ptr4, j4_o1, j4_o2, j4_o3, j4_c)
                            s[old_idx] = -s[old_idx]
                            dE += _deltaE_flip(new_idx, s, h_dense, ptr2, j2_idx, j2_c,
                                               ptr3, j3_o1, j3_o2, j3_c,
                                               ptr4, j4_o1, j4_o2, j4_o3, j4_c)
                            s[new_idx] = -s[new_idx]

                        c_candidate = cur_contacts
                        dE_test = dE
                        if contact_guided_accept == 1 and can_decode == 1:
                            c_candidate = _contacts_from_pos(pos_tmp, N, nbrs, seq_is_H)
                            dE_test = _contact_guided_delta(
                                dE, cur_contacts, c_candidate, step, steps,
                                contact_bias, contact_bias_final_frac, contact_paving_weight,
                                qubo_polish_frac, contact_hist
                            )
                        accept_move = 0
                        if dE_test <= 0.0:
                            accept_move = 1
                        else:
                            rng, u = _rand_float01(rng)
                            if u < math.exp(-dE_test / max(1e-12, T)):
                                accept_move = 1
                        if accept_move == 1:
                            accepted += 1
                            e += dE
                            for r2 in range(move_start_regrow, move_end_regrow):
                                old_site = int(pos[r2])
                                new_site = int(pos_tmp[r2])
                                if old_site != new_site:
                                    occ[old_site] = 0
                                    occ[new_site] = 1
                                    pos[r2] = new_site
                            cur_contacts = c_candidate
                            if cur_contacts >= 0 and cur_contacts < contact_hist.shape[0]:
                                contact_hist[cur_contacts] += 1
                            did_move = 1
                        else:
                            for r2 in range(move_end_regrow - 1, move_start_regrow - 1, -1):
                                old_site = int(pos[r2])
                                new_site = int(pos_tmp[r2])
                                if old_site == new_site:
                                    continue
                                old_idx = r2 * S + old_site
                                new_idx = r2 * S + new_site
                                s[new_idx] = -s[new_idx]
                                s[old_idx] = -s[old_idx]

                # With small probability, attempt a non-local pivot move.
                # Pivot rotates either side of the pivot and includes 90/180/270 degree rotations.
                rng, up = _rand_float01(rng)
                if did_move == 0 and up < pivot_prob and N > 3:
                    # choose a pivot residue k (avoid endpoints)
                    rng, k = _rand_int(rng, N - 2)
                    k = k + 1  # now k in [1, N-2]
                    # choose moved side: 0=head/N-terminal side, 1=tail/C-terminal side
                    rng, pivot_side = _rand_int(rng, 2)
                    # choose rotation: 0=cw, 1=ccw, 2=180
                    rng, rot = _rand_int(rng, 3)

                    # moved segment. pivot_max_tail=0 means full selected side.
                    move_start = k + 1
                    move_end = N
                    if pivot_side == 0:
                        move_start = 0
                        move_end = k
                        if pivot_max_tail > 0:
                            move_start = k - pivot_max_tail
                            if move_start < 0:
                                move_start = 0
                    else:
                        move_start = k + 1
                        move_end = N
                        if pivot_max_tail > 0:
                            move_end = k + 1 + pivot_max_tail
                            if move_end > N:
                                move_end = N

                    # pivot point coordinates
                    pk = int(pos[k])
                    xk = int(coords_x[pk]); yk = int(coords_y[pk])

                    # propose new sites for residues in selected moved segment
                    ok_pivot = 1
                    # use a small local buffer (we reuse pos_tmp as scratch indices for new sites)
                    # store new site for each moved residue at pos_tmp[r]
                    for r2 in range(move_start, move_end):
                        oldp = int(pos[r2])
                        xr = int(coords_x[oldp]); yr = int(coords_y[oldp])
                        dx = xr - xk
                        dy = yr - yk
                        if rot == 0:
                            # clockwise: (dx,dy)->(dy,-dx)
                            nx = xk + dy
                            ny = yk - dx
                        else:
                            # ccw: (dx,dy)->(-dy,dx)
                            nx = xk - dy
                            ny = yk + dx
                        if rot == 2:
                            # 180: (dx,dy)->(-dx,-dy)
                            nx = xk - dx
                            ny = yk - dy

                        if nx < 0 or ny < 0 or nx >= xy2s.shape[0] or ny >= xy2s.shape[1]:
                            ok_pivot = 0
                            break
                        ns = int(xy2s[nx, ny])
                        if ns < 0:
                            ok_pivot = 0
                            break
                        pos_tmp[r2] = ns

                    if ok_pivot == 1:
                        # If pivot_max_tail creates a partial segment, verify chain continuity
                        # across the stationary/moved boundary. Full-side pivots preserve this by geometry.
                        if move_start > 0:
                            ns = int(pos_tmp[move_start])
                            left = int(pos[move_start - 1])
                            if abs(int(coords_x[ns]) - int(coords_x[left])) + abs(int(coords_y[ns]) - int(coords_y[left])) != 1:
                                ok_pivot = 0
                        if ok_pivot == 1 and move_end < N:
                            ns = int(pos_tmp[move_end - 1])
                            right = int(pos[move_end])
                            if abs(int(coords_x[ns]) - int(coords_x[right])) + abs(int(coords_y[ns]) - int(coords_y[right])) != 1:
                                ok_pivot = 0

                    if ok_pivot == 1:
                        # collision check against fixed part and within moved segment
                        # mark fixed occupancy excluding the moving segment sites
                        for j in range(S):
                            occ_tmp[j] = occ[j]
                        for r2 in range(move_start, move_end):
                            occ_tmp[int(pos[r2])] = 0

                        for r2 in range(move_start, move_end):
                            ns = int(pos_tmp[r2])
                            if occ_tmp[ns] != 0:
                                ok_pivot = 0
                                break
                            occ_tmp[ns] = 1

                    if ok_pivot == 1:
                        # Apply multi-residue update virtually (flip spins in-place) and compute Î”E via sequential flips.
                        dE = 0.0
                        for r2 in range(move_start, move_end):
                            old_site = int(pos[r2])
                            new_site = int(pos_tmp[r2])
                            if old_site == new_site:
                                continue
                            old_idx = r2 * S + old_site
                            new_idx = r2 * S + new_site

                            dE += _deltaE_flip(old_idx, s, h_dense, ptr2, j2_idx, j2_c,
                                               ptr3, j3_o1, j3_o2, j3_c,
                                               ptr4, j4_o1, j4_o2, j4_o3, j4_c)
                            s[old_idx] = -s[old_idx]
                            dE += _deltaE_flip(new_idx, s, h_dense, ptr2, j2_idx, j2_c,
                                               ptr3, j3_o1, j3_o2, j3_c,
                                               ptr4, j4_o1, j4_o2, j4_o3, j4_c)
                            s[new_idx] = -s[new_idx]

                        # Metropolis for pivot
                        c_candidate = cur_contacts
                        dE_test = dE
                        if contact_guided_accept == 1 and can_decode == 1:
                            for r2 in range(0, move_start):
                                pos_tmp[r2] = pos[r2]
                            for r2 in range(move_end, N):
                                pos_tmp[r2] = pos[r2]
                            c_candidate = _contacts_from_pos(pos_tmp, N, nbrs, seq_is_H)
                            dE_test = _contact_guided_delta(
                                dE, cur_contacts, c_candidate, step, steps,
                                contact_bias, contact_bias_final_frac, contact_paving_weight,
                                qubo_polish_frac, contact_hist
                            )
                        accept_move = 0
                        if dE_test <= 0.0:
                            accept_move = 1
                        else:
                            rng, u = _rand_float01(rng)
                            if u < math.exp(-dE_test / max(1e-12, T)):
                                accept_move = 1

                        if accept_move == 1:
                            accepted += 1
                            e += dE
                            # update occupancy and positions
                            for r2 in range(move_start, move_end):
                                old_site = int(pos[r2])
                                new_site = int(pos_tmp[r2])
                                if old_site != new_site:
                                    occ[old_site] = 0
                                    occ[new_site] = 1
                                    pos[r2] = new_site
                            cur_contacts = c_candidate
                            if cur_contacts >= 0 and cur_contacts < contact_hist.shape[0]:
                                contact_hist[cur_contacts] += 1
                            did_move = 1
                        else:
                            # revert flips
                            for r2 in range(move_end - 1, move_start - 1, -1):
                                old_site = int(pos[r2])
                                new_site = int(pos_tmp[r2])
                                if old_site == new_site:
                                    continue
                                old_idx = r2 * S + old_site
                                new_idx = r2 * S + new_site
                                s[new_idx] = -s[new_idx]
                                s[old_idx] = -s[old_idx]

                        if did_move == 1:
                            # skip local move this step
                            pass
                        else:
                            # fall through to local move attempts
                            pass

                if did_move == 0:
                    # If a pivot already landed, do not also take a local residue move in the same SA step.
                    for _try in range(4):
                        rng, r = _rand_int(rng, N)
                        old_site = int(pos[r])

                        # candidate sites that preserve chain adjacency + collision-free
                        cand0 = -1
                        cand1 = -1
                        cand2 = -1
                        cand3 = -1
                        cnt = 0

                        if r == 0:
                            ref = int(pos[1])
                            for k in range(4):
                                nb = int(nbrs[ref, k])
                                if nb >= 0 and nb != old_site and occ[nb] == 0:
                                    if cnt == 0:
                                        cand0 = nb
                                    elif cnt == 1:
                                        cand1 = nb
                                    elif cnt == 2:
                                        cand2 = nb
                                    else:
                                        cand3 = nb
                                    cnt += 1
                        elif r == N - 1:
                            ref = int(pos[N - 2])
                            for k in range(4):
                                nb = int(nbrs[ref, k])
                                if nb >= 0 and nb != old_site and occ[nb] == 0:
                                    if cnt == 0:
                                        cand0 = nb
                                    elif cnt == 1:
                                        cand1 = nb
                                    elif cnt == 2:
                                        cand2 = nb
                                    else:
                                        cand3 = nb
                                    cnt += 1
                        else:
                            a = int(pos[r - 1])
                            b = int(pos[r + 1])
                            # intersection neighbors(a) âˆ© neighbors(b)
                            for k in range(4):
                                nb = int(nbrs[a, k])
                                if nb < 0 or nb == old_site or occ[nb] != 0:
                                    continue
                                # check if nb is a neighbor of b
                                is_nb = 0
                                for t2 in range(4):
                                    if int(nbrs[b, t2]) == nb:
                                        is_nb = 1
                                        break
                                if is_nb == 1:
                                    if cnt == 0:
                                        cand0 = nb
                                    elif cnt == 1:
                                        cand1 = nb
                                    elif cnt == 2:
                                        cand2 = nb
                                    else:
                                        cand3 = nb
                                    cnt += 1

                        if cnt == 0:
                            continue

                        rng, pick = _rand_int(rng, cnt)
                        if pick == 0:
                            new_site = cand0
                        elif pick == 1:
                            new_site = cand1
                        elif pick == 2:
                            new_site = cand2
                        else:
                            new_site = cand3

                        old_idx = r * S + old_site
                        new_idx = r * S + new_site

                        # exact Î”E via two sequential flips
                        dE1 = _deltaE_flip(old_idx, s, h_dense, ptr2, j2_idx, j2_c, ptr3, j3_o1, j3_o2, j3_c, ptr4, j4_o1, j4_o2, j4_o3, j4_c)
                        s[old_idx] = -s[old_idx]
                        dE2 = _deltaE_flip(new_idx, s, h_dense, ptr2, j2_idx, j2_c, ptr3, j3_o1, j3_o2, j3_c, ptr4, j4_o1, j4_o2, j4_o3, j4_c)
                        s[old_idx] = -s[old_idx]
                        dE = dE1 + dE2

                        # Metropolis
                        c_candidate = cur_contacts
                        dE_test = dE
                        if contact_guided_accept == 1 and can_decode == 1:
                            for r2 in range(N):
                                pos_tmp[r2] = pos[r2]
                            pos_tmp[r] = new_site
                            c_candidate = _contacts_from_pos(pos_tmp, N, nbrs, seq_is_H)
                            dE_test = _contact_guided_delta(
                                dE, cur_contacts, c_candidate, step, steps,
                                contact_bias, contact_bias_final_frac, contact_paving_weight,
                                qubo_polish_frac, contact_hist
                            )
                        accept_move = 0
                        if dE_test <= 0.0:
                            accept_move = 1
                        else:
                            rng, u = _rand_float01(rng)
                            if u < math.exp(-dE_test / max(1e-12, T)):
                                accept_move = 1

                        if accept_move == 1:
                            # commit
                            accepted += 1
                            s[old_idx] = -s[old_idx]
                            s[new_idx] = -s[new_idx]
                            e += dE
                            occ[old_site] = 0
                            occ[new_site] = 1
                            pos[r] = new_site
                            cur_contacts = c_candidate
                            if cur_contacts >= 0 and cur_contacts < contact_hist.shape[0]:
                                contact_hist[cur_contacts] += 1
                        did_move = 1
                        break  # done with this step

            else:
                # single-bit flip SA
                rng, idx = _rand_int(rng, V)
                dE = _deltaE_flip(idx, s, h_dense, ptr2, j2_idx, j2_c, ptr3, j3_o1, j3_o2, j3_c, ptr4, j4_o1, j4_o2, j4_o3, j4_c)

                accept_move = 0
                if dE <= 0.0:
                    accept_move = 1
                else:
                    rng, u = _rand_float01(rng)
                    if u < math.exp(-dE / max(1e-12, T)):
                        accept_move = 1
                if accept_move == 1:
                    accepted += 1
                    s[idx] = -s[idx]
                    e += dE
                did_move = 1

            # update best
            if contact_priority_best == 1 and can_decode == 1:
                if did_move == 1 and (contact_check_every <= 1 or (step % contact_check_every) == 0):
                    c_now = _contacts_from_pos(pos, N, nbrs, seq_is_H)
                    if c_now > c_best_trial or (c_now == c_best_trial and e < e_best):
                        c_best_trial = c_now
                        e_best = e
                        for i in range(V):
                            s_best_trial[i] = s[i]
                        last_improve = step
            else:
                if did_move == 1 and e < e_best:
                    e_best = e
                    for i in range(V):
                        s_best_trial[i] = s[i]
                    last_improve = step
            # global best updated at end-of-trial (to keep it feasible)

            # reheating
            if reheat_every > 0 and reheat_factor > 1.0 and (step - last_improve) >= reheat_every:
                T = min(t_init, T * reheat_factor)
                last_improve = step

            T *= beta

        # decode contacts for best-of-trial (strict)
        if can_decode == 1:
            c, feas = _decode_contacts_onehot_strict(s_best_trial, seq_is_H, coords_x, coords_y, N, S, pos_tmp, x_tmp, y_tmp, occ_tmp)
        else:
            c, feas = -1, 0

        # Update global best only using the per-trial best state (feasible by construction in residue mode).
        # This avoids 'melting' a good seed mid-trial and makes warm-start more reliable.
        if contact_priority_best == 1 and feas == 1:
            if int(c) > global_best_contacts or (int(c) == global_best_contacts and e_best < global_best_e):
                global_best_e = e_best
                global_has_best = 1
                global_best_contacts = int(c)
                for i in range(V):
                    s_best_global[i] = s_best_trial[i]
        else:
            if e_best < global_best_e:
                global_best_e = e_best
                global_has_best = 1
                global_best_contacts = int(c)
                for i in range(V):
                    s_best_global[i] = s_best_trial[i]
            elif e_best == global_best_e and int(c) > global_best_contacts:
                # tie-break: prefer higher-contact solution at equal energy
                global_best_contacts = int(c)
                for i in range(V):
                    s_best_global[i] = s_best_trial[i]

        # Seed archive: keep near-best but structurally diverse seeds
        # rather than repeatedly replaying only the highest-contact basin.
        if feas == 1 and archive_size > 0:
            archive_count = _archive_update_diverse(
                archive_spins, archive_e, archive_contacts, archive_count, archive_size,
                s_best_trial, e_best, int(c), V, archive_distance_den,
                archive_min_hamming_frac, archive_contact_slack
            )

        trial_best_e[tr] = e_best
        trial_best_contacts[tr] = int(c)
        trial_best_feasible[tr] = int(feas)
        trial_accept_rate[tr] = float(accepted) / float(max(1, steps))
        best_so_far[tr] = global_best_e

        if stop_on_target == 1 and target_contacts > 0 and feas == 1 and int(c) >= target_contacts:
            return global_best_e, s_best_global, trial_best_e, trial_best_contacts, trial_best_feasible, trial_accept_rate, best_so_far, tr + 1

    return global_best_e, s_best_global, trial_best_e, trial_best_contacts, trial_best_feasible, trial_accept_rate, best_so_far, trials


def main():
    ap = argparse.ArgumentParser(description="Numba-JIT CPU SA with sequence-aware SAW init, diverse seed archive, pivot, pull-like, endpoint regrow, and internal fragment regrow moves.")
    ap.add_argument("--ising", required=True, type=str, help="Path to exported QUBO/Ising JSON.")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--t_init", type=float, default=4000.0)
    ap.add_argument("--t_final", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")  # accepted for runner compatibility; CPU only here
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    ap.add_argument("--move_mode", type=str, default="residue", choices=["single", "residue"])
    ap.add_argument("--block_size", type=int, default=0, help="If 0, inferred from JSON S (coords length).")
    ap.add_argument("--reseed_each_trial", action="store_true")
    ap.add_argument("--warm_start_best", action="store_true")
    ap.add_argument("--best_t_scale", type=float, default=1.0)
    ap.add_argument("--warm_start_prob", type=float, default=0.30,
                    help="Probability of using warm-start when available (default 0.30).")
    ap.add_argument("--warm_start_min_contacts", type=int, default=13,
                    help="Only warm-start if best-so-far has at least this many contacts (default 13).")
    ap.add_argument("--pivot_prob", type=float, default=0.02,
                    help="Probability of proposing a pivot move (default 0.02).")
    ap.add_argument("--pivot_max_tail", type=int, default=0,
                    help="If >0, limit pivot tail length to this many residues for speed (default 0 = no limit).")
    ap.add_argument("--pull_prob", type=float, default=0.0,
                    help="Probability of proposing a pull-like move before pivot/local moves.")
    ap.add_argument("--reptation_prob", type=float, default=0.0,
                    help="Probability of proposing a slithering-snake/reptation move before pull/pivot/local moves.")
    ap.add_argument("--regrow_prob", type=float, default=0.0,
                    help="Probability of proposing an endpoint segment regrowth move before pivot/local moves.")
    ap.add_argument("--frag_regrow_prob", type=float, default=0.0,
                    help="Probability of proposing an internal bridge-fragment regrowth move before pivot/local moves.")
    ap.add_argument("--regrow_max_len", type=int, default=12,
                    help="Maximum segment length for endpoint and internal regrowth proposals.")
    ap.add_argument("--init_mode", type=str, default="seq_greedy",
                    choices=["random_saw", "seq_greedy"],
                    help="Initialization mode for non-warm trials.")
    ap.add_argument("--archive_size", type=int, default=8,
                    help="Number of high-contact seeds kept per worker.")
    ap.add_argument("--archive_min_hamming_frac", type=float, default=0.06,
                    help="Minimum spin-vector Hamming fraction used before adding a distinct archive basin.")
    ap.add_argument("--archive_contact_slack", type=int, default=3,
                    help="Archive keeps candidates within this contact gap from the current archive best.")
    ap.add_argument("--contact_priority_best", action="store_true",
                    help="Save trial/global best by feasible contact count first, then QUBO energy.")
    ap.add_argument("--contact_check_every", type=int, default=1,
                    help="Accepted-move contact-best check period in SA steps when contact_priority_best is enabled.")
    ap.add_argument("--target_contacts", type=int, default=0,
                    help="Target contact count used by --stop_on_target.")
    ap.add_argument("--stop_on_target", action="store_true",
                    help="Stop this worker after a completed trial reaches --target_contacts.")
    ap.add_argument("--contact_guided_accept", action="store_true",
                    help="Use fix7 contact-guided effective Metropolis acceptance during exploration.")
    ap.add_argument("--contact_bias", type=float, default=0.0,
                    help="Effective-energy reward per contact increase during contact-guided exploration.")
    ap.add_argument("--contact_bias_final_frac", type=float, default=0.10,
                    help="Lower bound fraction for contact_bias before the pure-QUBO polish phase.")
    ap.add_argument("--contact_paving_weight", type=float, default=0.0,
                    help="Penalty for revisiting the same contact-count band during contact-guided exploration.")
    ap.add_argument("--qubo_polish_frac", type=float, default=0.20,
                    help="Final fraction of each anneal that ignores contact guidance and uses pure QUBO energy.")
    ap.add_argument("--reheat_every", type=int, default=0)
    ap.add_argument("--reheat_factor", type=float, default=1.0)
    ap.add_argument("--warm_start_file", type=str, default="",
                    help="Optional JSON file whose spins seed global best before trial 0.")
    ap.add_argument("--save_best", type=str, default="", help="Write best JSON to this path.")
    # accept --gpu to avoid runner failures (ignored on CPU)
    ap.add_argument("--gpu", type=str, default="", help="(ignored)")

    args = ap.parse_args()

    dtype = np.float32 if args.dtype == "float32" else np.float64
    data = load_ising(args.ising)

    coords_x, coords_y, seq_is_H = _prepare_coords_and_seq(data)
    nbrs = _prepare_neighbors(coords_x, coords_y)
    xy2s = _prepare_xy2s(coords_x, coords_y)
    xy2s = _prepare_xy2s(coords_x, coords_y)

    terms = prepare_terms_cpu(data, dtype=dtype)
    V = int(terms["V"])
    S = int(coords_x.shape[0]) if coords_x.shape[0] > 0 else 0
    block_size = int(args.block_size) if args.block_size > 0 else S

    move_mode = 1 if args.move_mode == "residue" else 0
    seed_best_spins, seed_best_has, seed_best_contacts = _load_warm_start_seed(str(args.warm_start_file), V, dtype, data)
    warm_start_flag = 1 if (args.warm_start_best or args.warm_start_file) else 0

    t0 = time.time()
    best_e, best_spins, trial_best_e, trial_best_contacts, trial_best_feasible, trial_accept_rate, best_so_far, completed_trials = _anneal_trials_saw_chain(
        int(args.trials), int(args.steps), float(args.t_init), float(args.t_final),
        int(args.seed),
        1 if args.reseed_each_trial else 0,
        int(warm_start_flag),
        float(args.best_t_scale),
        float(args.warm_start_prob),
        int(args.warm_start_min_contacts),
        float(args.pivot_prob),
        int(args.pivot_max_tail),
        float(args.pull_prob),
        float(args.reptation_prob),
        float(args.regrow_prob),
        float(args.frag_regrow_prob),
        int(args.regrow_max_len),
        0 if args.init_mode == "random_saw" else 1,
        int(args.archive_size),
        float(args.archive_min_hamming_frac),
        int(args.archive_contact_slack),
        1 if args.contact_priority_best else 0,
        max(1, int(args.contact_check_every)),
        int(args.target_contacts),
        1 if args.stop_on_target else 0,
        1 if args.contact_guided_accept else 0,
        float(args.contact_bias),
        float(args.contact_bias_final_frac),
        float(args.contact_paving_weight),
        float(args.qubo_polish_frac),
        int(move_mode),
        int(block_size),
        int(args.reheat_every),
        float(args.reheat_factor),
        seed_best_spins,
        int(seed_best_has),
        int(seed_best_contacts),
        terms["h_dense"],
        terms["J2_a"], terms["J2_b"], terms["J2_c"],
        terms["J3_a"], terms["J3_b"], terms["J3_k"], terms["J3_c"],
        terms["J4_a"], terms["J4_b"], terms["J4_k"], terms["J4_l"], terms["J4_c"],
        terms["ptr2"], terms["j2_idx"], terms["j2_c"],
        terms["ptr3"], terms["j3_o1"], terms["j3_o2"], terms["j3_c"],
        terms["ptr4"], terms["j4_o1"], terms["j4_o2"], terms["j4_o3"], terms["j4_c"],
        nbrs, xy2s, coords_x, coords_y, seq_is_H
    )
    elapsed = time.time() - t0

    # Build JSON output (compatible with your parallel runner merger)
    best_spins_host = [float(x) for x in best_spins.tolist()]  # store as +/-1
    completed_trials = int(completed_trials)
    trace = [{"trial": int(i), "best": float(best_so_far[i])} for i in range(completed_trials)]
    trial_trace = []
    for i in range(completed_trials):
        trial_trace.append({
            "trial": int(i),
            "trial_best": float(trial_best_e[i]),
            "trial_best_contacts": int(trial_best_contacts[i]) if int(trial_best_contacts[i]) >= 0 else None,
            "trial_best_feasible": bool(int(trial_best_feasible[i]) == 1),
            "accept_rate": float(trial_accept_rate[i]),
        })

    # decode best contacts (strict) for top-level summary if possible
    best_contacts = None
    best_pairs = []
    if coords_x.shape[0] == block_size and seq_is_H.shape[0] == (V // block_size):
        # compute contacts by decoding positions from spins and counting (python-side)
        coords = data.get("coords") or []
        seq = data.get("seq") or ""
        if coords and seq:
            S2 = len(coords)
            N2 = len(seq)
            pos = []
            ok = True
            for r in range(N2):
                base = r * S2
                active = [j for j in range(S2) if best_spins_host[base + j] > 0]
                if len(active) != 1:
                    ok = False
                    break
                pos.append(active[0])
            if ok and len(set(pos)) == len(pos):
                def manh(a, b):
                    ax, ay = coords[a]; bx, by = coords[b]
                    return abs(ax - bx) + abs(ay - by)
                if all(manh(pos[r], pos[r+1]) == 1 for r in range(N2 - 1)):
                    c = 0
                    pairs = []
                    for i in range(N2):
                        if seq[i] != "H":
                            continue
                        for j in range(i + 2, N2):
                            if seq[j] != "H":
                                continue
                            if manh(pos[i], pos[j]) == 1:
                                c += 1
                                pairs.append([int(i), int(j)])
                    best_contacts = int(c)
                    best_pairs = pairs

    out = dict(
        energy=float(best_e),
        spins=best_spins_host,
        trace=trace,
        trial_trace=trial_trace,
        wall_seconds=float(elapsed),
        wall_total_seconds=float(elapsed),
        contacts=best_contacts,
        pairs=best_pairs,
        meta=dict(
            impl="cpu_numba_saw_chain_pivot_fix7_contactguided",
            ising=str(args.ising),
            trials=int(args.trials),
            completed_trials=int(completed_trials),
            steps=int(args.steps),
            t_init=float(args.t_init),
            t_final=float(args.t_final),
            seed=int(args.seed),
            device="cpu",
            dtype=str(args.dtype),
            move_mode=str(args.move_mode),
            block_size=int(block_size),
            reheat_every=int(args.reheat_every),
            reheat_factor=float(args.reheat_factor),
            warm_start_file=str(args.warm_start_file),
            init_mode=str(args.init_mode),
            archive_size=int(args.archive_size),
            archive_min_hamming_frac=float(args.archive_min_hamming_frac),
            archive_contact_slack=int(args.archive_contact_slack),
            contact_priority_best=bool(args.contact_priority_best),
            contact_check_every=int(args.contact_check_every),
            target_contacts=int(args.target_contacts),
            stop_on_target=bool(args.stop_on_target),
            contact_guided_accept=bool(args.contact_guided_accept),
            contact_bias=float(args.contact_bias),
            contact_bias_final_frac=float(args.contact_bias_final_frac),
            contact_paving_weight=float(args.contact_paving_weight),
            qubo_polish_frac=float(args.qubo_polish_frac),
            pull_prob=float(args.pull_prob),
            reptation_prob=float(args.reptation_prob),
            regrow_prob=float(args.regrow_prob),
            frag_regrow_prob=float(args.frag_regrow_prob),
            regrow_max_len=int(args.regrow_max_len),
        )
    )

    if args.save_best:
        with open(args.save_best, "w") as f:
            json.dump(out, f)
    else:
        print(json.dumps(out)[:2000])


if __name__ == "__main__":
    main()

