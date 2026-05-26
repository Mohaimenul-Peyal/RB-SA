#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HP_ssa_manybody_parallel_runner_cpu_fast.py

Parallel "trial farming" wrapper for SA solvers.

This version is tuned for CPU throughput:
  - Sets BLAS/OpenMP thread env vars to 1 per worker to avoid oversubscription.
  - Merges per-trial traces from workers into best_overall.json so you can compute hit rates.

Designed to run with:
  HP_ssa_manybody_ising_cpu_fast.py
but it can also run any solver script that accepts the same CLI flags.

This runner also passes through any unknown CLI arguments to the solver script (e.g., --pivot_prob).
"""

import argparse
import json
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _infer_block_size(ising_path: Path) -> int:
    data = _read_json(ising_path)
    if "S" in data and isinstance(data["S"], int) and data["S"] > 0:
        return int(data["S"])
    coords = data.get("coords")
    if isinstance(coords, list) and len(coords) > 0:
        return int(len(coords))
    raise ValueError("Could not infer block_size: JSON missing S and coords.")


def _score_result(obj: Dict[str, Any]) -> float:
    """
    Worker ranking score: prefer higher contacts if available, then lower energy.
    """
    contacts = obj.get("contacts")
    energy = obj.get("energy")
    if isinstance(contacts, int):
        # higher better
        return 1e9 + float(contacts) * 1e6 - (float(energy) if isinstance(energy, (int, float)) else 0.0)
    # fallback: lower energy better
    if isinstance(energy, (int, float)):
        return -float(energy)
    return float("-inf")


def _set_cpu_thread_env(env: Dict[str, str], threads: str = "1") -> None:
    """
    Force common math libs to single-thread to avoid oversubscription
    when running many processes (trial farming).
    """
    for k in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ]:
        env[k] = threads


def _close_log_handle(proc: subprocess.Popen) -> None:
    if hasattr(proc, "_log_handle"):
        try:
            proc._log_handle.close()  # type: ignore[attr-defined]
        except Exception:
            pass


def _result_hits_target(path: Path, target_contacts: int) -> bool:
    if target_contacts <= 0 or not path.exists():
        return False
    try:
        obj = _read_json(path)
    except Exception:
        return False
    best = obj.get("best", obj)
    contacts = best.get("contacts", obj.get("contacts"))
    return isinstance(contacts, int) and contacts >= target_contacts


def _start_worker(
    worker_id: int,
    solver_script: Path,
    ising: Path,
    trials: int,
    steps: int,
    t_init: float,
    t_final: float,
    seed: int,
    device: str,
    dtype: str,
    move_mode: str,
    block_size: int,
    reseed_each_trial: bool,
    warm_start_best: bool,
    best_t_scale: float,
    reheat_every: int,
    reheat_factor: float,
    solver_extra: List[str],
    gpu_id: Optional[str],
    outdir: Path,
) -> Tuple[subprocess.Popen, Path, Path]:
    out_best = outdir / f"best_worker_{worker_id}.json"
    out_log = outdir / f"log_worker_{worker_id}.txt"

    cmd = [
        sys.executable,
        str(solver_script),
        "--ising", str(ising),
        "--trials", str(trials),
        "--steps", str(steps),
        "--t_init", str(t_init),
        "--t_final", str(t_final),
        "--seed", str(seed),
        "--device", str(device),
        "--dtype", str(dtype),
        "--move_mode", str(move_mode),
        "--block_size", str(block_size),
        "--reheat_every", str(reheat_every),
        "--reheat_factor", str(reheat_factor),
        "--save_best", str(out_best),
    ]
    if solver_extra:
        cmd += list(solver_extra)

    if reseed_each_trial:
        cmd.append("--reseed_each_trial")
    if warm_start_best:
        cmd.append("--warm_start_best")
        cmd += ["--best_t_scale", str(best_t_scale)]

    env = os.environ.copy()

    # CPU: avoid BLAS threads inside each process
    if str(device).lower() in ("cpu", "auto"):
        _set_cpu_thread_env(env, "1")

    # GPU pinning (if requested)
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # unbuffered logs
    env["PYTHONUNBUFFERED"] = "1"

    f = out_log.open("w", encoding="utf-8")
    f.write("[cmd] " + " ".join(cmd) + "\n")
    f.flush()
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    proc._log_handle = f  # type: ignore[attr-defined]
    return proc, out_best, out_log


def main():
    ap = argparse.ArgumentParser(description="Parallel runner for manybody SA (CPU fast).")
    ap.add_argument("--ising", required=True, help="Path to manybody QUBO/Ising JSON")
    ap.add_argument("--solver_script", required=True, help="Path to solver script (python file)")
    ap.add_argument("--total_trials", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gpus", type=str, default="", help="Comma-separated GPU ids (optional)")
    ap.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "gpu"])
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--t_init", type=float, default=10.0)
    ap.add_argument("--t_final", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--move_mode", choices=["single", "residue"], default="single")
    ap.add_argument("--block_size", type=int, default=0)
    ap.add_argument("--reseed_each_trial", action="store_true")
    ap.add_argument("--warm_start_best", action="store_true")
    ap.add_argument("--best_t_scale", type=float, default=0.5)
    ap.add_argument("--reheat_every", type=int, default=0)
    ap.add_argument("--reheat_factor", type=float, default=1.0)
    ap.add_argument("--outdir", type=str, default="runs/out")
    ap.add_argument("--target_contacts", type=int, default=0,
                    help="Target contact count for optional target-aware early stopping.")
    ap.add_argument("--stop_on_target", action="store_true",
                    help="Stop remaining workers once any worker reaches --target_contacts.")
    ap.add_argument("--poll_interval", type=float, default=2.0,
                    help="Seconds between worker checks when --stop_on_target is enabled.")

    args, solver_extra = ap.parse_known_args()
    if solver_extra and solver_extra[0] == '--':
        solver_extra = solver_extra[1:]
    if solver_extra:
        print('[parallel] passing through extra solver args:', ' '.join(solver_extra))

    if args.stop_on_target and int(args.target_contacts) > 0:
        if "--target_contacts" not in solver_extra:
            solver_extra += ["--target_contacts", str(int(args.target_contacts))]
        if "--stop_on_target" not in solver_extra:
            solver_extra += ["--stop_on_target"]


    ising = Path(args.ising)
    solver_script = Path(args.solver_script)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    block_size = args.block_size if args.block_size > 0 else _infer_block_size(ising)

    total_trials = int(args.total_trials)
    workers = int(args.workers)
    if workers <= 0:
        raise ValueError("--workers must be > 0")

    steps = int(args.steps)

    print(f"[parallel] ising={ising.name}  total_trials={total_trials}  workers={workers}  steps={steps}")
    print(f"[parallel] move_mode={args.move_mode}  block_size={block_size}")
    if args.device == "gpu":
        gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
        if not gpus:
            raise ValueError("--device gpu requires --gpus list")
        print(f"[parallel] gpus={gpus}  (using first {workers})")
    else:
        gpus = []

    # Split trials across workers
    base = total_trials // workers
    rem = total_trials % workers
    trials_per_worker = [base + (1 if i < rem else 0) for i in range(workers)]

    procs: List[subprocess.Popen] = []
    best_paths: List[Path] = []
    start = time.time()

    for wid in range(workers):
        ntr = trials_per_worker[wid]
        if ntr <= 0:
            continue

        gpu_id = None
        if args.device == "gpu":
            gpu_id = gpus[wid % len(gpus)]

        w_seed = int(args.seed) + wid * 100000
        print(f"[launch] worker={wid} trials={ntr} seed={w_seed} gpu={gpu_id}")

        proc, out_best, out_log = _start_worker(
            worker_id=wid,
            solver_script=solver_script,
            ising=ising,
            trials=ntr,
            steps=steps,
            t_init=float(args.t_init),
            t_final=float(args.t_final),
            seed=w_seed,
            device=str(args.device),
            dtype=str(args.dtype),
            move_mode=str(args.move_mode),
            block_size=int(block_size),
            reseed_each_trial=bool(args.reseed_each_trial),
            warm_start_best=bool(args.warm_start_best),
            best_t_scale=float(args.best_t_scale),
            reheat_every=int(args.reheat_every),
            reheat_factor=float(args.reheat_factor),
            solver_extra=solver_extra,
            gpu_id=gpu_id,
            outdir=outdir,
        )
        procs.append(proc)
        best_paths.append(out_best)

    # Wait. In target-aware mode, each worker exits after a completed trial reaches
    # the target; the parent then terminates the still-running workers.
    target_early_stop_hit = False
    target_hit_worker: Optional[int] = None
    if bool(args.stop_on_target) and int(args.target_contacts) > 0:
        remaining = set(range(len(procs)))
        while remaining:
            finished_this_poll: List[int] = []
            for idx in list(remaining):
                proc = procs[idx]
                rc = proc.poll()
                if rc is None:
                    continue
                _close_log_handle(proc)
                finished_this_poll.append(idx)
                if rc == 0 and _result_hits_target(best_paths[idx], int(args.target_contacts)):
                    target_early_stop_hit = True
                    target_hit_worker = idx
                    print(f"[parallel] target reached by worker={idx}; terminating remaining workers")
                    for j in list(remaining):
                        if j == idx:
                            continue
                        pj = procs[j]
                        if pj.poll() is None:
                            pj.terminate()
                    for j in list(remaining):
                        if j == idx:
                            continue
                        pj = procs[j]
                        try:
                            pj.wait(timeout=5.0)
                        except subprocess.TimeoutExpired:
                            pj.kill()
                            pj.wait()
                        _close_log_handle(pj)
                    remaining.clear()
                    break
            for idx in finished_this_poll:
                remaining.discard(idx)
            if remaining:
                time.sleep(max(0.1, float(args.poll_interval)))
    else:
        for proc in procs:
            proc.wait()
            _close_log_handle(proc)

    # Validate outputs
    existing_best_paths = [p for p in best_paths if p.exists()]
    if not existing_best_paths:
        raise RuntimeError("No worker produced a best JSON.")
    if not target_early_stop_hit:
        for out_best in best_paths:
            if not out_best.exists():
                raise RuntimeError(f"Worker did not produce {out_best}")

    # Merge per-trial traces from all workers (if available).
    merged_trial_trace: List[Dict[str, Any]] = []
    global_trial = 0
    for wid, p in enumerate(best_paths):
        if not p.exists():
            continue
        obj = _read_json(p)
        ttrace = obj.get("trial_trace", [])
        if isinstance(ttrace, list) and ttrace:
            for entry in ttrace:
                if not isinstance(entry, dict):
                    continue
                e2 = dict(entry)
                e2["worker"] = wid
                e2["global_trial"] = global_trial + int(e2.get("trial", 0))
                merged_trial_trace.append(e2)
            global_trial += len(ttrace)
        else:
            meta = obj.get("meta", {})
            tr_count = meta.get("trials") if isinstance(meta, dict) else None
            if isinstance(tr_count, int):
                global_trial += tr_count

    # Aggregate best overall
    best_obj = None
    best_path = None
    for p in best_paths:
        if not p.exists():
            continue
        obj = _read_json(p)
        if best_obj is None or _score_result(obj) > _score_result(best_obj):
            best_obj = obj
            best_path = p

    elapsed = time.time() - start
    if best_obj is None:
        raise RuntimeError("No worker results found.")

    merged = {
        "source": "HP_ssa_manybody_parallel_runner_cpu_fast.py",
        "ising": str(ising),
        "total_trials": total_trials,
        "workers": workers,
        "completed_worker_files": len(existing_best_paths),
        "target_contacts": int(args.target_contacts),
        "stop_on_target": bool(args.stop_on_target),
        "target_early_stop_hit": bool(target_early_stop_hit),
        "target_hit_worker": target_hit_worker,
        "steps": int(args.steps),
        "t_init": float(args.t_init),
        "t_final": float(args.t_final),
        "device": str(args.device),
        "dtype": str(args.dtype),
        "move_mode": str(args.move_mode),
        "block_size": int(block_size),
        "elapsed_wall_seconds": float(elapsed),
        "best_worker_file": str(best_path) if best_path else None,
        "best": best_obj,
        "trial_trace": merged_trial_trace,
        "trial_trace_schema": {
            "trial_best": "best energy found within this trial (lower is better)",
            "trial_best_contacts": "H-H contacts for best-in-trial state (None if infeasible for one-hot decode)",
            "trial_best_feasible": "True if one-hot decode succeeded; else None/False",
            "accept_rate": "accepted_moves / steps",
            "trial_wall_seconds": "wall time for this trial",
            "worker": "worker id",
            "global_trial": "sequential trial index across workers",
        },
    }

    out_merged = outdir / "best_overall.json"
    out_merged.write_text(json.dumps(merged, indent=2))
    print(f"[done] elapsed={elapsed:.2f}s  best_overall={out_merged}")
    print(f"       contacts={best_obj.get('contacts')}  energy={best_obj.get('energy')}")


if __name__ == "__main__":
    main()

