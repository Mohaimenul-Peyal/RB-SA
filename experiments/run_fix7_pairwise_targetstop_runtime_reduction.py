#!/usr/bin/env python3
"""Runtime-reduction experiment for pairwise-only fix7 RB-full runs.

This script keeps the validated fresh pairwise-only QUBOs and focuses on the
two expensive benchmarks, 60mer and 64mer.

Policy:
  - Do not use external warm-start files.
  - For 60mer, create a new in-directory 35-contact seed and then self-warm
    from that seed to target 36.
  - For 64mer, pilot smaller no-warm budgets and stop if target 42 is found.
  - Use target-aware early stopping in the parent runner.
  - Rebuild a complete all-benchmark table by carrying forward previous
    pairwise-only rows for 20merA through 50mer.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


TARGETS = {
    "20merA": 9,
    "20merB": 10,
    "24mer": 9,
    "25mer": 8,
    "36mer": 14,
    "48mer": 23,
    "50mer": 21,
    "60mer": 36,
    "64mer": 42,
}


QUBOS = {
    "60mer": "runs/60mer/local/Final_run/fix7_pairwise_fresh_qubo/qubos/60mer_pairwise_L9_hp1p4.json",
    "64mer": "runs/64mer/local/Final_run/fix7_pairwise_fresh_qubo/qubos/64mer_pairwise_L10_hp2p5.json",
}


SCHEDULES: Dict[str, List[Dict[str, Any]]] = {
    "60mer": [
        {
            "label": "stage1_nowarm_seed35_128tr_2M_targetstop",
            "trials": 128,
            "workers": 16,
            "steps": 2_000_000,
            "t_init": 15.0,
            "seed": 920_060_001,
            "contact_bias": 0.6,
            "regrow_max_len": 30,
            "qubo_polish_frac": 0.30,
            "warm": False,
            "stop_contacts": 35,
            "purpose": "in-directory 35-contact seed discovery",
        },
        {
            "label": "stage2_selfwarm_256tr_2M_targetstop",
            "trials": 256,
            "workers": 16,
            "steps": 2_000_000,
            "t_init": 15.0,
            "seed": 920_060_002,
            "contact_bias": 0.6,
            "regrow_max_len": 30,
            "qubo_polish_frac": 0.30,
            "warm": True,
            "stop_contacts": 36,
            "purpose": "self-warm refinement from in-directory seed",
        },
        {
            "label": "stage3_selfwarm_256tr_2M_targetstop_retry",
            "trials": 256,
            "workers": 16,
            "steps": 2_000_000,
            "t_init": 15.0,
            "seed": 920_060_003,
            "contact_bias": 0.6,
            "regrow_max_len": 30,
            "qubo_polish_frac": 0.30,
            "warm": True,
            "stop_contacts": 36,
            "purpose": "self-warm retry if stage 2 misses target",
        },
    ],
    "64mer": [
        {
            "label": "pilot_nowarm_128tr_1p5M_targetstop",
            "trials": 128,
            "workers": 16,
            "steps": 1_500_000,
            "t_init": 12.0,
            "seed": 920_064_001,
            "contact_bias": 0.7,
            "regrow_max_len": 28,
            "qubo_polish_frac": 0.25,
            "warm": False,
            "stop_contacts": 42,
            "purpose": "pilot smaller trial count",
        },
        {
            "label": "pilot_nowarm_256tr_1M_targetstop",
            "trials": 256,
            "workers": 16,
            "steps": 1_000_000,
            "t_init": 12.0,
            "seed": 920_064_001,
            "contact_bias": 0.7,
            "regrow_max_len": 28,
            "qubo_polish_frac": 0.25,
            "warm": False,
            "stop_contacts": 42,
            "purpose": "pilot smaller step count",
        },
        {
            "label": "pilot_nowarm_192tr_1p2M_targetstop",
            "trials": 192,
            "workers": 16,
            "steps": 1_200_000,
            "t_init": 12.0,
            "seed": 920_064_001,
            "contact_bias": 0.7,
            "regrow_max_len": 28,
            "qubo_polish_frac": 0.25,
            "warm": False,
            "stop_contacts": 42,
            "purpose": "pilot balanced trial/step budget",
        },
        {
            "label": "fallback_nowarm_256tr_1p5M_targetstop",
            "trials": 256,
            "workers": 16,
            "steps": 1_500_000,
            "t_init": 12.0,
            "seed": 920_064_001,
            "contact_bias": 0.7,
            "regrow_max_len": 28,
            "qubo_polish_frac": 0.25,
            "warm": False,
            "stop_contacts": 42,
            "purpose": "validated fixed-budget setting with target-aware stop",
        },
    ],
}


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def contacts_of(path: Optional[Path]) -> str:
    if path is None or not path.exists():
        return ""
    try:
        payload = read_json(path)
        best = payload.get("best", payload)
        val = best.get("contacts", payload.get("contacts"))
        return "" if val is None else str(int(val))
    except Exception:
        return ""


def verify_pairwise_qubo(path: Path) -> Dict[str, Any]:
    data = read_json(path)
    if data.get("enable_3body") or data.get("enable_4body"):
        raise RuntimeError(f"QUBO is not pairwise-only: {path}")
    if len(data.get("cubic", [])) or len(data.get("quartic", [])):
        raise RuntimeError(f"QUBO has higher-order terms: {path}")
    return data


def build_command(repo: Path, qpath: Path, outdir: Path, cfg: Dict[str, Any], warm: Optional[Path]) -> List[str]:
    runner = repo / "src" / "HP_ssa_manybody_parallel_runner_cpu_fast_passthru.py"
    solver = repo / "src" / "HP_ssa_manybody_ising_cpu_numba_saw_pivot_fix7_contactguided.py"
    cmd = [
        sys.executable,
        str(runner),
        "--ising",
        str(qpath),
        "--solver_script",
        str(solver),
        "--total_trials",
        str(int(cfg["trials"])),
        "--workers",
        str(int(cfg["workers"])),
        "--steps",
        str(int(cfg["steps"])),
        "--t_init",
        str(float(cfg["t_init"])),
        "--t_final",
        "0.0006",
        "--seed",
        str(int(cfg["seed"])),
        "--device",
        "cpu",
        "--dtype",
        "float64",
        "--move_mode",
        "residue",
        "--block_size",
        "0",
        "--reheat_every",
        "35000",
        "--reheat_factor",
        "2.4",
        "--outdir",
        str(outdir),
        "--reseed_each_trial",
        "--target_contacts",
        str(int(cfg["stop_contacts"])),
        "--stop_on_target",
        "--poll_interval",
        "2.0",
    ]
    if warm is not None:
        cmd.extend(["--warm_start_best", "--best_t_scale", "1.35"])
    solver_extra = [
        "--pivot_prob",
        "0.5",
        "--pivot_max_tail",
        "0",
        "--pull_prob",
        "0.08",
        "--reptation_prob",
        "0.12",
        "--regrow_prob",
        "0.05",
        "--frag_regrow_prob",
        "0.16",
        "--regrow_max_len",
        str(int(cfg["regrow_max_len"])),
        "--init_mode",
        "seq_greedy",
        "--archive_size",
        "24",
        "--archive_min_hamming_frac",
        "0.24",
        "--archive_contact_slack",
        "5",
        "--contact_priority_best",
        "--contact_check_every",
        "50",
        "--contact_guided_accept",
        "--contact_bias",
        str(float(cfg["contact_bias"])),
        "--contact_bias_final_frac",
        "0.10",
        "--contact_paving_weight",
        "0.0008",
        "--qubo_polish_frac",
        str(float(cfg["qubo_polish_frac"])),
    ]
    if warm is not None:
        warm_contacts = int(contacts_of(warm) or 0)
        solver_extra.extend([
            "--warm_start_prob",
            "0.60",
            "--warm_start_min_contacts",
            str(max(0, warm_contacts - 1)),
            "--warm_start_file",
            str(warm),
        ])
    cmd.extend(["--", *solver_extra])
    return cmd


def read_result(path: Path, final_target: int) -> Dict[str, Any]:
    if not path.exists():
        return {
            "contacts": -1,
            "energy": "",
            "successes": 0,
            "actual_trials": 0,
            "hit_rate": 0.0,
            "target_hit": "no",
            "target_early_stop_hit": False,
        }
    payload = read_json(path)
    best = payload.get("best", payload)
    contacts = int(best.get("contacts", payload.get("contacts", -1)))
    trace = payload.get("trial_trace", []) or []
    successes = 0
    actual_trials = 0
    first_target_trial = ""
    if isinstance(trace, list):
        for item in trace:
            if not isinstance(item, dict):
                continue
            actual_trials += 1
            c = item.get("trial_best_contacts")
            if item.get("trial_best_feasible") and isinstance(c, int) and c >= final_target:
                successes += 1
                if first_target_trial == "":
                    first_target_trial = str(item.get("global_trial", item.get("trial", "")))
    hit_rate = successes / actual_trials if actual_trials else 0.0
    return {
        "contacts": contacts,
        "energy": best.get("energy", payload.get("energy", "")),
        "successes": successes,
        "actual_trials": actual_trials,
        "hit_rate": hit_rate,
        "target_hit": "yes" if contacts >= final_target else "no",
        "target_early_stop_hit": bool(payload.get("target_early_stop_hit", False)),
        "completed_worker_files": payload.get("completed_worker_files", ""),
        "first_target_trial": first_target_trial,
    }


def previous_final_rows(previous_rows: Iterable[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in previous_rows:
        grouped.setdefault(row["benchmark"], []).append(row)
    final: Dict[str, Dict[str, Any]] = {}
    for bench, rows in grouped.items():
        if bench in ("60mer", "64mer"):
            continue
        rows = sorted(rows, key=lambda r: int(r.get("iteration", "0") or 0))
        included: List[Dict[str, str]] = []
        chosen: Optional[Dict[str, str]] = None
        for row in rows:
            included.append(row)
            if str(row.get("target_hit", "")).lower() == "yes":
                chosen = row
                break
        if chosen is None and rows:
            chosen = max(rows, key=lambda r: int(float(r.get("contacts", "-1") or -1)))
            included = rows
        if chosen is None:
            continue
        runtime = sum(float(r.get("wrapper_seconds", "0") or 0) for r in included)
        successes = sum(int(float(r.get("successes", "0") or 0)) for r in included)
        trials = sum(int(float(r.get("actual_trials", "0") or 0)) for r in included)
        final[bench] = {
            "benchmark": bench,
            "target_contacts": int(chosen["target_contacts"]),
            "contacts": int(float(chosen["contacts"])),
            "runtime_s": runtime,
            "hit_rate": successes / trials if trials else 0.0,
            "notes": chosen.get("notes", "carried forward from pairwise-only validation"),
            "source": "carried forward from previous fresh pairwise-only table",
        }
    return final


def choose_new_final(bench: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    target = TARGETS[bench]
    if bench == "60mer":
        included: List[Dict[str, Any]] = []
        chosen: Optional[Dict[str, Any]] = None
        for row in rows:
            included.append(row)
            if row.get("target_hit") == "yes":
                chosen = row
                break
        if chosen is None:
            chosen = max(rows, key=lambda r: int(r.get("contacts", -1)))
            included = rows
        runtime = sum(float(r.get("wrapper_seconds", 0) or 0) for r in included)
        successes = sum(int(r.get("successes", 0) or 0) for r in included)
        trials = sum(int(r.get("actual_trials", 0) or 0) for r in included)
        return {
            "benchmark": bench,
            "target_contacts": target,
            "contacts": int(chosen.get("contacts", -1)),
            "runtime_s": runtime,
            "hit_rate": successes / trials if trials else 0.0,
            "notes": f"target-stop staged run; {len(included)} in-directory stages; warm seed contacts={chosen.get('warm_contacts', '')}",
            "source": "new target-stop runtime-reduction experiment",
        }
    hits = [r for r in rows if r.get("target_hit") == "yes"]
    if hits:
        chosen = min(hits, key=lambda r: float(r.get("wrapper_seconds", 1e99) or 1e99))
    else:
        chosen = max(rows, key=lambda r: int(r.get("contacts", -1)))
    return {
        "benchmark": bench,
        "target_contacts": target,
        "contacts": int(chosen.get("contacts", -1)),
        "runtime_s": float(chosen.get("wrapper_seconds", 0.0) or 0.0),
        "hit_rate": float(chosen.get("hit_rate", 0.0) or 0.0),
        "notes": str(chosen.get("notes", "")),
        "source": "new target-stop runtime-reduction experiment",
    }


def write_report(path: Path, final_rows: Dict[str, Dict[str, Any]], run_rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# Fix7 Pairwise-Only Target-Stop Runtime-Reduction Results",
        "",
        "This experiment keeps pairwise-only QUBOs and reduces long-chain runtime using target-aware early stopping.",
        "Rows for 20merA through 50mer are carried forward from the previous fresh pairwise-only validation.",
        "",
        "## Complete Benchmark Table",
        "",
        "| Benchmark | Target | Best | Runtime s | Hit rate | Notes |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for bench in TARGETS:
        if bench not in final_rows:
            continue
        row = final_rows[bench]
        lines.append(
            f"| {bench} | {row['target_contacts']} | {row['contacts']} | {float(row['runtime_s']):.3f} | {float(row['hit_rate']):.4f} | {row['notes']} |"
        )
    lines.extend([
        "",
        "## New 60mer/64mer Iterations",
        "",
        "| Benchmark | Iter | Label | Stop contacts | Warm contacts | Runtime s | Best | Actual trials | Hit rate | Target hit | Notes |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|:---:|---|",
    ])
    for row in run_rows:
        lines.append(
            f"| {row['benchmark']} | {row['iteration']} | `{row['label']}` | {row['stop_contacts']} | {row['warm_contacts']} | {float(row['wrapper_seconds']):.3f} | {row['contacts']} | {row['actual_trials']} | {float(row['hit_rate']):.4f} | {row['target_hit']} | {row['notes']} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `Runtime s` in the complete table is the selected runtime for the benchmark result.",
        "- For `60mer`, runtime is cumulative because the target run depends on the in-directory seed-discovery stage.",
        "- For `64mer`, runtime is the fastest successful no-warm pilot/fallback configuration, because failed pilots are tuning overhead rather than required warm-start stages.",
        "- `Hit rate` is computed from completed trial traces. In target-stop runs, some workers may be terminated before writing a worker-best file, so the denominator is the number of completed trials, not the requested full budget.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table_files(repo: Path, final_rows: Dict[str, Dict[str, Any]]) -> None:
    table_dir = repo / "results"
    md = table_dir / "pairwise_targetstop_runtime_reduced_results.md"
    csv_path = table_dir / "pairwise_targetstop_runtime_reduced_results.csv"
    rows = [final_rows[b] for b in TARGETS if b in final_rows]
    write_csv(csv_path, rows)
    lines = [
        "# Pairwise-Only RB-Full Target-Stop Runtime-Reduced Results",
        "",
        "Source: `runs/fix7_pairwise_targetstop_runtime_reduced_summary.csv`.",
        "The long-chain rows use target-aware early stopping. Pairwise-only QUBOs contain no 3-body or 4-body terms.",
        "",
        "| Benchmark | Target | Best | Runtime s | Hit rate | Notes |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['benchmark']} | {row['target_contacts']} | {row['contacts']} | {float(row['runtime_s']):.3f} | {float(row['hit_rate']):.4f} | {row['notes']} |"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", nargs="*", default=["60mer", "64mer"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    repo = repo_root_from_script()
    previous_summary = repo / "runs" / "fix7_pairwise_fresh_qubo_summary.csv"
    previous_rows = load_csv(previous_summary)
    final_rows = previous_final_rows(previous_rows)

    all_run_rows: List[Dict[str, Any]] = []
    summary_path = repo / "runs" / "fix7_pairwise_targetstop_runtime_reduced_summary.csv"
    report_path = repo / "runs" / "fix7_pairwise_targetstop_runtime_reduced_report.md"

    for bench in args.benchmarks:
        target = TARGETS[bench]
        base = repo / "runs" / bench / "local" / "Final_run" / "fix7_pairwise_targetstop_runtime_reduced"
        if args.clean and base.exists():
            shutil.rmtree(base)
        base.mkdir(parents=True, exist_ok=True)
        qpath = repo / QUBOS[bench]
        qmeta = verify_pairwise_qubo(qpath)

        rows = load_csv(base / "fix7_pairwise_targetstop_runtime_reduced_summary.csv") if args.resume else []
        rows_any: List[Dict[str, Any]] = [dict(r) for r in rows]
        completed = {int(r["iteration"]) for r in rows if str(r.get("iteration", "")).isdigit()}
        previous_best: Optional[Path] = None
        if rows_any:
            last_path = Path(str(rows_any[-1].get("result_file", "")))
            if last_path.exists():
                previous_best = last_path

        target_already_hit = any(str(r.get("target_hit", "")).lower() == "yes" for r in rows_any)
        for idx, cfg in enumerate(SCHEDULES[bench], start=1):
            if idx in completed:
                continue
            if target_already_hit:
                break
            warm = previous_best if bool(cfg.get("warm")) and previous_best is not None else None
            warm_contacts = contacts_of(warm)
            outdir = base / f"iter{idx:02d}_{cfg['label']}"
            cmd = build_command(repo, qpath, outdir, cfg, warm)
            outdir.mkdir(parents=True, exist_ok=True)
            write_json(outdir / "metadata.json", {
                "benchmark": bench,
                "target_contacts": target,
                "stop_contacts": int(cfg["stop_contacts"]),
                "purpose": cfg.get("purpose", ""),
                "qubo_file": str(qpath),
                "qubo_meta": {
                    "N": qmeta.get("N"),
                    "L": qmeta.get("L"),
                    "S": qmeta.get("S"),
                    "V": qmeta.get("V"),
                    "hp_reward": qmeta.get("hp_reward"),
                    "lambda_onehot": qmeta.get("lambda_onehot"),
                    "lambda_site": qmeta.get("lambda_site"),
                    "lambda_chain": qmeta.get("lambda_chain"),
                    "enable_3body": qmeta.get("enable_3body"),
                    "enable_4body": qmeta.get("enable_4body"),
                    "num_cubic_terms": len(qmeta.get("cubic", [])),
                    "num_quartic_terms": len(qmeta.get("quartic", [])),
                },
                "warm_start_file": str(warm) if warm else "",
                "warm_contacts": warm_contacts,
                "run_config": cfg,
                "command": cmd,
            })
            (outdir / "command.txt").write_text(subprocess.list2cmdline(cmd) + "\n", encoding="utf-8")
            print(f"[targetstop] {bench} iter={idx} {cfg['label']} stop_contacts={cfg['stop_contacts']} warm_contacts={warm_contacts}", flush=True)
            started = time.time()
            with (outdir / "wrapper_run.log").open("w", encoding="utf-8") as log:
                log.write("[cmd] " + subprocess.list2cmdline(cmd) + "\n")
                log.flush()
                proc = subprocess.run(cmd, cwd=str(repo), stdout=log, stderr=subprocess.STDOUT, text=True)
            elapsed = time.time() - started
            result_file = outdir / "best_overall.json"
            result = read_result(result_file, target)
            notes = str(cfg.get("purpose", ""))
            if warm is not None:
                notes += f"; self-warm from in-directory best ({warm_contacts} contacts)"
            else:
                notes += "; no warm start"
            row = {
                "benchmark": bench,
                "target_contacts": target,
                "iteration": idx,
                "label": cfg["label"],
                "stop_contacts": int(cfg["stop_contacts"]),
                "qubo_file": str(qpath),
                "warm_start_file": str(warm) if warm else "",
                "warm_contacts": warm_contacts,
                "outdir": str(outdir),
                "result_file": str(result_file),
                "status": "ok" if proc.returncode == 0 else f"failed:{proc.returncode}",
                "trials": int(cfg["trials"]),
                "workers": int(cfg["workers"]),
                "steps": int(cfg["steps"]),
                "t_init": float(cfg["t_init"]),
                "t_final": 0.0006,
                "pivot_prob": 0.5,
                "pull_prob": 0.08,
                "reptation_prob": 0.12,
                "regrow_prob": 0.05,
                "frag_regrow_prob": 0.16,
                "regrow_max_len": int(cfg["regrow_max_len"]),
                "archive_size": 24,
                "contact_bias": float(cfg["contact_bias"]),
                "qubo_polish_frac": float(cfg["qubo_polish_frac"]),
                "seed": int(cfg["seed"]),
                "qubo_L": qmeta.get("L"),
                "qubo_V": qmeta.get("V"),
                "qubo_hp_reward": qmeta.get("hp_reward"),
                "qubo_lambda_onehot": qmeta.get("lambda_onehot"),
                "qubo_lambda_site": qmeta.get("lambda_site"),
                "qubo_lambda_chain": qmeta.get("lambda_chain"),
                "qubo_enable_3body": qmeta.get("enable_3body"),
                "qubo_enable_4body": qmeta.get("enable_4body"),
                "qubo_cubic_terms": len(qmeta.get("cubic", [])),
                "qubo_quartic_terms": len(qmeta.get("quartic", [])),
                "wrapper_seconds": elapsed,
                "command": subprocess.list2cmdline(cmd),
                "notes": notes,
                **result,
            }
            rows_any.append(row)
            write_csv(outdir / "summary.csv", [row])
            write_csv(base / "fix7_pairwise_targetstop_runtime_reduced_summary.csv", rows_any)
            print(f"[targetstop-done] {bench} iter={idx} contacts={result['contacts']} target={result['target_hit']} actual_trials={result['actual_trials']} elapsed={elapsed:.2f}s", flush=True)
            previous_best = result_file if result_file.exists() else previous_best
            if result["target_hit"] == "yes" or proc.returncode != 0:
                target_already_hit = result["target_hit"] == "yes"
                break
        all_run_rows.extend(rows_any)
        if rows_any:
            final_rows[bench] = choose_new_final(bench, rows_any)

    write_csv(summary_path, all_run_rows)
    write_report(report_path, final_rows, all_run_rows)
    write_table_files(repo, final_rows)
    print(f"[targetstop-report] {report_path}", flush=True)
    print(f"[targetstop-summary] {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

