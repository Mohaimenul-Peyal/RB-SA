#!/usr/bin/env python3
"""Run fresh pairwise-only QUBO + fix7 RB-full experiments for all benchmarks.

Policy for this experiment:
  - Export a new QUBO for every benchmark.
  - Do not enable 3-body or 4-body terms.
  - Do not use warm starts from previous experiment folders.
  - If a benchmark needs a warm start, use only best_overall.json generated
    earlier inside the same new experiment root.

Outputs:
  runs/<benchmark>/local/Final_run/fix7_pairwise_fresh_qubo/
  runs/fix7_pairwise_fresh_qubo_summary.csv
  runs/fix7_pairwise_fresh_qubo_report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "20merA": {
        "seq": "HPHPPHHPHPPHPHHPPHPH",
        "target": 9,
        "L": 6,
        "hp_reward": -1.0,
        "penalty": None,
        "schedule": [
            {"label": "nowarm_4tr_20k", "trials": 4, "workers": 4, "steps": 20_000, "t_init": 8.0, "seed": 920_020_001, "contact_bias": 0.4, "regrow_max_len": 10},
            {"label": "selfwarm_8tr_30k", "trials": 8, "workers": 4, "steps": 30_000, "t_init": 8.0, "seed": 920_020_002, "contact_bias": 0.4, "regrow_max_len": 10},
        ],
    },
    "20merB": {
        "seq": "HHHPPHPHPHPPHPHPHPPH",
        "target": 10,
        "L": 6,
        "hp_reward": -1.0,
        "penalty": None,
        "schedule": [
            {"label": "nowarm_16tr_50k", "trials": 16, "workers": 8, "steps": 50_000, "t_init": 8.0, "seed": 920_020_101, "contact_bias": 0.4, "regrow_max_len": 10},
            {"label": "selfwarm_32tr_80k", "trials": 32, "workers": 8, "steps": 80_000, "t_init": 8.0, "seed": 920_020_102, "contact_bias": 0.4, "regrow_max_len": 10},
        ],
    },
    "24mer": {
        "seq": "HHPPHPPHPPHPPHPPHPPHPPHH",
        "target": 9,
        "L": 6,
        "hp_reward": -1.0,
        "penalty": None,
        "schedule": [
            {"label": "nowarm_24tr_60k", "trials": 24, "workers": 8, "steps": 60_000, "t_init": 8.0, "seed": 920_024_001, "contact_bias": 0.4, "regrow_max_len": 12},
            {"label": "selfwarm_48tr_100k", "trials": 48, "workers": 8, "steps": 100_000, "t_init": 8.0, "seed": 920_024_002, "contact_bias": 0.4, "regrow_max_len": 12},
        ],
    },
    "25mer": {
        "seq": "PPHPPHHPPPPHHPPPPHHPPPPHH",
        "target": 8,
        "L": 7,
        "hp_reward": -1.0,
        "penalty": None,
        "schedule": [
            {"label": "nowarm_24tr_60k", "trials": 24, "workers": 8, "steps": 60_000, "t_init": 8.0, "seed": 920_025_001, "contact_bias": 0.4, "regrow_max_len": 12},
            {"label": "selfwarm_48tr_100k", "trials": 48, "workers": 8, "steps": 100_000, "t_init": 8.0, "seed": 920_025_002, "contact_bias": 0.4, "regrow_max_len": 12},
        ],
    },
    "36mer": {
        "seq": "PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPP",
        "target": 14,
        "L": 8,
        "hp_reward": -1.4,
        "penalty": 360.0,
        "schedule": [
            {"label": "nowarm_32tr_100k", "trials": 32, "workers": 8, "steps": 100_000, "t_init": 10.0, "seed": 920_036_001, "contact_bias": 0.6, "regrow_max_len": 16},
            {"label": "selfwarm_32tr_100k", "trials": 32, "workers": 8, "steps": 100_000, "t_init": 10.0, "seed": 920_036_002, "contact_bias": 0.6, "regrow_max_len": 16},
        ],
    },
    "48mer": {
        "seq": "PPHPPHHPPHHPPPPPHHHHHHHHHHPPPPPPHHPPHHPPHPPHHHHH",
        "target": 23,
        "L": 11,
        "hp_reward": -1.4,
        "penalty": 620.0,
        "schedule": [
            {"label": "nowarm_64tr_500k", "trials": 64, "workers": 16, "steps": 500_000, "t_init": 10.0, "seed": 920_048_001, "contact_bias": 0.6, "regrow_max_len": 20},
            {"label": "selfwarm_64tr_500k", "trials": 64, "workers": 16, "steps": 500_000, "t_init": 10.0, "seed": 920_048_002, "contact_bias": 0.6, "regrow_max_len": 20},
        ],
    },
    "50mer": {
        "seq": "HHPHPHPHPHHHHPHPPPHPPPHPPPPHPPPHPPPHPHHHHPHPHPHPHH",
        "target": 21,
        "L": 11,
        "hp_reward": -3.2,
        "penalty": 300.0,
        "schedule": [
            {"label": "nowarm_64tr_300k", "trials": 64, "workers": 16, "steps": 300_000, "t_init": 12.0, "seed": 920_050_001, "contact_bias": 0.6, "regrow_max_len": 24},
            {"label": "selfwarm_64tr_300k", "trials": 64, "workers": 16, "steps": 300_000, "t_init": 12.0, "seed": 920_050_002, "contact_bias": 0.6, "regrow_max_len": 24},
            {"label": "selfwarm_128tr_500k", "trials": 128, "workers": 16, "steps": 500_000, "t_init": 12.0, "seed": 920_050_003, "contact_bias": 0.6, "regrow_max_len": 24},
        ],
    },
    "60mer": {
        "seq": "PPHHHPHHHHHHHHPPPHHHHHHHHHHPHPPPHHHHHHHHHHHHPPPPHHHHHHPHHPHP",
        "target": 36,
        "L": 9,
        "hp_reward": -1.4,
        "penalty": None,
        "schedule": [
            {"label": "nowarm_256tr_2M", "trials": 256, "workers": 16, "steps": 2_000_000, "t_init": 15.0, "seed": 920_060_001, "contact_bias": 0.6, "regrow_max_len": 30},
            {"label": "selfwarm_256tr_2M", "trials": 256, "workers": 16, "steps": 2_000_000, "t_init": 15.0, "seed": 920_060_002, "contact_bias": 0.6, "regrow_max_len": 30},
            {"label": "selfwarm_512tr_2M", "trials": 512, "workers": 16, "steps": 2_000_000, "t_init": 15.0, "seed": 920_060_003, "contact_bias": 0.6, "regrow_max_len": 30},
        ],
    },
    "64mer": {
        "seq": "HHHHHHHHHHHHPHPHPPHHPPHHPPHPPHHPPHHPPHPPHHPPHHPPHPHPHHHHHHHHHHHH",
        "target": 42,
        "L": 10,
        "hp_reward": -2.5,
        "penalty": 500.0,
        "schedule": [
            {"label": "nowarm_256tr_1p5M", "trials": 256, "workers": 16, "steps": 1_500_000, "t_init": 12.0, "seed": 920_064_001, "contact_bias": 0.7, "regrow_max_len": 28, "qubo_polish_frac": 0.25},
            {"label": "selfwarm_256tr_1p5M", "trials": 256, "workers": 16, "steps": 1_500_000, "t_init": 12.0, "seed": 920_064_002, "contact_bias": 0.7, "regrow_max_len": 28, "qubo_polish_frac": 0.25},
            {"label": "selfwarm_512tr_1p5M", "trials": 512, "workers": 16, "steps": 1_500_000, "t_init": 12.0, "seed": 920_064_003, "contact_bias": 0.7, "regrow_max_len": 28, "qubo_polish_frac": 0.25},
        ],
    },
}


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


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


def load_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def export_qubo(repo: Path, bench: str, cfg: Dict[str, Any], qubo_path: Path, force: bool) -> None:
    if qubo_path.exists() and not force:
        data = read_json(qubo_path)
        if data.get("enable_3body") or data.get("enable_4body") or data.get("cubic") or data.get("quartic"):
            raise RuntimeError(f"Existing QUBO is not pairwise-only: {qubo_path}")
        return
    cmd = [
        sys.executable,
        str(repo / "src" / "HP_export_manybody_ising.py"),
        "--seq",
        str(cfg["seq"]),
        "--l_size",
        str(int(cfg["L"])),
        "--hp_reward",
        str(float(cfg["hp_reward"])),
        "--scale_target",
        "1000",
        "--out",
        str(qubo_path),
    ]
    if cfg.get("penalty") is not None:
        penalty = str(float(cfg["penalty"]))
        cmd.extend([
            "--penalty_mode",
            "manual",
            "--lambda_onehot",
            penalty,
            "--lambda_site",
            penalty,
            "--lambda_chain",
            penalty,
        ])
    else:
        cmd.extend(["--penalty_mode", "auto"])
    qubo_path.parent.mkdir(parents=True, exist_ok=True)
    (qubo_path.parent / f"export_{bench}_pairwise_command.txt").write_text(subprocess.list2cmdline(cmd) + "\n", encoding="utf-8")
    subprocess.run(cmd, cwd=str(repo), check=True)
    data = read_json(qubo_path)
    if data.get("enable_3body") or data.get("enable_4body") or len(data.get("cubic", [])) or len(data.get("quartic", [])):
        raise RuntimeError(f"Exported QUBO is not pairwise-only: {qubo_path}")


def build_command(repo: Path, qpath: Path, outdir: Path, row: Dict[str, Any], warm: Optional[Path]) -> List[str]:
    solver = repo / "src" / "HP_ssa_manybody_ising_cpu_numba_saw_pivot_fix7_contactguided.py"
    runner = repo / "src" / "HP_ssa_manybody_parallel_runner_cpu_fast_passthru.py"
    cmd = [
        sys.executable,
        str(runner),
        "--ising",
        str(qpath),
        "--solver_script",
        str(solver),
        "--total_trials",
        str(int(row["trials"])),
        "--workers",
        str(int(row["workers"])),
        "--steps",
        str(int(row["steps"])),
        "--t_init",
        str(float(row["t_init"])),
        "--t_final",
        "0.0006",
        "--seed",
        str(int(row["seed"])),
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
        str(int(row["regrow_max_len"])),
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
        str(float(row["contact_bias"])),
        "--contact_bias_final_frac",
        "0.10",
        "--contact_paving_weight",
        "0.0008",
        "--qubo_polish_frac",
        str(float(row.get("qubo_polish_frac", 0.30))),
    ]
    if warm is not None:
        solver_extra.extend([
            "--warm_start_prob",
            "0.60",
            "--warm_start_min_contacts",
            str(max(0, int(contacts_of(warm) or 0) - 1)),
            "--warm_start_file",
            str(warm),
        ])
    cmd.extend(["--", *solver_extra])
    return cmd


def read_result(result_file: Path, target: int) -> Dict[str, Any]:
    if not result_file.exists():
        return {"contacts": -1, "energy": "", "successes": 0, "actual_trials": 0, "hit_rate": 0.0, "target_hit": "no"}
    payload = read_json(result_file)
    best = payload.get("best", payload)
    contacts = int(best.get("contacts", payload.get("contacts", -1)))
    successes = 0
    actual_trials = 0
    trace = payload.get("trial_trace", []) or []
    if isinstance(trace, list):
        for item in trace:
            if not isinstance(item, dict):
                continue
            actual_trials += 1
            c = item.get("trial_best_contacts")
            if item.get("trial_best_feasible") and isinstance(c, int) and c >= target:
                successes += 1
    if actual_trials == 0:
        actual_trials = int(payload.get("actual_trials", payload.get("total_trials", 0)) or 0)
        successes = int(payload.get("successes", 0) or 0)
    hit_rate = successes / actual_trials if actual_trials else 0.0
    return {
        "contacts": contacts,
        "energy": best.get("energy", payload.get("energy", "")),
        "successes": successes,
        "actual_trials": actual_trials,
        "hit_rate": hit_rate,
        "target_hit": "yes" if contacts >= target else "no",
    }


def write_report(path: Path, rows: List[Dict[str, Any]]) -> None:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["benchmark"], []).append(row)
    final_rows: Dict[str, Dict[str, Any]] = {}
    for bench, br in grouped.items():
        br_sorted = sorted(br, key=lambda r: int(r["iteration"]))
        chosen = None
        included: List[Dict[str, Any]] = []
        for row in br_sorted:
            included.append(row)
            if row.get("target_hit") == "yes":
                chosen = row
                break
        if chosen is None:
            chosen = max(br_sorted, key=lambda r: int(r.get("contacts", -1)))
            included = br_sorted
        total_runtime = sum(float(r.get("wrapper_seconds", 0.0) or 0.0) for r in included)
        total_successes = sum(int(float(r.get("successes", 0) or 0)) for r in included)
        total_trials = sum(int(float(r.get("actual_trials", 0) or 0)) for r in included)
        out = dict(chosen)
        out["cumulative_runtime_seconds"] = total_runtime
        out["cumulative_successes"] = total_successes
        out["cumulative_actual_trials"] = total_trials
        out["cumulative_hit_rate"] = total_successes / total_trials if total_trials else 0.0
        if len(included) > 1:
            out["final_notes"] = f"fresh pairwise QUBO; target reached after {len(included)} in-directory iterations; warm seed contacts={out.get('warm_contacts', '')}"
        else:
            out["final_notes"] = out["notes"]
        final_rows[bench] = out
    lines = [
        "# Fix7 Pairwise-Only Fresh-QUBO Self-Warm Experiment",
        "",
        "Every QUBO in this experiment was freshly exported with 3-body and 4-body terms disabled.",
        "Warm starts are only allowed from `best_overall.json` files generated inside the same benchmark's `fix7_pairwise_fresh_qubo` directory.",
        "",
        "## Final Benchmark Table",
        "",
        "| Benchmark | Target | Best | Runtime s | Hit rate | Notes |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for bench in BENCHMARKS:
        if bench not in final_rows:
            continue
        row = final_rows[bench]
        lines.append(
            f"| {bench} | {row['target_contacts']} | {row['contacts']} | {float(row['cumulative_runtime_seconds']):.3f} | {float(row['cumulative_hit_rate']):.4f} | {row['final_notes']} |"
        )
    lines.extend([
        "",
        "## All Iterations",
        "",
        "| Benchmark | Iter | Label | Warm contacts | Runtime s | Contacts | Hit rate | Target hit |",
        "|---|---:|---|---:|---:|---:|---:|:---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['benchmark']} | {row['iteration']} | `{row['label']}` | {row['warm_contacts']} | {float(row['wrapper_seconds']):.3f} | {row['contacts']} | {float(row['hit_rate']):.4f} | {row['target_hit']} |"
        )
    lines.extend([
        "",
        "## Files",
        "",
        "- Combined summary CSV: `runs/fix7_pairwise_fresh_qubo_summary.csv`",
        "- Per-benchmark folders: `runs/<benchmark>/local/Final_run/fix7_pairwise_fresh_qubo/`",
        "- Pairwise-only QUBOs: `runs/<benchmark>/local/Final_run/fix7_pairwise_fresh_qubo/qubos/`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", nargs="*", default=list(BENCHMARKS.keys()))
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    repo = repo_root_from_script()
    combined_path = repo / "runs" / "fix7_pairwise_fresh_qubo_summary.csv"
    report_path = repo / "runs" / "fix7_pairwise_fresh_qubo_report.md"
    combined_rows = load_rows(combined_path) if args.resume else []

    for bench in args.benchmarks:
        cfg = BENCHMARKS[bench]
        target = int(cfg["target"])
        base = repo / "runs" / bench / "local" / "Final_run" / "fix7_pairwise_fresh_qubo"
        qpath = base / "qubos" / f"{bench}_pairwise_L{cfg['L']}_hp{str(abs(float(cfg['hp_reward']))).replace('.', 'p')}.json"
        export_qubo(repo, bench, cfg, qpath, args.force_export)
        qmeta = read_json(qpath)
        rows = load_rows(base / "fix7_pairwise_fresh_qubo_summary.csv") if args.resume else []
        completed = {int(r["iteration"]) for r in rows if str(r.get("iteration", "")).isdigit()}
        if any(str(r.get("target_hit", "")).lower() == "yes" for r in rows):
            print(f"[pairwise-skip] {bench} already hit target")
            continue

        previous_best: Optional[Path] = None
        if rows:
            last = rows[-1]
            p = Path(last.get("result_file", ""))
            if p.exists():
                previous_best = p

        for idx, row_cfg in enumerate(cfg["schedule"], start=1):
            if idx in completed:
                continue
            warm = previous_best if idx > 1 and previous_best is not None else None
            warm_contacts = contacts_of(warm)
            outdir = base / f"iter{idx:02d}_{row_cfg['label']}"
            cmd = build_command(repo, qpath, outdir, row_cfg, warm)
            outdir.mkdir(parents=True, exist_ok=True)
            write_json(outdir / "metadata.json", {
                "benchmark": bench,
                "target_contacts": target,
                "purpose": "fresh pairwise-only QUBO fix7 RB-full run",
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
                "run_config": row_cfg,
                "command": cmd,
            })
            (outdir / "command.txt").write_text(subprocess.list2cmdline(cmd) + "\n", encoding="utf-8")
            print(f"[pairwise] {bench} iter={idx} {row_cfg['label']} warm_contacts={warm_contacts}")
            started = time.time()
            with (outdir / "wrapper_run.log").open("w", encoding="utf-8") as log:
                log.write("[cmd] " + subprocess.list2cmdline(cmd) + "\n")
                log.flush()
                proc = subprocess.run(cmd, cwd=str(repo), stdout=log, stderr=subprocess.STDOUT, text=True)
            elapsed = time.time() - started
            result_file = outdir / "best_overall.json"
            result = read_result(result_file, target)
            notes = "fresh pairwise QUBO; no warm start" if warm is None else f"fresh pairwise QUBO; self-warm from previous in-directory best ({warm_contacts} contacts)"
            row_out: Dict[str, Any] = {
                "benchmark": bench,
                "target_contacts": target,
                "iteration": idx,
                "label": row_cfg["label"],
                "qubo_file": str(qpath),
                "warm_start_file": str(warm) if warm else "",
                "warm_contacts": warm_contacts,
                "outdir": str(outdir),
                "result_file": str(result_file),
                "status": "ok" if proc.returncode == 0 else f"failed:{proc.returncode}",
                "trials": int(row_cfg["trials"]),
                "workers": int(row_cfg["workers"]),
                "steps": int(row_cfg["steps"]),
                "t_init": float(row_cfg["t_init"]),
                "t_final": 0.0006,
                "pivot_prob": 0.5,
                "pull_prob": 0.08,
                "reptation_prob": 0.12,
                "regrow_prob": 0.05,
                "frag_regrow_prob": 0.16,
                "regrow_max_len": int(row_cfg["regrow_max_len"]),
                "archive_size": 24,
                "contact_bias": float(row_cfg["contact_bias"]),
                "qubo_polish_frac": float(row_cfg.get("qubo_polish_frac", 0.30)),
                "seed": int(row_cfg["seed"]),
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
            rows.append(row_out)
            combined_rows.append(row_out)
            write_csv(outdir / "summary.csv", [row_out])
            write_csv(base / "fix7_pairwise_fresh_qubo_summary.csv", rows)
            write_csv(combined_path, combined_rows)
            write_report(report_path, combined_rows)
            print(f"[pairwise-done] {bench} iter={idx} contacts={result['contacts']} target={result['target_hit']} elapsed={elapsed:.2f}s")
            previous_best = result_file if result_file.exists() else previous_best
            if result.get("target_hit") == "yes" or proc.returncode != 0:
                break
    write_report(report_path, combined_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

