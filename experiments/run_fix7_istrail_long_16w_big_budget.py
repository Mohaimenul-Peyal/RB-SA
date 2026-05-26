#!/usr/bin/env python3
"""16-worker big-budget fix7 RB-full attempts for Istrail-long chains.

This is a follow-up to the warm-budget target attempt. It intentionally uses
all 16 local workers and larger annealing budgets. Warm starts are still only
RB-full-produced best_overall.json files, not contact-native search outputs.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "85mer": {
        "target": 53,
        "qubo": "runs/85mer/local/Final_run/fix7_istrail_long_warm_budget/qubos/85mer_pairwise_L12_hp3p0.json",
        "source_warm": "runs/85mer/local/Final_run/fix7_istrail_long_warm_budget/iter01_warm96_1p2M_hp3/best_overall.json",
        "schedule": [
            {"label": "warm256_2M_16w", "trials": 256, "workers": 16, "steps": 2_000_000, "t_init": 20.0, "seed": 950_085_001, "contact_bias": 1.20, "regrow_max_len": 56},
            {"label": "warm512_2p5M_16w_retry", "trials": 512, "workers": 16, "steps": 2_500_000, "t_init": 20.0, "seed": 950_085_002, "contact_bias": 1.25, "regrow_max_len": 60},
        ],
    },
    "100merA": {
        "target": 48,
        "qubo": "runs/100merA/local/Final_run/fix7_istrail_long_warm_budget/qubos/100merA_pairwise_L12_hp3p0.json",
        "source_warm": "runs/100merA/local/Final_run/fix7_istrail_long_warm_budget/iter02_warm128_1p5M_hp3_retry/best_overall.json",
        "schedule": [
            {"label": "warm256_2M_16w", "trials": 256, "workers": 16, "steps": 2_000_000, "t_init": 20.0, "seed": 950_100_101, "contact_bias": 1.20, "regrow_max_len": 60},
            {"label": "warm512_2p5M_16w_retry", "trials": 512, "workers": 16, "steps": 2_500_000, "t_init": 20.0, "seed": 950_100_102, "contact_bias": 1.25, "regrow_max_len": 64},
            {"label": "warm768_3M_16w_final_refine", "trials": 768, "workers": 16, "steps": 3_000_000, "t_init": 22.0, "seed": 950_100_103, "contact_bias": 1.35, "regrow_max_len": 72},
        ],
    },
    "100merB": {
        "target": 50,
        "qubo": "runs/100merB/local/Final_run/fix7_istrail_long_warm_budget/qubos/100merB_pairwise_L12_hp3p0.json",
        "source_warm": "runs/100merB/local/Final_run/fix7_istrail_long_warm_budget/iter02_warm128_1p5M_hp3_retry/best_overall.json",
        "schedule": [
            {"label": "warm256_2M_16w", "trials": 256, "workers": 16, "steps": 2_000_000, "t_init": 20.0, "seed": 950_100_201, "contact_bias": 1.20, "regrow_max_len": 60},
            {"label": "warm512_2p5M_16w_retry", "trials": 512, "workers": 16, "steps": 2_500_000, "t_init": 20.0, "seed": 950_100_202, "contact_bias": 1.25, "regrow_max_len": 64},
            {"label": "warm768_3M_16w_final_refine", "trials": 768, "workers": 16, "steps": 3_000_000, "t_init": 22.0, "seed": 950_100_203, "contact_bias": 1.35, "regrow_max_len": 72},
        ],
    },
}


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def load_warm_module(repo: Path):
    path = repo / "experiments" / "run_fix7_istrail_long_warm_budget.py"
    spec = importlib.util.spec_from_file_location("warm_budget_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import helper module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def choose_final(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    hits = [r for r in rows if r.get("target_hit") == "yes"]
    if hits:
        return min(hits, key=lambda r: float(r.get("cumulative_runtime_s", 1e99) or 1e99))
    return max(rows, key=lambda r: int(r.get("contacts", -1)))


def write_outputs(repo: Path, final_rows: List[Dict[str, Any]], run_rows: List[Dict[str, Any]]) -> None:
    write_csv(repo / "runs" / "istrail_long_fix7_16w_big_budget_summary.csv", run_rows)
    write_csv(repo / "runs" / "istrail_long_fix7_16w_big_budget_final_table.csv", final_rows)
    report = repo / "runs" / "istrail_long_fix7_16w_big_budget_report.md"
    lines = [
        "# Istrail-Long Fix7 16-Worker Big-Budget Target Attempt",
        "",
        "This run uses all 16 local workers and larger annealing budgets with RB-full-produced warm starts only.",
        "It reuses the stronger pairwise-only `hp_reward=-3.0`, `L=12` QUBOs from the warm-budget experiment.",
        "",
        "## Summary",
        "",
        "| Benchmark | Target | Best | Target hit | Runtime s to selected best | Attempted runtime s | Hit rate | Workers | Trials | Steps | BBox | Notes |",
        "|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in final_rows:
        lines.append(
            f"| {row['Benchmark']} | {row['Target contacts']} | {row['RB-full best contacts']} | {row['Target hit']} | {float(row['Runtime s']):.3f} | {float(row['Attempted runtime s']):.3f} | {float(row['Hit rate']):.4f} | {row['Workers']} | {row['Requested trials']} | {row['Steps']} | {row['Best fold bbox']} | {row['Notes']} |"
        )
    lines.extend([
        "",
        "## Iterations",
        "",
        "| Benchmark | Iter | Label | Warm contacts | Runtime s | Best | Trials | Target hit | Command file |",
        "|---|---:|---|---:|---:|---:|---:|:---:|---|",
    ])
    for row in run_rows:
        lines.append(
            f"| {row['benchmark']} | {row['iteration']} | `{row['label']}` | {row['warm_contacts']} | {float(row['wrapper_seconds']):.3f} | {row['contacts']} | {row['actual_trials']} | {row['target_hit']} | `{row['command_file']}` |"
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    table_dir = repo / "results" / "tables"
    fig_dir = repo / "results" / "figure_tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    write_csv(table_dir / "table19_istrail_long_16w_big_budget_target_attempt.csv", final_rows)
    md_lines = [
        "# Table 19. Istrail-Long RB-Full 16-Worker Big-Budget Target Attempt",
        "",
        "Source: `runs/istrail_long_fix7_16w_big_budget_summary.csv`.",
        "This attempt uses all 16 local workers and RB-full-produced warm starts only.",
        "",
        "| Benchmark | Target contacts | RB-full best contacts | Target hit | Runtime s to selected best | Attempted runtime s | Hit rate | Workers | Trials | Steps | L | hp reward | Penalty | Best fold bbox | Notes |",
        "|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in final_rows:
        md_lines.append(
            f"| {row['Benchmark']} | {row['Target contacts']} | {row['RB-full best contacts']} | {row['Target hit']} | {float(row['Runtime s']):.3f} | {float(row['Attempted runtime s']):.3f} | {float(row['Hit rate']):.4f} | {row['Workers']} | {row['Requested trials']} | {row['Steps']} | {row['L']} | {row['hp_reward']} | {row['lambda_onehot']} | {row['Best fold bbox']} | {row['Notes']} |"
        )
    (table_dir / "table19_istrail_long_16w_big_budget_target_attempt.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    fig_rows = [
        {
            "Benchmark": row["Benchmark"],
            "Target contacts": row["Target contacts"],
            "RB-full best contacts": row["RB-full best contacts"],
            "Target hit": row["Target hit"],
            "Runtime s": row["Runtime s"],
            "Best fold bbox": row["Best fold bbox"],
            "Best fold PNG": row["Best fold PNG"],
            "Best fold PDF": row["Best fold PDF"],
            "Best JSON": row["Best JSON"],
            "QUBO JSON": row["QUBO JSON"],
        }
        for row in final_rows
    ]
    write_csv(fig_dir / "fig11_istrail_long_16w_big_budget_fold_sources.csv", fig_rows)
    fig_lines = [
        "# Figure 11. Istrail-Long 16-Worker Big-Budget Fold Sources",
        "",
        "| Benchmark | Target | Best | Target hit | Runtime s | BBox | PNG | Best JSON |",
        "|---|---:|---:|:---:|---:|---|---|---|",
    ]
    for row in fig_rows:
        fig_lines.append(
            f"| {row['Benchmark']} | {row['Target contacts']} | {row['RB-full best contacts']} | {row['Target hit']} | {float(row['Runtime s']):.3f} | {row['Best fold bbox']} | `{row['Best fold PNG']}` | `{row['Best JSON']}` |"
        )
    (fig_dir / "fig11_istrail_long_16w_big_budget_fold_sources.md").write_text("\n".join(fig_lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", nargs="*", default=list(BENCHMARKS))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    repo = repo_root_from_script()
    helper = load_warm_module(repo)
    run_rows: List[Dict[str, Any]] = []
    final_rows: List[Dict[str, Any]] = []

    for bench in args.benchmarks:
        cfg = BENCHMARKS[bench]
        target = int(cfg["target"])
        root = repo / "runs" / bench / "local" / "Final_run" / "fix7_istrail_long_16w_big_budget"
        root.mkdir(parents=True, exist_ok=True)
        qpath = repo / str(cfg["qubo"])
        warm_source = repo / str(cfg["source_warm"])
        if not qpath.exists():
            raise FileNotFoundError(qpath)
        if not warm_source.exists():
            raise FileNotFoundError(warm_source)
        qmeta = helper.read_qubo_header(qpath)
        rows = load_csv(root / "fix7_istrail_long_16w_big_budget_summary.csv") if args.resume else []
        bench_rows: List[Dict[str, Any]] = [dict(r) for r in rows]
        previous_best = warm_source
        if bench_rows:
            last = Path(str(bench_rows[-1].get("result_file", "")))
            if last.exists():
                previous_best = last
        target_hit = any(str(r.get("target_hit", "")).lower() == "yes" for r in bench_rows)

        for idx, run_cfg in enumerate(cfg["schedule"], start=1):
            if target_hit:
                break
            if any(str(r.get("iteration")) == str(idx) for r in bench_rows):
                continue
            warm = previous_best
            warm_contacts = helper.contacts_of(warm)
            outdir = root / f"iter{idx:02d}_{run_cfg['label']}"
            cmd = helper.build_command(repo, qpath, outdir, run_cfg, warm, target)
            outdir.mkdir(parents=True, exist_ok=True)
            command_file = outdir / "command.txt"
            command_file.write_text(subprocess.list2cmdline(cmd) + "\n", encoding="utf-8")
            helper.write_json(outdir / "metadata.json", {
                "benchmark": bench,
                "target_contacts": target,
                "qubo_file": str(qpath),
                "qubo_meta": qmeta,
                "warm_start_file": str(warm),
                "warm_contacts": warm_contacts,
                "run_config": run_cfg,
                "command": cmd,
                "note": "16-worker big-budget run using RB-full-produced warm start only",
            })
            print(f"[16w-big] {bench} iter={idx} label={run_cfg['label']} warm_contacts={warm_contacts}", flush=True)
            started = time.time()
            with (outdir / "wrapper_run.log").open("w", encoding="utf-8") as log:
                log.write("[cmd] " + subprocess.list2cmdline(cmd) + "\n")
                log.flush()
                proc = subprocess.run(cmd, cwd=str(repo), stdout=log, stderr=subprocess.STDOUT, text=True)
            elapsed = time.time() - started
            result_file = outdir / "best_overall.json"
            result = helper.read_result(result_file, target)
            cumulative_runtime = sum(float(r.get("wrapper_seconds", 0) or 0) for r in bench_rows) + elapsed
            row = {
                "benchmark": bench,
                "target_contacts": target,
                "iteration": idx,
                "label": run_cfg["label"],
                "warm_start_file": str(warm),
                "warm_contacts": warm_contacts,
                "outdir": str(outdir),
                "result_file": str(result_file),
                "command_file": str(command_file),
                "status": "ok" if proc.returncode == 0 else f"failed:{proc.returncode}",
                "requested_trials": int(run_cfg["trials"]),
                "workers": int(run_cfg["workers"]),
                "steps": int(run_cfg["steps"]),
                "t_init": float(run_cfg["t_init"]),
                "t_final": 0.0005,
                "seed": int(run_cfg["seed"]),
                "L": qmeta.get("L"),
                "S": qmeta.get("S"),
                "V": qmeta.get("V"),
                "hp_reward": qmeta.get("hp_reward"),
                "lambda_onehot": qmeta.get("lambda_onehot"),
                "lambda_site": qmeta.get("lambda_site"),
                "lambda_chain": qmeta.get("lambda_chain"),
                "scale_alpha": qmeta.get("scale_alpha"),
                "qubo_size_mb": qmeta.get("file_size_mb"),
                "wrapper_seconds": elapsed,
                "cumulative_runtime_s": cumulative_runtime,
                "command": subprocess.list2cmdline(cmd),
                "notes": "16 workers; RB-full-produced warm start; hp_reward=-3.0 QUBO",
                **result,
            }
            bench_rows.append(row)
            write_csv(outdir / "summary.csv", [row])
            write_csv(root / "fix7_istrail_long_16w_big_budget_summary.csv", bench_rows)
            print(f"[16w-big-done] {bench} iter={idx} contacts={result['contacts']} target_hit={result['target_hit']} elapsed={elapsed:.1f}s status={row['status']}", flush=True)
            previous_best = result_file if result_file.exists() else previous_best
            if result["target_hit"] == "yes" or proc.returncode != 0:
                target_hit = result["target_hit"] == "yes"
                break

        selected = choose_final(bench_rows)
        attempted_runtime = sum(float(r.get("wrapper_seconds", 0) or 0) for r in bench_rows)
        qmeta_with_seq = dict(qmeta)
        if "seq" not in qmeta_with_seq:
            qmeta_with_seq["seq"] = json.loads(qpath.read_text(encoding="utf-8"))["seq"]
        plot = helper.plot_best_fold(Path(str(selected["result_file"])), qmeta_with_seq, Path(str(selected["outdir"])) / f"{bench}_16w_big_budget_best_fold")
        final_rows.append({
            "Benchmark": bench,
            "N": int(qmeta.get("N", 0)),
            "Target contacts": target,
            "Target HP energy": -target,
            "RB-full best contacts": int(selected.get("contacts", -1)),
            "RB-full best HP energy": -int(selected.get("contacts", -1)) if int(selected.get("contacts", -1)) >= 0 else "",
            "Target hit": selected.get("target_hit", "no"),
            "Runtime s": float(selected.get("cumulative_runtime_s", selected.get("wrapper_seconds", 0)) or 0),
            "Attempted runtime s": attempted_runtime,
            "Hit rate": float(selected.get("hit_rate", 0) or 0),
            "Completed trials": int(float(selected.get("actual_trials", 0) or 0)),
            "Requested trials": int(float(selected.get("requested_trials", 0) or 0)),
            "Workers": int(float(selected.get("workers", 0) or 0)),
            "Steps": int(float(selected.get("steps", 0) or 0)),
            "L": qmeta.get("L"),
            "S": qmeta.get("S"),
            "V": qmeta.get("V"),
            "hp_reward": qmeta.get("hp_reward"),
            "lambda_onehot": qmeta.get("lambda_onehot"),
            "lambda_site": qmeta.get("lambda_site"),
            "lambda_chain": qmeta.get("lambda_chain"),
            "scale_alpha": qmeta.get("scale_alpha"),
            "QUBO size MB": qmeta.get("file_size_mb"),
            "Best fold bbox": plot.get("bbox", ""),
            "Best fold PNG": plot.get("plot_png", ""),
            "Best fold PDF": plot.get("plot_pdf", ""),
            "Best JSON": selected.get("result_file", ""),
            "QUBO JSON": str(qpath),
            "Run folder": selected.get("outdir", ""),
            "Notes": "Target reached by 16-worker RB-full big-budget run" if selected.get("target_hit") == "yes" else "Target not reached; 16-worker RB-full big-budget attempt",
        })
        run_rows.extend(bench_rows)
        write_outputs(repo, final_rows, run_rows)

    write_outputs(repo, final_rows, run_rows)
    print("[16w-big-report] runs/istrail_long_fix7_16w_big_budget_report.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

