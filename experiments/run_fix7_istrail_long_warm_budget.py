#!/usr/bin/env python3
"""Warm-budget fix7 RB-full target attempts for Istrail-long chains.

The source warm starts are RB-full/fix7 outputs from the previous
`fix7_istrail_long_scalability` experiment. No external coordinate-space target-fold search
native folds are used here.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "85mer": {
        "seq": "HHHHPPPPHHHHHHHHHHHHPPPPPPHHHHHHHHHHHHPPPHHHHHHHHHHHHPPPHHHHHHHHHHHHPPPHPPHHPPHHPPHPH",
        "target": 53,
        "L": 12,
        "hp_reward": -3.0,
        "penalty": 600.0,
        "source_warm": "runs/85mer/local/Final_run/fix7_istrail_long_scalability/iter03_nowarm_64tr_800k_seed_sweep/best_overall.json",
        "schedule": [
            {"label": "warm96_1p2M_hp3", "trials": 96, "workers": 8, "steps": 1_200_000, "t_init": 18.0, "seed": 940_085_001, "contact_bias": 1.05, "regrow_max_len": 48, "warm": True},
            {"label": "warm128_1p5M_hp3_retry", "trials": 128, "workers": 8, "steps": 1_500_000, "t_init": 18.0, "seed": 940_085_002, "contact_bias": 1.10, "regrow_max_len": 52, "warm": True},
        ],
    },
    "100merA": {
        "seq": "PPPPPPHPHHPPPPPHHHPHHHHHPHHPPPPHHPPHHPHHHHHPHHHHHHHHHHPHHPHHHHHHHPPPPPPPPPPPHHHHHHHPPHPHHHPPPPPPHPHH",
        "target": 48,
        "L": 12,
        "hp_reward": -3.0,
        "penalty": 650.0,
        "source_warm": "runs/100merA/local/Final_run/fix7_istrail_long_scalability/iter01_nowarm_24tr_400k/best_overall.json",
        "schedule": [
            {"label": "warm96_1p2M_hp3", "trials": 96, "workers": 8, "steps": 1_200_000, "t_init": 18.0, "seed": 940_100_101, "contact_bias": 1.05, "regrow_max_len": 52, "warm": True},
            {"label": "warm128_1p5M_hp3_retry", "trials": 128, "workers": 8, "steps": 1_500_000, "t_init": 18.0, "seed": 940_100_102, "contact_bias": 1.10, "regrow_max_len": 56, "warm": True},
        ],
    },
    "100merB": {
        "seq": "PPPHHPPHHHHPPHHHPHHPHHPHHHHPPPPPPPPHHHHHHPPHHHHHHPPPPPPPPPHPHHPHHHHHHHHHHHPPHHHPHHPHPPHPHHHPPPPPPHHH",
        "target": 50,
        "L": 12,
        "hp_reward": -3.0,
        "penalty": 650.0,
        "source_warm": "runs/100merB/local/Final_run/fix7_istrail_long_scalability/iter01_nowarm_24tr_400k/best_overall.json",
        "schedule": [
            {"label": "warm96_1p2M_hp3", "trials": 96, "workers": 8, "steps": 1_200_000, "t_init": 18.0, "seed": 940_100_201, "contact_bias": 1.05, "regrow_max_len": 52, "warm": True},
            {"label": "warm128_1p5M_hp3_retry", "trials": 128, "workers": 8, "steps": 1_500_000, "t_init": 18.0, "seed": 940_100_202, "contact_bias": 1.10, "regrow_max_len": 56, "warm": True},
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


def load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def slug_float(value: float) -> str:
    return str(value).replace("-", "").replace(".", "p")


def export_qubo(repo: Path, bench: str, cfg: Dict[str, Any], force: bool) -> Path:
    hp_slug = slug_float(abs(float(cfg["hp_reward"])))
    qpath = repo / "runs" / bench / "local" / "Final_run" / "fix7_istrail_long_warm_budget" / "qubos" / f"{bench}_pairwise_L{cfg['L']}_hp{hp_slug}.json"
    if qpath.exists() and not force:
        return qpath
    cmd = [
        sys.executable,
        str(repo / "src" / "HP_export_manybody_ising.py"),
        "--seq",
        str(cfg["seq"]),
        "--l_size",
        str(int(cfg["L"])),
        "--hp_reward",
        str(float(cfg["hp_reward"])),
        "--penalty_mode",
        "manual",
        "--lambda_onehot",
        str(float(cfg["penalty"])),
        "--lambda_site",
        str(float(cfg["penalty"])),
        "--lambda_chain",
        str(float(cfg["penalty"])),
        "--scale_target",
        "1000",
        "--out",
        str(qpath),
    ]
    qpath.parent.mkdir(parents=True, exist_ok=True)
    (qpath.parent / f"export_{bench}_pairwise_command.txt").write_text(subprocess.list2cmdline(cmd) + "\n", encoding="utf-8")
    subprocess.run(cmd, cwd=str(repo), check=True)
    return qpath


def read_qubo_header(path: Path) -> Dict[str, Any]:
    text_parts: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if '"coords"' in line:
                break
            text_parts.append(line)
    text = "".join(text_parts)
    out: Dict[str, Any] = {"file_size_mb": round(path.stat().st_size / (1024 * 1024), 3)}
    patterns = {
        "seq": r'"seq"\s*:\s*"([^"]+)"',
        "N": r'"N"\s*:\s*(\d+)',
        "L": r'"L"\s*:\s*(\d+)',
        "S": r'"S"\s*:\s*(\d+)',
        "V": r'"V"\s*:\s*(\d+)',
        "hp_reward": r'"hp_reward"\s*:\s*([-+0-9.eE]+)',
        "lambda_onehot": r'"lambda_onehot"\s*:\s*([-+0-9.eE]+)',
        "lambda_site": r'"lambda_site"\s*:\s*([-+0-9.eE]+)',
        "lambda_chain": r'"lambda_chain"\s*:\s*([-+0-9.eE]+)',
        "scale_alpha": r'"scale_alpha"\s*:\s*([-+0-9.eE]+)',
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if not m:
            continue
        val = m.group(1)
        out[key] = val if key == "seq" else int(val) if key in {"N", "L", "S", "V"} else float(val)
    return out


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


def prepare_source_warm(repo: Path, bench: str, cfg: Dict[str, Any], root: Path) -> Optional[Path]:
    source = repo / str(cfg["source_warm"])
    if not source.exists():
        return None
    dest = root / "warm_starts" / "warm_start_from_scalability_best.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(source, dest)
    return dest


def build_command(repo: Path, qpath: Path, outdir: Path, cfg: Dict[str, Any], warm: Path, target: int) -> List[str]:
    runner = repo / "src" / "HP_ssa_manybody_parallel_runner_cpu_fast_passthru.py"
    solver = repo / "src" / "HP_ssa_manybody_ising_cpu_numba_saw_pivot_fix7_contactguided.py"
    warm_contacts = int(contacts_of(warm) or 0)
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
        "0.0005",
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
        "30000",
        "--reheat_factor",
        "2.6",
        "--outdir",
        str(outdir),
        "--reseed_each_trial",
        "--target_contacts",
        str(target),
        "--stop_on_target",
        "--poll_interval",
        "3.0",
        "--warm_start_best",
        "--best_t_scale",
        "1.45",
        "--",
        "--pivot_prob",
        "0.52",
        "--pivot_max_tail",
        "0",
        "--pull_prob",
        "0.10",
        "--reptation_prob",
        "0.12",
        "--regrow_prob",
        "0.06",
        "--frag_regrow_prob",
        "0.20",
        "--regrow_max_len",
        str(int(cfg["regrow_max_len"])),
        "--init_mode",
        "seq_greedy",
        "--archive_size",
        "40",
        "--archive_min_hamming_frac",
        "0.22",
        "--archive_contact_slack",
        "8",
        "--contact_priority_best",
        "--contact_check_every",
        "75",
        "--contact_guided_accept",
        "--contact_bias",
        str(float(cfg["contact_bias"])),
        "--contact_bias_final_frac",
        "0.10",
        "--contact_paving_weight",
        "0.0012",
        "--qubo_polish_frac",
        "0.18",
        "--warm_start_prob",
        "0.70",
        "--warm_start_min_contacts",
        str(max(0, warm_contacts - 3)),
        "--warm_start_file",
        str(warm),
    ]
    return cmd


def read_result(path: Path, target: int) -> Dict[str, Any]:
    if not path.exists():
        return {"contacts": -1, "energy": "", "successes": 0, "actual_trials": 0, "hit_rate": 0.0, "target_hit": "no", "target_early_stop_hit": False, "first_target_trial": ""}
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
            if item.get("trial_best_feasible") and isinstance(c, int) and c >= target:
                successes += 1
                if first_target_trial == "":
                    first_target_trial = str(item.get("global_trial", item.get("trial", "")))
    return {
        "contacts": contacts,
        "energy": best.get("energy", payload.get("energy", "")),
        "successes": successes,
        "actual_trials": actual_trials,
        "hit_rate": successes / actual_trials if actual_trials else 0.0,
        "target_hit": "yes" if contacts >= target else "no",
        "target_early_stop_hit": bool(payload.get("target_early_stop_hit", False)),
        "completed_worker_files": payload.get("completed_worker_files", ""),
        "first_target_trial": first_target_trial,
    }


def plot_best_fold(best_path: Path, qmeta: Dict[str, Any], out_prefix: Path) -> Dict[str, Any]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"plot_status": f"plot skipped: {exc}", "plot_png": "", "plot_pdf": "", "bbox": ""}
    if not best_path.exists():
        return {"plot_status": "plot skipped: missing best_overall.json", "plot_png": "", "plot_pdf": "", "bbox": ""}
    payload = read_json(best_path)
    best = payload.get("best", payload)
    spins = best.get("spins")
    seq = str(qmeta.get("seq", ""))
    L = int(qmeta.get("L", 0) or 0)
    S = int(qmeta.get("S", 0) or 0)
    N = int(qmeta.get("N", len(seq)) or len(seq))
    if not isinstance(spins, list) or len(spins) != N * S or not seq or L <= 0:
        return {"plot_status": "plot skipped: invalid spins/QUBO metadata", "plot_png": "", "plot_pdf": "", "bbox": ""}
    coords = []
    for r in range(N):
        block = spins[r * S : (r + 1) * S]
        site = None
        for i, val in enumerate(block):
            if val == 1:
                site = i
                break
        if site is None:
            return {"plot_status": f"plot skipped: residue {r} has no active site", "plot_png": "", "plot_pdf": "", "bbox": ""}
        coords.append((site // L, site % L))
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    bbox = f"{max(xs) - min(xs) + 1}x{max(ys) - min(ys) + 1}"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")
    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.plot(xs, ys, color="gray", lw=2.2, zorder=1)
    for i, (x, y) in enumerate(coords):
        color = "red" if seq[i] == "H" else "blue"
        ax.scatter(x, y, s=220, color=color, edgecolors="k", linewidths=0.45, zorder=2)
        ax.text(x, y, f"{i}", ha="center", va="center", color="white", fontsize=4.5, zorder=3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(min(xs) - 1.0, max(xs) + 1.0)
    ax.set_ylim(min(ys) - 1.0, max(ys) + 1.0)
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight", dpi=180)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"plot_status": "ok", "plot_png": str(png_path), "plot_pdf": str(pdf_path), "bbox": bbox}


def choose_final(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    hits = [r for r in rows if r.get("target_hit") == "yes"]
    if hits:
        return min(hits, key=lambda r: float(r.get("cumulative_runtime_s", 1e99) or 1e99))
    return max(rows, key=lambda r: int(r.get("contacts", -1)))


def write_outputs(repo: Path, final_rows: List[Dict[str, Any]], run_rows: List[Dict[str, Any]]) -> None:
    write_csv(repo / "runs" / "istrail_long_fix7_warm_budget_summary.csv", run_rows)
    write_csv(repo / "runs" / "istrail_long_fix7_warm_budget_final_table.csv", final_rows)
    report = repo / "runs" / "istrail_long_fix7_warm_budget_report.md"
    lines = [
        "# Istrail-Long Fix7 Warm-Budget Target Attempt",
        "",
        "This experiment uses RB-full-produced warm starts from the earlier scalability pilot, stronger pairwise contact reward (`hp_reward=-3.0`), and larger warm-start annealing budgets.",
        "No target-native `external coordinate-space target-fold search` folds are used.",
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
    write_csv(table_dir / "table18_istrail_long_warm_budget_target_attempt.csv", final_rows)
    table_lines = [
        "# Table 18. Istrail-Long RB-Full Warm-Budget Target Attempt",
        "",
        "Source: `runs/istrail_long_fix7_warm_budget_summary.csv`.",
        "Warm starts are RB-full-produced best folds from the earlier long-chain scalability pilot; no contact-native search warm starts are used.",
        "",
        "| Benchmark | Target contacts | RB-full best contacts | Target hit | Runtime s to selected best | Attempted runtime s | Hit rate | Workers | Trials | Steps | L | hp reward | Penalty | Best fold bbox | Notes |",
        "|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in final_rows:
        table_lines.append(
            f"| {row['Benchmark']} | {row['Target contacts']} | {row['RB-full best contacts']} | {row['Target hit']} | {float(row['Runtime s']):.3f} | {float(row['Attempted runtime s']):.3f} | {float(row['Hit rate']):.4f} | {row['Workers']} | {row['Requested trials']} | {row['Steps']} | {row['L']} | {row['hp_reward']} | {row['lambda_onehot']} | {row['Best fold bbox']} | {row['Notes']} |"
        )
    (table_dir / "table18_istrail_long_warm_budget_target_attempt.md").write_text("\n".join(table_lines) + "\n", encoding="utf-8")

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
    write_csv(fig_dir / "fig10_istrail_long_warm_budget_fold_sources.csv", fig_rows)
    fig_lines = [
        "# Figure 10. Istrail-Long Warm-Budget Fold Sources",
        "",
        "| Benchmark | Target | Best | Target hit | Runtime s | BBox | PNG | Best JSON |",
        "|---|---:|---:|:---:|---:|---|---|---|",
    ]
    for row in fig_rows:
        fig_lines.append(
            f"| {row['Benchmark']} | {row['Target contacts']} | {row['RB-full best contacts']} | {row['Target hit']} | {float(row['Runtime s']):.3f} | {row['Best fold bbox']} | `{row['Best fold PNG']}` | `{row['Best JSON']}` |"
        )
    (fig_dir / "fig10_istrail_long_warm_budget_fold_sources.md").write_text("\n".join(fig_lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", nargs="*", default=list(BENCHMARKS))
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    repo = repo_root_from_script()
    run_rows: List[Dict[str, Any]] = []
    final_rows: List[Dict[str, Any]] = []

    for bench in args.benchmarks:
        cfg = BENCHMARKS[bench]
        target = int(cfg["target"])
        root = repo / "runs" / bench / "local" / "Final_run" / "fix7_istrail_long_warm_budget"
        if args.clean and root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        qpath = export_qubo(repo, bench, cfg, force=bool(args.force_export))
        qmeta = read_qubo_header(qpath)
        qmeta["seq"] = cfg["seq"]
        source_warm = prepare_source_warm(repo, bench, cfg, root)
        if source_warm is None:
            raise FileNotFoundError(f"Missing source warm start for {bench}: {cfg['source_warm']}")

        rows = load_csv(root / "fix7_istrail_long_warm_budget_summary.csv") if args.resume else []
        bench_rows: List[Dict[str, Any]] = [dict(r) for r in rows]
        previous_best: Path = source_warm
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
            warm_contacts = contacts_of(warm)
            outdir = root / f"iter{idx:02d}_{run_cfg['label']}"
            cmd = build_command(repo, qpath, outdir, run_cfg, warm, target)
            outdir.mkdir(parents=True, exist_ok=True)
            command_file = outdir / "command.txt"
            command_file.write_text(subprocess.list2cmdline(cmd) + "\n", encoding="utf-8")
            write_json(outdir / "metadata.json", {
                "benchmark": bench,
                "target_contacts": target,
                "sequence": cfg["seq"],
                "qubo_file": str(qpath),
                "qubo_meta": qmeta,
                "warm_start_file": str(warm),
                "warm_contacts": warm_contacts,
                "run_config": run_cfg,
                "command": cmd,
            })
            print(f"[warm-budget] {bench} iter={idx} label={run_cfg['label']} warm_contacts={warm_contacts}", flush=True)
            started = time.time()
            with (outdir / "wrapper_run.log").open("w", encoding="utf-8") as log:
                log.write("[cmd] " + subprocess.list2cmdline(cmd) + "\n")
                log.flush()
                proc = subprocess.run(cmd, cwd=str(repo), stdout=log, stderr=subprocess.STDOUT, text=True)
            elapsed = time.time() - started
            result_file = outdir / "best_overall.json"
            result = read_result(result_file, target)
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
                "notes": "RB-full-produced warm start; stronger pairwise contact reward; no external coordinate-space warm start",
                **result,
            }
            bench_rows.append(row)
            write_csv(outdir / "summary.csv", [row])
            write_csv(root / "fix7_istrail_long_warm_budget_summary.csv", bench_rows)
            print(f"[warm-budget-done] {bench} iter={idx} contacts={result['contacts']} target_hit={result['target_hit']} elapsed={elapsed:.1f}s", flush=True)
            previous_best = result_file if result_file.exists() else previous_best
            if result["target_hit"] == "yes" or proc.returncode != 0:
                target_hit = result["target_hit"] == "yes"
                break

        selected = choose_final(bench_rows)
        attempted_runtime = sum(float(r.get("wrapper_seconds", 0) or 0) for r in bench_rows)
        plot = plot_best_fold(Path(str(selected["result_file"])), qmeta, Path(str(selected["outdir"])) / f"{bench}_warm_budget_best_fold")
        final_rows.append({
            "Benchmark": bench,
            "N": int(qmeta.get("N", len(cfg["seq"]))),
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
            "Notes": "Target reached by RB-full warm-budget run" if selected.get("target_hit") == "yes" else "Target not reached; RB-full warm-budget attempt",
        })
        run_rows.extend(bench_rows)

    write_outputs(repo, final_rows, run_rows)
    print("[warm-budget-report] runs/istrail_long_fix7_warm_budget_report.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

