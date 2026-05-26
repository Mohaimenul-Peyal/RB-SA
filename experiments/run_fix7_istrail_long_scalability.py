#!/usr/bin/env python3
"""Run fix7 RB-full scalability pilots for the long Istrail HP benchmarks.

This experiment is intentionally separated from the validated 20merA-64mer
tables. The 85mer/100merA/100merB instances have much larger pairwise QUBOs,
so this script uses fresh in-directory QUBOs, lower worker counts, target-aware
early stopping, and in-directory self-warm only.
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
        "hp_reward": -2.5,
        "penalty": 600.0,
        "schedule": [
            {"label": "nowarm_32tr_500k", "trials": 32, "workers": 4, "steps": 500_000, "t_init": 16.0, "seed": 930_085_001, "contact_bias": 0.75, "regrow_max_len": 36, "warm": False},
            {"label": "selfwarm_32tr_650k", "trials": 32, "workers": 4, "steps": 650_000, "t_init": 16.0, "seed": 930_085_002, "contact_bias": 0.75, "regrow_max_len": 36, "warm": True},
            {"label": "nowarm_64tr_800k_seed_sweep", "trials": 64, "workers": 4, "steps": 800_000, "t_init": 18.0, "seed": 930_085_003, "contact_bias": 1.00, "regrow_max_len": 42, "qubo_polish_frac": 0.15, "warm": False},
        ],
    },
    "100merA": {
        "seq": "PPPPPPHPHHPPPPPHHHPHHHHHPHHPPPPHHPPHHPHHHHHPHHHHHHHHHHPHHPHHHHHHHPPPPPPPPPPPHHHHHHHPPHPHHHPPPPPPHPHH",
        "target": 48,
        "L": 12,
        "hp_reward": -2.5,
        "penalty": 650.0,
        "schedule": [
            {"label": "nowarm_24tr_400k", "trials": 24, "workers": 4, "steps": 400_000, "t_init": 16.0, "seed": 930_100_101, "contact_bias": 0.75, "regrow_max_len": 40, "warm": False},
            {"label": "selfwarm_24tr_550k", "trials": 24, "workers": 4, "steps": 550_000, "t_init": 16.0, "seed": 930_100_102, "contact_bias": 0.75, "regrow_max_len": 40, "warm": True},
            {"label": "nowarm_48tr_700k_seed_sweep", "trials": 48, "workers": 4, "steps": 700_000, "t_init": 18.0, "seed": 930_100_103, "contact_bias": 1.00, "regrow_max_len": 44, "qubo_polish_frac": 0.15, "warm": False},
        ],
    },
    "100merB": {
        "seq": "PPPHHPPHHHHPPHHHPHHPHHPHHHHPPPPPPPPHHHHHHPPHHHHHHPPPPPPPPPHPHHPHHHHHHHHHHHPPHHHPHHPHPPHPHHHPPPPPPHHH",
        "target": 50,
        "L": 12,
        "hp_reward": -2.5,
        "penalty": 650.0,
        "schedule": [
            {"label": "nowarm_24tr_400k", "trials": 24, "workers": 4, "steps": 400_000, "t_init": 16.0, "seed": 930_100_201, "contact_bias": 0.75, "regrow_max_len": 40, "warm": False},
            {"label": "selfwarm_24tr_550k", "trials": 24, "workers": 4, "steps": 550_000, "t_init": 16.0, "seed": 930_100_202, "contact_bias": 0.75, "regrow_max_len": 40, "warm": True},
            {"label": "nowarm_48tr_700k_seed_sweep", "trials": 48, "workers": 4, "steps": 700_000, "t_init": 18.0, "seed": 930_100_203, "contact_bias": 1.00, "regrow_max_len": 44, "qubo_polish_frac": 0.15, "warm": False},
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
    qpath = repo / "runs" / bench / "local" / "Final_run" / "fix7_istrail_long_scalability" / "qubos" / f"{bench}_pairwise_L{cfg['L']}_hp{hp_slug}.json"
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
    """Read only the scalar JSON fields before the large coords/term arrays."""
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
        "enable_3body": r'"enable_3body"\s*:\s*(true|false)',
        "enable_4body": r'"enable_4body"\s*:\s*(true|false)',
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
        if key in {"seq"}:
            out[key] = val
        elif key in {"enable_3body", "enable_4body"}:
            out[key] = val == "true"
        elif key in {"N", "L", "S", "V"}:
            out[key] = int(val)
        else:
            out[key] = float(val)
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


def build_command(repo: Path, qpath: Path, outdir: Path, cfg: Dict[str, Any], warm: Optional[Path], target: int) -> List[str]:
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
        str(target),
        "--stop_on_target",
        "--poll_interval",
        "3.0",
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
        "0.18",
        "--regrow_max_len",
        str(int(cfg["regrow_max_len"])),
        "--init_mode",
        "seq_greedy",
        "--archive_size",
        "32",
        "--archive_min_hamming_frac",
        "0.24",
        "--archive_contact_slack",
        "6",
        "--contact_priority_best",
        "--contact_check_every",
        "75",
        "--contact_guided_accept",
        "--contact_bias",
        str(float(cfg["contact_bias"])),
        "--contact_bias_final_frac",
        "0.10",
        "--contact_paving_weight",
        "0.0010",
        "--qubo_polish_frac",
        str(float(cfg.get("qubo_polish_frac", 0.25))),
    ]
    if warm is not None:
        warm_contacts = int(contacts_of(warm) or 0)
        solver_extra.extend([
            "--warm_start_prob",
            "0.60",
            "--warm_start_min_contacts",
            str(max(0, warm_contacts - 2)),
            "--warm_start_file",
            str(warm),
        ])
    cmd.extend(["--", *solver_extra])
    return cmd


def read_result(path: Path, target: int) -> Dict[str, Any]:
    if not path.exists():
        return {
            "contacts": -1,
            "energy": "",
            "successes": 0,
            "actual_trials": 0,
            "hit_rate": 0.0,
            "target_hit": "no",
            "target_early_stop_hit": False,
            "first_target_trial": "",
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
    fig_size = 7.5 if N >= 85 else 6.0
    marker_size = 220 if N >= 85 else 500
    font_size = 4.5 if N >= 85 else 9
    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ax.plot(xs, ys, color="gray", lw=2.2, zorder=1)
    for i, (x, y) in enumerate(coords):
        color = "red" if seq[i] == "H" else "blue"
        ax.scatter(x, y, s=marker_size, color=color, edgecolors="k", linewidths=0.45, zorder=2)
        ax.text(x, y, f"{i}", ha="center", va="center", color="white", fontsize=font_size, zorder=3)
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


def choose_final(rows: List[Dict[str, Any]], target: int) -> Dict[str, Any]:
    hits = [r for r in rows if r.get("target_hit") == "yes"]
    if hits:
        return min(hits, key=lambda r: float(r.get("cumulative_runtime_s", 1e99) or 1e99))
    return max(rows, key=lambda r: int(r.get("contacts", -1)))


def write_report(repo: Path, final_rows: List[Dict[str, Any]], run_rows: List[Dict[str, Any]]) -> None:
    report = repo / "runs" / "istrail_long_fix7_scalability_report.md"
    lines = [
        "# Istrail-Long Fix7 RB-Full Scalability Experiment",
        "",
        "This experiment extends RB-full beyond the validated 20merA-64mer set to the Istrail 85mer, 100merA, and 100merB chains.",
        "The runs use fresh pairwise-only QUBOs and only in-directory self-warm seeds generated by the immediately preceding fix7 stage.",
        "",
        "## Summary",
        "",
        "| Benchmark | N | Target | Best | Target hit | Runtime s to selected best | Attempted runtime s | Hit rate | L | QUBO MB | Notes |",
        "|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in final_rows:
        lines.append(
            f"| {row['Benchmark']} | {row['N']} | {row['Target contacts']} | {row['RB-full best contacts']} | {row['Target hit']} | {float(row['Runtime s']):.3f} | {float(row['Attempted runtime s']):.3f} | {float(row['Hit rate']):.4f} | {row['L']} | {row['QUBO size MB']} | {row['Notes']} |"
        )
    lines.extend([
        "",
        "## Iteration Log",
        "",
        "| Benchmark | Iter | Label | Warm contacts | Runtime s | Best | Trials | Hit rate | Target hit | Command file |",
        "|---|---:|---|---:|---:|---:|---:|---:|:---:|---|",
    ])
    for row in run_rows:
        lines.append(
            f"| {row['benchmark']} | {row['iteration']} | `{row['label']}` | {row['warm_contacts']} | {float(row['wrapper_seconds']):.3f} | {row['contacts']} | {row['actual_trials']} | {float(row['hit_rate']):.4f} | {row['target_hit']} | `{row['command_file']}` |"
        )
    lines.extend([
        "",
        "## Paper-Use Note",
        "",
        "Do not merge these rows into the primary ground-state validation table unless `Target hit` is `yes`.",
        "If any target is missed, report the table as a scalability pilot and keep the validated 20merA-64mer table as the main evidence.",
    ])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manuscript_tables(repo: Path, final_rows: List[Dict[str, Any]]) -> None:
    table_dir = repo / "results" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    csv_path = table_dir / "table17_istrail_long_fix7_scalability_pilot.csv"
    md_path = table_dir / "table17_istrail_long_fix7_scalability_pilot.md"
    write_csv(csv_path, final_rows)
    lines = [
        "# Table 17. Istrail-Long RB-Full Scalability Pilot",
        "",
        "Source: `runs/istrail_long_fix7_scalability_summary.csv`.",
        "These rows extend the experiment to 85mer, 100merA, and 100merB using fresh pairwise-only QUBOs and in-directory fix7 self-warm only.",
        "They should be presented as an extension/scalability pilot unless every row reaches the reference target.",
        "",
        "| Benchmark | N | Target contacts | RB-full best contacts | Target hit | Runtime s to selected best | Attempted runtime s | Hit rate | Workers | Trials | Steps | L | hp reward | Penalty | Best fold bbox | Notes |",
        "|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in final_rows:
        lines.append(
            f"| {row['Benchmark']} | {row['N']} | {row['Target contacts']} | {row['RB-full best contacts']} | {row['Target hit']} | {float(row['Runtime s']):.3f} | {float(row['Attempted runtime s']):.3f} | {float(row['Hit rate']):.4f} | {row['Workers']} | {row['Requested trials']} | {row['Steps']} | {row['L']} | {row['hp_reward']} | {row['lambda_onehot']} | {row['Best fold bbox']} | {row['Notes']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure_table(repo: Path, final_rows: List[Dict[str, Any]]) -> None:
    fig_dir = repo / "results" / "figure_tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "Benchmark": r["Benchmark"],
            "N": r["N"],
            "Target contacts": r["Target contacts"],
            "RB-full best contacts": r["RB-full best contacts"],
            "Target hit": r["Target hit"],
            "Runtime s": r["Runtime s"],
            "Best fold bbox": r["Best fold bbox"],
            "Best fold PNG": r["Best fold PNG"],
            "Best fold PDF": r["Best fold PDF"],
            "Best JSON": r["Best JSON"],
            "QUBO JSON": r["QUBO JSON"],
            "Figure use": "Istrail-long scalability extension; do not label as ground state unless target hit is yes",
        }
        for r in final_rows
    ]
    csv_path = fig_dir / "fig9_istrail_long_scalability_fold_sources.csv"
    md_path = fig_dir / "fig9_istrail_long_scalability_fold_sources.md"
    write_csv(csv_path, rows)
    lines = [
        "# Figure 9. Istrail-Long Scalability Fold Sources",
        "",
        "This figure table lists the best fold plots from the 85mer-100merB RB-full scalability extension.",
        "",
        "| Benchmark | N | Target | Best | Target hit | Runtime s | BBox | PNG | Best JSON |",
        "|---|---:|---:|---:|:---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['Benchmark']} | {row['N']} | {row['Target contacts']} | {row['RB-full best contacts']} | {row['Target hit']} | {float(row['Runtime s']):.3f} | {row['Best fold bbox']} | `{row['Best fold PNG']}` | `{row['Best JSON']}` |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def upsert_inventory(repo: Path) -> None:
    table_inventory = repo / "results" / "tables" / "table_inventory.csv"
    rows = load_csv(table_inventory)
    rows = [r for r in rows if r.get("Table") != "Table 17"]
    rows.append({
        "Table": "Table 17",
        "File": "table17_istrail_long_fix7_scalability_pilot.md",
        "Status": "Scalability extension",
        "Recommended use": "Use for 85mer-100merB RB-full extension; present as pilot unless all target contacts are reached.",
    })
    write_csv(table_inventory, rows)
    md_path = table_inventory.with_suffix(".md")
    lines = [
        "# Table 16. Manuscript Table Inventory",
        "",
        "| Table | File | Status | Recommended use |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['Table']} | {r['File']} | {r['Status']} | {r['Recommended use']} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig_inventory = repo / "results" / "figure_tables" / "figure_table_inventory.csv"
    frows = load_csv(fig_inventory)
    frows = [r for r in frows if r.get("Figure table") != "fig9_istrail_long_scalability_fold_sources.md"]
    frows.append({
        "Figure table": "fig9_istrail_long_scalability_fold_sources.md",
        "Recommended figure": "Istrail-long best-fold and scalability extension panels",
        "Status": "pilot/extension",
        "Main source": "Table 17 and per-benchmark best_overall.json files",
    })
    write_csv(fig_inventory, frows)
    fmd_path = fig_inventory.with_suffix(".md")
    lines = [
        "# Figure Table Inventory",
        "",
        "| Figure table | Recommended figure | Status | Main source |",
        "|---|---|---|---|",
    ]
    for r in frows:
        lines.append(f"| {r['Figure table']} | {r['Recommended figure']} | {r['Status']} | {r['Main source']} |")
    fmd_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", nargs="*", default=list(BENCHMARKS))
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    repo = repo_root_from_script()
    all_run_rows: List[Dict[str, Any]] = []
    final_rows: List[Dict[str, Any]] = []

    for bench in args.benchmarks:
        cfg = BENCHMARKS[bench]
        target = int(cfg["target"])
        root = repo / "runs" / bench / "local" / "Final_run" / "fix7_istrail_long_scalability"
        if args.clean and root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)

        print(f"[istrail] exporting/checking QUBO for {bench}", flush=True)
        qpath = export_qubo(repo, bench, cfg, force=bool(args.force_export))
        qmeta = read_qubo_header(qpath)
        qmeta.update({"seq": cfg["seq"]})

        rows = load_csv(root / "fix7_istrail_long_scalability_summary.csv") if args.resume else []
        bench_rows: List[Dict[str, Any]] = [dict(r) for r in rows]
        previous_best: Optional[Path] = None
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
            warm = previous_best if bool(run_cfg.get("warm")) and previous_best is not None else None
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
                "warm_start_file": str(warm) if warm else "",
                "warm_contacts": warm_contacts,
                "run_config": run_cfg,
                "command": cmd,
            })
            print(f"[istrail-run] {bench} iter={idx} label={run_cfg['label']} warm_contacts={warm_contacts}", flush=True)
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
                "warm_start_file": str(warm) if warm else "",
                "warm_contacts": warm_contacts,
                "outdir": str(outdir),
                "result_file": str(result_file),
                "command_file": str(command_file),
                "status": "ok" if proc.returncode == 0 else f"failed:{proc.returncode}",
                "requested_trials": int(run_cfg["trials"]),
                "workers": int(run_cfg["workers"]),
                "steps": int(run_cfg["steps"]),
                "t_init": float(run_cfg["t_init"]),
                "t_final": 0.0006,
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
                "notes": "fresh pairwise-only QUBO; in-directory self-warm only" if warm else "fresh pairwise-only QUBO; no warm start",
                **result,
            }
            bench_rows.append(row)
            write_csv(outdir / "summary.csv", [row])
            write_csv(root / "fix7_istrail_long_scalability_summary.csv", bench_rows)
            print(f"[istrail-done] {bench} iter={idx} contacts={result['contacts']} target_hit={result['target_hit']} elapsed={elapsed:.1f}s", flush=True)
            previous_best = result_file if result_file.exists() else previous_best
            if result["target_hit"] == "yes" or proc.returncode != 0:
                target_hit = result["target_hit"] == "yes"
                break

        if bench_rows:
            selected = choose_final(bench_rows, target)
            plot = plot_best_fold(Path(str(selected["result_file"])), qmeta, Path(str(selected["outdir"])) / f"{bench}_best_fold")
            attempted_runtime = sum(float(r.get("wrapper_seconds", 0) or 0) for r in bench_rows)
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
                "Notes": "Target reached by RB-full fix7 extension" if selected.get("target_hit") == "yes" else "Scalability pilot; target not reached in this local budget",
            })
        all_run_rows.extend(bench_rows)

    write_csv(repo / "runs" / "istrail_long_fix7_scalability_summary.csv", all_run_rows)
    write_csv(repo / "runs" / "istrail_long_fix7_scalability_final_table.csv", final_rows)
    write_report(repo, final_rows, all_run_rows)
    write_manuscript_tables(repo, final_rows)
    write_figure_table(repo, final_rows)
    upsert_inventory(repo)
    print("[istrail-report] runs/istrail_long_fix7_scalability_report.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

