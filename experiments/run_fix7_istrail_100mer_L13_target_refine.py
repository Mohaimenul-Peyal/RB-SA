#!/usr/bin/env python3
"""L=13 target refinement for Istrail 100merA and 100merB.

This run is designed for the one-contact-short 100mer basins from the
16-worker L=12 experiment. It exports fresh L=13 pairwise-only QUBOs, remaps
the best RB-full L=12 spin warm starts by lattice coordinate, and then runs
16-worker fix7 refinement. No contact-native search warm starts are used.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "100merA": {
        "seq": "PPPPPPHPHHPPPPPHHHPHHHHHPHHPPPPHHPPHHPHHHHHPHHHHHHHHHHPHHPHHHHHHHPPPPPPPPPPPHHHHHHHPPHPHHHPPPPPPHPHH",
        "target": 48,
        "L": 13,
        "old_L": 12,
        "hp_reward": -4.0,
        "penalty": 650.0,
        "source_warm": "runs/100merA/local/Final_run/fix7_istrail_long_16w_big_budget/iter02_warm512_2p5M_16w_retry/best_overall.json",
        "schedule": [
            {"label": "L13_hp4_warm512_2p5M_16w", "trials": 512, "workers": 16, "steps": 2_500_000, "t_init": 22.0, "seed": 960_100_101, "contact_bias": 1.40, "regrow_max_len": 76},
            {"label": "L13_hp4_warm768_3M_16w_retry", "trials": 768, "workers": 16, "steps": 3_000_000, "t_init": 22.0, "seed": 960_100_102, "contact_bias": 1.50, "regrow_max_len": 84},
        ],
    },
    "100merB": {
        "seq": "PPPHHPPHHHHPPHHHPHHPHHPHHHHPPPPPPPPHHHHHHPPHHHHHHPPPPPPPPPHPHHPHHHHHHHHHHHPPHHHPHHPHPPHPHHHPPPPPPHHH",
        "target": 50,
        "L": 13,
        "old_L": 12,
        "hp_reward": -4.0,
        "penalty": 650.0,
        "source_warm": "runs/100merB/local/Final_run/fix7_istrail_long_16w_big_budget/iter02_warm512_2p5M_16w_retry/best_overall.json",
        "schedule": [
            {"label": "L13_hp4_warm512_2p5M_16w", "trials": 512, "workers": 16, "steps": 2_500_000, "t_init": 22.0, "seed": 960_100_201, "contact_bias": 1.40, "regrow_max_len": 76},
            {"label": "L13_hp4_warm768_3M_16w_retry", "trials": 768, "workers": 16, "steps": 3_000_000, "t_init": 22.0, "seed": 960_100_202, "contact_bias": 1.50, "regrow_max_len": 84},
        ],
    },
}


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def import_warm_helpers(repo: Path):
    path = repo / "experiments" / "run_fix7_istrail_long_warm_budget.py"
    spec = importlib.util.spec_from_file_location("warm_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import helper module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    qpath = repo / "runs" / bench / "local" / "Final_run" / "fix7_istrail_100mer_L13_target_refine" / "qubos" / f"{bench}_pairwise_L{cfg['L']}_hp{hp_slug}.json"
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
    parts: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if '"coords"' in line:
                break
            parts.append(line)
    text = "".join(parts)
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


def remap_warm_start(source: Path, dest: Path, old_L: int, new_L: int, seq: str) -> Path:
    if dest.exists():
        return dest
    raw = read_json(source)
    best = raw.get("best", raw)
    old_spins = best.get("spins")
    N = len(seq)
    old_S = old_L * old_L
    new_S = new_L * new_L
    if not isinstance(old_spins, list) or len(old_spins) != N * old_S:
        raise ValueError(f"Cannot remap warm start: invalid spin length in {source}")
    new_spins = [0] * (N * new_S)
    coords = []
    for r in range(N):
        block = old_spins[r * old_S : (r + 1) * old_S]
        try:
            old_site = block.index(1)
        except ValueError as exc:
            raise ValueError(f"Residue {r} has no active site in {source}") from exc
        x = old_site // old_L
        y = old_site % old_L
        new_site = x * new_L + y
        new_spins[r * new_S + new_site] = 1
        coords.append([x, y])
    obj = {
        "source": "remapped_lattice_warm_start",
        "source_file": str(source),
        "old_L": old_L,
        "new_L": new_L,
        "coords": coords,
        "best": {
            "spins": new_spins,
            "contacts": int(best.get("contacts", raw.get("contacts", -1))),
            "energy": best.get("energy", raw.get("energy", None)),
            "source_contacts": int(best.get("contacts", raw.get("contacts", -1))),
        },
    }
    write_json(dest, obj)
    return dest


def build_command(repo: Path, qpath: Path, outdir: Path, cfg: Dict[str, Any], warm: Path, target: int, warm_contacts: int) -> List[str]:
    runner = repo / "src" / "HP_ssa_manybody_parallel_runner_cpu_fast_passthru.py"
    solver = repo / "src" / "HP_ssa_manybody_ising_cpu_numba_saw_pivot_fix7_contactguided.py"
    return [
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
        "0.00045",
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
        "28000",
        "--reheat_factor",
        "2.8",
        "--outdir",
        str(outdir),
        "--reseed_each_trial",
        "--target_contacts",
        str(target),
        "--stop_on_target",
        "--poll_interval",
        "4.0",
        "--warm_start_best",
        "--best_t_scale",
        "1.55",
        "--",
        "--pivot_prob",
        "0.50",
        "--pivot_max_tail",
        "0",
        "--pull_prob",
        "0.11",
        "--reptation_prob",
        "0.12",
        "--regrow_prob",
        "0.06",
        "--frag_regrow_prob",
        "0.22",
        "--regrow_max_len",
        str(int(cfg["regrow_max_len"])),
        "--init_mode",
        "seq_greedy",
        "--archive_size",
        "48",
        "--archive_min_hamming_frac",
        "0.20",
        "--archive_contact_slack",
        "10",
        "--contact_priority_best",
        "--contact_check_every",
        "100",
        "--contact_guided_accept",
        "--contact_bias",
        str(float(cfg["contact_bias"])),
        "--contact_bias_final_frac",
        "0.10",
        "--contact_paving_weight",
        "0.0015",
        "--qubo_polish_frac",
        "0.12",
        "--warm_start_prob",
        "0.65",
        "--warm_start_min_contacts",
        str(max(0, warm_contacts - 3)),
        "--warm_start_file",
        str(warm),
    ]


def choose_final(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    hits = [r for r in rows if r.get("target_hit") == "yes"]
    if hits:
        return min(hits, key=lambda r: float(r.get("cumulative_runtime_s", 1e99) or 1e99))
    return max(rows, key=lambda r: int(r.get("contacts", -1)))


def write_outputs(repo: Path, final_rows: List[Dict[str, Any]], run_rows: List[Dict[str, Any]]) -> None:
    write_csv(repo / "runs" / "istrail_100mer_L13_target_refine_summary.csv", run_rows)
    write_csv(repo / "runs" / "istrail_100mer_L13_target_refine_final_table.csv", final_rows)
    report = repo / "runs" / "istrail_100mer_L13_target_refine_report.md"
    lines = [
        "# Istrail 100mer L=13 Fix7 Target Refinement",
        "",
        "This experiment targets only `100merA` and `100merB`. It exports L=13 pairwise-only QUBOs, remaps the best RB-full L=12 warm starts into L=13, and runs 16-worker fix7 refinement.",
        "No contact-native warm starts are used.",
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
    write_csv(table_dir / "table20_istrail_100mer_L13_target_refine.csv", final_rows)
    md_lines = [
        "# Table 20. Istrail 100mer L=13 RB-Full Target Refinement",
        "",
        "Source: `runs/istrail_100mer_L13_target_refine_summary.csv`.",
        "The warm starts are RB-full L=12 best folds remapped by coordinate into L=13.",
        "",
        "| Benchmark | Target contacts | RB-full best contacts | Target hit | Runtime s to selected best | Attempted runtime s | Hit rate | Workers | Trials | Steps | L | hp reward | Penalty | Best fold bbox | Notes |",
        "|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in final_rows:
        md_lines.append(
            f"| {row['Benchmark']} | {row['Target contacts']} | {row['RB-full best contacts']} | {row['Target hit']} | {float(row['Runtime s']):.3f} | {float(row['Attempted runtime s']):.3f} | {float(row['Hit rate']):.4f} | {row['Workers']} | {row['Requested trials']} | {row['Steps']} | {row['L']} | {row['hp_reward']} | {row['lambda_onehot']} | {row['Best fold bbox']} | {row['Notes']} |"
        )
    (table_dir / "table20_istrail_100mer_L13_target_refine.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

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
    write_csv(fig_dir / "fig12_istrail_100mer_L13_target_refine_fold_sources.csv", fig_rows)
    fig_lines = [
        "# Figure 12. Istrail 100mer L=13 Target-Refinement Fold Sources",
        "",
        "| Benchmark | Target | Best | Target hit | Runtime s | BBox | PNG | Best JSON |",
        "|---|---:|---:|:---:|---:|---|---|---|",
    ]
    for row in fig_rows:
        fig_lines.append(
            f"| {row['Benchmark']} | {row['Target contacts']} | {row['RB-full best contacts']} | {row['Target hit']} | {float(row['Runtime s']):.3f} | {row['Best fold bbox']} | `{row['Best fold PNG']}` | `{row['Best JSON']}` |"
        )
    (fig_dir / "fig12_istrail_100mer_L13_target_refine_fold_sources.md").write_text("\n".join(fig_lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", nargs="*", default=list(BENCHMARKS))
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    repo = repo_root_from_script()
    helper = import_warm_helpers(repo)
    run_rows: List[Dict[str, Any]] = []
    final_rows: List[Dict[str, Any]] = []

    for bench in args.benchmarks:
        cfg = BENCHMARKS[bench]
        target = int(cfg["target"])
        root = repo / "runs" / bench / "local" / "Final_run" / "fix7_istrail_100mer_L13_target_refine"
        root.mkdir(parents=True, exist_ok=True)
        print(f"[L13] exporting/checking QUBO for {bench}", flush=True)
        qpath = export_qubo(repo, bench, cfg, bool(args.force_export))
        qmeta = read_qubo_header(qpath)
        qmeta["seq"] = cfg["seq"]
        source_warm = repo / str(cfg["source_warm"])
        remapped = remap_warm_start(
            source_warm,
            root / "warm_starts" / f"{bench}_L12_best_remapped_to_L13.json",
            int(cfg["old_L"]),
            int(cfg["L"]),
            str(cfg["seq"]),
        )

        rows = load_csv(root / "fix7_istrail_100mer_L13_target_refine_summary.csv") if args.resume else []
        bench_rows: List[Dict[str, Any]] = [dict(r) for r in rows]
        previous_best: Path = remapped
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
            warm_contacts = int(helper.contacts_of(warm) or 0)
            outdir = root / f"iter{idx:02d}_{run_cfg['label']}"
            cmd = build_command(repo, qpath, outdir, run_cfg, warm, target, warm_contacts)
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
                "note": "L=13 refinement from RB-full L=12 warm start remapped by coordinate",
            })
            print(f"[L13-run] {bench} iter={idx} label={run_cfg['label']} warm_contacts={warm_contacts}", flush=True)
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
                "t_final": 0.00045,
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
                "notes": "L=13; hp_reward=-4.0; RB-full L=12 warm start remapped to L=13",
                **result,
            }
            bench_rows.append(row)
            write_csv(outdir / "summary.csv", [row])
            write_csv(root / "fix7_istrail_100mer_L13_target_refine_summary.csv", bench_rows)
            print(f"[L13-done] {bench} iter={idx} contacts={result['contacts']} target_hit={result['target_hit']} elapsed={elapsed:.1f}s status={row['status']}", flush=True)
            previous_best = result_file if result_file.exists() else previous_best
            if result["target_hit"] == "yes" or proc.returncode != 0:
                target_hit = result["target_hit"] == "yes"
                break

        selected = choose_final(bench_rows)
        attempted_runtime = sum(float(r.get("wrapper_seconds", 0) or 0) for r in bench_rows)
        plot = helper.plot_best_fold(Path(str(selected["result_file"])), qmeta, Path(str(selected["outdir"])) / f"{bench}_L13_target_refine_best_fold")
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
            "Notes": "Target reached by L=13 RB-full refinement" if selected.get("target_hit") == "yes" else "Target not reached by L=13 RB-full refinement",
        })
        run_rows.extend(bench_rows)
        write_outputs(repo, final_rows, run_rows)

    write_outputs(repo, final_rows, run_rows)
    print("[L13-report] runs/istrail_100mer_L13_target_refine_report.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

