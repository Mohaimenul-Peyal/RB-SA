import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt


def load_chain_coords(qubo_path: Path, best_path: Path) -> Tuple[str, List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Return (sequence, coords per residue, contact pairs) from QUBO + best JSON."""
    qubo = json.loads(qubo_path.read_text())
    best = json.loads(best_path.read_text())

    # Support both formats:
    #  - best_worker_*.json: {"spins": [...], "pairs": [...]}
    #  - best_overall.json: {"best": {"spins": [...], ...}, ...}
    best_payload = best.get("best", best)

    seq = qubo["seq"]
    coords = qubo["coords"]
    N = qubo["N"]
    S = qubo["S"]
    spins = best_payload["spins"]

    if len(spins) != N * S:
        raise ValueError(f"Spin vector length {len(spins)} != N*S ({N}*{S})")

    chain_xy: List[Tuple[int, int]] = []
    for r in range(N):
        block = spins[r * S : (r + 1) * S]
        try:
            idx = block.index(1)
        except ValueError:
            raise ValueError(f"No active site for residue {r}") from None
        chain_xy.append(tuple(coords[idx]))

    # "pairs" (if present) may live either at top-level or inside "best".
    pairs = best_payload.get("pairs", best.get("pairs", []))
    return seq, chain_xy, pairs


def plot_chain(seq: str, chain_xy: List[Tuple[int, int]], out_prefix: Path):
    """Match the HP_ssa_manybody_ising.py plotting style."""
    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(figsize=(6, 6))

    xs, ys = zip(*chain_xy)
    ax.plot(xs, ys, color="gray", lw=3, zorder=1)

    for i, (x, y) in enumerate(chain_xy):
        color = "red" if seq[i] == "H" else "blue"
        ax.scatter(x, y, s=500, color=color, edgecolors="k", zorder=2)
        ax.text(x, y, f"{i}:{seq[i]}", ha="center", va="center", color="white", fontsize=9, zorder=3)

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    margin = 1.0
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    #ax.set_title("HHHPPHPHPHPPHPHPHPPH", fontsize=22)
    fig.tight_layout()

    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight", dpi=160)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main():
    ap = argparse.ArgumentParser(description="Plot ground-state fold from best_*.json and QUBO/Ising JSON.")
    ap.add_argument("--best", required=True, type=Path, help="Path to best_*.json")
    ap.add_argument("--qubo", required=True, type=Path, help="Path to matching qubo_manybody_ising*.json")
    ap.add_argument("--out", type=Path, help="Output prefix (without extension). Defaults to best path stem in same dir.")
    args = ap.parse_args()

    out_prefix = args.out if args.out else args.best.with_suffix("")

    seq, chain_xy, contacts = load_chain_coords(args.qubo, args.best)
    png_path, pdf_path = plot_chain(seq, chain_xy, out_prefix)
    print(f"[ok] saved {png_path} and {pdf_path} (contacts={len(contacts)})")


if __name__ == "__main__":
    main()

