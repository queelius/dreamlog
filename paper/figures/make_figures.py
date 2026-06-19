#!/usr/bin/env python3
"""
Generate three paper figures for DreamLog paper.

Figure 1: compression_vs_recall.pdf
  Scatter of clause-compression amount vs held-out recovery (mean per domain)
  across EX38's 22 domains. Linear regression line + Pearson/Spearman r annotation.

Figure 2: extraction_crossover.pdf
  Heatmap of EX30's delta_bits(K, L) over K in 2..12 x L in 2..8.
  Diverging colormap centered at 0; crossover (delta_bits=0) contour highlighted.
  Negative = bits accepts abstraction; positive = bits rejects.

Figure 3: opi_coverage_recovery.pdf
  From EX37: coverage vs recovery contrasting closed-world (only fires at p=1.0)
  vs open-world (fires at coverage >= tau=0.5). Step/line plot; mark tau=0.5.
"""

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

try:
    from scipy.stats import pearsonr, spearmanr
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
FIG_DIR = REPO_ROOT / "paper" / "figures"
DATA_DIR = REPO_ROOT / "experiments" / "data"

# Font sizes
LABEL_FONTSIZE = 10
TICK_FONTSIZE = 9
ANNOT_FONTSIZE = 9


# ============================================================================
# Figure 1: compression_vs_recall scatter (EX38)
# ============================================================================

def load_ex38():
    latest = json.loads((DATA_DIR / "ex38" / "latest.json").read_text())
    run_dir = REPO_ROOT / latest["run_dir"]
    return json.loads((run_dir / "results.json").read_text())


def figure_compression_vs_recall():
    data = load_ex38()
    scatter = data["scatter_points"]
    agg = data["aggregate"]

    # x = compression amount (1 - clause_compression_ratio); y = mean_recovery
    x = np.array([pt["compression_amount"] for pt in scatter])
    y = np.array([pt["mean_recovery"] for pt in scatter])
    y_err = np.array([pt["std_recovery"] for pt in scatter])
    families = [pt["subfamily"] for pt in scatter]

    # Pearson and Spearman
    cc = agg["clause_compression_ratio_vs_recovery"]
    pearson_r = cc["pearson_r"]
    pearson_p = cc["pearson_p"]
    spearman_r = cc["spearman_r"]
    spearman_p = cc["spearman_p"]

    # Linear regression
    coeffs = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min() - 0.02, x.max() + 0.02, 100)
    y_line = np.polyval(coeffs, x_line)

    # Colorblind-friendly subfamily colors
    colors = {"A": "#0072B2", "B": "#E69F00", "C": "#009E73"}
    markers = {"A": "o", "B": "s", "C": "^"}

    fig, ax = plt.subplots(figsize=(4.5, 3.6))

    # Error bars per subfamily
    for sf in ("A", "B", "C"):
        idx = [i for i, f in enumerate(families) if f == sf]
        if not idx:
            continue
        ax.errorbar(
            x[idx], y[idx], yerr=y_err[idx],
            fmt=markers[sf], color=colors[sf], label="Subfamily %s" % sf,
            markersize=6, capsize=3, linewidth=1, elinewidth=0.8,
            zorder=3,
        )

    # Regression line
    ax.plot(x_line, y_line, color="#555555", linewidth=1.2,
            linestyle="--", zorder=2, label="Regression")

    # Annotation
    annot = (
        "Pearson r = %.3f (p = %.3f)\n"
        "Spearman r = %.3f (p = %.3f)"
        % (pearson_r, pearson_p, spearman_r, spearman_p)
    )
    ax.text(0.97, 0.05, annot,
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=ANNOT_FONTSIZE,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85))

    ax.set_xlabel("Clause compression amount (1 - after/before)",
                  fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Held-out recovery (mean, 5 seeds)",
                  fontsize=LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.legend(fontsize=ANNOT_FONTSIZE, framealpha=0.85)
    ax.set_xlim(left=max(0, x.min() - 0.05))
    ax.set_ylim([-0.05, 1.1])
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: "%.0f%%" % (v * 100)))
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    out = FIG_DIR / "compression_vs_recall.pdf"
    fig.savefig(str(out), format="pdf", bbox_inches="tight")
    plt.close(fig)
    print("Written: %s  (%d bytes)" % (out, out.stat().st_size))
    return out


# ============================================================================
# Figure 2: extraction crossover heatmap (EX30)
# ============================================================================

def load_ex30():
    latest = json.loads((DATA_DIR / "ex30" / "latest.json").read_text())
    run_dir = REPO_ROOT / latest["run_dir"]
    return json.loads((run_dir / "results.json").read_text())


def figure_extraction_crossover():
    data = load_ex30()
    surface = data["extraction_surface"]

    K_values = list(range(2, 13))
    L_values = [row["body_len"] for row in surface]

    # Build grid: rows = L (body_len), cols = K
    grid = np.zeros((len(L_values), len(K_values)))
    for i, row in enumerate(surface):
        for j, k in enumerate(K_values):
            grid[i, j] = row["delta_bits_by_k"][str(k)]

    # Clamp for display; actual range can be huge
    vmax = max(abs(grid.min()), abs(grid.max()))
    vmax = min(vmax, 80.0)   # cap at +-80 bits for readability

    fig, ax = plt.subplots(figsize=(5.5, 3.4))

    cmap = plt.get_cmap("RdBu")
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto",
                   origin="lower", interpolation="nearest")

    # Crossover contour: delta_bits = 0
    X, Y = np.meshgrid(np.arange(len(K_values)), np.arange(len(L_values)))
    cs = ax.contour(X, Y, grid, levels=[0.0], colors=["black"],
                    linewidths=[1.5])
    ax.clabel(cs, fmt={0.0: "delta=0"}, fontsize=ANNOT_FONTSIZE - 1,
              inline=True, inline_spacing=4)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("delta_bits (negative: accepts)", fontsize=LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

    ax.set_xticks(range(len(K_values)))
    ax.set_xticklabels([str(k) for k in K_values], fontsize=TICK_FONTSIZE)
    ax.set_yticks(range(len(L_values)))
    ax.set_yticklabels([str(l) for l in L_values], fontsize=TICK_FONTSIZE)
    ax.set_xlabel("K (rules sharing the body)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("L (shared body length)", fontsize=LABEL_FONTSIZE)

    # Mark crossover K* per row
    for i, row in enumerate(surface):
        k_star = row.get("crossover_k")
        if k_star is not None and k_star in K_values:
            j = K_values.index(k_star)
            ax.plot(j, i, "k*", markersize=7, zorder=5)

    # Add legend note for crossover marker
    ax.plot([], [], "k*", markersize=7, label="K* (crossover)")
    ax.legend(fontsize=ANNOT_FONTSIZE, loc="upper right", framealpha=0.85)

    plt.tight_layout()
    out = FIG_DIR / "extraction_crossover.pdf"
    fig.savefig(str(out), format="pdf", bbox_inches="tight")
    plt.close(fig)
    print("Written: %s  (%d bytes)" % (out, out.stat().st_size))
    return out


# ============================================================================
# Figure 3: Op I coverage vs recovery (EX37)
# ============================================================================

def load_ex37():
    latest = json.loads((DATA_DIR / "ex37" / "latest.json").read_text())
    run_dir = REPO_ROOT / latest["run_dir"]
    return json.loads((run_dir / "results.json").read_text())


def figure_opi_coverage_recovery():
    data = load_ex37()
    sweep = data["completeness_sweep"]
    TAU = 0.5

    chain_lengths = sorted(set(r["chain_len"] for r in sweep))
    # Colorblind-friendly palette for chain lengths
    cl_colors = {4: "#0072B2", 6: "#E69F00", 8: "#009E73"}
    cl_markers = {4: "o", 6: "s", 8: "^"}

    fig, ax = plt.subplots(figsize=(5.0, 3.6))

    for chain_len in chain_lengths:
        # Closed-world rows
        cw_rows = sorted(
            [r for r in sweep
             if r["chain_len"] == chain_len and not r["open_world"]],
            key=lambda r: r["coverage"]
        )
        # Open-world rows
        ow_rows = sorted(
            [r for r in sweep
             if r["chain_len"] == chain_len and r["open_world"]],
            key=lambda r: r["coverage"]
        )

        cw_x = [r["coverage"] for r in cw_rows]
        cw_y = [r["recovery"] for r in cw_rows]
        ow_x = [r["coverage"] for r in ow_rows]
        ow_y = [r["recovery"] for r in ow_rows]

        color = cl_colors.get(chain_len, "#333333")
        marker = cl_markers.get(chain_len, "o")

        # Closed-world: solid line
        ax.plot(cw_x, cw_y, color=color, linestyle="-",
                marker=marker, markersize=5, linewidth=1.4,
                label="L=%d (closed)" % chain_len)
        # Open-world: dashed line
        ax.plot(ow_x, ow_y, color=color, linestyle="--",
                marker=marker, markersize=5, linewidth=1.4, alpha=0.85,
                label="L=%d (open)" % chain_len)

    # Mark tau=0.5
    ax.axvline(x=TAU, color="#CC3333", linestyle=":", linewidth=1.4,
               label="tau=%.1f (open-world threshold)" % TAU)
    ax.text(TAU + 0.01, 0.08, "tau=%.1f" % TAU,
            color="#CC3333", fontsize=ANNOT_FONTSIZE, va="bottom")

    # Annotation: closed-world cliff
    ax.annotate(
        "closed-world cliff\n(fires only at p=1.0)",
        xy=(1.0, 0.95),
        xytext=(0.72, 0.65),
        fontsize=ANNOT_FONTSIZE - 1,
        arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8),
        color="#333333",
    )

    ax.set_xlabel("Closure coverage (fraction of R(B) present)",
                  fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Held-out recovery", fontsize=LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.set_xlim([0.35, 1.05])
    ax.set_ylim([-0.05, 1.15])
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: "%.0f%%" % (v * 100)))

    # Legend: 2-column to keep compact
    ax.legend(fontsize=ANNOT_FONTSIZE - 1, ncol=2, framealpha=0.85,
              loc="upper left")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    out = FIG_DIR / "opi_coverage_recovery.pdf"
    fig.savefig(str(out), format="pdf", bbox_inches="tight")
    plt.close(fig)
    print("Written: %s  (%d bytes)" % (out, out.stat().st_size))
    return out


# ============================================================================
# Main
# ============================================================================

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating paper figures -> %s" % FIG_DIR)
    out1 = figure_compression_vs_recall()
    out2 = figure_extraction_crossover()
    out3 = figure_opi_coverage_recovery()
    print()
    print("All three figures written:")
    for p in [out1, out2, out3]:
        print("  %s  (%d bytes)" % (p, p.stat().st_size))


if __name__ == "__main__":
    main()
