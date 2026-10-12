"""Render the deterministic RUGD architecture diagram."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE = "#DCEBFA"
TEAL = "#DDF4EE"
GOLD = "#FFF1CC"
PURPLE = "#ECE3F7"
RED = "#FBE3E5"
INK = "#17324D"


def box(ax, x, y, width, height, text, color, *, dashed=False, fontsize=9):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=1.3,
        edgecolor=INK,
        facecolor=color,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        color=INK,
        fontsize=fontsize,
        weight="semibold",
        linespacing=1.2,
    )
    return patch


def arrow(ax, start, end, *, dashed=False, color=INK):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.25,
            color=color,
            linestyle="--" if dashed else "-",
            connectionstyle="arc3,rad=0.0",
        )
    )


def render(output_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 6.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    ax.text(0.25, 5.82, "ShiftGuard-RUGD", fontsize=18, weight="bold", color=INK)
    ax.text(
        0.25,
        5.49,
        "Robust uncertainty-guided diffusion for zero-shot cross-city severe congestion",
        fontsize=10.5,
        color="#49657B",
    )

    box(ax, 0.25, 3.50, 1.45, 1.05, "Speed history\n+ observation mask", BLUE)
    box(ax, 0.25, 1.62, 1.45, 1.05, "Road topology\n+ calendar", BLUE)
    box(ax, 2.15, 3.50, 1.55, 1.05, "Shared masked\ntemporal GRU", TEAL)
    box(ax, 2.15, 1.62, 1.55, 1.05, "ID-free context\nencoder", TEAL)
    box(ax, 4.15, 2.57, 1.45, 1.05, "Local feature\nfusion", GOLD)
    box(ax, 6.12, 2.57, 1.70, 1.05, "Degree-normalized\ngraph diffusion", PURPLE)
    box(ax, 8.35, 2.57, 1.55, 1.05, "Residual spatial-\ntemporal fusion", PURPLE)
    box(ax, 10.43, 2.57, 1.55, 1.05, "Final Dirichlet\nclassifier", GOLD)
    box(ax, 12.48, 2.57, 1.25, 1.05, "Class scores\n+ vacuity", BLUE)

    box(ax, 4.08, 4.35, 1.60, 0.82, "Local Dirichlet head", GOLD, dashed=True)
    box(ax, 6.24, 4.35, 1.55, 0.82, "Auxiliary rare-event\nloss", RED, dashed=True, fontsize=8.5)
    box(ax, 8.35, 0.58, 1.55, 0.82, "Gradient reversal", RED, dashed=True)
    box(ax, 10.43, 0.58, 1.55, 0.82, "Source-city\ndiscriminator", RED, dashed=True)
    box(ax, 12.35, 4.30, 1.50, 0.92, "Source-only\ntemperature + APS", TEAL)

    arrow(ax, (1.70, 4.02), (2.15, 4.02))
    arrow(ax, (1.70, 2.14), (2.15, 2.14))
    arrow(ax, (3.70, 4.02), (4.15, 3.37))
    arrow(ax, (3.70, 2.14), (4.15, 2.83))
    arrow(ax, (5.60, 3.10), (6.12, 3.10))
    arrow(ax, (7.82, 3.10), (8.35, 3.10))
    arrow(ax, (9.90, 3.10), (10.43, 3.10))
    arrow(ax, (11.98, 3.10), (12.48, 3.10))
    arrow(ax, (4.88, 3.62), (4.88, 4.35), dashed=True)
    arrow(ax, (5.68, 4.76), (6.24, 4.76), dashed=True)
    arrow(ax, (9.12, 2.57), (9.12, 1.40), dashed=True)
    arrow(ax, (9.90, 0.99), (10.43, 0.99), dashed=True)
    arrow(ax, (13.10, 3.62), (13.10, 4.30))

    ax.text(6.97, 2.28, "repeat K steps", ha="center", fontsize=8.2, color="#49657B")
    ax.text(8.23, 5.00, "Dashed = source-training only", fontsize=8.8, color="#8A3440")
    ax.text(
        0.25,
        0.28,
        "All encoders are shared across cities and node counts; target-city labels are used only for final evaluation.",
        fontsize=9.3,
        color="#49657B",
    )

    fig.tight_layout(pad=0.35)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    render(Path("docs/figures/shiftguard_rugd_architecture.png"))
    print("Rendered final RUGD architecture PNG.")
