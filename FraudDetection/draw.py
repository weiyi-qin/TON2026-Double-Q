from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, StrMethodFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_FIGURE = SCRIPT_DIR / "fraud_detection.eps"

RESULT_SPECS = [
    ("Double-DQ", "double_dq_results.npz", "red", "-"),
    ("Double-Q", "double_queue_results.npz", "orange", "-."),
    ("[24]", "naive_surrogate_gd_results.npz", "black", ":"),
    ("[29]", "coco_results.npz", "blue", "--"),
]


def smooth_cumulative_curve(values, window=101):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    window = min(window, values.size if values.size % 2 == 1 else max(1, values.size - 1))
    if window <= 1:
        return values
    increments = np.diff(np.r_[0.0, values])
    kernel = np.ones(window) / window
    smooth_increments = np.convolve(increments, kernel, mode="same")
    smooth_increments = np.maximum(smooth_increments, 0.0)
    smoothed = np.cumsum(smooth_increments)
    if smoothed[-1] > 0:
        smoothed *= values[-1] / smoothed[-1]
    return smoothed


def load_results():
    missing_files = [
        file_name
        for _, file_name, _, _ in RESULT_SPECS
        if not (SCRIPT_DIR / file_name).exists()
    ]
    if missing_files:
        missing_text = ", ".join(missing_files)
        raise FileNotFoundError(
            f"Missing result files: {missing_text}. Run the four algorithm scripts first."
        )

    loaded_items = []
    total_steps = None

    for label, file_name, color, linestyle in RESULT_SPECS:
        result_path = SCRIPT_DIR / file_name
        with np.load(result_path) as data:
            loss = data["loss"]
            violation = data["violation"]
            current_total_steps = int(data["total_steps"].item())

        if total_steps is None:
            total_steps = current_total_steps
        elif current_total_steps != total_steps:
            raise ValueError(
                f"{file_name} has total_steps={current_total_steps}, "
                f"expected {total_steps}."
            )

        loaded_items.append((label, loss, violation, color, linestyle))

    return loaded_items, total_steps


def style_axis(axis, font_family, tick_fontsize, legend_fontsize, offset_fontsize):
    axis.grid(True, linestyle="--", alpha=0.55)
    axis.legend(
        prop={"family": font_family, "size": legend_fontsize},
        loc="upper left",
        framealpha=0.7,
        handlelength=1.5,
        handletextpad=0.3,
        columnspacing=0.5,
        borderaxespad=0.2,
    )
    axis.tick_params(axis="x", labelsize=tick_fontsize)
    axis.tick_params(axis="y", labelsize=tick_fontsize)
    for label in axis.get_xticklabels():
        label.set_fontname(font_family)
    for label in axis.get_yticklabels():
        label.set_fontname(font_family)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, min_n_ticks=5))
    axis.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    axis.yaxis.offsetText.set_fontsize(offset_fontsize)
    axis.yaxis.offsetText.set_fontname(font_family)


def draw(show=True):
    print("Start plotting...")
    plot_items, total_steps = load_results()

    font_family = "Times New Roman"
    label_fontsize = 30
    tick_fontsize = 25
    legend_fontsize = 30
    offset_fontsize = 30

    plt.rcParams.update(
        {
            "font.family": font_family,
            "axes.labelsize": label_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "legend.fontsize": legend_fontsize,
        }
    )

    fig, (ax_loss, ax_violation) = plt.subplots(1, 2, figsize=(15, 6))

    for label, loss_values, _, color, linestyle in plot_items:
        values = np.asarray(loss_values, dtype=float) / 5.0
        x_values = np.arange(1, len(values) + 1)
        ax_loss.plot(
            x_values,
            values,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2,
        )

    ax_loss.set_xlabel("Time", fontsize=label_fontsize, fontname=font_family)
    ax_loss.set_ylabel("Accumulated Loss", fontsize=label_fontsize, fontname=font_family)
    ax_loss.set_xlim(0, total_steps + 1)
    ax_loss.set_xticks(np.arange(0, 5001, 1000))
    ax_loss.set_ylim(0, None)
    style_axis(ax_loss, font_family, tick_fontsize, legend_fontsize, offset_fontsize)

    for label, _, violation_values, color, linestyle in plot_items:
        values = np.asarray(smooth_cumulative_curve(violation_values), dtype=float) * 3.0
        x_values = np.arange(1, len(values) + 1)
        ax_violation.plot(
            x_values,
            values,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2,
        )

    ax_violation.set_xlabel("Time", fontsize=label_fontsize, fontname=font_family)
    ax_violation.set_ylabel("Hard Violation", fontsize=label_fontsize, fontname=font_family)
    ax_violation.set_xlim(0, total_steps + 1)
    ax_violation.set_xticks(np.arange(0, 5001, 1000))
    ax_violation.set_ylim(0, None)
    style_axis(ax_violation, font_family, tick_fontsize, legend_fontsize, offset_fontsize)

    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.90, wspace=0.22)
    fig.savefig(OUTPUT_FIGURE, dpi=300, facecolor="white")
    print(f"Figure saved to {OUTPUT_FIGURE}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Draw fraud detection experiment results.")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="save the figure without opening an interactive matplotlib window",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    draw(show=not args.no_show)
