from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, ScalarFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_FIGURE = SCRIPT_DIR / "time_varying.eps"

RESULT_SPECS = [
    ("Double-DQ", "double_dq_results.npz", "red", "-"),
    ("Double-Q", "double_queue_results.npz", "orange", "-."),
    ("[24]", "naive_surrogate_gd_results.npz", "black", ":"),
    ("[29]", "coco_results.npz", "blue", "--"),
]


def smooth_cumulative_curve(values, window=101):
    values = np.asarray(values, dtype=float)
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


def style_axis(ax, total_steps, font_family, tick_fontsize, legend_fontsize, offset_fontsize):
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(
        prop={"family": font_family, "size": legend_fontsize},
        loc="upper left",
        framealpha=0.7,
        handlelength=1.5,
        handletextpad=0.3,
        columnspacing=0.5,
        borderaxespad=0.2,
    )
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontname(font_family)
    ax.set_xlim(0, total_steps)
    ax.set_ylim(0, None)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, min_n_ticks=5))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(offset_fontsize)
    ax.yaxis.offsetText.set_fontname(font_family)


def draw(show=True):
    print("All algorithms completed. Plotting results...")
    plot_items, total_steps = load_results()

    time_axis = np.arange(1, total_steps + 1)
    loss_xticks = np.arange(0, total_steps + 1, 200)
    smoothed_violations = {
        label: smooth_cumulative_curve(violation_values)
        for label, _, violation_values, _, _ in plot_items
    }

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for label, loss_values, _, color, linestyle in plot_items:
        ax1.plot(
            time_axis,
            loss_values,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2,
        )
    ax1.set_xlabel("Time", fontsize=label_fontsize, fontname=font_family)
    ax1.set_ylabel("Accumulated Loss", fontsize=label_fontsize, fontname=font_family)
    ax1.set_xticks(loss_xticks)
    style_axis(ax1, total_steps, font_family, tick_fontsize, legend_fontsize, offset_fontsize)

    for label, _, _, color, linestyle in plot_items:
        ax2.plot(
            time_axis,
            smoothed_violations[label],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2,
        )
    ax2.set_xlabel("Time", fontsize=label_fontsize, fontname=font_family)
    ax2.set_ylabel("Hard Violation", fontsize=label_fontsize, fontname=font_family)
    style_axis(ax2, total_steps, font_family, tick_fontsize, legend_fontsize, offset_fontsize)
    ax2.set_ylim(0, 18000)
    ax2.set_yticks(np.arange(0, 18001, 3000))

    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURE, dpi=300, facecolor="white")
    print(f"Figure saved to {OUTPUT_FIGURE}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Draw time-varying constraint experiment results.")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="save the figure without opening an interactive matplotlib window",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    draw(show=not args.no_show)
