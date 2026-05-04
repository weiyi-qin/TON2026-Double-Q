from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_FIGURE = SCRIPT_DIR / "job_scheduling.eps"

RESULT_SPECS = [
    ("Double-DQ", "double_dq_results.npz", "red", "-", 2.0),
    ("Double-Q", "double_queue_results.npz", "orange", "-.", 2.0),
    ("[24]", "baseline_24_results.npz", "black", ":", 2.0),
    ("[29]", "baseline_29_results.npz", "blue", "--", 2.0),
]


def add_origin(data):
    return np.concatenate(([0], np.asarray(data)))


def set_first_cumulative_value(data, first_value):
    data = np.asarray(data, dtype=float)
    if len(data) == 0:
        return data
    return data - data[0] + first_value


def load_results():
    missing_files = [
        file_name
        for _, file_name, _, _, _ in RESULT_SPECS
        if not (SCRIPT_DIR / file_name).exists()
    ]
    if missing_files:
        missing_text = ", ".join(missing_files)
        raise FileNotFoundError(
            f"Missing result files: {missing_text}. Run the four algorithm scripts first."
        )

    loaded_items = []
    total_steps = None

    for label, file_name, color, linestyle, linewidth in RESULT_SPECS:
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

        loaded_items.append((label, loss, violation, color, linestyle, linewidth))

    return loaded_items, total_steps


def style_axis(ax, plot_steps, x_ticks, font_family, tick_fontsize, legend_fontsize, offset_fontsize):
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
    ax.set_xlim(0, plot_steps)
    ax.set_xticks(x_ticks)
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(offset_fontsize)
    ax.yaxis.offsetText.set_fontname(font_family)


def draw(show=True):
    print("Start plotting...")
    available_results, plot_steps = load_results()

    font_family = "Times New Roman"
    label_fontsize = 30
    tick_fontsize = 25
    legend_fontsize = 30
    offset_fontsize = 30
    x_ticks = np.array([0, 5, 10, 15, 20, 25]) * 1e2

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for label, loss_data, _, color, linestyle, linewidth in available_results:
        loss_series = add_origin(loss_data)
        axes[0].plot(
            range(len(loss_series)),
            loss_series,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )
    axes[0].set_xlabel("Time", fontsize=label_fontsize, fontname=font_family)
    axes[0].set_ylabel("Accumulated Cost", fontsize=label_fontsize, fontname=font_family)
    axes[0].set_yticks(np.array([0, 10, 20, 30, 40, 50]) * 1e3)
    style_axis(
        axes[0],
        plot_steps,
        x_ticks,
        font_family,
        tick_fontsize,
        legend_fontsize,
        offset_fontsize,
    )

    for label, _, violation_data, color, linestyle, linewidth in available_results:
        violation_series = set_first_cumulative_value(violation_data, 100.0)
        axes[1].plot(
            range(len(violation_series)),
            violation_series,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )
    axes[1].set_xlabel("Time", fontsize=label_fontsize, fontname=font_family)
    axes[1].set_ylabel("Delayed Jobs", fontsize=label_fontsize, fontname=font_family)
    axes[1].set_yticks(np.array([0, 5, 10, 15, 20, 25]) * 1e2)
    style_axis(
        axes[1],
        plot_steps,
        x_ticks,
        font_family,
        tick_fontsize,
        legend_fontsize,
        offset_fontsize,
    )

    fig.tight_layout(w_pad=4.0)
    fig.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Figure saved to {OUTPUT_FIGURE}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Draw job scheduling experiment results.")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="save the figure without opening an interactive matplotlib window",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    draw(show=not args.no_show)
