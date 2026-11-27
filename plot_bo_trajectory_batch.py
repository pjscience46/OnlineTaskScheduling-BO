"""Plot BO trajectories for MULTIPLE combo_results CSV files in one run.

Each CSV generates one PNG file (no PDF) under the matching BO_Trajectory
folder. Initial evaluations are black X markers; subsequent BO suggestions
are blue dots connected in evaluation order; first optimum is a red star.
No title, legend, evaluation numbers, or text annotations are drawn.

Run:
    python plot_bo_trajectory_batch.py

Optional: Override the configured CSV list with one or more --csv arguments:
    python plot_bo_trajectory_batch.py --csv "C:\\...\\combo_results_comm.csv" \\
        --csv "C:\\...\\combo_results_General.csv"

Requires: pip install pandas matplotlib
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch


# ================== EDIT THESE SETTINGS ==================
PROJECT_ROOT = Path(r"C:\Thesis\Paper2\Online_Rebuttal\OnlineTaskScheduling-BO")

# Add or remove paths as needed. All 12 are included by default:
# 3 algorithms (MAST, MTSA, MTPA) x 4 models (comm, Amdahl, General, Roofline).
CSV_PATHS = [
    PROJECT_ROOT / "Area" / "BO_MAST" / "combo_results_comm.csv",
    PROJECT_ROOT / "Area" / "BO_MAST" / "combo_results_Amdahl.csv",
    PROJECT_ROOT / "Area" / "BO_MAST" / "combo_results_General.csv",
    PROJECT_ROOT / "Area" / "BO_MAST" / "combo_results_Roofline.csv",

    PROJECT_ROOT / "Area" / "BO_MTSA" / "combo_results_comm.csv",
    PROJECT_ROOT / "Area" / "BO_MTSA" / "combo_results_Amdahl.csv",
    PROJECT_ROOT / "Area" / "BO_MTSA" / "combo_results_General.csv",
    PROJECT_ROOT / "Area" / "BO_MTSA" / "combo_results_Roofline.csv",

    PROJECT_ROOT / "Area" / "BO_MTPA" / "combo_results_comm.csv",
    PROJECT_ROOT / "Area" / "BO_MTPA" / "combo_results_Amdahl.csv",
    PROJECT_ROOT / "Area" / "BO_MTPA" / "combo_results_General.csv",
    PROJECT_ROOT / "Area" / "BO_MTPA" / "combo_results_Roofline.csv",
]

N_INITIAL = 20               # First N evaluations plotted as black X's
# If any file uses a different number of initial samples, specify its name/path:
# e.g. {"combo_results_General.csv": 14} applies to every folder with that name.
N_INITIAL_OVERRIDES = {}     # Can also use full CSV path as a string as key
P1_LABEL = r"$\mu$"         # Vertical axis
P2_LABELS = {"BO_MAST": r"$\beta$", "BO_MTSA": r"$\alpha$", "BO_MTPA": r"$\gamma$"}
DEFAULT_P2_LABEL = r"$\beta$"
INVERT_P1_AXIS = True
SHOW_DIRECTION_ARROWS = True
P1_LIMITS = None             # Optional: (0.1, 0.9); otherwise from data
P2_LIMITS = None             # Optional: (1.0, 5.0); otherwise from data
FIGURE_SIZE = (9.5, 6.8)    # Inches
EQUAL_BORDER_INCHES = 0.9   # Equal distance (inches) from each axes border to figure edge
DPI = 300
# =========================================================


def initial_count_for(csv_path: Path, default: int) -> int:
    return int(N_INITIAL_OVERRIDES.get(str(csv_path),
               N_INITIAL_OVERRIDES.get(csv_path.name, default)))


def add_direction_arrows(ax, xs: list[float], ys: list[float]) -> None:
    """Add arrows along successive BO-only segments, excluding initial points."""
    for x0, y0, x1, y1 in zip(xs[:-1], ys[:-1], xs[1:], ys[1:]):
        if x0 == x1 and y0 == y1:
            continue
        start = (x0 + 0.42 * (x1 - x0), y0 + 0.42 * (y1 - y0))
        end = (x0 + 0.62 * (x1 - x0), y0 + 0.62 * (y1 - y0))
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=10,
            linewidth=0.9, color="#2563a5", alpha=0.7, zorder=3,
        ))


def expanded_limits(values: pd.Series, fixed_limits):
    if fixed_limits is not None:
        return tuple(fixed_limits)
    low, high = float(values.min()), float(values.max())
    pad = max((high - low) * 0.045, abs(high) * 0.005, 0.01)
    return low - pad, high + pad


def default_output_path(csv_path: Path) -> Path:
    """Mirror subfolders: Area/BO_MAST/combo_results_comm.csv -> BO_Trajectory/Area/BO_MAST/comm/*.png."""
    csv_path = csv_path.resolve()
    root = next((p for p in csv_path.parents if p.name.casefold() == "onlinetaskscheduling-bo"), None)
    if root is None:
        raise ValueError(
            f"Cannot locate 'OnlineTaskScheduling-BO' in path: {csv_path}. "
            "Place the CSV under the project directory."
        )
    prefix = "combo_results_"
    if not csv_path.stem.casefold().startswith(prefix):
        raise ValueError(f"Expected a filename starting with '{prefix}': {csv_path.name}")
    model = csv_path.stem[len(prefix):]
    if not model:
        raise ValueError(f"CSV filename does not identify a model: {csv_path.name}")
    relative_subfolder = csv_path.parent.relative_to(root)
    target_folder = root / "BO_Trajectory" / relative_subfolder / model
    return target_folder / f"{csv_path.stem}_BO_trajectory.png"


def p2_label_for(csv_path: Path) -> str:
    return P2_LABELS.get(csv_path.parent.name.upper(), DEFAULT_P2_LABEL)


def equal_axes_borders(fig, border_in: float) -> None:
    """Use physically equal outer margins for all four axes edges; avoid tight cropping."""
    width_in, height_in = fig.get_size_inches()
    if not (0 < border_in < min(width_in, height_in) / 2):
        raise ValueError("EQUAL_BORDER_INCHES is too large for the figure size")
    fig.subplots_adjust(
        left=border_in / width_in,
        right=1 - border_in / width_in,
        bottom=border_in / height_in,
        top=1 - border_in / height_in,
    )


def plot_trajectory(csv_path: Path, n_initial: int) -> Path:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = ["eval_id", "P1", "P2", "value"]
    missing = set(required).difference(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
    if df.empty:
        raise ValueError("CSV contains no evaluations")
    for name in required:
        df[name] = pd.to_numeric(df[name], errors="raise")
    if df[required].isna().any().any():
        raise ValueError("CSV contains missing evaluation IDs or numerical values")

    df = df.sort_values("eval_id", kind="stable").reset_index(drop=True)
    if df["eval_id"].duplicated().any():
        raise ValueError("eval_id values must be unique")
    if not 0 <= n_initial <= len(df):
        raise ValueError(f"Initial count {n_initial} must be between 0 and {len(df)}")

    initial = df.iloc[:n_initial]
    suggested = df.iloc[n_initial:]
    best = df.loc[df["value"].idxmin()]  # First best when values tie

    plt.rcParams.update({
        "font.family": "serif", "font.size": 12,
        "axes.labelsize": 17, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=False)

    if not initial.empty:
        ax.scatter(initial["P2"], initial["P1"], marker="x", s=100,
                   linewidths=2.1, color="black", zorder=4)

    if not suggested.empty:
        xs, ys = suggested["P2"].tolist(), suggested["P1"].tolist()
        ax.plot(xs, ys, color="#2563a5", linewidth=1.35, alpha=0.75, zorder=2)
        ax.scatter(xs, ys, marker=".", s=115, color="#2563a5", zorder=5)
        if SHOW_DIRECTION_ARROWS:
            add_direction_arrows(ax, xs, ys)

    ax.scatter([best["P2"]], [best["P1"]], marker="*", s=350,
               color="red", edgecolor="white", linewidth=0.85, zorder=10)
    ax.set_xlabel(p2_label_for(csv_path))
    ax.set_ylabel(P1_LABEL)
    ax.set_xlim(*expanded_limits(df["P2"], P2_LIMITS))
    low, high = expanded_limits(df["P1"], P1_LIMITS)
    ax.set_ylim(high, low) if INVERT_P1_AXIS else ax.set_ylim(low, high)
    ax.grid(True, linestyle="--", color="0.84", alpha=0.65, linewidth=0.7)

    # Equal spacing around all four sides of the plot. Do not call tight_layout,
    # constrained_layout, or savefig(bbox_inches='tight').
    equal_axes_borders(fig, EQUAL_BORDER_INCHES)
    output = default_output_path(csv_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=DPI)   # PNG ONLY, normal bounding box
    plt.close(fig)
    print(f"[OK] {csv_path.name} -> {output} "
          f"({len(initial)} initial, {len(suggested)} BO, best={best['value']:.6f})")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", action="append", type=Path, dest="csvs",
                        help="CSV file to plot; may be repeated. Overrides the CSV_PATHS list.")
    parser.add_argument("--initial", type=int, default=N_INITIAL,
                        help="Default number of initial evaluations")
    args = parser.parse_args()

    files = args.csvs if args.csvs else CSV_PATHS
    if not files:
        parser.error("CSV_PATHS is empty. Add at least one file path.")

    done, missing, failed = 0, 0, 0
    for path in files:
        if not path.is_file():
            print(f"[SKIP — FILE NOT FOUND] {path}")
            missing += 1
            continue
        try:
            plot_trajectory(path, initial_count_for(path, args.initial))
            done += 1
        except (OSError, ValueError, KeyError) as exc:
            failed += 1
            print(f"[ERROR] {path}: {exc}")
    print(f"\nFinished: {done} PNG(s) saved, {missing} missing file(s), {failed} error(s).")
    if not done:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
