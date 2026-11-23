"""CSV-initialized Bayesian optimization on Pegasus LIGO DAGs.

Input: previous_BO_samples.csv with EXACTLY two columns: P1,P2.
Every row is evaluated first (even repeated rows); no N_INITIAL/random draws.
The original GP / LCB / patience / max-evals logic is retained.
For FAIR, P2 may be blank and is ignored. Old y-values are NEVER reused.

Model number, algorithm version and priority are all selected interactively.
LIGO graph and task CSVs are loaded from LIGO/GRAPHS and LIGO/Tasks.
"""

import os
import time
import csv
import sys
import itertools
import logging
import math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple, Dict, Any, List

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from processors import *
from utils import *
from statistics import *
from models import RooflineModel, AmdahlModel, CommunicationModel, GeneralModel

# -------------------------------
# Inputs
# -------------------------------

# Keep the interactive selection from the original BO implementation.
version = int(input("Enter algorithm version number [0-MAST, 1-MTSA, 2-MTPA, 3-Fair] : "))
model_num = int(input("Enter the Model Number [0-Roofline, 1-Amdahl, 2-Communication, 3-General]: "))
priority_num = int(input("Enter the Priority Number [0-FCFS, 1-Allocation, 2-Length, 3-Area]: "))

# Model number selects the actual speedup model, as in the original
# utils.compute_and_save() function. task.py expects a speedup_model
# argument; LIGO/Tasks CSVs hold only w,p,d,c, not model objects.
MODEL_INFO = {
    0: "Roofline",
    1: "Amdahl",
    2: "Communication",
    3: "General",
}
if model_num not in MODEL_INFO:
    raise ValueError("Invalid model number. Enter 0, 1, 2, or 3.")
model_name = MODEL_INFO[model_num]
if priority_num == 0:
    priority_name = "FCFS"
elif priority_num == 1:
    priority_name = "Allocation"
elif priority_num == 2:
    priority_name = "Length"
elif priority_num == 3:
    priority_name = "Area"
else:
    raise ValueError("Invalid priority number")

# -------------------------------------------------------
# Input paths. The selected model/algorithm/priority determine output folders.
# -------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
WORKLOAD_DIR = SCRIPT_DIR / "MONTAGE"
SEED_CSV = SCRIPT_DIR / "previous_BO_samples.csv"
COMBINATIONS_CSV = SCRIPT_DIR / "combinations.csv"

ALGORITHM_INFO = {
    0: ("MAST", "beta"),
    1: ("MTSA", "alpha"),
    2: ("MTPA", "gamma"),
    3: ("Fair", None),
}
if version not in ALGORITHM_INFO:
    raise ValueError("Invalid version")
algorithm_name, parameter = ALGORITHM_INFO[version]

# Output naming matches the original code; speedup model is selected at runtime.
# Original BO files use "comm" as the Communication summary suffix.
SUMMARY_SUFFIX = "comm" if model_num == 2 else model_name
OUTPUT_DIR = WORKLOAD_DIR / priority_name / f"BO_{algorithm_name}"
result_directory = OUTPUT_DIR / f"Results_{algorithm_name.lower()}" / model_name
COMBO_RESULTS_CSV = OUTPUT_DIR / f"combo_results_{SUMMARY_SUFFIX}.csv"
TERMINAL_LOG_TXT = OUTPUT_DIR / f"TERMINAL_LOG_{model_name}_TXT.txt"

# -------------------------------
# Parameter ranges: preserve the explicit algorithm/model-specific
# configuration from the original code so each range can be adjusted
# independently in future experiments.
# -------------------------------
if version == 0:
    if model_num == 0:
        P1_MIN, P1_MAX, P1_STEP = 0.1, 0.90, 0.02
        P2_MIN, P2_MAX, P2_STEP = 1.0, 5.0, 0.2
    elif model_num == 1:
        P1_MIN, P1_MAX, P1_STEP = 0.1, 0.90, 0.02
        P2_MIN, P2_MAX, P2_STEP = 1.0, 5.0, 0.2
    elif model_num == 2:
        P1_MIN, P1_MAX, P1_STEP = 0.1, 0.90, 0.02
        P2_MIN, P2_MAX, P2_STEP = 1.0, 5.0, 0.2
    elif model_num == 3:
        P1_MIN, P1_MAX, P1_STEP = 0.1, 0.90, 0.02
        P2_MIN, P2_MAX, P2_STEP = 1.0, 5.0, 0.2

elif version == 1:
    if model_num == 0:
        P1_MIN, P1_MAX, P1_STEP = 0.1, 0.90, 0.02
        P2_MIN, P2_MAX, P2_STEP = 1.0, 5.0, 0.2
    elif model_num == 1:
        P1_MIN, P1_MAX, P1_STEP = 0.1, 0.90, 0.02
        P2_MIN, P2_MAX, P2_STEP = 1.0, 5.0, 0.2
    elif model_num == 2:
        P1_MIN, P1_MAX, P1_STEP = 0.1, 0.90, 0.02
        P2_MIN, P2_MAX, P2_STEP = 1.0, 5.0, 0.2
    elif model_num == 3:
        P1_MIN, P1_MAX, P1_STEP = 0.1, 0.90, 0.02
        P2_MIN, P2_MAX, P2_STEP = 1.0, 5.0, 0.2

elif version == 2:
    if model_num == 0:
        P1_MIN, P1_MAX, P1_STEP = 0.1, 0.90, 0.02
        P2_MIN, P2_MAX, P2_STEP = 0.0, 1.0, 0.05
    elif model_num == 1:
        P1_MIN, P1_MAX, P1_STEP = 0.1, 0.90, 0.02
        P2_MIN, P2_MAX, P2_STEP = 0.0, 1.0, 0.05
    elif model_num == 2:
        P1_MIN, P1_MAX, P1_STEP = 0.1, 0.90, 0.02
        P2_MIN, P2_MAX, P2_STEP = 0.0, 1.0, 0.05
    elif model_num == 3:
        P1_MIN, P1_MAX, P1_STEP = 0.1, 0.90, 0.02
        P2_MIN, P2_MAX, P2_STEP = 0.0, 1.0, 0.05

elif version == 3:  # FAIR has only P1 (mu), no P2
    P1_MIN, P1_MAX, P1_STEP = 0.1, 0.90, 0.02

# -------------------------------
# Bayesian Optimization settings (unchanged from original)
# -------------------------------
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

MAX_EVALS = 100
KAPPA = 1.0
SAT_PATIENCE = 10
MIN_IMPROVEMENT = 0.0

# -------------------------------
# Utility functions
# -------------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def frange_inclusive(vmin: float, vmax: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("step must be > 0")
    n = int(round((vmax - vmin) / step))
    vals = [round(vmin + i * step, 10) for i in range(n + 1)]
    if abs(vals[-1] - vmax) > 1e-9:
        vals.append(round(vmax, 10))
    vals = [v for v in vals if v <= vmax + 1e-9]
    return vals

def write_csv_header_if_missing(path: str, fieldnames: List[str]) -> None:
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

def append_csv_row(path: str, fieldnames: List[str], row: Dict[str, Any]) -> None:
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

class TeeStdout:
    def __init__(self, logfile_path: str):
        self.logfile_path = logfile_path
        self.terminal = sys.stdout
        self.log = open(logfile_path, "w", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

def create_empty_csv(mu, par, directory, file_name):
    file_path = os.path.join(directory, file_name)
    with open(file_path, "w") as file:
        pass

def build_index(values: List[float]) -> Dict[float, int]:
    return {v: i for i, v in enumerate(values)}

def grid_position(value: float, grid_index: Dict[float, int]) -> int:
    # An old seed may be in range but not on the exact grid used by new BO.
    # For the LCB tie-break, use the nearest grid index in that case.
    if value in grid_index:
        return grid_index[value]
    return grid_index[min(grid_index, key=lambda v: abs(v - value))]


def manhattan_grid_distance(
    a: Tuple[float, float],
    b: Tuple[float, float],
    idx1: Dict[float, int],
    idx2: Dict[float, int]
) -> int:
    return (abs(grid_position(a[0], idx1) - grid_position(b[0], idx1))
            + abs(grid_position(a[1], idx2) - grid_position(b[1], idx2)))

def to_X_2d(pairs: List[Tuple[float, float]]) -> np.ndarray:
    return np.array([[p1, p2] for (p1, p2) in pairs], dtype=float)

def to_X_1d(points: List[float]) -> np.ndarray:
    return np.array([[p1] for p1 in points], dtype=float)

def fit_surrogate(X: np.ndarray, y: np.ndarray):
    if not SKLEARN_OK:
        raise RuntimeError("scikit-learn not available. Install: pip install scikit-learn")

    n_features = X.shape[1]
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=[0.1] * n_features, nu=2.5)
        + WhiteKernel(1e-6)
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=0
    )
    gp.fit(X, y)
    return gp

def propose_next_2d(
    unevaluated: List[Tuple[float, float]],
    gp,
    global_best_pair: Tuple[float, float],
    idx1: Dict[float, int],
    idx2: Dict[float, int],
    kappa: float
) -> Tuple[float, float]:
    Xcand = to_X_2d(unevaluated)
    mu, sigma = gp.predict(Xcand, return_std=True)
    lcb = mu - kappa * sigma

    best_idx = None
    best_lcb = float("inf")
    best_dist = float("inf")

    for i, pair in enumerate(unevaluated):
        score = float(lcb[i])
        dist = manhattan_grid_distance(pair, global_best_pair, idx1, idx2)

        if score < best_lcb - 1e-12:
            best_lcb = score
            best_dist = dist
            best_idx = i
        elif abs(score - best_lcb) <= 1e-12 and dist < best_dist:
            best_dist = dist
            best_idx = i

    return unevaluated[best_idx]

def propose_next_1d(
    unevaluated: List[float],
    gp,
    global_best_p1: float,
    idx1: Dict[float, int],
    kappa: float
) -> float:
    Xcand = to_X_1d(unevaluated)
    mu, sigma = gp.predict(Xcand, return_std=True)
    lcb = mu - kappa * sigma

    best_idx = None
    best_lcb = float("inf")
    best_dist = float("inf")

    for i, p1 in enumerate(unevaluated):
        score = float(lcb[i])
        dist = abs(grid_position(p1, idx1) - grid_position(global_best_p1, idx1))

        if score < best_lcb - 1e-12:
            best_lcb = score
            best_dist = dist
            best_idx = i
        elif abs(score - best_lcb) <= 1e-12 and dist < best_dist:
            best_dist = dist
            best_idx = i

    return unevaluated[best_idx]

# -------------------------------
# Initial samples - every row of the two-column CSV, no random points
# -------------------------------
def load_previous_samples(path: Path, fair: bool) -> list:
    """Read all P1,P2 rows in file order. No fixed count, no deduplication.

    Previous makespan ratios are NOT loaded; each row is evaluated again
    on the current LIGO workload. For FAIR, P2 may be empty and is ignored.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Initial sample CSV not found: {path}")

    samples = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or [h.strip() for h in reader.fieldnames] != ["P1", "P2"]:
            raise ValueError("Previous BO sample CSV must have exactly two columns: P1,P2")

        for line, row in enumerate(reader, start=2):
            try:
                p1 = float(row["P1"])
                p2 = None if fair else float(row["P2"])
            except (ValueError, TypeError, KeyError):
                raise ValueError(f"Invalid P1/P2 value at CSV line {line}: {row}") from None
            if not math.isfinite(p1) or p1 < P1_MIN - 1e-9 or p1 > P1_MAX + 1e-9:
                raise ValueError(f"P1 outside [{P1_MIN}, {P1_MAX}] at line {line}: {p1}")
            if not fair and (not math.isfinite(p2)
                             or p2 < P2_MIN - 1e-9 or p2 > P2_MAX + 1e-9):
                raise ValueError(f"P2 outside [{P2_MIN}, {P2_MAX}] at line {line}: {p2}")
            samples.append(round(p1, 10) if fair
                           else (round(p1, 10), round(p2, 10)))

    if not samples:
        raise ValueError(f"CSV has no parameter rows: {path}")
    return samples


def parameter_filename(value: float) -> str:
    # Keep original two-decimal filenames for normal grid samples.
    # Preserve more digits when previous BO used a finer search grid.
    if abs(value - round(value, 2)) < 1e-10:
        return f"{value:.2f}"
    return f"{value:.10f}".rstrip("0").rstrip(".")


def unique_evaluation_csv(directory: Path, filename: str) -> Path:
    """Retain a detail CSV for every input CSV row, even duplicate seeds."""
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    n = 2
    while (directory / f"{candidate.stem}_repeat_{n}{candidate.suffix}").exists():
        n += 1
    return directory / f"{candidate.stem}_repeat_{n}{candidate.suffix}"


def load_workload_configuration():
    """Validate LIGO graph/task inputs before consuming BO evaluations."""
    if not COMBINATIONS_CSV.is_file():
        raise FileNotFoundError(f"Missing workload configuration: {COMBINATIONS_CSV}")
    combinations = pd.read_csv(COMBINATIONS_CSV)
    required = {"P", "n", "Index"}
    if not required.issubset(combinations.columns):
        raise ValueError(f"{COMBINATIONS_CSV} must include columns: P,n,Index")
    if combinations.empty:
        raise ValueError(f"Empty combinations file: {COMBINATIONS_CSV}")

    # The user's Pegasus graph indices are 1-based; task CSV indexing
    # must match the same Index values in combinations.csv.
    for _, row in combinations.iterrows():
        n = int(row["n"])
        index = int(row["Index"])
        graph = WORKLOAD_DIR / "GRAPHS" / f"n={n}" / f"Montage_output_{index}.csv"
        task = WORKLOAD_DIR / "Tasks" / f"n={n}" / f"{index}.csv"
        if not graph.is_file():
            raise FileNotFoundError(f"MONTAGE graph missing: {graph}")
        if not task.is_file():
            raise FileNotFoundError(
                f"MONTAGE task parameters missing: {task}\n"
                "Supply the corresponding MONTAGE/Tasks file before running BO."
            )
    return combinations

# -------------------------------
# Speedup model selected by the user (same mapping as utils.compute_and_save)
# -------------------------------
MODEL_CLASSES = {
    0: RooflineModel,
    1: AmdahlModel,
    2: CommunicationModel,
    3: GeneralModel,
}

# -------------------------------
# Evaluation
# -------------------------------
def compute_and_save_montage(result_directory, priority_num, mu, alpha,
                          beta, gamma, version, writer):
    """Existing scheduler objective, but with MONTAGE instead of DAGGEN paths.

    Uses the same objective (worst makespan ratio), algorithm calls,
    task loader and graph parser as the user's original utils.py.
    """
    # Actual selected execution-time model, not only an output label.
    # task.py requires this object for p_max(), time(), area, and lower bounds.
    model = MODEL_CLASSES[model_num]()
    combinations_df = load_workload_configuration.cached
    max_makespan_ratio = float("-inf")

    for row_number, (_, row) in enumerate(combinations_df.iterrows(), start=1):
        P = int(row["P"])
        n = int(row["n"])
        i = int(row["Index"])
        dag_file = WORKLOAD_DIR / "GRAPHS" / f"n={n}" / f"Montage_output_{i}.csv"
        node_file = WORKLOAD_DIR / "Tasks" / f"n={n}" / f"{i}.csv"

        nodes = load_nodes_from_csv(str(node_file))
        if len(nodes) != n:
            raise ValueError(f"Expected {n} tasks in {node_file}, found {len(nodes)}")
        edges = extract_dependencies_from_csv(str(dag_file))
        task_graph = Graph(nodes, edges)
        adjacency = task_graph.get_adjacency()
        time_opt = task_graph.get_T_opt(P, adjacency, speedup_model=model)
        processors = Processors(P)
        time_algo_1 = processors.online_scheduling_algorithm(
            task_graph, 1, alpha=alpha, beta=beta, gamma=gamma,
            adjacency=adjacency, mu=mu, speedup_model=model, P=P,
            version=version, priority_num=priority_num
        )
        # Preserve the second scheduler call from the user's original code.
        min_time = processors.online_scheduling_algorithm(
            task_graph, 2, alpha=alpha, beta=beta, gamma=gamma,
            adjacency=adjacency, mu=mu, speedup_model=model, P=P,
            version=version, priority_num=priority_num
        )
        makespan_ratio = time_algo_1 / time_opt
        writer.writerow([str(P), str(n), str(time_algo_1),
                         str(time_opt), str(makespan_ratio)])
        if makespan_ratio > max_makespan_ratio:
            max_makespan_ratio = makespan_ratio
        print(f"Completed computation for {row_number} "
              f"(mu={mu}, {parameter}={beta if version == 0 else alpha if version == 1 else gamma}) "
              f"Priority:{priority_name} P:{P} n:{n}")
    return max_makespan_ratio


def evaluate_one_combination_return_value(p1: float,
                                          p2: Optional[float] = None) -> float:
    mu, par = p1, p2
    if version == 0:
        file_name = f"mu_{parameter_filename(mu)}_beta_{parameter_filename(par)}.csv"
        alpha, beta, gamma = None, par, None
    elif version == 1:
        file_name = f"mu_{parameter_filename(mu)}_alpha_{parameter_filename(par)}.csv"
        alpha, beta, gamma = par, None, None
    elif version == 2:
        file_name = f"mu_{parameter_filename(mu)}_gamma_{parameter_filename(par)}.csv"
        alpha, beta, gamma = None, None, par
    else:
        file_name = f"mu_{parameter_filename(mu)}.csv"
        alpha, beta, gamma = None, None, None

    ensure_dir(str(result_directory))
    file_path = unique_evaluation_csv(result_directory, file_name)
    with file_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["P", "n", "Algorithm Time", "Optimal Time", "Makespan Ratio"])
        with ThreadPoolExecutor() as executor:
            future = executor.submit(compute_and_save_montage,
                                     result_directory, priority_num, mu,
                                     alpha, beta, gamma, version, writer)
            combo_max_value = future.result()
    print(f"Worst makespan ratio = {combo_max_value:.6g}")
    return combo_max_value

# -------------------------------
# Main
# -------------------------------
def main():
    initial = load_previous_samples(SEED_CSV, fair=(version == 3))
    # Fail early if graphs or task files are missing; no misleading empty output.
    load_workload_configuration.cached = load_workload_configuration()

    ensure_dir(str(OUTPUT_DIR))
    ensure_dir(str(result_directory))
    # Archive an old summary rather than throwing FileExistsError or
    # appending to unrelated runs. Keep the required canonical output name.
    if COMBO_RESULTS_CSV.exists():
        k = 1
        while True:
            old_file = COMBO_RESULTS_CSV.with_name(
                f"{COMBO_RESULTS_CSV.stem}_previous_{k}{COMBO_RESULTS_CSV.suffix}")
            if not old_file.exists():
                break
            k += 1
        COMBO_RESULTS_CSV.rename(old_file)
        print(f"Archived previous BO summary as {old_file}")

    tee = TeeStdout(str(TERMINAL_LOG_TXT))
    old_stdout = sys.stdout
    sys.stdout = tee

    try:
        p1_values = frange_inclusive(P1_MIN, P1_MAX, P1_STEP)
        idx1 = build_index(p1_values)

        # =====================================================
        # CASE 1: FAIR ALGORITHM
        # Fair has only one parameter: mu
        # =====================================================
        if version == 3:
            all_points = list(p1_values)

            fields = [
                "eval_id",
                "P1",
                "value",
                "global_best_value",
                "global_best_P1"
            ]

            write_csv_header_if_missing(str(COMBO_RESULTS_CSV), fields)

            evaluated = set()
            X_obs = []
            y_obs = []

            global_best_val = float("inf")
            global_best_p1 = None

            no_improve = 0
            eval_id = 0

            print("=== START Bayesian search for FAIR: 1D search ===")
            print(f"Grid size: {len(all_points)}")
            print(f"CSV initial samples: {len(initial)}, Max evals: {MAX_EVALS}, kappa={KAPPA}\n")
            if len(initial) >= MAX_EVALS:
                print("All CSV samples will be evaluated; no additional BO "
                      "evaluations fit under MAX_EVALS.")

            for p1 in initial:
                val = evaluate_one_combination_return_value(p1)

                eval_id += 1
                evaluated.add(p1)
                X_obs.append(p1)
                y_obs.append(val)

                if val + MIN_IMPROVEMENT < global_best_val:
                    global_best_val = val
                    global_best_p1 = p1
                    no_improve = 0
                else:
                    no_improve += 1

                append_csv_row(str(COMBO_RESULTS_CSV), fields, {
                    "eval_id": eval_id,
                    "P1": p1,
                    "value": val,
                    "global_best_value": global_best_val,
                    "global_best_P1": global_best_p1
                })

                print(
                    f"[init {eval_id}] mu={p1} -> {val:.6g} "
                    f"| best={global_best_val:.6g} @ mu={global_best_p1}"
                )


            if global_best_p1 is None:
                print("No evaluations were run.")
                return

            no_improve = 0

            # Bayesian loop for Fair
            while eval_id < min(MAX_EVALS, len(all_points)):
                if no_improve >= SAT_PATIENCE:
                    print(f"\nStopping: no improvement for {SAT_PATIENCE} steps.")
                    break

                X = to_X_1d(X_obs)
                y = np.array(y_obs, dtype=float)

                gp = fit_surrogate(X, y)

                unevaluated = [p1 for p1 in all_points if p1 not in evaluated]

                if not unevaluated:
                    print("\nNo more unseen values left.")
                    break

                next_p1 = propose_next_1d(
                    unevaluated=unevaluated,
                    gp=gp,
                    global_best_p1=global_best_p1,
                    idx1=idx1,
                    kappa=KAPPA
                )

                val = evaluate_one_combination_return_value(next_p1)

                eval_id += 1
                evaluated.add(next_p1)
                X_obs.append(next_p1)
                y_obs.append(val)

                improved = False
                if val + MIN_IMPROVEMENT < global_best_val:
                    global_best_val = val
                    global_best_p1 = next_p1
                    no_improve = 0
                    improved = True
                else:
                    no_improve += 1

                append_csv_row(str(COMBO_RESULTS_CSV), fields, {
                    "eval_id": eval_id,
                    "P1": next_p1,
                    "value": val,
                    "global_best_value": global_best_val,
                    "global_best_P1": global_best_p1
                })

                print(f"\n[step {eval_id}] proposed mu={next_p1} -> {val:.6g}")
                print(
                    f"best={global_best_val:.6g} @ mu={global_best_p1} "
                    f"| improved={improved} | no_improve={no_improve}/{SAT_PATIENCE}"
                )

            print("\n=== FINAL FAIR RESULT ===")
            print(f"Evaluations run: {eval_id}")
            print(f"Best makespan ratio: {global_best_val}")
            print(f"Best mu: {global_best_p1}")
            print(f"Saved CSV: {COMBO_RESULTS_CSV}")
            print(f"Saved terminal log: {TERMINAL_LOG_TXT}")

        # =====================================================
        # CASE 2: MAST / MTSA / MTPA
        # These algorithms have two parameters: mu and beta/alpha/gamma
        # =====================================================
        else:
            p2_values = frange_inclusive(P2_MIN, P2_MAX, P2_STEP)
            idx2 = build_index(p2_values)

            all_pairs = list(itertools.product(p1_values, p2_values))

            fields = [
                "eval_id",
                "P1",
                "P2",
                "value",
                "global_best_value",
                "global_best_P1",
                "global_best_P2"
            ]

            write_csv_header_if_missing(str(COMBO_RESULTS_CSV), fields)

            evaluated = set()
            X_obs = []
            y_obs = []

            global_best_val = float("inf")
            global_best_pair = None

            no_improve = 0
            eval_id = 0

            print("=== START Bayesian search: 2D search ===")
            print(f"Grid size: {len(all_pairs)}")
            print(f"CSV initial samples: {len(initial)}, Max evals: {MAX_EVALS}, kappa={KAPPA}\n")
            if len(initial) >= MAX_EVALS:
                print("All CSV samples will be evaluated; no additional BO "
                      "evaluations fit under MAX_EVALS.")

            for pair in initial:
                p1, p2 = pair

                val = evaluate_one_combination_return_value(p1, p2)

                eval_id += 1
                evaluated.add(pair)
                X_obs.append(pair)
                y_obs.append(val)

                if val + MIN_IMPROVEMENT < global_best_val:
                    global_best_val = val
                    global_best_pair = pair
                    no_improve = 0
                else:
                    no_improve += 1

                append_csv_row(str(COMBO_RESULTS_CSV), fields, {
                    "eval_id": eval_id,
                    "P1": p1,
                    "P2": p2,
                    "value": val,
                    "global_best_value": global_best_val,
                    "global_best_P1": global_best_pair[0],
                    "global_best_P2": global_best_pair[1],
                })

                print(
                    f"[init {eval_id}] mu={p1}, {parameter}={p2} -> {val:.6g} "
                    f"| best={global_best_val:.6g} @ {global_best_pair}"
                )


            if global_best_pair is None:
                print("No evaluations were run.")
                return

            no_improve = 0

            # Bayesian loop for MAST / MTSA / MTPA
            while eval_id < min(MAX_EVALS, len(all_pairs)):
                if no_improve >= SAT_PATIENCE:
                    print(f"\nStopping: no improvement for {SAT_PATIENCE} steps.")
                    break

                X = to_X_2d(X_obs)
                y = np.array(y_obs, dtype=float)

                gp = fit_surrogate(X, y)

                unevaluated = [pair for pair in all_pairs if pair not in evaluated]

                if not unevaluated:
                    print("\nNo more unseen combinations left.")
                    break

                next_pair = propose_next_2d(
                    unevaluated=unevaluated,
                    gp=gp,
                    global_best_pair=global_best_pair,
                    idx1=idx1,
                    idx2=idx2,
                    kappa=KAPPA
                )

                p1, p2 = next_pair

                val = evaluate_one_combination_return_value(p1, p2)

                eval_id += 1
                evaluated.add(next_pair)
                X_obs.append(next_pair)
                y_obs.append(val)

                improved = False
                if val + MIN_IMPROVEMENT < global_best_val:
                    global_best_val = val
                    global_best_pair = next_pair
                    no_improve = 0
                    improved = True
                else:
                    no_improve += 1

                append_csv_row(str(COMBO_RESULTS_CSV), fields, {
                    "eval_id": eval_id,
                    "P1": p1,
                    "P2": p2,
                    "value": val,
                    "global_best_value": global_best_val,
                    "global_best_P1": global_best_pair[0],
                    "global_best_P2": global_best_pair[1],
                })

                print(f"\n[step {eval_id}] proposed mu={p1}, {parameter}={p2} -> {val:.6g}")
                print(
                    f"best={global_best_val:.6g} @ {global_best_pair} "
                    f"| improved={improved} | no_improve={no_improve}/{SAT_PATIENCE}"
                )

            print("\n=== FINAL RESULT ===")
            print(f"Evaluations run: {eval_id}")
            print(f"Best makespan ratio: {global_best_val}")
            print(f"Best mu: {global_best_pair[0]}")
            print(f"Best {parameter}: {global_best_pair[1]}")
            print(f"Saved CSV: {COMBO_RESULTS_CSV}")
            print(f"Saved terminal log: {TERMINAL_LOG_TXT}")

    finally:
        sys.stdout = old_stdout
        tee.close()

if __name__ == "__main__":
    main()