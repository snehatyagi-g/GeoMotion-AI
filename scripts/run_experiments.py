"""Run Phase 3 algorithms in Python; outputs are isolated from JS reference files."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adaptive_radius import AdaptiveRadiusMatcher, DEFAULT_ADAPTIVE_PARAMETERS
from src.csv_utils import read_csv, write_csv
from src.evaluation import build_metric_tables, evaluation_rows, load_scenario_networks, load_split_tracks, trajectory_accuracy_score
from src.hmm_map_matching import HMMViterbiMatcher, DEFAULT_HMM_PARAMETERS
from src.nearest_road import NearestRoadMatcher

DATA = ROOT / "data" / "processed" / "synthetic_v2"
OUT_METRICS = ROOT / "results" / "python_metrics"
OUT_FIGURES = ROOT / "results" / "python_figures"
INTERVALS = [1, 5, 15, 30, 45, 60, 120]
ALGORITHMS = ["Nearest Road", "HMM/Viterbi", "Adaptive Radius"]
CSV_COLUMNS = ["algorithm", "scope", "context", "noise_tier", "sampling_interval_s", "observations", "trajectories",
               "edge_accuracy", "edge_precision_micro", "edge_recall_micro", "edge_f1_micro", "edge_precision_macro",
               "edge_recall_macro", "edge_f1_macro", "mean_matching_distance_m", "trajectory_accuracy_mean",
               "trajectory_exact_match_rate", "candidate_count_mean", "processing_ms_per_observation"]


def run_matcher(track, algorithm, parameters=None):
    if algorithm == "Nearest Road":
        return NearestRoadMatcher().match(track["observations"], track["network"])
    if algorithm == "HMM/Viterbi":
        return HMMViterbiMatcher(parameters).match(track["observations"], track["network"])
    return AdaptiveRadiusMatcher(parameters).match(track["observations"], track["network"], track["features"])


def tune_parameters(tracks, algorithm, grid):
    trials, best = [], None
    for parameters in grid:
        rows = []
        for track in tracks:
            result = run_matcher(track, algorithm, parameters)
            rows.extend(evaluation_rows(track, result, algorithm, result["elapsedMs"]))
        score = trajectory_accuracy_score(rows)
        candidate_mean = sum(float(r["candidate_count"]) for r in rows) / len(rows) if rows else 0.0
        trial = {"parameters": parameters, "validation_mean_trajectory_accuracy": score,
                 "validation_candidate_count_mean": candidate_mean}
        trials.append(trial)
        if best is None or score > best["validation_mean_trajectory_accuracy"] + 1e-12 or (
                abs(score - best["validation_mean_trajectory_accuracy"]) <= 1e-12 and
                candidate_mean < best["validation_candidate_count_mean"]):
            best = trial
    return {"selected": best, "trials": trials}


def ensure_same_keys(rows):
    by_algorithm = {name: set() for name in ALGORITHMS}
    for row in rows:
        by_algorithm[row["algorithm"]].add("|".join(str(row[k]) for k in
            ("scenario_id", "trajectory_id", "sampling_interval_s", "seq", "timestamp")))
    expected = by_algorithm[ALGORITHMS[0]]
    for algorithm in ALGORITHMS[1:]:
        if by_algorithm[algorithm] != expected:
            raise ValueError(f"Algorithms did not process identical test observations: {algorithm}")
    return len(expected)


def render_figures(detailed):
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    overall = detailed["overall"]
    fig, ax = plt.subplots(figsize=(9, 5))
    names = [r["algorithm"] for r in overall]
    ax.bar(names, [r["metrics"]["edge_accuracy"] for r in overall], color=["#5470c6", "#91cc75", "#fac858"])
    ax.set_ylim(0, 1); ax.set_ylabel("Edge accuracy"); ax.set_title("Phase 3 algorithm comparison"); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(OUT_FIGURES / "algorithm_comparison.png", dpi=160); plt.close(fig)
    for key, field, title, filename in (("by_context", "context", "Road context", "context_wise_comparison.png"),
                                       ("by_sampling_interval", "sampling_interval_s", "Sampling interval (s)", "sampling_interval_comparison.png")):
        groups = sorted({r[field] for r in detailed[key]}, key=lambda x: float(x) if field == "sampling_interval_s" else str(x))
        fig, ax = plt.subplots(figsize=(11, 5))
        width = .8 / len(ALGORITHMS)
        for index, algorithm in enumerate(ALGORITHMS):
            lookup = {r[field]: r["metrics"]["edge_accuracy"] for r in detailed[key] if r["algorithm"] == algorithm}
            xs = [i + (index - (len(ALGORITHMS)-1)/2) * width for i in range(len(groups))]
            ax.bar(xs, [lookup[g] for g in groups], width=width, label=algorithm)
        ax.set_xticks(range(len(groups)), [str(g) for g in groups]); ax.set_ylim(0, 1)
        ax.set_ylabel("Edge accuracy"); ax.set_xlabel(title); ax.set_title(f"Edge accuracy by {title.lower()}")
        ax.legend(); ax.grid(axis="y", alpha=.25); fig.tight_layout()
        fig.savefig(OUT_FIGURES / filename, dpi=160); plt.close(fig)


def main():
    OUT_METRICS.mkdir(parents=True, exist_ok=True)
    manifest_rows = read_csv(DATA / "scenario_manifest.csv")
    validation_ids = {r["scenario_id"] for r in manifest_rows if r["split"] == "validation"}
    validation_networks = load_scenario_networks(DATA, validation_ids)
    validation = load_split_tracks(DATA, "validation", INTERVALS, validation_networks)
    hmm_grid = [{**DEFAULT_HMM_PARAMETERS, "candidateSigmaMultiplier": sigma, "transitionBetaMultiplier": beta}
                for sigma in (1.96, 2.5, 3, 4) for beta in (1, 2, 4)]
    adaptive_grid = [{**DEFAULT_ADAPTIVE_PARAMETERS, "sigmaMultiplier": sigma, "densityWeight": density,
                      "parallelAmbiguitySigma": ambiguity}
                     for sigma in (1.5, 1.96, 2.3) for density in (.15, .35, .55) for ambiguity in (1, 1.5, 2)]
    hmm_tuning = tune_parameters(validation["tracks"], "HMM/Viterbi", hmm_grid)
    adaptive_tuning = tune_parameters(validation["tracks"], "Adaptive Radius", adaptive_grid)
    selected = {"hmm": hmm_tuning["selected"]["parameters"], "adaptive": adaptive_tuning["selected"]["parameters"]}

    # Test rows/networks/labels are not loaded until after validation-only selection has finished.
    test_ids = {r["scenario_id"] for r in manifest_rows if r["split"] == "test"}
    test_networks = load_scenario_networks(DATA, test_ids)
    test = load_split_tracks(DATA, "test", INTERVALS, test_networks)
    test_rows, started = [], perf_counter()
    for track in test["tracks"]:
        for algorithm in ALGORITHMS:
            result = run_matcher(track, algorithm, selected.get("hmm") if algorithm == "HMM/Viterbi" else selected.get("adaptive"))
            for match in result["matches"]:
                if match["edge_id"] not in track["network"].edge_by_id:
                    raise ValueError(f"{algorithm} returned unknown edge {match['edge_id']} in {track['scenario']['scenario_id']}")
            test_rows.extend(evaluation_rows(track, result, algorithm, result["elapsedMs"]))
    wall_ms = (perf_counter() - started) * 1000
    eligible_count = ensure_same_keys(test_rows)
    per_algorithm = {name: sum(r["algorithm"] == name for r in test_rows) for name in ALGORITHMS}
    if set(per_algorithm.values()) != {eligible_count}:
        raise ValueError("Algorithm test observation counts differ")
    tables = build_metric_tables(test_rows, ALGORITHMS)
    write_csv(OUT_METRICS / "algorithm_comparison.csv", tables["records"], CSV_COLUMNS)
    detailed = {
        "project": "GeoMotion AI", "phase": "Phase 3 Python migration", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "data/processed/synthetic_v2", "benchmark_type": "synthetic; generator-defined labels",
        "evaluation_split": "test", "validation_split": "validation", "training_split": "train",
        "training_used_for_fitting": False, "test_tuning_or_selection_used": False,
        "test_observation_keys_identical_for_all_algorithms": True,
        "test_observations_labels_and_networks_loaded_after_validation_selection": True,
        "eligible_test_observations_per_algorithm": per_algorithm, "unique_eligible_test_keys": eligible_count,
        "radius_label_files_loaded_by_matchers": False, "radius_target_diagnostic_used_for_tuning": False,
        "test_elapsed_wall_ms_all_methods": wall_ms,
        "validation_tuning": {"score": "mean trajectory edge accuracy, equally weighted by scenario/interval trajectory",
                              "hmm": hmm_tuning, "adaptive_radius": adaptive_tuning},
        "metric_definitions": {"micro_precision_recall_f1": "Single-label multiclass micro scores; equal to edge accuracy.",
            "mean_matching_distance_m": "Metric distance from noisy observation to predicted road edge, not true-coordinate error.",
            "trajectory_accuracy_mean": "Mean per-track pointwise edge accuracy.",
            "processing_ms_per_observation": "Matcher wall time per observation; excludes input loading/network construction."},
        "test_results": tables["detailed"],
        "integrity": {"test_sampling_intervals_s": INTERVALS, "raw_and_synthetic_v2_modified": False,
                      "reference_results_modified": False, "radius_label_files_read": False}}
    (OUT_METRICS / "detailed_metrics.json").write_text(json.dumps(detailed, indent=2) + "\n", encoding="utf-8")
    render_figures(tables["detailed"])
    print(json.dumps({"test_observations_per_algorithm": per_algorithm, "selected_hmm_parameters": hmm_tuning["selected"],
        "selected_adaptive_parameters": adaptive_tuning["selected"], "overall_metrics": tables["detailed"]["overall"],
        "metrics_dir": str(OUT_METRICS), "figures_dir": str(OUT_FIGURES)}, indent=2))


if __name__ == "__main__":
    main()
