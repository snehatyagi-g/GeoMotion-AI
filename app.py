"""GeoMotion AI: interactive dashboard for a synthetic map-matching benchmark."""
from __future__ import annotations

import json
import math
import re
import sys
import threading
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "processed" / "synthetic_v2"
PY_METRICS_ROOT = ROOT / "results" / "python_metrics"
JS_METRICS_ROOT = ROOT / "results" / "metrics"
PY_FIGURES_ROOT = ROOT / "results" / "python_figures"
INTERVALS = [1, 5, 15, 30, 45, 60, 120]
ALGORITHMS = ["Nearest Road", "HMM/Viterbi", "Adaptive Radius"]
ALGORITHM_COLORS = {
    "Nearest Road": [43, 108, 176, 210],
    "HMM/Viterbi": [226, 139, 46, 220],
    "Adaptive Radius": [38, 148, 112, 220],
    "Observed GPS": [35, 94, 190, 220],
    "Ground truth": [164, 79, 156, 210],
}
_LIVE_PLOT_LOCK = threading.RLock()

sys.path.insert(0, str(ROOT))
from src.adaptive_radius import AdaptiveRadiusMatcher  # noqa: E402
from src.evaluation import calculate_metrics  # noqa: E402
from src.hmm_map_matching import HMMViterbiMatcher  # noqa: E402
from src.live_matching import (  # noqa: E402
    adaptive_diagnostics,
    point_on_edge,
    run_matcher,
    validate_observation_rows,
)
from src.nearest_road import NearestRoadMatcher  # noqa: E402
from src.road_network import RoadNetwork  # noqa: E402


st.set_page_config(
    page_title="GeoMotion AI",
    page_icon=":material/directions:",
    layout="wide",
    initial_sidebar_state="expanded",
)


class DashboardDataError(RuntimeError):
    """Raised when an expected dashboard input is missing or malformed."""


def _modified_ns(path_text: str) -> int:
    path = Path(path_text)
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return -1


@st.cache_data(ttl=300, max_entries=64, show_spinner=False)
def _cached_csv(path_text: str, _modified: int) -> pd.DataFrame:
    path = Path(path_text)
    if not path.is_file():
        raise DashboardDataError(f"Required CSV file was not found: {path}")
    try:
        frame = pd.read_csv(path, keep_default_na=False)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise DashboardDataError(f"Could not read {path.name}: {exc}") from exc
    if frame.empty:
        raise DashboardDataError(f"CSV file contains no data rows: {path}")
    return frame


def load_csv(path: Path) -> pd.DataFrame:
    return _cached_csv(str(path), _modified_ns(str(path)))


@st.cache_data(ttl=300, max_entries=16, show_spinner=False)
def _cached_json(path_text: str, _modified: int) -> dict:
    path = Path(path_text)
    if not path.is_file():
        raise DashboardDataError(f"Required JSON file was not found: {path}")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise DashboardDataError(f"Could not read {path.name}: {exc}") from exc
    if not isinstance(result, dict):
        raise DashboardDataError(f"Expected a JSON object in {path}")
    return result


def load_json(path: Path) -> dict:
    return _cached_json(str(path), _modified_ns(str(path)))


def _optional_csv(path: Path) -> pd.DataFrame:
    try:
        return load_csv(path)
    except DashboardDataError:
        return pd.DataFrame()


def _optional_json(path: Path) -> dict:
    try:
        return load_json(path)
    except DashboardDataError:
        return {}


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise DashboardDataError(f"{label} is missing required columns: {', '.join(sorted(missing))}")


def parse_linestring_wkt(value: str) -> list[list[float]]:
    """Return a WKT LINESTRING as PyDeck [longitude, latitude] positions."""
    match = re.match(r"^\s*LINESTRING\s*\((.+)\)\s*$", str(value), re.IGNORECASE)
    if not match:
        return []
    path = []
    try:
        for pair in match.group(1).split(","):
            lon, lat = (float(part) for part in pair.strip().split()[:2])
            if not (math.isfinite(lon) and math.isfinite(lat)):
                return []
            path.append([lon, lat])
    except (TypeError, ValueError):
        return []
    return path if len(path) >= 2 else []


def _network_paths(network_rows: pd.DataFrame, edge_ids: set[str] | None = None, color=None) -> list[dict]:
    rows = []
    for record in network_rows.to_dict("records"):
        edge_id = str(record.get("edge_id", ""))
        if edge_ids is not None and edge_id not in edge_ids:
            continue
        path = parse_linestring_wkt(record.get("geometry_wkt", ""))
        if not path:
            try:
                path = [[float(record["from_lon"]), float(record["from_lat"])],
                        [float(record["to_lon"]), float(record["to_lat"])]]
            except (KeyError, TypeError, ValueError):
                continue
        rows.append({
            "edge_id": edge_id,
            "road_type": str(record.get("highway", "road")),
            "layer": int(float(record.get("layer", 0) or 0)),
            "path": path,
            "color": color or [106, 119, 133, 125],
        })
    return rows


def _read_manifest() -> pd.DataFrame:
    frame = load_csv(DATA_ROOT / "scenario_manifest.csv")
    _require_columns(frame, {"scenario_id", "split", "scenario_type", "gps_noise_tier"}, "Scenario manifest")
    return frame


def _load_test_result_tables() -> tuple[pd.DataFrame, dict, pd.DataFrame, dict]:
    py_metrics = _optional_csv(PY_METRICS_ROOT / "algorithm_comparison.csv")
    py_detail = _optional_json(PY_METRICS_ROOT / "detailed_metrics.json")
    js_metrics = _optional_csv(JS_METRICS_ROOT / "algorithm_comparison.csv")
    js_detail = _optional_json(JS_METRICS_ROOT / "detailed_metrics.json")
    return py_metrics, py_detail, js_metrics, js_detail


def _metric_rows(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    if frame.empty or "scope" not in frame:
        return pd.DataFrame()
    return frame.loc[frame["scope"].eq(scope)].copy()


def _metric_value(detail: dict, algorithm: str, metric: str, default=None):
    try:
        row = next(row for row in detail["test_results"]["overall"] if row["algorithm"] == algorithm)
        return row["metrics"][metric]
    except (KeyError, StopIteration, TypeError):
        return default


def _algorithm_explanations() -> None:
    descriptions = [
        ("Nearest Road", "For each GPS point, choose the closest road segment. It is fast and simple, but treats each point independently."),
        ("HMM / Viterbi", "Consider several nearby road candidates for each point, then choose the sequence that best fits both GPS proximity and connected movement through the road network."),
        ("Adaptive Radius", "Adjust how far to search using reported GPS uncertainty and nearby road density/layer features. It then ranks candidates using distance, heading, and continuity."),
    ]
    with st.container(horizontal=True):
        for title, explanation in descriptions:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(explanation)


def _render_map(layers: list, center_lat: float, center_lon: float, zoom: float, alt: str, height: int = 520) -> None:
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom, pitch=28),
        map_style=None,
        tooltip={"text": "{edge_id}\nRoad: {road_type}\nLayer: {layer}"},
    )
    st.pydeck_chart(deck, height=height, width="stretch", alt=alt)


def _page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def render_overview() -> None:
    _page_header("GeoMotion AI", "Adaptive radius-based map matching for complex urban road networks")
    py_metrics, py_detail, _, _ = _load_test_result_tables()
    manifest = _optional_csv(DATA_ROOT / "scenario_manifest.csv")
    observations_count = py_detail.get("unique_eligible_test_keys")
    scenarios_count = len(manifest) if not manifest.empty else None
    test_scenarios = int(manifest["split"].eq("test").sum()) if not manifest.empty and "split" in manifest else None
    overall = _metric_rows(py_metrics, "overall")
    if not overall.empty:
        values = overall.set_index("algorithm")
        accuracy = values["edge_accuracy"].to_dict()
    else:
        accuracy = {}

    with st.container(horizontal=True):
        st.metric("Eligible test observations", f"{observations_count:,}" if isinstance(observations_count, int) else "Unavailable", border=True)
        st.metric("Scenarios", f"{scenarios_count:,}" if scenarios_count is not None else "Unavailable", border=True)
        st.metric("Test scenarios", str(test_scenarios) if test_scenarios is not None else "Unavailable", border=True)
        st.metric("Sampling intervals", "7", border=True)

    st.subheader("Test-set edge accuracy")
    if accuracy:
        display = pd.DataFrame([{"Algorithm": name, "Edge accuracy": float(accuracy[name])} for name in ALGORITHMS if name in accuracy])
        st.bar_chart(display, x="Algorithm", y="Edge accuracy", color="Algorithm", height=300,
                     alt="Test edge accuracy for the three map-matching algorithms")
        if "Adaptive Radius" in accuracy and "HMM/Viterbi" in accuracy:
            st.caption("Adaptive Radius improves on Nearest Road overall in this benchmark, but does not exceed HMM/Viterbi overall.")
    else:
        st.info("Python experiment metrics are unavailable. Run the Phase 3 Python experiment before exploring results.")

    st.subheader("The three matchers")
    _algorithm_explanations()
    st.subheader("Study workflow")
    st.markdown("Explore the synthetic scenarios and road graph, run the existing Python matchers on one selected trajectory, compare saved test metrics, and export result tables for your report or viva.")


def render_dataset_explorer() -> None:
    _page_header("Dataset explorer", "Inspect the benchmark tables and their records without editing source files.")
    st.info("Ground-truth and radius-label tables are available only for inspection/evaluation. They are never matcher inputs.", icon=":material/visibility:")
    left, right = st.columns(2)
    with left:
        table_kind = st.selectbox("Dataset table", [
            "Noisy GPS observations", "Inference-time features", "Ground-truth labels",
            "Radius-label diagnostics", "Road network", "Road-context features", "Scenario manifest",
        ], key="explorer_table")
        split = st.selectbox("Split", ["train", "validation", "test"], index=2, key="explorer_split")
    with right:
        interval = st.selectbox("Sampling interval", INTERVALS, index=0, format_func=lambda value: f"{value} s", key="explorer_interval")
        show_rows = st.selectbox("Preview rows", [100, 500, 1000, 5000], index=1, key="explorer_preview_rows")

    if table_kind == "Noisy GPS observations":
        path = DATA_ROOT / split / "observations" / f"traj_dt_{interval}s.csv"
    elif table_kind == "Inference-time features":
        path = DATA_ROOT / split / "features" / f"features_dt_{interval}s.csv"
    elif table_kind == "Ground-truth labels":
        path = DATA_ROOT / split / "ground_truth" / f"ground_truth_dt_{interval}s.csv"
    elif table_kind == "Radius-label diagnostics":
        path = DATA_ROOT / split / "radius_labels" / f"radius_labels_dt_{interval}s.csv"
    elif table_kind == "Road network":
        path = DATA_ROOT / "road_network.csv"
    elif table_kind == "Road-context features":
        path = DATA_ROOT / "road_context_features.csv"
    else:
        path = DATA_ROOT / "scenario_manifest.csv"

    try:
        frame = load_csv(path)
    except DashboardDataError as exc:
        st.error(str(exc))
        return
    if "split" in frame and table_kind not in ("Road network", "Road-context features", "Scenario manifest"):
        frame = frame.loc[frame["split"].eq(split)]
    if table_kind not in ("Road network", "Road-context features", "Scenario manifest") and "sampling_interval_s" in frame:
        frame = frame.loc[pd.to_numeric(frame["sampling_interval_s"], errors="coerce").eq(interval)]
    st.caption(f"Source: `{path.relative_to(ROOT)}` · {len(frame):,} records · {len(frame.columns)} columns")
    with st.container(horizontal=True):
        st.metric("Records", f"{len(frame):,}", border=True)
        st.metric("Columns", f"{len(frame.columns)}", border=True)
        missing = frame.isna() | frame.astype(str).eq("")
        st.metric("Missing cells", f"{int(missing.sum().sum()):,}", border=True)
        st.metric("Duplicate rows", f"{int(frame.duplicated().sum()):,}", border=True)
    if frame.empty:
        st.warning("No records match the current selection.")
        return
    scenarios = sorted(frame["scenario_id"].astype(str).unique()) if "scenario_id" in frame else []
    if scenarios:
        selected_scenario = st.selectbox("Scenario", ["All scenarios", *scenarios], key="explorer_scenario")
        if selected_scenario != "All scenarios":
            frame = frame.loc[frame["scenario_id"].astype(str).eq(selected_scenario)]
    st.dataframe(frame.head(show_rows), hide_index=True, height=430, width="stretch", alt=f"Preview of {table_kind.lower()}")
    csv_bytes = frame.to_csv(index=False).encode("utf-8")
    st.download_button("Download selected table as CSV", csv_bytes, file_name=f"{path.stem}_preview.csv", mime="text/csv", key="explorer_download")
    st.caption("For large tables, the screen preview is capped; the download contains all filtered records.")


def render_road_network() -> None:
    _page_header("Road network", "View the scenario-specific synthetic road graph and inspect its edge attributes.")
    try:
        manifest = _read_manifest()
        network_path = DATA_ROOT / "road_network.csv"
        network_frame = load_csv(network_path)
        _require_columns(network_frame, {"scenario_id", "edge_id", "from_lat", "from_lon", "to_lat", "to_lon", "geometry_wkt"}, "Road network")
    except DashboardDataError as exc:
        st.error(str(exc))
        return
    scenario_meta = manifest.copy()
    scenario_meta["scenario_label"] = scenario_meta.apply(
        lambda row: f"{row['scenario_id']} · {row.get('scenario_type', 'scenario')} · {row.get('split', '')}", axis=1)
    labels = scenario_meta["scenario_label"].tolist()
    chosen_label = st.selectbox("Synthetic scenario", labels, key="network_scenario")
    scenario = scenario_meta.loc[scenario_meta["scenario_label"].eq(chosen_label)].iloc[0]
    edges = network_frame.loc[network_frame["scenario_id"].astype(str).eq(str(scenario["scenario_id"]))].copy()
    if edges.empty:
        st.warning("No road edges were found for this scenario.")
        return
    road_types = sorted(edges["highway"].astype(str).unique()) if "highway" in edges else []
    selected_types = st.multiselect("Road classes", road_types, default=road_types, key="network_types")
    visible = edges.loc[edges["highway"].astype(str).isin(selected_types)] if "highway" in edges else edges
    with st.container(horizontal=True):
        st.metric("Scenario context", str(scenario.get("scenario_type", "Unknown")), border=True)
        st.metric("GPS noise", str(scenario.get("gps_noise_tier", "Unknown")), border=True)
        st.metric("Edges in graph", f"{len(edges):,}", border=True)
        st.metric("Edges displayed", f"{len(visible):,}", border=True)
    positions = pd.concat([visible["from_lat"], visible["to_lat"]]).astype(float), pd.concat([visible["from_lon"], visible["to_lon"]]).astype(float)
    center_lat = float(positions[0].mean())
    center_lon = float(positions[1].mean())
    paths = _network_paths(visible)
    if paths:
        _render_map([pdk.Layer("PathLayer", data=paths, get_path="path", get_color="color", width_min_pixels=2,
                               pickable=True, auto_highlight=True)], center_lat, center_lon, 12.5,
                    "Synthetic road edges in the selected scenario")
    else:
        st.warning("This network has no drawable LINESTRING geometry.")
    columns = [c for c in ["edge_id", "highway", "length_m", "speed_limit_kmh", "oneway", "bridge", "tunnel", "layer", "from_node", "to_node"] if c in visible]
    st.dataframe(visible[columns].head(500), hide_index=True, width="stretch", alt="Road edge attributes for the selected synthetic scenario")
    st.caption("Coordinates are supplied as longitude/latitude. Matching distances use the project's local metric approximation in metres.")


def _key_columns(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["scenario_id", "trajectory_id", "sampling_interval_s", "seq", "timestamp"]
    _require_columns(frame, set(keys), "Trajectory table")
    result = frame.copy()
    for key in keys:
        result[key] = result[key].astype(str)
    return result


def load_aligned_track(split: str, interval: int, scenario_id: str, trajectory_id: str):
    """Read and align one trajectory; matcher receives observations/features only."""
    base = DATA_ROOT / split
    observations = _key_columns(load_csv(base / "observations" / f"traj_dt_{interval}s.csv"))
    features = _key_columns(load_csv(base / "features" / f"features_dt_{interval}s.csv"))
    truth = _key_columns(load_csv(base / "ground_truth" / f"ground_truth_dt_{interval}s.csv"))
    keys = ["scenario_id", "trajectory_id", "sampling_interval_s", "seq", "timestamp"]
    if observations.duplicated(keys).any() or features.duplicated(keys).any() or truth.duplicated(keys).any():
        raise DashboardDataError("Duplicate observation keys prevent safe trajectory alignment.")
    merged = observations.merge(features, on=keys, how="inner", validate="one_to_one", suffixes=("", "_feature"))
    merged = merged.merge(truth, on=keys, how="inner", validate="one_to_one", suffixes=("", "_truth"))
    selected = merged.loc[merged["scenario_id"].eq(scenario_id) & merged["trajectory_id"].eq(trajectory_id)].copy()
    if selected.empty:
        raise DashboardDataError("No aligned observations, inference features, and ground truth were found for this trajectory.")
    selected = selected.sort_values("seq", key=lambda series: pd.to_numeric(series, errors="coerce"), kind="stable")
    network_frame = load_csv(DATA_ROOT / "road_network.csv")
    network_rows = network_frame.loc[network_frame["scenario_id"].astype(str).eq(scenario_id)].to_dict("records")
    network = RoadNetwork(scenario_id, network_rows)
    labels = {str(row["edge_id_true"]) for row in selected.to_dict("records")}
    unresolved = labels - set(network.edge_by_id)
    if unresolved:
        raise DashboardDataError(f"Ground-truth edges do not resolve in this network: {', '.join(sorted(unresolved)[:8])}")
    records = selected.to_dict("records")
    # Keep matching inputs separate. Ground-truth edge IDs are attached only after prediction.
    observations_inference = [{
        "lat": float(row["lat"]), "lon": float(row["lon"]), "sigma_m": float(row["sigma_m"]),
        "heading_deg": float(row["heading_deg"]), "speed_kmh": float(row.get("speed_kmh", 0) or 0),
    } for row in records]
    features_inference = [{
        "nearby_edge_count_100m": float(row.get("nearby_edge_count_100m", 0) or 0),
        "layer_count_100m": float(row.get("layer_count_100m", 0) or 0),
    } for row in records]
    truth_ids = [str(row["edge_id_true"]) for row in records]
    coords = [{"lat": point["lat"], "lon": point["lon"]} for point in observations_inference]
    metadata = [{key: row[key] for key in keys} for row in records]
    return network, observations_inference, features_inference, truth_ids, coords, metadata


def _run_selected_trajectory(network, observations, features, truth_ids, selected_params: dict):
    """Run each existing Python matcher without passing evaluation labels to it."""
    runs = {
        "Nearest Road": NearestRoadMatcher().match(observations, network),
        "HMM/Viterbi": HMMViterbiMatcher(selected_params.get("hmm", {})).match(observations, network),
        "Adaptive Radius": AdaptiveRadiusMatcher(selected_params.get("adaptive", {})).match(observations, network, features),
    }
    rows = []
    for algorithm, result in runs.items():
        for index, match in enumerate(result["matches"]):
            if match["edge_id"] not in network.edge_by_id:
                raise DashboardDataError(f"{algorithm} returned an unknown road edge: {match['edge_id']}")
            # Attach truth only for after-the-fact display/evaluation.
            rows.append({"algorithm": algorithm, "seq_index": index + 1, "edge_id_true": truth_ids[index],
                         "edge_id_pred": match["edge_id"], "correct": match["edge_id"] == truth_ids[index],
                         "matching_distance_m": match["matching_distance_m"], "candidate_count": match["candidate_count"],
                         "search_radius_m": match.get("search_radius_m")})
    return runs, pd.DataFrame(rows)


def render_map_matching() -> None:
    _page_header("Map matching", "Run the existing Python matchers on one selected synthetic trajectory.")
    st.info("The algorithms use noisy observations, road geometry, and inference-time features. Ground truth is attached only after matching for evaluation.", icon=":material/lock:")
    try:
        manifest = _read_manifest()
        py_detail = _optional_json(PY_METRICS_ROOT / "detailed_metrics.json")
        selected_params = {
            "hmm": py_detail.get("validation_tuning", {}).get("hmm", {}).get("selected", {}).get("parameters", {}),
            "adaptive": py_detail.get("validation_tuning", {}).get("adaptive_radius", {}).get("selected", {}).get("parameters", {}),
        }
        split = st.selectbox("Dataset split", ["test", "validation", "train"], key="matching_split")
        interval = st.selectbox("Sampling interval", INTERVALS, format_func=lambda value: f"{value} s", key="matching_interval")
        eligible_scenarios = manifest.loc[manifest["split"].eq(split)].copy()
        scenario_options = eligible_scenarios["scenario_id"].astype(str).tolist()
        if not scenario_options:
            st.warning(f"No scenarios are listed for the {split} split.")
            return
        scenario_id = st.selectbox("Scenario", scenario_options, key="matching_scenario")
        observations_df = load_csv(DATA_ROOT / split / "observations" / f"traj_dt_{interval}s.csv")
        scenario_obs = observations_df.loc[observations_df["scenario_id"].astype(str).eq(scenario_id)]
        trajectory_options = sorted(scenario_obs["trajectory_id"].astype(str).unique().tolist())
        if not trajectory_options:
            st.warning("There are no observations for this scenario and interval.")
            return
        trajectory_id = st.selectbox("Trajectory", trajectory_options, key="matching_trajectory")
    except DashboardDataError as exc:
        st.error(str(exc))
        return

    run_key = (split, int(interval), scenario_id, trajectory_id)
    if st.button("Run all three matchers", type="primary", key="run_matchers"):
        try:
            with st.spinner("Matching trajectory against its scenario road network..."):
                network, observations, features, truth_ids, coords, metadata = load_aligned_track(*run_key)
                runs, comparison = _run_selected_trajectory(network, observations, features, truth_ids, selected_params)
                st.session_state["dashboard_match_result"] = {
                    "key": run_key, "network_rows": pd.DataFrame(network.edges), "coords": coords,
                    "metadata": metadata, "runs": runs, "comparison": comparison,
                    "scenario_context": str(eligible_scenarios.loc[eligible_scenarios["scenario_id"].astype(str).eq(scenario_id)].iloc[0].get("scenario_type", "Unknown")),
                }
        except (DashboardDataError, KeyError, ValueError, TypeError) as exc:
            st.error(f"Could not run map matching: {exc}")
            return

    result = st.session_state.get("dashboard_match_result")
    if not result or result.get("key") != run_key:
        st.caption("Choose a split, interval, scenario, and trajectory, then select **Run all three matchers**.")
        return
    comparison = result["comparison"]
    overall = []
    for algorithm in ALGORITHMS:
        rows = comparison.loc[comparison["algorithm"].eq(algorithm)]
        acc = float(rows["correct"].mean()) if not rows.empty else 0.0
        overall.append({"Algorithm": algorithm, "Edge accuracy": acc,
                        "Correct points": int(rows["correct"].sum()), "Observations": int(len(rows))})
    st.caption(f"Scenario context: {result['scenario_context']} · {len(result['coords']):,} observations · {interval}s sampling")
    st.dataframe(pd.DataFrame(overall), hide_index=True, width="stretch", alt="Matching scores for the selected trajectory")
    with st.expander("Algorithm explanations and parameter source"):
        _algorithm_explanations()
        st.markdown("HMM and Adaptive Radius parameters, when present, come from Phase 3 validation-only selection. The matcher does not read ground-truth radius labels.")
        st.json(selected_params)

    chosen_algorithm = st.selectbox("Highlight algorithm on map", ALGORITHMS, key="matching_highlight")
    network_rows = result["network_rows"]
    network_paths = _network_paths(network_rows)
    matched_ids = set(comparison.loc[comparison["algorithm"].eq(chosen_algorithm), "edge_id_pred"].astype(str))
    layers = [pdk.Layer("PathLayer", data=network_paths, get_path="path", get_color="color", width_min_pixels=1,
                        pickable=True, auto_highlight=True)]
    layers.append(pdk.Layer("PathLayer", data=_network_paths(network_rows, matched_ids, ALGORITHM_COLORS[chosen_algorithm]),
                            get_path="path", get_color="color", width_min_pixels=5, pickable=True))
    coordinate_path = [[coord["lon"], coord["lat"]] for coord in result["coords"]]
    if len(coordinate_path) > 1:
        layers.append(pdk.Layer("PathLayer", data=[{"path": coordinate_path, "edge_id": "Observed GPS", "road_type": "GPS trajectory", "layer": 0, "color": ALGORITHM_COLORS["Observed GPS"]}],
                                get_path="path", get_color="color", width_min_pixels=3))
    layers.append(pdk.Layer("ScatterplotLayer", data=[{"position": [coord["lon"], coord["lat"]], "edge_id": "GPS observation"} for coord in result["coords"]],
                            get_position="position", get_fill_color=ALGORITHM_COLORS["Observed GPS"], get_radius=14,
                            radius_min_pixels=3, radius_max_pixels=7, pickable=True))
    positions = pd.concat([network_rows["from_lat"], network_rows["to_lat"]]).astype(float), pd.concat([network_rows["from_lon"], network_rows["to_lon"]]).astype(float)
    _render_map(layers, float(positions[0].mean()), float(positions[1].mean()), 13,
                f"Observed GPS points and {chosen_algorithm} matched road edges")
    display_rows = comparison.loc[comparison["algorithm"].eq(chosen_algorithm)].copy()
    display_rows = display_rows.rename(columns={"seq_index": "Point", "edge_id_true": "Ground-truth edge (evaluation)",
        "edge_id_pred": "Predicted edge", "correct": "Correct", "matching_distance_m": "GPS-to-edge distance (m)",
        "candidate_count": "Candidates", "search_radius_m": "Search radius (m)"})
    st.dataframe(display_rows.drop(columns=["algorithm"]), hide_index=True, width="stretch",
                 alt=f"Per-observation {chosen_algorithm} predictions and evaluation labels")


def _live_network(scenario_id: str):
    network_frame = load_csv(DATA_ROOT / "road_network.csv")
    rows = network_frame.loc[network_frame["scenario_id"].astype(str).eq(str(scenario_id))].to_dict("records")
    if not rows:
        raise DashboardDataError(f"No road network is available for {scenario_id}.")
    return RoadNetwork(str(scenario_id), rows), pd.DataFrame(rows)


def _network_covers_points(network, observations: list[dict], margin_m: float = 1000.0) -> bool:
    projected = [(network.project(point)) for point in observations]
    edge_xy = [network.project({"lat": float(row[key_lat]), "lon": float(row[key_lon])})
               for row in network.edges for key_lat, key_lon in (("from_lat", "from_lon"), ("to_lat", "to_lon"))]
    min_x, max_x = min(p[0] for p in edge_xy) - margin_m, max(p[0] for p in edge_xy) + margin_m
    min_y, max_y = min(p[1] for p in edge_xy) - margin_m, max(p[1] for p in edge_xy) + margin_m
    return all(min_x <= x <= max_x and min_y <= y <= max_y for x, y in projected)


def _live_match_key(mode: str, selection: dict, algorithm: str) -> str:
    """Stable key for the selected benchmark trajectory and matcher."""
    import hashlib
    encoded = json.dumps({"mode": mode, "selection": selection, "algorithm": algorithm}, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _live_row_table(observations: list[dict], matches: list[dict], network) -> pd.DataFrame:
    rows = []
    edges = [network.edge_by_id[match["edge_id"]] for match in matches]
    road_names = [_live_edge_attribute(edge, ("name", "road_name"), "") for edge in edges]
    include_road_names = bool(road_names) and all(road_names)
    for index, (observation, match) in enumerate(zip(observations, matches), start=1):
        edge = network.edge_by_id[match["edge_id"]]
        row = {
            "Observation ID": observation.get("observation_id", index),
            "Latitude": observation["lat"], "Longitude": observation["lon"],
            "Timestamp": observation.get("timestamp", "") or "Unavailable",
            "GPS Sigma (m)": observation.get("sigma_m"),
            "Matched Road ID": match["edge_id"],
            "Road Class": edge.get("highway", "Unavailable"),
            "Matching Distance (m)": match["matching_distance_m"],
            "Search Radius (m)": match.get("search_radius_m"),
            "Candidate Count": match["candidate_count"], "Matching Status": "Matched",
        }
        if include_road_names:
            row["Matched Road Name"] = road_names[index - 1]
        rows.append(row)
    return pd.DataFrame(rows)


def _load_live_demo_track(split: str, interval: int, scenario_id: str, trajectory_id: str):
    """Load only noisy GPS observations for one benchmark trajectory, in sequence order."""
    source = load_csv(DATA_ROOT / split / "observations" / f"traj_dt_{interval}s.csv")
    required = {"scenario_id", "trajectory_id", "split", "sampling_interval_s", "seq", "timestamp", "lat", "lon"}
    _require_columns(source, required, f"{split}/{interval}s GPS observations")
    track = source.loc[
        source["scenario_id"].astype(str).eq(str(scenario_id))
        & source["trajectory_id"].astype(str).eq(str(trajectory_id))
    ].copy()
    if track.empty:
        raise DashboardDataError(f"No GPS observations were found for {scenario_id}/{trajectory_id}.")
    track["_seq_order"] = pd.to_numeric(track["seq"], errors="coerce")
    if track["_seq_order"].isna().any():
        raise DashboardDataError(f"Trajectory {trajectory_id} contains a nonnumeric observation sequence.")
    track = track.sort_values(["_seq_order", "timestamp"], kind="stable").drop(columns="_seq_order").reset_index(drop=True)
    fields = {"lat", "lon", "timestamp", "sigma_m", "heading_deg", "speed_kmh"}
    observations = validate_observation_rows([{
        "observation_id": row["seq"],
        **{field: row.get(field) for field in fields if field in row},
    } for row in track.to_dict("records")], require_timestamp=True)
    return track, observations


def _live_candidate_table(network, observation: dict, match: dict, algorithm: str) -> pd.DataFrame:
    """List real nearby candidates in distance order; selection follows matcher scope."""
    candidates = network.candidates(observation)
    radius = match.get("search_radius_m")
    if algorithm == "Rule-Based Adaptive Radius" and radius is not None:
        shown = [candidate for candidate in candidates if candidate["distanceMeters"] <= float(radius)]
        if not shown:
            shown = candidates[:1]
    elif algorithm == "HMM / Viterbi" and radius is not None:
        shown = [candidate for candidate in candidates if candidate["distanceMeters"] <= float(radius)]
        shown = shown[:int(match.get("candidate_count", len(shown)))] or candidates[:1]
    else:
        shown = candidates
    nearest_id = candidates[0]["edgeId"] if candidates else None
    selected_id = str(match["edge_id"])
    return pd.DataFrame([{
        "Road ID": candidate["edgeId"],
        "Road class": candidate["edge"].get("highway", "Unavailable"),
        "Distance (m)": candidate["distanceMeters"],
        "Candidate role": "Selected match" if candidate["edgeId"] == selected_id else (
            "Nearest candidate" if candidate["edgeId"] == nearest_id else "Other candidate"
        ),
    } for candidate in shown[:15]])


def _live_edge_attribute(edge: dict, fields: tuple[str, ...], fallback: str) -> str:
    for field in fields:
        value = edge.get(field)
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "null"}:
            return text
    return fallback


def _live_demo_playback_transition(action: str, index: int, total: int) -> tuple[int, bool]:
    """Deterministic playback state transition shared by controls and animation ticks."""
    if total <= 1:
        return 0, False
    index = max(0, min(total - 1, int(index)))
    if action == "play":
        return (0 if index == total - 1 else index), True
    if action in ("pause", "previous", "next", "restart"):
        return {
            "pause": (index, False),
            "previous": (max(0, index - 1), False),
            "next": (min(total - 1, index + 1), False),
            "restart": (0, False),
        }[action]
    if action == "tick":
        if index >= total - 1:
            return index, False
        return index + 1, True
    raise ValueError(f"Unknown playback action: {action}")


def _live_matched_path_rows(network_rows: pd.DataFrame, matches: list[dict]) -> list[dict]:
    """Return route pieces made only from the actual predicted road geometries."""
    rows_by_id = {str(row["edge_id"]): row for row in network_rows.to_dict("records")}
    route_pieces = []
    for match in matches:
        edge_id = str(match["edge_id"])
        if route_pieces and route_pieces[-1]["edge_id"] == edge_id:
            route_pieces[-1]["candidates"].append(match["_candidate"])
        else:
            route_pieces.append({"edge_id": edge_id, "candidates": [match["_candidate"]]})
    matched_paths = []
    for piece in route_pieces:
        row = rows_by_id.get(piece["edge_id"])
        if not row:
            continue
        path = parse_linestring_wkt(row.get("geometry_wkt", ""))
        if not path:
            fallback = _network_paths(pd.DataFrame([row]))
            path = fallback[0]["path"] if fallback else []
        positions = piece["candidates"]
        if len(positions) > 1 and positions[-1]["alongMeters"] < positions[0]["alongMeters"]:
            path.reverse()
        if len(path) >= 2:
            matched_paths.append({"edge_id": piece["edge_id"], "road_type": str(row.get("highway", "road")),
                                  "layer": int(float(row.get("layer", 0) or 0)), "path": path,
                                  "color": [48, 142, 255, 245]})
    return matched_paths


def _live_directional_arrows(paths: list[list[list[float]]], limit: int = 14, size_m: float = 24.0) -> list[dict]:
    """Create sparse white arrow polygons from real adjacent trajectory coordinates."""
    import math

    segments = [(path[index], path[index + 1]) for path in paths for index in range(len(path) - 1)]
    if not segments:
        return []
    stride = max(1, math.ceil(len(segments) / limit))
    arrows = []
    for index in range(stride // 2, len(segments), stride):
        start, end = segments[index]
        lat_scale = 110540.0
        lon_scale = 111320.0 * math.cos(math.radians((start[1] + end[1]) / 2))
        x1, y1 = start[0] * lon_scale, start[1] * lat_scale
        x2, y2 = end[0] * lon_scale, end[1] * lat_scale
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1:
            continue
        ux, uy = dx / length, dy / length
        px, py = -uy, ux
        cx, cy = x1 + dx * 0.64, y1 + dy * 0.64
        triangle = [(cx + ux * size_m * 0.65, cy + uy * size_m * 0.65),
                    (cx - ux * size_m * 0.45 + px * size_m * 0.48, cy - uy * size_m * 0.45 + py * size_m * 0.48),
                    (cx - ux * size_m * 0.45 - px * size_m * 0.48, cy - uy * size_m * 0.45 - py * size_m * 0.48)]
        arrows.append({"polygon": [[x / lon_scale, y / lat_scale] for x, y in triangle]})
        if len(arrows) >= limit:
            break
    return arrows


def _live_path_point(path: list[list[float]], fraction: float) -> list[float] | None:
    """Choose an actual point along a rendered trajectory for a map callout."""
    if not path:
        return None
    return list(path[round(max(0.0, min(1.0, fraction)) * (len(path) - 1))])


def _live_match_positions(matches: list[dict], network) -> list[list[float]]:
    """Return matcher-selected road positions as [longitude, latitude] pairs."""
    return [[lon, lat] for lat, lon in
            (point_on_edge(match["_candidate"], network) for match in matches)]


def _live_vehicle_bearing(positions: list[list[float]], index: int) -> float:
    """Return a north-based clockwise bearing for an upward-facing vehicle icon."""
    if len(positions) < 2:
        return 0.0
    current = positions[index]
    neighbor = positions[index + 1] if index < len(positions) - 1 else positions[index - 1]
    if index == len(positions) - 1:
        current, neighbor = neighbor, current
    lat_mid = math.radians((current[1] + neighbor[1]) / 2)
    dx = (neighbor[0] - current[0]) * 111320.0 * math.cos(lat_mid)
    dy = (neighbor[1] - current[1]) * 110540.0
    if math.hypot(dx, dy) < 1e-6:
        return 0.0
    return math.degrees(math.atan2(dx, dy)) % 360.0


def _live_callout_position(anchor: list[float], path: list[list[float]], side: float,
                           offset_m: float) -> list[float]:
    """Offset a label from the local route direction in approximate projected metres."""
    if len(path) < 2:
        return [anchor[0], anchor[1] + side * offset_m / 110540.0]
    nearest = min(range(len(path)), key=lambda index:
                  (path[index][0] - anchor[0]) ** 2 + (path[index][1] - anchor[1]) ** 2)
    before, after = path[max(0, nearest - 1)], path[min(len(path) - 1, nearest + 1)]
    lat_mid = math.radians(anchor[1])
    dx = (after[0] - before[0]) * 111320.0 * math.cos(lat_mid)
    dy = (after[1] - before[1]) * 110540.0
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return [anchor[0], anchor[1] + side * offset_m / 110540.0]
    normal_x, normal_y = -dy / length * side, dx / length * side
    return [anchor[0] + normal_x * offset_m / (111320.0 * math.cos(lat_mid)),
            anchor[1] + normal_y * offset_m / 110540.0]


def _render_live_map(network, network_rows: pd.DataFrame, observations: list[dict], matches: list[dict],
                     current_index: int, algorithm: str) -> None:
    """Render a dark, offline navigation map from the selected road geometry."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon as PlotPolygon

    chosen = matches[current_index]
    candidates = network.candidates(observations[current_index])
    radius = chosen.get("search_radius_m")
    if radius is None:
        candidate_rows = candidates[:min(5, int(chosen["candidate_count"]))]
    else:
        candidate_rows = [item for item in candidates if item["distanceMeters"] <= float(radius)][:min(30, int(chosen["candidate_count"]))]
        if not candidate_rows and candidates:
            candidate_rows = candidates[:1]
    rows_by_id = {str(row["edge_id"]): row for row in network_rows.to_dict("records")}
    matched_paths = _live_matched_path_rows(network_rows, matches)
    raw_path = [[float(point["lon"]), float(point["lat"])] for point in observations]
    matched_positions = _live_match_positions(matches, network)
    matched_path = [point for path in matched_paths for point in path["path"]]
    extent_points = list(raw_path) + list(matched_path)
    projected = [network.project({"lon": point[0], "lat": point[1]}) for point in extent_points]
    callout_offset = 0.20 * max(80.0,
        max(point[0] for point in projected) - min(point[0] for point in projected),
        max(point[1] for point in projected) - min(point[1] for point in projected)) if projected else 80.0
    callouts = []
    if len(raw_path) > 1:
        anchor = _live_path_point(raw_path, 0.28)
        callouts.append(("Raw GPS Trajectory", anchor,
                         _live_callout_position(anchor, raw_path, 1.0, callout_offset)))
    if matched_path and len(raw_path) > 1:
        anchor = _live_path_point(matched_path, 0.72)
        callouts.append(("Map-Matched Route", anchor,
                         _live_callout_position(anchor, matched_path, -1.0, callout_offset)))
    extent_points.extend(point for _, _, label_position in callouts for point in [label_position])
    extent_points.extend(point for edge in candidate_rows for point in
                         parse_linestring_wkt(rows_by_id[str(edge["edgeId"])].get("geometry_wkt", "")))

    center_lat = sum(point[1] for point in extent_points) / len(extent_points)
    center_lon = sum(point[0] for point in extent_points) / len(extent_points)
    lon_scale = 111320.0 * math.cos(math.radians(center_lat))

    def project(point):
        return ((float(point[0]) - center_lon) * lon_scale,
                (float(point[1]) - center_lat) * 110540.0)

    def project_path(path):
        return [project(point) for point in path]

    with _LIVE_PLOT_LOCK:
        fig, ax = plt.subplots(figsize=(14, 8), facecolor="#101925")
        ax.set_facecolor("#101925")
        ax.set_axis_off()
        for road in _network_paths(network_rows):
            x, y = zip(*project_path(road["path"]))
            ax.plot(x, y, color="#72869e", linewidth=1.45, alpha=0.82, zorder=1)
        for candidate in candidate_rows:
            edge = rows_by_id.get(str(candidate["edgeId"]))
            path = parse_linestring_wkt(edge.get("geometry_wkt", "")) if edge else []
            if len(path) >= 2:
                x, y = zip(*project_path(path))
                ax.plot(x, y, color="#c0d1e3", linewidth=2.4, alpha=0.96, zorder=2)
        for road in matched_paths:
            x, y = zip(*project_path(road["path"]))
            casing, = ax.plot(x, y, color="#07111f", linewidth=9.5, zorder=4)
            casing.set_gid(f"matched-road-casing:{road['edge_id']}")
            route, = ax.plot(x, y, color="#2f9fff", linewidth=5.8, solid_capstyle="round", zorder=5)
            route.set_gid(f"matched-road:{road['edge_id']}")
        if len(raw_path) > 1:
            x, y = zip(*project_path(raw_path))
            gps_line, = ax.plot(x, y, color="#ff354b", linewidth=3.8, solid_capstyle="round", zorder=6)
            gps_line.set_gid("raw-gps-trajectory")
        gps_x, gps_y = zip(*project_path(raw_path))
        ax.scatter(gps_x, gps_y, s=30, color="#ff354b", edgecolors="white", linewidths=0.8, zorder=7)
        current_gps = project(raw_path[current_index])
        ax.add_patch(Circle(current_gps, 18, facecolor="#ff354b", edgecolor="white", linewidth=1.7, zorder=8))

        arrows = _live_directional_arrows([raw_path], limit=12) if len(raw_path) > 1 else []
        if len(raw_path) > 1:
            arrows.extend(_live_directional_arrows([matched_positions], limit=12))
        for arrow in arrows:
            arrow_patch = PlotPolygon(project_path(arrow["polygon"]), closed=True,
                                      facecolor="white", edgecolor="#101925", linewidth=1.6, zorder=9)
            arrow_patch.set_gid("direction-arrow")
            ax.add_patch(arrow_patch)

        car_x, car_y = project(matched_positions[current_index])
        ax.add_patch(Circle((car_x, car_y), 58, facecolor="#2f9fff", edgecolor="#eaf6ff",
                            linewidth=1.6, alpha=0.36, zorder=10))
        angle = math.radians(_live_vehicle_bearing(matched_positions, current_index))
        forward, right = (math.sin(angle), math.cos(angle)), (math.cos(angle), -math.sin(angle))
        def car_point(front, side):
            return (car_x + forward[0] * front + right[0] * side,
                    car_y + forward[1] * front + right[1] * side)
        body = PlotPolygon([car_point(46, 0), car_point(28, 18), car_point(-23, 18),
                            car_point(-34, 12), car_point(-34, -12), car_point(-23, -18),
                            car_point(28, -18)], closed=True, facecolor="#f5f9ff",
                           edgecolor="#10213a", linewidth=2.2, zorder=12)
        body.set_gid("vehicle-marker")
        ax.add_patch(body)
        ax.add_patch(PlotPolygon([car_point(25, 11), car_point(25, -11), car_point(4, -11),
                                  car_point(4, 11)], closed=True, facecolor="#39a9ff",
                                 edgecolor="#10213a", linewidth=1.0, zorder=13))
        for label, anchor, label_position in callouts:
            ax.annotate(label, xy=project(anchor), xytext=project(label_position), color="white",
                        fontsize=11, fontweight="bold", zorder=15,
                        arrowprops={"arrowstyle": "-", "color": "white", "lw": 1.5},
                        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#1a2a40",
                              "edgecolor": "#d6e5f5", "linewidth": 1.0})

        px, py = zip(*(project(point) for point in extent_points))
        x_span, y_span = max(px) - min(px), max(py) - min(py)
        cx, cy = (max(px) + min(px)) / 2, (max(py) + min(py)) / 2
        aspect = 14 / 8
        x_span = max(x_span, y_span * aspect, 120.0) * 1.15
        y_span = x_span / aspect
        ax.set_xlim(cx - x_span / 2, cx + x_span / 2)
        ax.set_ylim(cy - y_span / 2, cy + y_span / 2)
        ax.set_aspect("equal", adjustable="box")
        ax.text(0.018, 0.965, f"LIVE MAP MATCHING  ·  {algorithm.upper()}", transform=ax.transAxes,
                color="#dceafa", fontsize=9, fontweight="bold", va="top",
                bbox={"boxstyle": "round,pad=0.45", "facecolor": "#1b2a3d", "edgecolor": "#34475e"})
        fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.015)
        st.pyplot(fig, width="stretch",
                  alt="Dark offline road map with the red GPS trajectory, blue matched route, white direction arrows, and moving vehicle")
        plt.close(fig)

def _evaluate_live_benchmark(split: str, interval: int, scenario_id: str, trajectory_id: str,
                             observations: list[dict], matches: list[dict], elapsed_ms: float,
                             network=None) -> dict | None:
    """Evaluate completed live predictions against aligned labels, after matching only."""
    ground_truth_path = DATA_ROOT / split / "ground_truth" / f"ground_truth_dt_{interval}s.csv"
    try:
        truth = load_csv(ground_truth_path)
    except DashboardDataError:
        return None
    required = {"scenario_id", "trajectory_id", "split", "sampling_interval_s", "seq", "timestamp", "edge_id_true"}
    if not required.issubset(truth.columns):
        return None
    selected = truth.loc[
        truth["scenario_id"].astype(str).eq(str(scenario_id))
        & truth["trajectory_id"].astype(str).eq(str(trajectory_id))
        & truth["split"].astype(str).eq(str(split))
        & pd.to_numeric(truth["sampling_interval_s"], errors="coerce").eq(int(interval))
    ].copy()
    selected["_seq_order"] = pd.to_numeric(selected["seq"], errors="coerce")
    if selected["_seq_order"].isna().any():
        return None
    selected = selected.sort_values(["_seq_order", "timestamp"], kind="stable").reset_index(drop=True)
    if len(selected) != len(observations) or len(matches) != len(observations):
        return None
    rows = []
    for observation, match, label in zip(observations, matches, selected.to_dict("records")):
        if str(observation.get("observation_id")) != str(label.get("seq")) or str(observation.get("timestamp")) != str(label.get("timestamp")):
            return None
        true_edge = str(label["edge_id_true"])
        if network is not None and true_edge not in network.edge_by_id:
            return None
        rows.append({
            "trajectory_id": str(trajectory_id),
            "edge_id_true": true_edge,
            "edge_id_pred": str(match["edge_id"]),
            "matching_distance_m": float(match["matching_distance_m"]),
            "candidate_count": int(match["candidate_count"]),
            "processing_ms_per_observation": float(elapsed_ms) / max(1, len(matches)),
        })
    return calculate_metrics(rows)


def _render_live_demo() -> None:
    _page_header("Live Map-Matching Demo",
                 "Replay a recorded synthetic GPS trajectory and compare its noisy path with predictions from the existing Python matchers.")
    st.caption("Recorded benchmark playback · not a live GPS feed or real-world accuracy claim.")
    try:
        manifest = _read_manifest()
        top = st.columns([1, 1.3, 1])
        with top[0]:
            split = st.selectbox("Dataset split", ["test", "validation", "train"], key="demo_split")
        scenario_ids = manifest.loc[manifest["split"].eq(split), "scenario_id"].astype(str).tolist()
        if not scenario_ids:
            st.warning(f"No scenarios are available in the {split} split.")
            return
        with top[1]:
            scenario_id = st.selectbox("Scenario", scenario_ids, key="demo_scenario")
        with top[2]:
            interval = st.selectbox("Sampling interval", INTERVALS, index=len(INTERVALS) - 1,
                                   format_func=lambda value: f"{value} s", key="demo_interval")
        source_path = DATA_ROOT / split / "observations" / f"traj_dt_{interval}s.csv"
        source = load_csv(source_path)
        in_scenario = source.loc[source["scenario_id"].astype(str).eq(str(scenario_id))]
        trajectory_ids = sorted(in_scenario["trajectory_id"].astype(str).unique())
        if not trajectory_ids:
            st.warning("No trajectories are available for this scenario and sampling interval.")
            return
        trajectory_id = st.selectbox("Trajectory", trajectory_ids, key="demo_trajectory")
        algorithm = st.selectbox("Map-matching algorithm",
            ["Nearest Road", "HMM / Viterbi", "Rule-Based Adaptive Radius"], key="demo_algorithm")
        algorithm_help = {
            "Nearest Road": "Selects the closest road segment independently for each GPS point.",
            "HMM / Viterbi": "Uses the ordered observations and road-network transitions to choose a likely sequence.",
            "Rule-Based Adaptive Radius": "Sets the candidate radius from GPS uncertainty and nearby road complexity at prediction time.",
        }
        st.caption(algorithm_help[algorithm])

        track, observations = _load_live_demo_track(split, interval, scenario_id, trajectory_id)
        st.caption(f"{scenario_id} · {trajectory_id} · {len(observations):,} observations · {interval}s sampling")
        preview_fields = [field for field in ("seq", "timestamp", "lat", "lon", "sigma_m", "heading_deg", "speed_kmh") if field in track.columns]
        with st.expander("GPS observations and timestamps", expanded=False):
            st.dataframe(track[preview_fields], hide_index=True, height=230, width="stretch",
                         alt="Selected trajectory's original synthetic GPS observations")

        selection = {"split": split, "scenario": scenario_id, "interval": int(interval),
                     "trajectory": trajectory_id}
        run_key = _live_match_key("synthetic_benchmark", selection, algorithm)
        if st.session_state.get("demo_active_key") != run_key:
            st.session_state["demo_active_key"] = run_key
            st.session_state["demo_playing"] = False
            st.session_state["demo_observation_index"] = 0
            st.session_state["demo_last_rendered_index"] = 0
            st.session_state.pop("demo_result", None)

        parameters_detail = _optional_json(PY_METRICS_ROOT / "detailed_metrics.json")
        matcher_parameters = {
            "hmm": parameters_detail.get("validation_tuning", {}).get("hmm", {}).get("selected", {}).get("parameters", {}),
            "adaptive": parameters_detail.get("validation_tuning", {}).get("adaptive_radius", {}).get("selected", {}).get("parameters", {}),
        }
        if st.button("Run Map Matching", type="primary", key="demo_run_match"):
            try:
                network, network_rows = _live_network(scenario_id)
                with st.spinner(f"Running {algorithm} on {len(observations):,} observations..."):
                    # Matcher inputs contain noisy GPS observations only, never reference labels.
                    prediction = run_matcher(algorithm, observations, network, parameters=matcher_parameters)
                matches = prediction["matches"]
                if len(matches) != len(observations):
                    raise DashboardDataError("Matcher output length does not match the selected GPS trajectory.")
                unresolved = {str(match["edge_id"]) for match in matches} - set(network.edge_by_id)
                if unresolved:
                    raise DashboardDataError(f"Matcher returned edges outside the selected network: {sorted(unresolved)[:5]}")
                st.session_state["demo_result"] = {
                    "key": run_key, "algorithm": algorithm, "observations": observations,
                    "matches": matches, "network": network, "network_rows": network_rows,
                    "elapsed_ms": float(prediction["elapsedMs"]),
                }
                st.session_state["demo_playing"] = False
                st.session_state["demo_observation_index"] = 0
                st.session_state["demo_last_rendered_index"] = 0
            except (DashboardDataError, ValueError, KeyError, TypeError) as exc:
                st.error(f"Map matching could not be completed: {exc}")
                return

        result = st.session_state.get("demo_result")
        if not result or result.get("key") != run_key:
            st.info("Choose a scenario, trajectory, and algorithm, then run matching to generate predictions from the existing implementation.")
            return
        observations = result["observations"]
        matches = result["matches"]
        network = result["network"]
        n_observations = len(observations)
        if n_observations < 1:
            st.info("This trajectory has no usable GPS observations.")
            return

        speed_col, status_col = st.columns([1, 3])
        with speed_col:
            playback_speed = st.selectbox("Playback speed", [0.5, 1.0, 2.0, 4.0], index=1,
                                          format_func=lambda value: f"{value:g}×", key="demo_playback_speed")
        with status_col:
            st.caption(f"{algorithm} · predictions computed from {scenario_id} road geometry")

        if n_observations > 1:
            is_playing = bool(st.session_state.get("demo_playing", False))

            @st.fragment(run_every=(1.0 / float(playback_speed)) if is_playing else None,
                         key="live_demo_playback")
            def render_live_demo_playback():
                current_before_slider = int(st.session_state.get("demo_observation_index", 0))
                last_rendered = int(st.session_state.get("demo_last_rendered_index", current_before_slider))
                manual_seek = current_before_slider != last_rendered
                if manual_seek:
                    st.session_state["demo_playing"] = False
                    st.session_state["demo_last_rendered_index"] = current_before_slider
                elif st.session_state.get("demo_playing", False):
                    current_after_tick, playing = _live_demo_playback_transition("tick", current_before_slider, n_observations)
                    st.session_state["demo_observation_index"] = current_after_tick
                    st.session_state["demo_playing"] = playing
                    st.session_state["demo_last_rendered_index"] = current_after_tick
                    current_before_slider = current_after_tick
                    if not playing:
                        st.rerun()

                current_index = int(st.session_state.get("demo_observation_index", 0))
                current_observation = observations[current_index]
                current_match = matches[current_index]
                current_edge = network.edge_by_id[current_match["edge_id"]]
                road_name = _live_edge_attribute(current_edge, ("name", "road_name"), "")
                snapped = point_on_edge(current_match["_candidate"], network)
                radius_text = "not used" if current_match.get("search_radius_m") is None else f"{float(current_match['search_radius_m']):.2f} m"
                _render_live_map(network, result["network_rows"], observations, matches, current_index, algorithm)
                st.markdown(
                    "<div style='display:flex;flex-wrap:wrap;gap:10px 22px;align-items:center;"
                    "padding:10px 14px;margin:2px 0 10px;border:1px solid #263449;border-radius:12px;"
                    "background:linear-gradient(105deg,#111a27,#182437);color:#eaf1fa;font-size:0.9rem'>"
                    "<span><b style='color:#ff3b4e'>━━</b> Raw GPS trajectory</span>"
                    "<span><b style='color:#399fff'>━━</b> Map-matched road route</span>"
                    "<span><b style='color:#fff'>➤</b> Direction of travel</span>"
                    "<span><b style='color:#55baff'>●</b> Moving vehicle</span></div>",
                    unsafe_allow_html=True,
                )

                action = None
                controls = st.columns(5)
                if controls[0].button("Play", key="demo_play", disabled=bool(st.session_state.get("demo_playing", False))):
                    action = "play"
                elif controls[1].button("Pause", key="demo_pause", disabled=not bool(st.session_state.get("demo_playing", False))):
                    action = "pause"
                elif controls[2].button("Previous", key="demo_previous"):
                    action = "previous"
                elif controls[3].button("Next", key="demo_next"):
                    action = "next"
                elif controls[4].button("Restart", key="demo_restart"):
                    action = "restart"
                if action:
                    current, playing = _live_demo_playback_transition(action, current_index, n_observations)
                    st.session_state["demo_observation_index"] = current
                    st.session_state["demo_playing"] = playing
                    st.session_state["demo_last_rendered_index"] = current
                    st.rerun()

                current_index = st.slider("Trajectory progress", 0, n_observations - 1,
                                          key="demo_observation_index",
                                          format="Observation %d of " + str(n_observations))
                st.session_state["demo_last_rendered_index"] = int(current_index)
                st.progress((current_index + 1) / n_observations,
                            text=f"Observation {current_index + 1} of {n_observations}")
                current_observation = observations[current_index]
                current_match = matches[current_index]
                current_edge = network.edge_by_id[current_match["edge_id"]]
                road_name = _live_edge_attribute(current_edge, ("name", "road_name"), "")
                snapped = point_on_edge(current_match["_candidate"], network)
                st.subheader("Current observation")
                detail_columns = st.columns(4)
                detail_columns[0].metric("Observation", f"{current_index + 1} / {n_observations}")
                detail_columns[1].metric("GPS", f"{current_observation['lat']:.6f}, {current_observation['lon']:.6f}")
                detail_columns[2].metric("GPS sigma", f"{current_observation['sigma_m']:.2f} m")
                detail_columns[3].metric("Distance to road", f"{current_match['matching_distance_m']:.2f} m")
                road_description = f"{road_name} · " if road_name else ""
                st.caption(f"{current_observation.get('timestamp') or 'Timestamp unavailable'} · {algorithm} · "
                           f"Road {current_match['edge_id']} · {road_description}{current_edge.get('highway', 'class unavailable')} · "
                           f"matched point {snapped[1]:.6f}, {snapped[0]:.6f} (lat, lon) · "
                           f"search radius {radius_text} · "
                           f"{int(current_match['candidate_count'])} candidates · matched")
                candidate_table = _live_candidate_table(network, current_observation, current_match, algorithm)
                with st.expander("Current observation candidates", expanded=False):
                    st.dataframe(candidate_table, hide_index=True, height=250, width="stretch",
                                 alt="Actual candidate road IDs and calculated GPS-to-road distances")
                if algorithm == "Rule-Based Adaptive Radius":
                    diagnostics = adaptive_diagnostics(current_observation, network, matcher_parameters.get("adaptive", {}))
                    with st.expander("Adaptive radius diagnostics", expanded=False):
                        st.dataframe(pd.DataFrame([diagnostics]), hide_index=True, width="stretch",
                                     alt="Adaptive radius and candidate filtering diagnostics")

            render_live_demo_playback()
        else:
            observation, match = observations[0], matches[0]
            st.info("This trajectory contains one observation. The map shows the GPS point and its match; no multi-point route or playback is implied.")
            st.markdown("**GPS point (red)**　·　**Matched road (blue)**　·　**Vehicle on matched road**")
            _render_live_map(network, result["network_rows"], observations, matches, 0, algorithm)
            single_edge = network.edge_by_id[match["edge_id"]]
            snapped = point_on_edge(match["_candidate"], network)
            detail_columns = st.columns(4)
            detail_columns[0].metric("Observation", "1 / 1")
            detail_columns[1].metric("GPS", f"{observation['lat']:.6f}, {observation['lon']:.6f}")
            detail_columns[2].metric("GPS sigma", f"{observation['sigma_m']:.2f} m")
            detail_columns[3].metric("Distance to road", f"{match['matching_distance_m']:.2f} m")
            st.caption(f"{observation.get('timestamp') or 'Timestamp unavailable'} · {algorithm} · Road {match['edge_id']} · "
                       f"{single_edge.get('highway', 'class unavailable')} · matched point {snapped[1]:.6f}, {snapped[0]:.6f} (lat, lon) · "
                       f"{int(match['candidate_count'])} candidates · matched")
            single_candidates = _live_candidate_table(network, observation, match, algorithm)
            with st.expander("Current observation candidates", expanded=False):
                st.dataframe(single_candidates, hide_index=True, width="stretch",
                             alt="Actual candidates for the single GPS observation")
            if algorithm == "Rule-Based Adaptive Radius":
                diagnostics = adaptive_diagnostics(observation, network, matcher_parameters.get("adaptive", {}))
                with st.expander("Adaptive radius diagnostics", expanded=False):
                    st.dataframe(pd.DataFrame([diagnostics]), hide_index=True, width="stretch",
                                 alt="Adaptive radius and candidate filtering diagnostics")

        # Ground truth is read only after a prediction is complete and is used only for scoring.
        evaluation = _evaluate_live_benchmark(split, interval, scenario_id, trajectory_id,
                                              observations, matches, result["elapsed_ms"], network)
        st.subheader("Selected trajectory performance")
        st.caption(f"Evaluation scope: {split} split · {scenario_id}/{trajectory_id} · {interval}s · labels are evaluation-only")
        metrics = evaluation or {}
        score_columns = st.columns(4)
        score_values = [
            ("Edge accuracy", f"{metrics['edge_accuracy']:.3f}" if metrics else "Not available"),
            ("Trajectory accuracy", f"{metrics['trajectory_accuracy_mean']:.3f}" if metrics and n_observations > 1 else "Not available"),
            ("Edge F1 (macro)", f"{metrics['edge_f1_macro']:.3f}" if metrics else "Not available"),
            ("Execution time", f"{result['elapsed_ms']:.2f} ms"),
            ("Mean candidates", f"{metrics['candidate_count_mean']:.2f}" if metrics else f"{sum(int(m['candidate_count']) for m in matches) / n_observations:.2f}"),
            ("Mean GPS-to-road distance", f"{metrics['mean_matching_distance_m']:.2f} m" if metrics else f"{sum(float(m['matching_distance_m']) for m in matches) / n_observations:.2f} m"),
        ]
        for column, (label, value) in zip(score_columns, score_values[:4]):
            column.metric(label, value)
        second_score_columns = st.columns(2)
        for column, (label, value) in zip(second_score_columns, score_values[4:]):
            column.metric(label, value)
        if evaluation is None:
            st.caption("Ground-truth labels were unavailable or could not be aligned; only prediction-derived distance and candidate diagnostics are shown.")
        st.markdown("**Matched road sequence**")
        st.code(" → ".join(match["edge_id"] for match in matches), language=None)
        output = _live_row_table(observations, matches, network)
        output["Selected Algorithm"] = algorithm
        st.subheader("Observation-wise predictions")
        st.dataframe(output, hide_index=True, width="stretch", alt="Live predictions with GPS coordinates, matched road IDs and distances")
        st.download_button("Download matching results as CSV", output.to_csv(index=False).encode("utf-8"),
                           file_name=f"{trajectory_id}_{algorithm.lower().replace(' ', '_').replace('/', '-')}_matches.csv",
                           mime="text/csv", key="demo_results_csv")
    except DashboardDataError as exc:
        st.error(str(exc))

def render_algorithm_comparison() -> None:
    _page_header("Algorithm comparison", "Compare saved Python test metrics across contexts, GPS noise levels, and sampling rates.")
    st.warning("These scores are from a synthetic benchmark and are not evidence of real-world performance.", icon=":material/science:")
    metrics, detail, js_metrics, js_detail = _load_test_result_tables()
    if metrics.empty or not detail:
        st.error("Python experiment results are unavailable. Expected `results/python_metrics/algorithm_comparison.csv` and `detailed_metrics.json`.")
        return
    scope_choice = st.selectbox("Comparison view", ["Overall", "Road context", "GPS noise tier", "Sampling interval"], key="comparison_scope")
    scope_map = {"Overall": ("overall", None), "Road context": ("road_context", "context"),
                 "GPS noise tier": ("gps_noise", "noise_tier"), "Sampling interval": ("sampling_interval", "sampling_interval_s")}
    scope, group_field = scope_map[scope_choice]
    selected = _metric_rows(metrics, scope)
    if selected.empty:
        st.warning(f"No Python metric rows are available for {scope_choice.lower()}.")
        return
    if group_field:
        col_name = {"context": "context", "noise_tier": "noise_tier", "sampling_interval_s": "sampling_interval_s"}[group_field]
        values = selected[col_name].astype(str).unique().tolist()
        if group_field == "context":
            chosen = st.selectbox("Road context", values, key="comparison_context")
        elif group_field == "noise_tier":
            chosen = st.selectbox("GPS noise level", values, key="comparison_noise")
        else:
            chosen = st.selectbox("Sampling interval", sorted(values, key=int), format_func=lambda value: f"{value} s", key="comparison_interval")
        selected = selected.loc[selected[col_name].astype(str).eq(str(chosen))]
    else:
        selected = selected.sort_values("algorithm", key=lambda series: series.map({name: index for index, name in enumerate(ALGORITHMS)}))

    chart_metric = st.selectbox("Metric to chart", ["edge_accuracy", "edge_f1_macro", "trajectory_accuracy_mean", "trajectory_exact_match_rate"],
                                format_func=lambda value: {"edge_accuracy": "Edge accuracy", "edge_f1_macro": "Macro F1", "trajectory_accuracy_mean": "Mean trajectory accuracy", "trajectory_exact_match_rate": "Exact trajectory match rate"}[value],
                                key="comparison_metric")
    plotted = selected[["algorithm", chart_metric]].copy().set_index("algorithm")
    plotted = plotted.reindex([name for name in ALGORITHMS if name in plotted.index])
    label = {"edge_accuracy": "Edge accuracy", "edge_f1_macro": "Macro F1", "trajectory_accuracy_mean": "Mean trajectory accuracy", "trajectory_exact_match_rate": "Exact trajectory match rate"}[chart_metric]
    st.bar_chart(plotted, height=360, color=["#2B6CB0"], alt=f"{label} comparison for the selected evaluation group")
    shown_columns = ["algorithm", "observations", "trajectories", "edge_accuracy", "edge_precision_macro", "edge_recall_macro", "edge_f1_macro",
                     "mean_matching_distance_m", "trajectory_accuracy_mean", "trajectory_exact_match_rate", "candidate_count_mean", "processing_ms_per_observation"]
    shown_columns = [column for column in shown_columns if column in selected]
    st.dataframe(selected[shown_columns], hide_index=True, width="stretch", alt=f"Detailed comparison of {scope_choice.lower()} metrics")

    if scope == "overall":
        js_hmm = _metric_value(js_detail, "HMM/Viterbi", "edge_accuracy")
        py_hmm = _metric_value(detail, "HMM/Viterbi", "edge_accuracy")
        js_exact = _metric_value(js_detail, "HMM/Viterbi", "trajectory_exact_match_rate")
        py_exact = _metric_value(detail, "HMM/Viterbi", "trajectory_exact_match_rate")
        if all(value is not None for value in (py_hmm, js_hmm, py_exact, js_exact)):
            st.info(
                "Parity note: the saved Python HMM result is close to, but not identical to, the JavaScript reference. "
                f"Edge accuracy: Python {py_hmm:.6f} vs JavaScript {js_hmm:.6f}; exact-trajectory rate: "
                f"Python {py_exact:.6f} vs JavaScript {js_exact:.6f}. The documented comparison attributes the small gap to possible Viterbi route-cost/tie ordering differences; observation-level predictions are not exported, so the cause is not confirmed.",
                icon=":material/info:",
            )
        else:
            st.info("HMM parity note: the JavaScript comparison metrics are unavailable, so the saved Python–JavaScript gap cannot be summarized here.", icon=":material/info:")
    adaptive = selected.loc[selected["algorithm"].eq("Adaptive Radius"), "edge_accuracy"]
    if scope == "overall" and not adaptive.empty:
        nearest = selected.loc[selected["algorithm"].eq("Nearest Road"), "edge_accuracy"]
        hmm = selected.loc[selected["algorithm"].eq("HMM/Viterbi"), "edge_accuracy"]
        if not nearest.empty and not hmm.empty:
            st.caption(f"In the overall test summary, Adaptive Radius is {float(adaptive.iloc[0]):.4f}, Nearest Road is {float(nearest.iloc[0]):.4f}, and HMM/Viterbi is {float(hmm.iloc[0]):.4f}. No single method is declared the winner for all conditions.")


def render_results_export() -> None:
    _page_header("Results and export", "Review the saved Phase 3 outputs and download reproducible result files.")
    py_csv_path = PY_METRICS_ROOT / "algorithm_comparison.csv"
    py_json_path = PY_METRICS_ROOT / "detailed_metrics.json"
    try:
        metrics = load_csv(py_csv_path)
        detail = load_json(py_json_path)
    except DashboardDataError as exc:
        st.error(str(exc))
        return
    st.caption(f"Python test metrics · generated {detail.get('generated_at_utc', 'timestamp unavailable')} · split: {detail.get('evaluation_split', 'unknown')}")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Download Python metrics CSV", metrics.to_csv(index=False).encode("utf-8"),
                           file_name="geomotion_python_algorithm_comparison.csv", mime="text/csv", key="results_csv_download")
    with col2:
        json_bytes = json.dumps(detail, indent=2, ensure_ascii=False).encode("utf-8")
        st.download_button("Download detailed metrics JSON", json_bytes, file_name="geomotion_python_detailed_metrics.json",
                           mime="application/json", key="results_json_download")
    st.subheader("Saved Python comparison figures")
    figure_files = [
        ("algorithm_comparison.png", "Overall algorithm comparison"),
        ("context_wise_comparison.png", "Road-context comparison"),
        ("sampling_interval_comparison.png", "Sampling-interval comparison"),
    ]
    available = [(PY_FIGURES_ROOT / filename, title) for filename, title in figure_files if (PY_FIGURES_ROOT / filename).is_file()]
    if available:
        cols = st.columns(min(2, len(available)))
        for index, (path, title) in enumerate(available):
            with cols[index % len(cols)]:
                st.image(str(path), caption=title, width="stretch", alt=title)
    else:
        st.info("No saved Python figures were found. The metrics table below remains available.")
    st.subheader("Saved Python metric table")
    scope_options = metrics["scope"].astype(str).unique().tolist() if "scope" in metrics else []
    if scope_options:
        scope = st.selectbox("Rows to display and export", scope_options, key="export_scope")
        view = metrics.loc[metrics["scope"].astype(str).eq(scope)].copy()
    else:
        view = metrics
    st.dataframe(view, hide_index=True, width="stretch", alt="Saved Python algorithm comparison metrics")
    st.download_button("Download this metric view as CSV", view.to_csv(index=False).encode("utf-8"),
                       file_name=f"geomotion_metrics_{scope if scope_options else 'all'}.csv", mime="text/csv", key="metric_scope_download")
    st.subheader("Python–JavaScript parity note")
    st.markdown("Nearest Road and Adaptive Radius match the saved JavaScript aggregate results. HMM/Viterbi has a small difference in edge accuracy and trajectory metrics; the current outputs do not contain keyed prediction exports, so exact observation-level causes remain unverified.")
    st.caption(f"Full comparison notes: `{(ROOT / 'reports' / 'phase3_python_js_comparison.md').relative_to(ROOT)}`")


def _sidebar_status() -> None:
    st.sidebar.markdown("### Project status")
    st.sidebar.caption("Synthetic benchmark · Python implementation")
    for label, path in (("Benchmark", DATA_ROOT / "scenario_manifest.csv"),
                        ("Python metrics", PY_METRICS_ROOT / "algorithm_comparison.csv")):
        if path.is_file():
            st.sidebar.success(f"{label} available", icon=":material/check_circle:")
        else:
            st.sidebar.warning(f"{label} unavailable", icon=":material/warning:")
    st.sidebar.caption("Raw and processed source datasets are read-only in this dashboard.")


def main() -> None:
    st.markdown("#### GeoMotion AI  ·  B.Tech AI/ML project")
    st.warning("Synthetic benchmark results · not real-world validation", icon=":material/science:")
    _sidebar_status()
    pages = [
        st.Page(render_overview, title="Overview", icon=":material/home:", default=True),
        st.Page(render_dataset_explorer, title="Dataset Explorer", icon=":material/table_view:"),
        st.Page(render_road_network, title="Road Network", icon=":material/account_tree:"),
        st.Page(render_map_matching, title="Map Matching", icon=":material/alt_route:"),
        st.Page(_render_live_demo, title="Live Map-Matching Demo", icon=":material/my_location:"),
        st.Page(render_algorithm_comparison, title="Algorithm Comparison", icon=":material/analytics:"),
        st.Page(render_results_export, title="Results and Export", icon=":material/download:"),
    ]
    page = st.navigation(pages, position="sidebar")
    page.run()


if __name__ == "__main__":
    main()
