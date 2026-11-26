"""Read-only functional audit of GeoMotion AI dashboard data and interactions."""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from app import ALGORITHMS, DATA_ROOT, INTERVALS, PY_METRICS_ROOT, load_aligned_track, parse_linestring_wkt

ROOT = Path(__file__).resolve().parents[1]
PAGE = "import app\napp.{}()"
DATASET_CHOICES = [
    "Noisy GPS observations", "Inference-time features", "Ground-truth labels",
    "Radius-label diagnostics", "Road network", "Road-context features", "Scenario manifest",
]


def _expected_explorer_path(kind, split="test", interval=1):
    if kind == "Noisy GPS observations":
        return DATA_ROOT / split / "observations" / f"traj_dt_{interval}s.csv"
    if kind == "Inference-time features":
        return DATA_ROOT / split / "features" / f"features_dt_{interval}s.csv"
    if kind == "Ground-truth labels":
        return DATA_ROOT / split / "ground_truth" / f"ground_truth_dt_{interval}s.csv"
    if kind == "Radius-label diagnostics":
        return DATA_ROOT / split / "radius_labels" / f"radius_labels_dt_{interval}s.csv"
    if kind == "Road network":
        return DATA_ROOT / "road_network.csv"
    if kind == "Road-context features":
        return DATA_ROOT / "road_context_features.csv"
    return DATA_ROOT / "scenario_manifest.csv"


AUDIT_PREFIX = '''
import streamlit as st
import app
st.session_state["_dashboard_audit"] = {"maps": [], "charts": [], "downloads": []}
_original_map = st.pydeck_chart
_original_bar = st.bar_chart
_original_download = st.download_button
def _map_audit(deck, *args, **kwargs):
    layers = []
    for layer in deck.layers:
        data = layer.data
        rows = data.to_dict("records") if hasattr(data, "to_dict") else list(data)
        layers.append({"type": layer.type, "rows": rows})
    st.session_state["_dashboard_audit"]["maps"].append(layers)
    return _original_map(deck, *args, **kwargs)
def _bar_audit(*args, **kwargs):
    data = args[0] if args else kwargs.get("data")
    st.session_state["_dashboard_audit"]["charts"].append(data.copy() if hasattr(data, "copy") else data)
    return _original_bar(*args, **kwargs)
def _download_audit(*args, **kwargs):
    label = args[0] if args else kwargs.get("label")
    payload = args[1] if len(args) > 1 else kwargs.get("data")
    filename = kwargs.get("file_name")
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    st.session_state["_dashboard_audit"]["downloads"].append({"label": label, "file_name": filename, "payload": payload})
    return _original_download(*args, **kwargs)
st.pydeck_chart = _map_audit
st.bar_chart = _bar_audit
st.download_button = _download_audit
try:
    PAGE_CALL
finally:
    st.pydeck_chart = _original_map
    st.bar_chart = _original_bar
    st.download_button = _original_download
'''


def _run_page(function):
    script = AUDIT_PREFIX.replace("PAGE_CALL", f"app.{function}()")
    return AppTest.from_string(script, default_timeout=45).run()


def test_every_synthetic_v2_csv_has_rows_headers_and_expected_schema():
    files = sorted(DATA_ROOT.rglob("*.csv"))
    assert len(files) == 91
    for path in files:
        frame = pd.read_csv(path, keep_default_na=False)
        assert len(frame) > 0, f"No records in {path.relative_to(DATA_ROOT)}"
        assert len(frame.columns) > 0 and all(str(column).strip() for column in frame.columns)
        if "observations" in path.parts:
            assert {"lat", "lon", "sigma_m", "heading_deg"}.issubset(frame.columns)
        elif "features" in path.parts:
            assert {"nearby_edge_count_100m", "layer_count_100m"}.issubset(frame.columns)
        elif "ground_truth" in path.parts:
            assert "edge_id_true" in frame.columns
        elif "radius_labels" in path.parts:
            assert {"recommended_radius_m", "min_ok_radius_m", "max_ok_radius_m"}.issubset(frame.columns)
        elif path.name == "road_network.csv":
            assert {"scenario_id", "edge_id", "geometry_wkt"}.issubset(frame.columns)


def test_dataset_explorer_loads_all_seven_table_selectors_and_download_controls():
    app = _run_page("render_dataset_explorer")
    assert not app.exception
    selector = app.selectbox(key="explorer_table")
    assert selector.options == DATASET_CHOICES
    for kind in DATASET_CHOICES:
        app.selectbox(key="explorer_table").select(kind).run()
        assert not app.exception, f"{kind}: {app.exception}"
        expected = pd.read_csv(_expected_explorer_path(kind), keep_default_na=False)
        visible = app.dataframe[0].value
        assert list(visible.columns) == list(expected.columns), kind
        assert len(visible) > 0, kind
        assert len(app.download_button) == 1, kind
        assert app.download_button[0].label == "Download selected table as CSV"
        download = app.session_state["_dashboard_audit"]["downloads"][-1]
        downloaded = pd.read_csv(BytesIO(download["payload"]), keep_default_na=False)
        assert list(downloaded.columns) == list(expected.columns), kind
        assert len(downloaded) == len(expected), kind

    app.selectbox(key="explorer_table").select("Noisy GPS observations").run()
    app.selectbox(key="explorer_preview_rows").select(100).run()
    scenario_id = str(app.selectbox(key="explorer_scenario").options[1])
    app.selectbox(key="explorer_scenario").select(scenario_id).run()
    assert not app.exception
    filtered = pd.read_csv(_expected_explorer_path("Noisy GPS observations"), keep_default_na=False)
    filtered = filtered.loc[filtered["scenario_id"].eq(scenario_id)]
    assert len(app.dataframe[0].value) == min(100, len(filtered))
    exported = pd.read_csv(BytesIO(app.session_state["_dashboard_audit"]["downloads"][-1]["payload"]), keep_default_na=False)
    assert len(exported) == len(filtered)
    assert set(exported["scenario_id"]) == {scenario_id}


def test_dataset_explorer_split_and_interval_filters_select_correct_records():
    app = _run_page("render_dataset_explorer")
    for split in ("train", "validation", "test"):
        app.selectbox(key="explorer_split").select(split).run()
        for interval in INTERVALS:
            app.selectbox(key="explorer_interval").select(interval).run()
            assert not app.exception
            expected = pd.read_csv(_expected_explorer_path("Noisy GPS observations", split, interval), keep_default_na=False)
            expected_count = len(expected)
            records_metric = next(metric for metric in app.metric if metric.label == "Records")
            assert int(records_metric.value.replace(",", "")) == expected_count
            assert list(app.dataframe[0].value.columns) == list(expected.columns)


def test_all_road_network_scenarios_render_maps_and_road_class_filter_works():
    manifest = pd.read_csv(DATA_ROOT / "scenario_manifest.csv", keep_default_na=False)
    assert len(manifest) == 48
    app = _run_page("render_road_network")
    assert not app.exception
    assert len(app.session_state["_dashboard_audit"]["maps"]) == 1
    labels = [f"{r.scenario_id} · {r.scenario_type} · {r.split}" for r in manifest.itertuples()]
    for label in labels:
        app.selectbox(key="network_scenario").select(label).run()
        assert not app.exception, f"Scenario {label}: {app.exception}"
        assert len(app.session_state["_dashboard_audit"]["maps"]) == 1, label
        assert len(app.dataframe[0].value) > 0, label
        captured = app.session_state["_dashboard_audit"]["maps"][0]
        assert captured and sum(len(layer["rows"]) for layer in captured) > 0
    classes = app.multiselect(key="network_types").options
    if classes:
        app.multiselect(key="network_types").set_value([classes[0]]).run()
        assert not app.exception
        selected_frame = app.dataframe[0].value
        assert set(selected_frame["highway"].astype(str)) <= {str(classes[0])}
        assert len(app.session_state["_dashboard_audit"]["maps"]) == 1


def test_all_map_matching_intervals_produce_three_valid_edge_sequences():
    app = _run_page("render_map_matching")
    assert not app.exception
    assert not app.session_state["_dashboard_audit"]["maps"]  # map is rendered after explicit run only
    split = "test"
    first_scenario = pd.read_csv(DATA_ROOT / "scenario_manifest.csv", keep_default_na=False).query("split == @split").iloc[0].scenario_id
    network_edges = set(pd.read_csv(DATA_ROOT / "road_network.csv", keep_default_na=False)
                        .query("scenario_id == @first_scenario")["edge_id"].astype(str))
    app.selectbox(key="matching_scenario").select(first_scenario).run()
    for interval in INTERVALS:
        app.selectbox(key="matching_interval").select(interval).run()
        assert not app.exception
        app.button(key="run_matchers").click().run()
        assert not app.exception, f"{interval}s matching: {app.exception}"
        assert len(app.session_state["_dashboard_audit"]["maps"]) == 1
        result = app.session_state["dashboard_match_result"]
        assert result["key"] == ("test", interval, first_scenario, result["key"][3])
        assert result["coords"] and len(result["coords"]) == len(result["metadata"])
        rows = result["comparison"]
        assert set(rows["algorithm"]) == set(ALGORITHMS)
        counts = rows.groupby("algorithm").size()
        assert counts.nunique() == 1
        for algorithm in ALGORITHMS:
            algorithm_rows = rows.loc[rows["algorithm"].eq(algorithm)]
            predicted = set(algorithm_rows["edge_id_pred"].astype(str))
            assert predicted and predicted.issubset(network_edges), f"{algorithm}, {interval}s has unresolved edges"
            assert (algorithm_rows["matching_distance_m"] >= 0).all()
            assert (algorithm_rows["candidate_count"] > 0).all()


def test_map_matching_split_and_scenario_controls_align_and_render():
    app = _run_page("render_map_matching")
    manifest = pd.read_csv(DATA_ROOT / "scenario_manifest.csv", keep_default_na=False)
    for split in ("train", "validation", "test"):
        scenario = manifest.loc[manifest["split"].eq(split), "scenario_id"].iloc[-1]
        app.selectbox(key="matching_split").select(split).run()
        assert not app.exception
        app.selectbox(key="matching_scenario").select(scenario).run()
        app.selectbox(key="matching_interval").select(120).run()
        app.button(key="run_matchers").click().run()
        assert not app.exception
        assert app.session_state["dashboard_match_result"]["key"][0:3] == (split, 120, scenario)
        assert len(app.session_state["_dashboard_audit"]["maps"]) == 1


def test_comparison_metrics_and_charts_match_saved_python_results():
    saved = pd.read_csv(PY_METRICS_ROOT / "algorithm_comparison.csv", keep_default_na=False)
    app = _run_page("render_algorithm_comparison")
    assert not app.exception
    assert len(app.session_state["_dashboard_audit"]["charts"]) == 1
    chart_data = app.session_state["_dashboard_audit"]["charts"][-1]
    assert list(chart_data.columns) == ["edge_accuracy"]
    assert list(chart_data.index) == ALGORITHMS
    assert chart_data["edge_accuracy"].tolist() == saved.loc[saved["scope"].eq("overall")].set_index("algorithm").loc[ALGORITHMS, "edge_accuracy"].tolist()
    overall = app.dataframe[0].value
    expected = saved.loc[saved["scope"].eq("overall")].reset_index(drop=True)
    pd.testing.assert_frame_equal(overall.reset_index(drop=True), expected.loc[:, overall.columns],
                                  check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    for choice, scope, key, value in (
        ("Road context", "road_context", "comparison_context", "urban_canyon"),
        ("GPS noise tier", "gps_noise", "comparison_noise", "high"),
        ("Sampling interval", "sampling_interval", "comparison_interval", "120"),
    ):
        app.selectbox(key="comparison_scope").select(choice).run()
        app.selectbox(key=key).select(value).run()
        assert not app.exception
        assert len(app.session_state["_dashboard_audit"]["charts"]) == 1
        display = app.dataframe[0].value
        assert set(display["algorithm"]) == set(ALGORITHMS)
        chart_data = app.session_state["_dashboard_audit"]["charts"][-1]
        chart_values = chart_data["edge_accuracy"].to_dict()
        assert chart_values == dict(zip(display["algorithm"], display["edge_accuracy"], strict=True))
        expected_rows = saved.loc[saved["scope"].eq(scope)]
        if scope == "road_context":
            expected_rows = expected_rows.loc[expected_rows["context"].eq(value)]
        elif scope == "gps_noise":
            expected_rows = expected_rows.loc[expected_rows["noise_tier"].eq(value)]
        else:
            expected_rows = expected_rows.loc[expected_rows["sampling_interval_s"].astype(str).eq(value)]
        assert display["observations"].tolist() == expected_rows["observations"].tolist()
        pd.testing.assert_series_equal(display["edge_accuracy"].reset_index(drop=True), expected_rows["edge_accuracy"].reset_index(drop=True),
                                       check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)


def test_results_exports_match_saved_csv_json_and_figures():
    saved_csv = pd.read_csv(PY_METRICS_ROOT / "algorithm_comparison.csv", keep_default_na=False)
    saved_json = json.loads((PY_METRICS_ROOT / "detailed_metrics.json").read_text(encoding="utf-8"))
    app = _run_page("render_results_export")
    assert not app.exception
    labels = {button.label for button in app.download_button}
    assert "Download Python metrics CSV" in labels
    assert "Download detailed metrics JSON" in labels
    assert "Download this metric view as CSV" in labels
    assert len(app.image) == 3
    displayed = app.dataframe[0].value
    pd.testing.assert_frame_equal(displayed.reset_index(drop=True), saved_csv.loc[saved_csv["scope"].eq("overall")].reset_index(drop=True),
                                  check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    downloads = {item["label"]: item["payload"] for item in app.session_state["_dashboard_audit"]["downloads"]}
    exported_csv = pd.read_csv(BytesIO(downloads["Download Python metrics CSV"]), keep_default_na=False)
    pd.testing.assert_frame_equal(exported_csv, saved_csv, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    exported_json = json.loads(downloads["Download detailed metrics JSON"].decode("utf-8"))
    assert exported_json == saved_json
    assert saved_json["unique_eligible_test_keys"] == 12944
    assert saved_json["test_results"]["overall"]
    assert "HMM" in " ".join(app.markdown[i].value for i in range(len(app.markdown)))
    app.selectbox(key="export_scope").select("sampling_interval").run()
    export_scope = pd.read_csv(BytesIO(app.session_state["_dashboard_audit"]["downloads"][-1]["payload"]), keep_default_na=False)
    expected_scope = saved_csv.loc[saved_csv["scope"].eq("sampling_interval")].reset_index(drop=True)
    assert set(export_scope["scope"]) == {"sampling_interval"}
    pd.testing.assert_frame_equal(export_scope.astype(str).reset_index(drop=True), expected_scope.astype(str), check_dtype=False,
                                  check_exact=False, rtol=1e-12, atol=1e-12)


def test_network_geometry_and_selected_truth_resolve_for_each_split():
    manifest = pd.read_csv(DATA_ROOT / "scenario_manifest.csv", keep_default_na=False)
    for split in ("train", "validation", "test"):
        scenario = manifest.loc[manifest["split"].eq(split), "scenario_id"].iloc[0]
        for interval in (1, 120):
            trajectory = pd.read_csv(DATA_ROOT / split / "observations" / f"traj_dt_{interval}s.csv", keep_default_na=False)
            trajectory = trajectory.loc[trajectory["scenario_id"].eq(scenario)].iloc[0]
            network, observations, features, truth_ids, coords, metadata = load_aligned_track(
                split, interval, scenario, trajectory.trajectory_id)
            assert len(observations) == len(features) == len(truth_ids) == len(coords) == len(metadata)
            assert set(truth_ids).issubset(network.edge_by_id)
            assert all(len(parse_linestring_wkt(edge["geometry_wkt"])) >= 2 for edge in network.edges)


def test_missing_data_is_reported_instead_of_leaving_road_or_explorer_pages_blank(tmp_path):
    missing_root = str(tmp_path / "missing_synthetic_v2")
    road = AppTest.from_string(
        f"import app\nfrom pathlib import Path\napp.DATA_ROOT=Path({missing_root!r})\napp.render_road_network()",
        default_timeout=30,
    ).run()
    assert not road.exception
    assert road.error and "scenario_manifest.csv" in road.error[0].value
    explorer = AppTest.from_string(
        f"import app\nfrom pathlib import Path\napp.DATA_ROOT=Path({missing_root!r})\napp.render_dataset_explorer()",
        default_timeout=30,
    ).run()
    assert not explorer.exception
    assert explorer.error and "Required CSV file was not found" in explorer.error[0].value

    missing_results = str(tmp_path / "missing_python_metrics")
    comparison = AppTest.from_string(
        f"import app\nfrom pathlib import Path\napp.PY_METRICS_ROOT=Path({missing_results!r})\napp.render_algorithm_comparison()",
        default_timeout=30,
    ).run()
    assert not comparison.exception
    assert comparison.error and "Python experiment results are unavailable" in comparison.error[0].value
    results = AppTest.from_string(
        f"import app\nfrom pathlib import Path\napp.PY_METRICS_ROOT=Path({missing_results!r})\napp.render_results_export()",
        default_timeout=30,
    ).run()
    assert not results.exception
    assert results.error and "Required CSV file was not found" in results.error[0].value
