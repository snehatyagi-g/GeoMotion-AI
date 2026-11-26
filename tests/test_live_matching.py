"""Validation and rendering tests for the synthetic Live Map-Matching Demo."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app import (DATA_ROOT, _live_candidate_table, _live_demo_playback_transition,
                 _live_row_table, _load_live_demo_track)
import app as dashboard
from app import _render_live_map
from src.live_matching import (
    adaptive_diagnostics,
    inference_features,
    point_on_edge,
    prepare_uploaded_observations,
    run_matcher,
    validate_coordinates,
    validate_observation_rows,
)
from src.road_network import RoadNetwork

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def use_project_dataset_root(monkeypatch):
    # Earlier AppTest scripts may assign a temporary DATA_ROOT on the shared app module.
    monkeypatch.setattr(dashboard, "DATA_ROOT", ROOT / "data" / "processed" / "synthetic_v2")


def _sample_track():
    scenario = "SC005"
    obs = pd.read_csv(DATA_ROOT / "test" / "observations" / "traj_dt_120s.csv", keep_default_na=False)
    track = obs.loc[obs["scenario_id"].eq(scenario)].sort_values("seq").head(3)
    observations = validate_observation_rows([{
        "observation_id": row.seq, "lat": row.lat, "lon": row.lon, "timestamp": row.timestamp,
        "sigma_m": row.sigma_m, "heading_deg": row.heading_deg, "speed_kmh": row.speed_kmh,
    } for row in track.itertuples()], require_timestamp=True)
    road_rows = pd.read_csv(DATA_ROOT / "road_network.csv", keep_default_na=False)
    network = RoadNetwork(scenario, road_rows.loc[road_rows["scenario_id"].eq(scenario)].to_dict("records"))
    return scenario, observations, network


def test_coordinate_and_observation_validation_rejects_invalid_values():
    assert validate_coordinates("28.6", "77.2") == (28.6, 77.2)
    with pytest.raises(ValueError, match="Latitude"):
        validate_coordinates(91, 77)
    with pytest.raises(ValueError, match="finite"):
        validate_coordinates(float("nan"), 77)
    with pytest.raises(ValueError, match="positive"):
        validate_observation_rows([{"lat": 28, "lon": 77, "sigma_m": 0}])
    with pytest.raises(ValueError, match="heading"):
        validate_observation_rows([{"lat": 28, "lon": 77, "heading_deg": 400}])


def test_uploaded_csv_mapping_skips_invalid_rows_and_defaults_optional_features():
    frame = pd.DataFrame({"y": [28.6, "bad", None], "x": [77.2, 77.3, 77.4], "time": ["t1", "t2", "t3"]})
    rows, rejected = prepare_uploaded_observations(frame, latitude_column="y", longitude_column="x", timestamp_column="time")
    assert len(rows) == 1 and rejected == 2
    assert rows[0]["sigma_m"] == 10.0
    assert rows[0]["heading_deg"] == 0.0
    with pytest.raises(ValueError, match="No rows"):
        prepare_uploaded_observations(frame.iloc[1:], latitude_column="y", longitude_column="x")
    out_of_order = pd.DataFrame({"lat": [28.61, 28.62], "lon": [77.21, 77.22],
                                 "time": ["2026-01-02T00:00:00Z", "2026-01-01T00:00:00Z"]})
    ordered, _ = prepare_uploaded_observations(out_of_order, latitude_column="lat", longitude_column="lon",
                                                timestamp_column="time")
    assert [point["lat"] for point in ordered] == [28.62, 28.61]


def test_existing_algorithms_run_sequentially_and_return_edges_in_selected_network():
    _, observations, network = _sample_track()
    features = inference_features(observations, network)
    assert len(features) == len(observations)
    for algorithm in ("Nearest Road", "HMM / Viterbi", "Rule-Based Adaptive Radius"):
        run = run_matcher(algorithm, observations, network)
        assert len(run["matches"]) == len(observations)
        assert {match["edge_id"] for match in run["matches"]} <= set(network.edge_by_id)
        assert all(match["matching_distance_m"] >= 0 for match in run["matches"])
        assert all(match["candidate_count"] > 0 for match in run["matches"])
    diagnostic = adaptive_diagnostics(observations[0], network)
    assert diagnostic["adaptive_search_radius_m"] > 0
    assert diagnostic["candidate_roads_after_filter"] <= diagnostic["candidate_roads_before_filter"]
    snapped = point_on_edge(run["matches"][0]["_candidate"], network)
    assert -90 <= snapped[0] <= 90 and -180 <= snapped[1] <= 180


@pytest.mark.parametrize("algorithm", ["Nearest Road", "HMM / Viterbi", "Rule-Based Adaptive Radius"])
def test_offline_map_uses_actual_gps_matched_roads_arrows_and_current_vehicle(algorithm, monkeypatch):
    _, observations, network = _sample_track()
    result = run_matcher(algorithm, observations, network)
    road_rows = pd.read_csv(DATA_ROOT / "road_network.csv", keep_default_na=False)
    road_rows = road_rows.loc[road_rows["scenario_id"].eq(network.scenario_id)]
    captured = []
    monkeypatch.setattr(dashboard.st, "pyplot", lambda figure, **kwargs: captured.append(figure))

    _render_live_map(network, road_rows, observations, result["matches"], 1, algorithm)

    assert len(captured) == 1
    axes = captured[0].axes[0]
    lines = {line.get_gid(): line for line in axes.lines if line.get_gid()}
    assert len(lines["raw-gps-trajectory"].get_xdata()) == len(observations)
    predicted_ids = {match["edge_id"] for match in result["matches"]}
    route_lines = {key.split(":", 1)[1]: line for key, line in lines.items() if key.startswith("matched-road:")}
    assert set(route_lines) == predicted_ids
    actual_by_id = road_rows.set_index("edge_id")["geometry_wkt"].to_dict()
    from app import parse_linestring_wkt
    for edge_id, line in route_lines.items():
        assert len(line.get_xdata()) == len(parse_linestring_wkt(actual_by_id[edge_id]))
    patches = [patch for patch in axes.patches if patch.get_gid()]
    assert any(patch.get_gid() == "vehicle-marker" for patch in patches)
    assert len([patch for patch in patches if patch.get_gid() == "direction-arrow"]) >= 2
    assert {text.get_text() for text in axes.texts} >= {"Raw GPS Trajectory", "Map-Matched Route"}
    assert axes.get_facecolor()[:3] == pytest.approx((16 / 255, 25 / 255, 37 / 255))


def test_single_observation_map_shows_point_and_match_without_trajectory_lines(monkeypatch):
    _, observations, network = _sample_track()
    observation = observations[0]
    result = run_matcher("Nearest Road", [observation], network)
    road_rows = pd.read_csv(DATA_ROOT / "road_network.csv", keep_default_na=False)
    road_rows = road_rows.loc[road_rows["scenario_id"].eq(network.scenario_id)]
    captured = []
    monkeypatch.setattr(dashboard.st, "pyplot", lambda figure, **kwargs: captured.append(figure))

    _render_live_map(network, road_rows, [observation], result["matches"], 0, "Nearest Road")

    axes = captured[0].axes[0]
    assert not any(line.get_gid() == "raw-gps-trajectory" for line in axes.lines)
    assert any(collection.get_offsets().shape[0] == 1 for collection in axes.collections)
    assert any(patch.get_gid() == "vehicle-marker" for patch in axes.patches)
    assert not any(patch.get_gid() == "direction-arrow" for patch in axes.patches)

def test_demo_loads_one_synthetic_trajectory_in_observation_order_without_labels():
    track, observations = _load_live_demo_track("test", 120, "SC005", "SC005_T120")
    assert len(observations) == len(track)
    assert [int(row["observation_id"]) for row in observations] == sorted(
        int(value) for value in track["seq"].tolist()
    )
    assert all(set(row).issubset({"observation_id", "lat", "lon", "timestamp", "sigma_m", "heading_deg", "speed_kmh"})
               for row in observations)
    first = track.iloc[0]
    assert observations[0]["lat"] == float(first["lat"])
    assert observations[0]["lon"] == float(first["lon"])


def test_candidate_table_uses_roadnetwork_distances_and_marks_selected_candidate():
    _, observations, network = _sample_track()
    prediction = run_matcher("Rule-Based Adaptive Radius", observations, network)
    match = prediction["matches"][0]
    table = _live_candidate_table(network, observations[0], match, "Rule-Based Adaptive Radius")
    actual = network.candidates(observations[0])
    actual_by_id = {row["edgeId"]: row["distanceMeters"] for row in actual}
    assert all(row["Road ID"] in actual_by_id for _, row in table.iterrows())
    assert all(abs(row["Distance (m)"] - actual_by_id[row["Road ID"]]) < 1e-9
               for _, row in table.iterrows())
    assert match["edge_id"] in table.loc[table["Candidate role"].eq("Selected match"), "Road ID"].values


def test_live_demo_playback_controls_transition_one_observation_and_stop_at_end():
    assert _live_demo_playback_transition("play", 0, 5) == (0, True)
    assert _live_demo_playback_transition("tick", 0, 5) == (1, True)
    assert _live_demo_playback_transition("pause", 2, 5) == (2, False)
    assert _live_demo_playback_transition("previous", 2, 5) == (1, False)
    assert _live_demo_playback_transition("next", 2, 5) == (3, False)
    assert _live_demo_playback_transition("restart", 4, 5) == (0, False)
    assert _live_demo_playback_transition("tick", 4, 5) == (4, False)
    assert _live_demo_playback_transition("previous", 0, 5) == (0, False)
    assert _live_demo_playback_transition("next", 0, 1) == (0, False)


def test_live_benchmark_evaluation_uses_aligned_labels_after_matching():
    _, observations = _load_live_demo_track("test", 120, "SC005", "SC005_T120")
    network, _ = dashboard._live_network("SC005")
    result = run_matcher("Nearest Road", observations, network)
    metrics = dashboard._evaluate_live_benchmark(
        "test", 120, "SC005", "SC005_T120", observations, result["matches"], result["elapsedMs"], network
    )
    assert metrics is not None
    assert metrics["observations"] == len(observations)
    assert 0 <= metrics["edge_accuracy"] <= 1
    assert 0 <= metrics["edge_f1_macro"] <= 1
    assert metrics["mean_matching_distance_m"] >= 0


def test_external_network_coverage_is_not_assumed_for_unrelated_coordinates():
    _, observations, network = _sample_track()
    from app import _network_covers_points
    assert _network_covers_points(network, observations)
    assert not _network_covers_points(network, [{"lat": 0.0, "lon": 0.0}])


def test_live_matching_export_table_contains_actual_predictions_and_available_metadata():
    _, observations, network = _sample_track()
    result = run_matcher("Nearest Road", observations, network)
    output = _live_row_table(observations, result["matches"], network)
    assert len(output) == len(observations)
    assert output["Latitude"].tolist() == [point["lat"] for point in observations]
    assert output["Matched Road ID"].tolist() == [match["edge_id"] for match in result["matches"]]
    assert "Area / Locality" not in output.columns
    assert "Matched Road Name" not in output.columns
    roundtrip = pd.read_csv(pd.io.common.StringIO(output.to_csv(index=False)))
    assert list(roundtrip.columns) == list(output.columns)
    assert roundtrip["Matched Road ID"].tolist() == output["Matched Road ID"].tolist()
    assert roundtrip["Matching Distance (m)"].ge(0).all()


def test_existing_dashboard_pages_remain_with_one_rebuilt_live_demo_entry():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    expected = {
        "Overview", "Dataset Explorer", "Road Network", "Map Matching",
        "Algorithm Comparison", "Results and Export", "Live Map-Matching Demo",
    }
    assert all(f'title="{page}"' in source for page in expected)
    assert 'title="Vehicle Tracking & Live Map Matching"' not in source
    section = source[source.index("def _render_live_demo()") : source.index("def render_algorithm_comparison()")]
    assert "file_uploader" not in section
    assert "Manual GPS Coordinate Input" not in section
    assert "External CSV Upload" not in section
