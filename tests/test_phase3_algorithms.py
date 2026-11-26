"""Focused deterministic unit tests for the Python Phase 3 port."""
import math

from src.adaptive_radius import AdaptiveRadiusMatcher, adaptive_search_radius
from src.evaluation import calculate_metrics
from src.hmm_map_matching import HMMViterbiMatcher
from src.nearest_road import NearestRoadMatcher
from src.road_network import RoadNetwork, circular_difference_deg, haversine_meters


def network():
    rows = [
        {"edge_id": "A", "from_node": "n0", "to_node": "n1", "from_lat": "28.0", "from_lon": "77.0", "to_lat": "28.0", "to_lon": "77.001", "oneway": "1", "layer": "0", "geometry_wkt": "LINESTRING (77.0 28.0, 77.001 28.0)"},
        {"edge_id": "B", "from_node": "n1", "to_node": "n2", "from_lat": "28.0", "from_lon": "77.001", "to_lat": "28.001", "to_lon": "77.001", "oneway": "1", "layer": "0", "geometry_wkt": "LINESTRING (77.001 28.0, 77.001 28.001)"},
    ]
    return RoadNetwork("unit", rows)


def observations():
    return [{"lat": 28.0, "lon": 77.0001, "sigma_m": 5, "heading_deg": 90},
            {"lat": 28.0001, "lon": 77.001, "sigma_m": 5, "heading_deg": 0}]


def test_projected_distances_and_haversine_are_metres():
    net = network()
    assert 9 < net.distance_to_edge({"lat": 28.0001, "lon": 77.0005}, net.edge_by_id["A"])["distanceMeters"] < 12
    assert 9.5 < haversine_meters({"lat": 28, "lon": 77}, {"lat": 28, "lon": 77.0001}) < 10.2


def test_nearest_hmm_and_adaptive_return_valid_network_edges():
    net, obs = network(), observations()
    results = [NearestRoadMatcher().match(obs, net), HMMViterbiMatcher().match(obs, net),
               AdaptiveRadiusMatcher().match(obs, net, [{"nearby_edge_count_100m": 2, "layer_count_100m": 1}] * 2)]
    for result in results:
        assert len(result["matches"]) == len(obs)
        assert all(m["edge_id"] in net.edge_by_id and m["matching_distance_m"] >= 0 for m in result["matches"])


def test_adaptive_radius_includes_close_parallel_candidate():
    result = adaptive_search_radius(sigma=10, nearest_distance=3, second_distance=8, nearby_count=2,
        layer_count=1, nearest_heading=89, second_heading=91,
        parameters={"sigmaMultiplier": 1, "densityWeight": 0, "layerWeight": 0, "parallelAmbiguitySigma": 1})
    assert result["parallelAmbiguity"] is True
    assert result["radiusMeters"] >= 8.5


def test_circular_heading_difference_wraps_at_north():
    assert circular_difference_deg(359, 1) == 2


def test_single_label_micro_metrics_equal_accuracy():
    rows = [{"edge_id_true": "A", "edge_id_pred": "A", "trajectory_id": "t", "matching_distance_m": 2,
             "candidate_count": 1, "processing_ms_per_observation": .01},
            {"edge_id_true": "B", "edge_id_pred": "A", "trajectory_id": "t", "matching_distance_m": 3,
             "candidate_count": 2, "processing_ms_per_observation": .01}]
    metrics = calculate_metrics(rows)
    assert metrics["edge_accuracy"] == metrics["edge_precision_micro"] == metrics["edge_recall_micro"] == metrics["edge_f1_micro"] == .5
    assert math.isclose(metrics["mean_matching_distance_m"], 2.5)
