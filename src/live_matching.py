"""Input validation and inference helpers for the interactive live-matching page."""
from __future__ import annotations

import math
from typing import Iterable

import pandas as pd

from .adaptive_radius import AdaptiveRadiusMatcher, adaptive_search_radius
from .hmm_map_matching import HMMViterbiMatcher
from .nearest_road import NearestRoadMatcher


def validate_coordinates(latitude: float, longitude: float) -> tuple[float, float]:
    """Return finite WGS84 coordinates or raise a clear ValueError."""
    lat, lon = float(latitude), float(longitude)
    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise ValueError("Latitude and longitude must be finite numbers.")
    if not -90 <= lat <= 90:
        raise ValueError("Latitude must be between -90 and 90 degrees.")
    if not -180 <= lon <= 180:
        raise ValueError("Longitude must be between -180 and 180 degrees.")
    return lat, lon


def validate_observation_rows(rows: Iterable[dict], *, require_timestamp: bool = False) -> list[dict]:
    """Normalize inference observations; no labels or radius targets are accepted."""
    clean = []
    for index, row in enumerate(rows, start=1):
        try:
            lat, lon = validate_coordinates(row["lat"], row["lon"])
            sigma = float(row.get("sigma_m", 10.0))
            heading = float(row.get("heading_deg", 0.0))
            speed = float(row.get("speed_kmh", 0.0))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Observation {index}: {exc}") from exc
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"Observation {index}: GPS accuracy/sigma must be a positive finite number.")
        if not math.isfinite(heading) or not 0 <= heading <= 360:
            raise ValueError(f"Observation {index}: heading must be between 0 and 360 degrees.")
        if not math.isfinite(speed) or speed < 0:
            raise ValueError(f"Observation {index}: speed must be a non-negative finite number.")
        timestamp = row.get("timestamp")
        if require_timestamp and (timestamp is None or not str(timestamp).strip()):
            raise ValueError(f"Observation {index}: timestamp is required.")
        clean.append({
            "lat": lat, "lon": lon, "sigma_m": sigma, "heading_deg": heading,
            "speed_kmh": speed, "timestamp": "" if timestamp is None else str(timestamp),
            "observation_id": str(row.get("observation_id", index)),
        })
    if not clean:
        raise ValueError("No valid GPS observations were supplied.")
    return clean


def prepare_uploaded_observations(
    frame: pd.DataFrame,
    *, latitude_column: str,
    longitude_column: str,
    timestamp_column: str | None = None,
    sigma_column: str | None = None,
    heading_column: str | None = None,
    speed_column: str | None = None,
) -> tuple[list[dict], int]:
    """Map user-selected CSV fields to inference inputs and report rejected rows."""
    if latitude_column not in frame or longitude_column not in frame:
        raise ValueError("Choose existing latitude and longitude columns.")
    indexed_rows = list(enumerate((row for _, row in frame.iterrows()), start=1))
    if timestamp_column:
        parsed_times = pd.to_datetime([row[timestamp_column] for _, row in indexed_rows], errors="coerce", utc=True,
                                      format="mixed")
        if parsed_times.notna().all():
            indexed_rows = [pair for _, pair in sorted(zip(parsed_times, indexed_rows), key=lambda item: item[0])]
    mapped = []
    for source_position, row in indexed_rows:
        mapped.append({
            "observation_id": str(source_position),
            "lat": row[latitude_column], "lon": row[longitude_column],
            "timestamp": row[timestamp_column] if timestamp_column else "",
            "sigma_m": row[sigma_column] if sigma_column else 10.0,
            "heading_deg": row[heading_column] if heading_column else 0.0,
            "speed_kmh": row[speed_column] if speed_column else 0.0,
        })
    accepted, rejected = [], 0
    for row in mapped:
        try:
            accepted.extend(validate_observation_rows([row]))
        except ValueError:
            rejected += 1
    if not accepted:
        raise ValueError("No rows have valid coordinates and supported input values.")
    return accepted, rejected


def inference_features(observations: list[dict], network) -> list[dict]:
    """Derive adaptive matcher features from noisy points and the selected network."""
    features = []
    for observation in observations:
        distances = network.candidates(observation)
        nearby = [candidate for candidate in distances if candidate["distanceMeters"] <= 100]
        features.append({
            "nearby_edge_count_100m": len(nearby),
            "layer_count_100m": len({candidate["edge"]["layer"] for candidate in nearby}),
        })
    return features


def run_matcher(label: str, observations: list[dict], network, *, parameters: dict | None = None):
    """Run one existing GeoMotion matcher against an ordered observation sequence."""
    parameters = parameters or {}
    if label == "Nearest Road":
        return NearestRoadMatcher().match(observations, network)
    if label == "HMM / Viterbi":
        return HMMViterbiMatcher(parameters.get("hmm", {})).match(observations, network)
    if label == "Rule-Based Adaptive Radius":
        features = inference_features(observations, network)
        return AdaptiveRadiusMatcher(parameters.get("adaptive", {})).match(observations, network, features)
    raise ValueError(f"Unsupported map-matching algorithm: {label}")


def adaptive_diagnostics(observation: dict, network, parameters: dict | None = None) -> dict:
    """Explain adaptive radius calculation for one point using inference-time data."""
    candidates = network.candidates(observation)
    nearest = candidates[0]
    second = candidates[1] if len(candidates) > 1 else {
        "distanceMeters": math.inf, "headingDeg": nearest["headingDeg"]
    }
    nearby = [item for item in candidates if item["distanceMeters"] <= 100]
    layers = {item["edge"]["layer"] for item in nearby}
    radius = adaptive_search_radius(
        sigma=float(observation["sigma_m"]),
        nearest_distance=nearest["distanceMeters"], second_distance=second["distanceMeters"],
        nearby_count=len(nearby), layer_count=len(layers),
        nearest_heading=nearest["headingDeg"], second_heading=second["headingDeg"],
        parameters=parameters,
    )
    return {
        "gps_uncertainty_m": float(observation["sigma_m"]),
        "nearby_road_count_100m": len(nearby), "road_layer_count_100m": len(layers),
        "adaptive_search_radius_m": radius["radiusMeters"],
        "candidate_roads_before_filter": len(candidates),
        "candidate_roads_after_filter": max(1, sum(c["distanceMeters"] <= radius["radiusMeters"] for c in candidates)),
        "parallel_road_ambiguity": radius["parallelAmbiguity"],
    }


def point_on_edge(candidate: dict, network) -> tuple[float, float]:
    """Interpolate the snapped point on an edge and return (latitude, longitude)."""
    edge = candidate["edge"]
    along = float(candidate["alongMeters"])
    for segment in edge["segments"]:
        if along <= segment["offset"] + segment["len"] or segment is edge["segments"][-1]:
            fraction = 0 if segment["len"] == 0 else max(0, min(1, (along - segment["offset"]) / segment["len"]))
            x = segment["a"][0] + fraction * (segment["b"][0] - segment["a"][0])
            y = segment["a"][1] + fraction * (segment["b"][1] - segment["a"][1])
            lat = network.origin_lat + y / 110540.0
            lon_scale = 111320.0 * math.cos(math.radians(network.origin_lat))
            lon = network.origin_lon + (x / lon_scale if lon_scale else 0)
            return lat, lon
    return float(edge["from_lat"]), float(edge["from_lon"])
