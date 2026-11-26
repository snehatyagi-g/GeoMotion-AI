"""Deterministic inference-time adaptive candidate search and scoring."""
from time import perf_counter
import math

from .road_network import circular_difference_deg, haversine_meters

DEFAULT_ADAPTIVE_PARAMETERS = {"sigmaMultiplier": 1.96, "densityWeight": 0.35, "layerWeight": 0.75,
                               "parallelAmbiguitySigma": 1.5, "headingWeight": 0.2, "transitionWeight": 0.35}


def adaptive_search_radius(*, sigma, nearest_distance, second_distance, nearby_count, layer_count,
                           nearest_heading, second_heading, parameters=None):
    p = {**DEFAULT_ADAPTIVE_PARAMETERS, **(parameters or {})}
    density_term = math.log2(1 + max(0, nearby_count))
    layer_term = max(0, layer_count - 1)
    radius = max(nearest_distance + 0.5, p["sigmaMultiplier"] * sigma + p["densityWeight"] * sigma * density_term
                 + p["layerWeight"] * sigma * layer_term)
    gap = max(0.0, second_distance - nearest_distance)
    parallel = (math.isfinite(second_distance) and gap <= p["parallelAmbiguitySigma"] * sigma
                and circular_difference_deg(nearest_heading, second_heading) <= 20)
    if parallel:
        radius = max(radius, second_distance + 0.5)
    return {"radiusMeters": radius, "parallelAmbiguity": parallel,
            "localComplexity": density_term + p["layerWeight"] * layer_term}


class AdaptiveRadiusMatcher:
    def __init__(self, parameters=None):
        self.parameters = {**DEFAULT_ADAPTIVE_PARAMETERS, **(parameters or {})}

    def match(self, observations, network, feature_rows=None):
        started = perf_counter()
        feature_rows = feature_rows or []
        matches, previous = [], None
        for i, obs in enumerate(observations):
            all_candidates = network.candidates(obs)
            nearest = all_candidates[0]
            second = all_candidates[1] if len(all_candidates) > 1 else {"distanceMeters": math.inf, "headingDeg": nearest["headingDeg"]}
            feature = feature_rows[i] if i < len(feature_rows) else {}
            nearby_count = int(float(feature.get("nearby_edge_count_100m", sum(c["distanceMeters"] <= 100 for c in all_candidates))))
            layer_count = int(float(feature.get("layer_count_100m", len({c["edge"]["layer"] for c in all_candidates if c["distanceMeters"] <= 100}))))
            radius = adaptive_search_radius(sigma=float(obs["sigma_m"]), nearest_distance=nearest["distanceMeters"],
                second_distance=second["distanceMeters"], nearby_count=nearby_count, layer_count=layer_count,
                nearest_heading=nearest["headingDeg"], second_heading=second["headingDeg"], parameters=self.parameters)
            candidates = [c for c in all_candidates if c["distanceMeters"] <= radius["radiusMeters"]] or [nearest]
            displacement = haversine_meters(previous["observation"], obs) if previous else 0.0
            previous_sigma = float(previous["observation"]["sigma_m"]) if previous else float(obs["sigma_m"])
            beta = max(10.0, (previous_sigma + float(obs["sigma_m"])) / 2 + 0.1 * displacement)
            scored = []
            for candidate in candidates:
                distance_cost = 0.5 * (candidate["distanceMeters"] / max(1.0, float(obs["sigma_m"]))) ** 2
                heading_cost = self.parameters["headingWeight"] * (circular_difference_deg(candidate["headingDeg"], obs["heading_deg"]) / 45) ** 2
                continuity_cost = 0.0
                if previous:
                    transition = network.transition_distance(previous["candidate"], candidate)
                    if transition["connected"]:
                        continuity_cost = self.parameters["transitionWeight"] * abs(transition["distanceMeters"] - displacement) / beta
                        if transition["reverse"]:
                            continuity_cost += 1.5
                    else:
                        continuity_cost = 3 + self.parameters["transitionWeight"] * displacement / beta
                scored.append((distance_cost + heading_cost + continuity_cost, candidate))
            chosen = min(scored, key=lambda item: item[0])[1]
            matches.append({"edge_id": chosen["edgeId"], "matching_distance_m": chosen["distanceMeters"],
                            "candidate_count": len(candidates), "search_radius_m": radius["radiusMeters"],
                            "parallel_ambiguity": radius["parallelAmbiguity"], "_candidate": chosen})
            previous = {"candidate": chosen, "observation": obs}
        return {"matches": matches, "elapsedMs": (perf_counter() - started) * 1000}
