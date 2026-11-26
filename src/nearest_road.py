"""Nearest road segment baseline."""
from time import perf_counter


class NearestRoadMatcher:
    def match(self, observations, network):
        start = perf_counter()
        matches = []
        for obs in observations:
            all_candidates = network.candidates(obs)
            best = all_candidates[0]
            matches.append({"edge_id": best["edgeId"], "matching_distance_m": best["distanceMeters"],
                            "candidate_count": len(all_candidates), "search_radius_m": None, "_candidate": best})
        return {"matches": matches, "elapsedMs": (perf_counter() - start) * 1000}
