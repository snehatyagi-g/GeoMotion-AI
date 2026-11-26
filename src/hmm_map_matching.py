"""Gaussian-emission, network-transition HMM/Viterbi matcher."""
from time import perf_counter
import math

from .road_network import haversine_meters

DEFAULT_HMM_PARAMETERS = {"candidateSigmaMultiplier": 3, "maxCandidates": 8, "transitionBetaMultiplier": 2}


class HMMViterbiMatcher:
    def __init__(self, parameters=None):
        self.parameters = {**DEFAULT_HMM_PARAMETERS, **(parameters or {})}

    def match(self, observations, network):
        started = perf_counter()
        if not observations:
            return {"matches": [], "elapsedMs": (perf_counter() - started) * 1000}
        states = []
        for obs in observations:
            all_candidates = network.candidates(obs)
            radius = max(float(obs["sigma_m"]) * self.parameters["candidateSigmaMultiplier"], 1.96 * float(obs["sigma_m"]))
            candidates = [c for c in all_candidates if c["distanceMeters"] <= radius]
            if not candidates:
                candidates = [all_candidates[0]]
            states.append({"radius": radius, "candidates": candidates[: int(self.parameters["maxCandidates"])]})
        score_rows, back_pointers = [], []
        for i, obs in enumerate(observations):
            candidates = states[i]["candidates"]
            sigma = max(1.0, float(obs["sigma_m"]))
            emissions = [-0.5 * (c["distanceMeters"] / sigma) ** 2 - math.log(sigma) for c in candidates]
            if i == 0:
                score_rows.append(emissions)
                back_pointers.append([-1] * len(candidates))
                continue
            previous = states[i - 1]["candidates"]
            displacement = haversine_meters(observations[i - 1], obs)
            avg_sigma = (float(observations[i - 1]["sigma_m"]) + float(obs["sigma_m"])) / 2
            beta = max(10.0, self.parameters["transitionBetaMultiplier"] * avg_sigma + 0.1 * displacement)
            row_scores, row_back = [], []
            for candidate in candidates:
                best_score, best_index = -math.inf, 0
                for k, previous_candidate in enumerate(previous):
                    transition = network.transition_distance(previous_candidate, candidate)
                    route_distance = transition["distanceMeters"] if transition["connected"] else displacement
                    log_transition = -abs(route_distance - displacement) / beta
                    if not transition["connected"]:
                        log_transition -= 7
                    if transition["reverse"]:
                        log_transition -= 2
                    score = score_rows[i - 1][k] + log_transition + emissions[len(row_scores)]
                    if score > best_score:
                        best_score, best_index = score, k
                row_scores.append(best_score)
                row_back.append(best_index)
            score_rows.append(row_scores)
            back_pointers.append(row_back)
        state = max(range(len(score_rows[-1])), key=score_rows[-1].__getitem__)
        path = [0] * len(observations)
        for i in range(len(observations) - 1, -1, -1):
            path[i] = state
            state = back_pointers[i][state]
            if state < 0 and i > 0:
                state = 0
        matches = []
        for i, candidate_index in enumerate(path):
            item = states[i]["candidates"][candidate_index]
            matches.append({"edge_id": item["edgeId"], "matching_distance_m": item["distanceMeters"],
                            "candidate_count": len(states[i]["candidates"]), "search_radius_m": states[i]["radius"],
                            "_candidate": item})
        return {"matches": matches, "elapsedMs": (perf_counter() - started) * 1000}
