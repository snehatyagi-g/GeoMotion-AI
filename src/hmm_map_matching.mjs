import { performance } from 'node:perf_hooks';
import { haversineMeters } from './road_network.mjs';

export const DEFAULT_HMM_PARAMETERS = Object.freeze({ candidateSigmaMultiplier: 3, maxCandidates: 8, transitionBetaMultiplier: 2 });

export class HMMViterbiMatcher {
  constructor(parameters = {}) { this.parameters = { ...DEFAULT_HMM_PARAMETERS, ...parameters }; }

  match(observations, network) {
    const started = performance.now();
    if (!observations.length) return { matches: [], elapsedMs: performance.now() - started };
    const candidatesByPoint = observations.map(obs => {
      const all = network.candidates(obs);
      const radius = Math.max(Number(obs.sigma_m) * this.parameters.candidateSigmaMultiplier, 1.96 * Number(obs.sigma_m));
      let candidates = all.filter(c => c.distanceMeters <= radius);
      if (!candidates.length) candidates = [all[0]]; // Safe nearest-road fallback if the radius returns nothing.
      candidates = candidates.slice(0, this.parameters.maxCandidates);
      return { radius, candidates };
    });

    const scoreRows = [], backPointers = [];
    for (let i = 0; i < observations.length; i++) {
      const candidates = candidatesByPoint[i].candidates;
      const sigma = Math.max(1, Number(observations[i].sigma_m));
      const emissions = candidates.map(c => -0.5 * (c.distanceMeters / sigma) ** 2 - Math.log(sigma));
      if (i === 0) { scoreRows.push(emissions); backPointers.push(candidates.map(() => -1)); continue; }
      const previous = candidatesByPoint[i - 1].candidates;
      const displacement = haversineMeters(observations[i - 1], observations[i]);
      const avgSigma = (Number(observations[i - 1].sigma_m) + Number(observations[i].sigma_m)) / 2;
      const beta = Math.max(10, this.parameters.transitionBetaMultiplier * avgSigma + 0.1 * displacement);
      const rowScores = [], rowBack = [];
      for (let j = 0; j < candidates.length; j++) {
        let bestScore = -Infinity, bestIndex = 0;
        for (let k = 0; k < previous.length; k++) {
          const transition = network.transitionDistance(previous[k], candidates[j]);
          const routeDistance = transition.connected ? transition.distanceMeters : displacement;
          let logTransition = -Math.abs(routeDistance - displacement) / beta;
          if (!transition.connected) logTransition -= 7; // Disconnected transitions are possible but strongly disfavored.
          if (transition.reverse) logTransition -= 2;
          const score = scoreRows[i - 1][k] + logTransition + emissions[j];
          if (score > bestScore) { bestScore = score; bestIndex = k; }
        }
        rowScores.push(bestScore); rowBack.push(bestIndex);
      }
      scoreRows.push(rowScores); backPointers.push(rowBack);
    }

    let state = scoreRows.at(-1).reduce((best, value, i, arr) => value > arr[best] ? i : best, 0);
    const path = Array(observations.length);
    for (let i = observations.length - 1; i >= 0; i--) {
      path[i] = state;
      state = backPointers[i][state];
      if (state < 0 && i > 0) state = 0;
    }
    const matches = path.map((candidateIndex, i) => {
      const item = candidatesByPoint[i].candidates[candidateIndex];
      return {
        edge_id: item.edgeId,
        matching_distance_m: item.distanceMeters,
        candidate_count: candidatesByPoint[i].candidates.length,
        search_radius_m: candidatesByPoint[i].radius,
        _candidate: item,
      };
    });
    return { matches, elapsedMs: performance.now() - started };
  }
}

