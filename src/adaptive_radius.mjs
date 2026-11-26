import { performance } from 'node:perf_hooks';
import { circularDifferenceDeg, haversineMeters } from './road_network.mjs';

export const DEFAULT_ADAPTIVE_PARAMETERS = Object.freeze({
  sigmaMultiplier: 1.96,
  densityWeight: 0.35,
  layerWeight: 0.75,
  parallelAmbiguitySigma: 1.5,
  headingWeight: 0.2,
  transitionWeight: 0.35,
});

export function adaptiveSearchRadius({ sigma, nearestDistance, secondDistance, nearbyCount, layerCount, nearestHeading, secondHeading }, parameters = DEFAULT_ADAPTIVE_PARAMETERS) {
  const densityTerm = Math.log2(1 + Math.max(0, nearbyCount));
  const layerTerm = Math.max(0, layerCount - 1);
  let radius = Math.max(
    nearestDistance + 0.5,
    parameters.sigmaMultiplier * sigma + parameters.densityWeight * sigma * densityTerm + parameters.layerWeight * sigma * layerTerm,
  );
  const gap = Math.max(0, secondDistance - nearestDistance);
  const parallel = Number.isFinite(secondDistance) && gap <= parameters.parallelAmbiguitySigma * sigma &&
    circularDifferenceDeg(nearestHeading, secondHeading) <= 20;
  if (parallel) radius = Math.max(radius, secondDistance + 0.5);
  return { radiusMeters: radius, parallelAmbiguity: parallel, localComplexity: densityTerm + parameters.layerWeight * layerTerm };
}

export class AdaptiveRadiusMatcher {
  constructor(parameters = {}) { this.parameters = { ...DEFAULT_ADAPTIVE_PARAMETERS, ...parameters }; }

  match(observations, network, featureRows = []) {
    const started = performance.now();
    const matches = [];
    let previous = null;
    for (let i = 0; i < observations.length; i++) {
      const obs = observations[i];
      const all = network.candidates(obs);
      const nearest = all[0], second = all[1] ?? { distanceMeters: Infinity, headingDeg: nearest.headingDeg };
      const feature = featureRows[i] ?? {};
      const nearbyCount = Number(feature.nearby_edge_count_100m ?? all.filter(c => c.distanceMeters <= 100).length);
      const layerCount = Number(feature.layer_count_100m ?? new Set(all.filter(c => c.distanceMeters <= 100).map(c => c.edge.layer)).size);
      const radius = adaptiveSearchRadius({
        sigma: Number(obs.sigma_m), nearestDistance: nearest.distanceMeters, secondDistance: second.distanceMeters,
        nearbyCount, layerCount, nearestHeading: nearest.headingDeg, secondHeading: second.headingDeg,
      }, this.parameters);
      let candidates = all.filter(c => c.distanceMeters <= radius.radiusMeters);
      if (!candidates.length) candidates = [nearest]; // The base formula includes the closest road; retain a guard for numeric edge cases.
      const displacement = previous ? haversineMeters(previous.observation, obs) : 0;
      const beta = Math.max(10, (Number(previous?.observation.sigma_m ?? obs.sigma_m) + Number(obs.sigma_m)) / 2 + 0.1 * displacement);
      const scored = candidates.map(c => {
        const distanceCost = 0.5 * (c.distanceMeters / Math.max(1, Number(obs.sigma_m))) ** 2;
        const headingCost = this.parameters.headingWeight * (circularDifferenceDeg(c.headingDeg, obs.heading_deg) / 45) ** 2;
        let continuityCost = 0;
        if (previous) {
          const transition = network.transitionDistance(previous.candidate, c);
          if (transition.connected) continuityCost = this.parameters.transitionWeight * Math.abs(transition.distanceMeters - displacement) / beta + (transition.reverse ? 1.5 : 0);
          else continuityCost = 3 + this.parameters.transitionWeight * displacement / beta;
        }
        return { candidate: c, cost: distanceCost + headingCost + continuityCost };
      }).sort((a, b) => a.cost - b.cost);
      const chosen = scored[0].candidate;
      matches.push({ edge_id: chosen.edgeId, matching_distance_m: chosen.distanceMeters, candidate_count: candidates.length, search_radius_m: radius.radiusMeters, parallel_ambiguity: radius.parallelAmbiguity, _candidate: chosen });
      previous = { candidate: chosen, observation: obs };
    }
    return { matches, elapsedMs: performance.now() - started };
  }
}

