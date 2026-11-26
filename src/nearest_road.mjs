import { performance } from 'node:perf_hooks';

export class NearestRoadMatcher {
  match(observations, network) {
    const start = performance.now();
    const matches = observations.map(obs => {
      const all = network.candidates(obs);
      const best = all[0];
      return {
        edge_id: best.edgeId,
        matching_distance_m: best.distanceMeters,
        candidate_count: all.length,
        search_radius_m: null,
        _candidate: best,
      };
    });
    return { matches, elapsedMs: performance.now() - start };
  }
}

