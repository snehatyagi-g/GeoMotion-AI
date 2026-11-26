import test from 'node:test';
import assert from 'node:assert/strict';
import { RoadNetwork } from '../src/road_network.mjs';
import { NearestRoadMatcher } from '../src/nearest_road.mjs';
import { HMMViterbiMatcher } from '../src/hmm_map_matching.mjs';
import { AdaptiveRadiusMatcher, adaptiveSearchRadius } from '../src/adaptive_radius.mjs';
import { calculateMetrics } from '../src/evaluation.mjs';

const origin = { lat: 28.7, lon: 77.2 };
function geo(x, y) {
  return { lat: origin.lat + y / 110540, lon: origin.lon + x / (111320 * Math.cos(origin.lat * Math.PI / 180)) };
}
function row(id, from, to, p1, p2, extra = {}) {
  const a = geo(...p1), b = geo(...p2);
  return {
    scenario_id: 'TEST', edge_id: id, from_node: from, to_node: to,
    from_lat: a.lat, from_lon: a.lon, to_lat: b.lat, to_lon: b.lon,
    highway: 'primary', length_m: Math.hypot(p2[0] - p1[0], p2[1] - p1[1]), oneway: 1, bridge: 0, tunnel: 0, layer: 0,
    geometry_wkt: `LINESTRING (${a.lon} ${a.lat}, ${b.lon} ${b.lat})`, ...extra,
  };
}
function fixture(connected = true) {
  const edges = [
    row('A', 'n0', 'n1', [0, 0], [100, 0]),
    row('B', 'n1', 'n2', [100, 0], [200, 0]),
    row('C', 'n0', 'p0', [0, 0], [0, 20]),
    row('D', 'p0', 'p1', [0, 20], [200, 20]),
    row('E', 'p1', 'n2', [200, 20], [200, 0]),
  ];
  if (!connected) edges[4] = row('E', 'q0', 'q1', [200, 20], [250, 20]);
  return new RoadNetwork('TEST', edges);
}
function observation(x, y, seq, sigma = 5) {
  return { ...geo(x, y), seq, sigma_m: sigma, timestamp: `2026-01-01T00:00:${String(seq).padStart(2, '0')}Z`, speed_kmh: 10, heading_deg: 90 };
}

test('point-to-segment distance is returned in metric metres', () => {
  const network = fixture();
  const candidate = network.distanceToEdge(geo(50, 10), network.edgeById.get('A'));
  assert.ok(candidate.distanceMeters > 9.8 && candidate.distanceMeters < 10.2, `${candidate.distanceMeters} m`);
});

test('Nearest Road returns the nearest valid network edge and searched candidate count', () => {
  const network = fixture();
  const run = new NearestRoadMatcher().match([observation(40, 2, 1)], network);
  assert.equal(run.matches[0].edge_id, 'A');
  assert.equal(run.matches[0].candidate_count, network.edges.length);
  assert.ok(network.edgeById.has(run.matches[0].edge_id));
});

test('HMM/Viterbi returns a connected, ordered road sequence', () => {
  const network = fixture();
  const obs = [observation(30, 2, 1), observation(90, 1, 2), observation(140, 2, 3)];
  const run = new HMMViterbiMatcher().match(obs, network);
  assert.equal(run.matches.length, obs.length);
  assert.deepEqual(run.matches.map(m => m.edge_id), ['A', 'A', 'B']);
  assert.ok(run.matches.every(m => network.edgeById.has(m.edge_id)));
});

test('HMM/Viterbi handles disconnected transitions without throwing', () => {
  const network = fixture(false);
  const obs = [observation(30, 0, 1), observation(225, 20, 2)];
  const run = new HMMViterbiMatcher({ candidateSigmaMultiplier: 1 }).match(obs, network);
  assert.equal(run.matches.length, 2);
  assert.ok(run.matches.every(m => network.edgeById.has(m.edge_id)));
});

test('adaptive radius increases with uncertainty/density and includes close parallel candidates', () => {
  const pars = { sigmaMultiplier: 1.96, densityWeight: 0.35, layerWeight: 0.75, parallelAmbiguitySigma: 1.5 };
  const sparseLow = adaptiveSearchRadius({ sigma: 5, nearestDistance: 2, secondDistance: 40, nearbyCount: 1, layerCount: 1, nearestHeading: 90, secondHeading: 90 }, pars);
  const denseHigh = adaptiveSearchRadius({ sigma: 25, nearestDistance: 6, secondDistance: 15, nearbyCount: 10, layerCount: 2, nearestHeading: 90, secondHeading: 92 }, pars);
  assert.ok(denseHigh.radiusMeters > sparseLow.radiusMeters);
  assert.equal(denseHigh.parallelAmbiguity, true);
  assert.ok(denseHigh.radiusMeters >= 15.5);
});

test('Adaptive Radius matches within an input-derived radius and returns valid edge IDs', () => {
  const network = fixture();
  const obs = [observation(30, 2, 1), observation(90, 1, 2), observation(140, 2, 3)];
  const features = obs.map(() => ({ nearby_edge_count_100m: 3, layer_count_100m: 1 }));
  const run = new AdaptiveRadiusMatcher().match(obs, network, features);
  assert.equal(run.matches.length, obs.length);
  assert.ok(run.matches.every(m => network.edgeById.has(m.edge_id) && m.search_radius_m > 0));
});

test('evaluation metrics are correct for perfect edge predictions', () => {
  const rows = [
    { trajectory_id: 't1', edge_id_true: 'A', edge_id_pred: 'A', matching_distance_m: 1, candidate_count: 2, processing_ms_per_observation: 0.1 },
    { trajectory_id: 't1', edge_id_true: 'B', edge_id_pred: 'B', matching_distance_m: 2, candidate_count: 3, processing_ms_per_observation: 0.2 },
  ];
  const m = calculateMetrics(rows);
  assert.equal(m.edge_accuracy, 1);
  assert.equal(m.edge_f1_macro, 1);
  assert.equal(m.trajectory_exact_match_rate, 1);
  assert.equal(m.mean_matching_distance_m, 1.5);
});

