import path from 'node:path';
import { readCsv } from './csv_utils.mjs';
import { RoadNetwork } from './road_network.mjs';

function key(row) { return `${row.scenario_id}|${row.trajectory_id}|${row.sampling_interval_s}|${row.seq}|${row.timestamp}`; }

function indexUnique(rows, name) {
  const map = new Map();
  for (const row of rows) {
    const k = key(row);
    if (map.has(k)) throw new Error(`Duplicate row key in ${name}: ${k}`);
    map.set(k, row);
  }
  return map;
}

export function loadScenarioNetworks(processedRoot, allowedScenarioIds = null) {
  const rows = readCsv(path.join(processedRoot, 'road_network.csv'));
  const grouped = new Map();
  for (const row of rows) {
    if (!row.scenario_id.startsWith('SC')) continue;
    if (allowedScenarioIds && !allowedScenarioIds.has(row.scenario_id)) continue;
    if (!grouped.has(row.scenario_id)) grouped.set(row.scenario_id, []);
    grouped.get(row.scenario_id).push(row);
  }
  return new Map([...grouped].map(([id, edges]) => [id, new RoadNetwork(id, edges)]));
}

export function loadSplitTracks(processedRoot, split, intervals, networkByScenario, { requireLabels = true } = {}) {
  const manifestRows = readCsv(path.join(processedRoot, 'scenario_manifest.csv')).filter(r => r.split === split);
  const manifest = new Map(manifestRows.map(r => [r.scenario_id, r]));
  const tracks = [];
  const eligibleKeys = new Set();
  for (const interval of intervals) {
    const prefix = `dt_${interval}s.csv`;
    const observations = readCsv(path.join(processedRoot, split, 'observations', `traj_${prefix}`));
    const features = readCsv(path.join(processedRoot, split, 'features', `features_${prefix}`));
    const gt = requireLabels ? readCsv(path.join(processedRoot, split, 'ground_truth', `ground_truth_${prefix}`)) : [];
    const obsByKey = indexUnique(observations, `observations/${split}/${interval}s`);
    const featByKey = indexUnique(features, `features/${split}/${interval}s`);
    const gtByKey = requireLabels ? indexUnique(gt, `ground_truth/${split}/${interval}s`) : new Map();
    if (obsByKey.size !== featByKey.size || (requireLabels && obsByKey.size !== gtByKey.size)) throw new Error(`Table row count mismatch for ${split}/${interval}s`);
    const grouped = new Map();
    for (const [k, obs] of obsByKey) {
      const feature = featByKey.get(k), truth = gtByKey.get(k);
      if (!feature || (requireLabels && !truth)) throw new Error(`Missing aligned row for key ${k}`);
      if (obs.split !== split || feature.split !== split || (truth && truth.split !== split)) throw new Error(`Split mismatch for ${k}`);
      const scenario = manifest.get(obs.scenario_id);
      if (!scenario) throw new Error(`Scenario ${obs.scenario_id} not in ${split} manifest`);
      const network = networkByScenario.get(obs.scenario_id);
      if (!network) throw new Error(`No scenario network for ${obs.scenario_id}`);
      if (truth && !network.edgeById.has(truth.edge_id_true)) throw new Error(`Unresolved truth edge ${truth.edge_id_true} for ${k}`);
      if (!grouped.has(obs.trajectory_id)) grouped.set(obs.trajectory_id, { scenario, network, interval, observations: [], features: [], groundTruth: [] });
      const track = grouped.get(obs.trajectory_id);
      track.observations.push(obs); track.features.push(feature);
      if (truth) track.groundTruth.push(truth);
      eligibleKeys.add(k);
    }
    for (const [trajectoryId, track] of grouped) {
      track.trajectoryId = trajectoryId;
      const order = track.observations.map((_, i) => i).sort((a, b) => Number(track.observations[a].seq) - Number(track.observations[b].seq));
      for (const field of ['observations', 'features', 'groundTruth']) if (track[field].length) track[field] = order.map(i => track[field][i]);
      tracks.push(track);
    }
  }
  return { tracks, eligibleKeys };
}

export function evaluationRows(track, predictions, algorithm, elapsedMs) {
  if (track.groundTruth.length !== predictions.matches.length) throw new Error(`Prediction/GT length mismatch: ${track.trajectoryId}`);
  const perObsMs = track.observations.length ? elapsedMs / track.observations.length : 0;
  return predictions.matches.map((match, i) => ({
    algorithm,
    scenario_id: track.scenario.scenario_id,
    trajectory_id: track.trajectoryId,
    split: track.scenario.split,
    scenario_type: track.scenario.scenario_type,
    gps_noise_tier: track.scenario.gps_noise_tier,
    sampling_interval_s: Number(track.interval),
    seq: Number(track.observations[i].seq),
    timestamp: track.observations[i].timestamp,
    edge_id_true: track.groundTruth[i].edge_id_true,
    edge_id_pred: match.edge_id,
    matching_distance_m: match.matching_distance_m,
    candidate_count: match.candidate_count,
    search_radius_m: match.search_radius_m,
    processing_ms_per_observation: perObsMs,
  }));
}

