function mean(values) { return values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0; }

export function trajectoryAccuracyScore(rows) {
  const groups = new Map();
  for (const r of rows) {
    if (!groups.has(r.trajectory_id)) groups.set(r.trajectory_id, []);
    groups.get(r.trajectory_id).push(r);
  }
  return mean([...groups.values()].map(track => mean(track.map(r => r.edge_id_pred === r.edge_id_true ? 1 : 0))));
}

export function calculateMetrics(rows) {
  const classes = new Set();
  const byTrue = new Map(), byPred = new Map(), truePred = new Map();
  let correct = 0;
  for (const r of rows) {
    classes.add(r.edge_id_true); classes.add(r.edge_id_pred);
    if (r.edge_id_true === r.edge_id_pred) correct++;
    byTrue.set(r.edge_id_true, (byTrue.get(r.edge_id_true) ?? 0) + 1);
    byPred.set(r.edge_id_pred, (byPred.get(r.edge_id_pred) ?? 0) + 1);
    const key = `${r.edge_id_true}\u0000${r.edge_id_pred}`;
    truePred.set(key, (truePred.get(key) ?? 0) + 1);
  }
  const macroP = [], macroR = [], macroF = [];
  for (const id of classes) {
    const tp = truePred.get(`${id}\u0000${id}`) ?? 0;
    const fp = (byPred.get(id) ?? 0) - tp;
    const fn = (byTrue.get(id) ?? 0) - tp;
    const p = tp + fp ? tp / (tp + fp) : 0;
    const r = tp + fn ? tp / (tp + fn) : 0;
    macroP.push(p); macroR.push(r); macroF.push(p + r ? 2 * p * r / (p + r) : 0);
  }
  const tracks = new Map();
  for (const r of rows) {
    if (!tracks.has(r.trajectory_id)) tracks.set(r.trajectory_id, []);
    tracks.get(r.trajectory_id).push(r);
  }
  const perTrackAcc = [...tracks.values()].map(track => mean(track.map(r => r.edge_id_pred === r.edge_id_true ? 1 : 0)));
  const exactTracks = [...tracks.values()].filter(track => track.every(r => r.edge_id_pred === r.edge_id_true)).length;
  const acc = rows.length ? correct / rows.length : 0;
  return {
    observations: rows.length,
    trajectories: tracks.size,
    edge_accuracy: acc,
    edge_precision_micro: acc,
    edge_recall_micro: acc,
    edge_f1_micro: acc,
    edge_precision_macro: mean(macroP),
    edge_recall_macro: mean(macroR),
    edge_f1_macro: mean(macroF),
    mean_matching_distance_m: mean(rows.map(r => r.matching_distance_m)),
    trajectory_accuracy_mean: mean(perTrackAcc),
    trajectory_exact_match_rate: tracks.size ? exactTracks / tracks.size : 0,
    candidate_count_mean: mean(rows.map(r => r.candidate_count)),
    processing_ms_per_observation: mean(rows.map(r => r.processing_ms_per_observation)),
  };
}

function summarizeGroup(rows, algorithm, scope, groupKey = null) {
  const selected = rows.filter(r => r.algorithm === algorithm && (!groupKey || groupKey.predicate(r)));
  if (!selected.length) return null;
  const context = groupKey?.context ?? '';
  const noise = groupKey?.noise ?? '';
  const interval = groupKey?.interval ?? '';
  return { algorithm, scope, context, noise_tier: noise, sampling_interval_s: interval, ...calculateMetrics(selected) };
}

export function buildMetricTables(rows, algorithms) {
  const records = [];
  const overall = [], byContext = [], byNoise = [], byInterval = [];
  for (const algorithm of algorithms) {
    const o = summarizeGroup(rows, algorithm, 'overall'); records.push(o); overall.push({ algorithm, metrics: calculateMetrics(rows.filter(r => r.algorithm === algorithm)) });
    const contexts = [...new Set(rows.map(r => r.scenario_type))].sort();
    for (const context of contexts) {
      const r = summarizeGroup(rows, algorithm, 'road_context', { context, predicate: row => row.scenario_type === context });
      records.push(r); byContext.push({ context, algorithm, metrics: calculateMetrics(rows.filter(row => row.algorithm === algorithm && row.scenario_type === context)) });
    }
    const noises = [...new Set(rows.map(r => r.gps_noise_tier))].sort();
    for (const noise of noises) {
      const r = summarizeGroup(rows, algorithm, 'gps_noise', { noise, predicate: row => row.gps_noise_tier === noise });
      records.push(r); byNoise.push({ noise_tier: noise, algorithm, metrics: calculateMetrics(rows.filter(row => row.algorithm === algorithm && row.gps_noise_tier === noise)) });
    }
    const intervals = [...new Set(rows.map(r => Number(r.sampling_interval_s)))].sort((a, b) => a - b);
    for (const interval of intervals) {
      const r = summarizeGroup(rows, algorithm, 'sampling_interval', { interval, predicate: row => Number(row.sampling_interval_s) === interval });
      records.push(r); byInterval.push({ sampling_interval_s: interval, algorithm, metrics: calculateMetrics(rows.filter(row => row.algorithm === algorithm && Number(row.sampling_interval_s) === interval)) });
    }
  }
  return { records, detailed: { overall, by_context: byContext, by_noise: byNoise, by_sampling_interval: byInterval } };
}

