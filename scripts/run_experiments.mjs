import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { readCsv, writeCsv } from '../src/csv_utils.mjs';
import { loadScenarioNetworks, loadSplitTracks, evaluationRows } from '../src/data_loader.mjs';
import { NearestRoadMatcher } from '../src/nearest_road.mjs';
import { HMMViterbiMatcher, DEFAULT_HMM_PARAMETERS } from '../src/hmm_map_matching.mjs';
import { AdaptiveRadiusMatcher, DEFAULT_ADAPTIVE_PARAMETERS } from '../src/adaptive_radius.mjs';
import { buildMetricTables, trajectoryAccuracyScore } from '../src/evaluation.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DATA = path.join(ROOT, 'data', 'processed', 'synthetic_v2');
const METRICS_DIR = path.join(ROOT, 'results', 'metrics');
const FIGURES_DIR = path.join(ROOT, 'results', 'figures');
const REPORT = path.join(ROOT, 'reports', 'phase3_experiment_report.md');
const INTERVALS = [1, 5, 15, 30, 45, 60, 120];
const ALGORITHMS = ['Nearest Road', 'HMM/Viterbi', 'Adaptive Radius'];

function mean(values) { return values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0; }

function runMatcher(track, algorithm, parameters = {}) {
  if (algorithm === 'Nearest Road') return new NearestRoadMatcher().match(track.observations, track.network);
  if (algorithm === 'HMM/Viterbi') return new HMMViterbiMatcher(parameters).match(track.observations, track.network);
  if (algorithm === 'Adaptive Radius') return new AdaptiveRadiusMatcher(parameters).match(track.observations, track.network, track.features);
  throw new Error(`Unknown algorithm: ${algorithm}`);
}

function tuneParameters(tracks, algorithm, grid) {
  let best = null;
  const trials = [];
  for (const parameters of grid) {
    const predictions = [];
    for (const track of tracks) {
      const result = runMatcher(track, algorithm, parameters);
      predictions.push(...evaluationRows(track, result, algorithm, result.elapsedMs));
    }
    const score = trajectoryAccuracyScore(predictions);
    const candidateMean = mean(predictions.map(r => r.candidate_count));
    const trial = { parameters, validation_mean_trajectory_accuracy: score, validation_candidate_count_mean: candidateMean };
    trials.push(trial);
    if (!best || score > best.validation_mean_trajectory_accuracy + 1e-12 ||
      (Math.abs(score - best.validation_mean_trajectory_accuracy) <= 1e-12 && candidateMean < best.validation_candidate_count_mean)) best = trial;
  }
  return { selected: best, trials };
}

function ensureSameKeys(rows) {
  const byAlgorithm = new Map(ALGORITHMS.map(a => [a, new Set()]));
  for (const row of rows) byAlgorithm.get(row.algorithm).add(`${row.scenario_id}|${row.trajectory_id}|${row.sampling_interval_s}|${row.seq}|${row.timestamp}`);
  const expected = byAlgorithm.get(ALGORITHMS[0]);
  for (const algorithm of ALGORITHMS.slice(1)) {
    const current = byAlgorithm.get(algorithm);
    if (current.size !== expected.size || [...expected].some(k => !current.has(k))) throw new Error(`Algorithms did not process identical test observations: ${algorithm}`);
  }
  return expected.size;
}

function fmt(n, digits = 4) { return Number(n).toFixed(digits); }
function summaryRow(list, algorithm) { return list.find(r => r.algorithm === algorithm); }

function buildReport({ rawCount, eligibleCount, testRows, metrics, tuning, testElapsed, parameters }) {
  const over = metrics.detailed.overall;
  const byContext = metrics.detailed.by_context;
  const byNoise = metrics.detailed.by_noise;
  const byInterval = metrics.detailed.by_sampling_interval;
  const adaptive = summaryRow(over, 'Adaptive Radius');
  const nearest = summaryRow(over, 'Nearest Road');
  const hmm = summaryRow(over, 'HMM/Viterbi');
  const winners = new Map();
  for (const context of ['open_highway', 'urban_canyon', 'complex_intersection', 'airport_multilevel']) {
    const rows = byContext.filter(r => r.context === context);
    const top = [...rows].sort((a, b) => b.metrics.edge_accuracy - a.metrics.edge_accuracy)[0];
    winners.set(context, top?.algorithm ?? 'n/a');
  }
  const challenge = [...byInterval].filter(r => r.algorithm === 'Adaptive Radius').sort((a, b) => a.metrics.edge_accuracy - b.metrics.edge_accuracy)[0];
  const splitCounts = Object.fromEntries(['train', 'validation', 'test'].map(s => [s, readCsv(path.join(DATA, 'scenario_manifest.csv')).filter(r => r.split === s).length]));
  const rowsOverall = over.map(r => `| ${r.algorithm} | ${fmt(r.metrics.edge_accuracy)} | ${fmt(r.metrics.edge_precision_macro)} | ${fmt(r.metrics.edge_recall_macro)} | ${fmt(r.metrics.edge_f1_macro)} | ${fmt(r.metrics.mean_matching_distance_m, 2)} | ${fmt(r.metrics.trajectory_accuracy_mean)} | ${fmt(r.metrics.candidate_count_mean, 2)} | ${fmt(r.metrics.processing_ms_per_observation, 4)} |`).join('\n');
  const contextRows = ['open_highway', 'urban_canyon', 'complex_intersection', 'airport_multilevel'].map(context => {
    const cell = byContext.filter(r => r.context === context).map(r => `${r.algorithm}: ${fmt(r.metrics.edge_accuracy)}`).join('; ');
    return `| ${context} | ${cell} |`;
  }).join('\n');
  const noiseRows = ['low', 'high'].map(noise => {
    const cell = byNoise.filter(r => r.noise_tier === noise).map(r => `${r.algorithm}: ${fmt(r.metrics.edge_accuracy)} (macro F1 ${fmt(r.metrics.edge_f1_macro)})`).join('; ');
    return `| ${noise} | ${cell} |`;
  }).join('\n');
  const intervalRows = INTERVALS.map(interval => {
    const cell = byInterval.filter(r => Number(r.sampling_interval_s) === interval).map(r => `${r.algorithm}: ${fmt(r.metrics.edge_accuracy)}`).join('; ');
    return `| ${interval}s | ${cell} |`;
  }).join('\n');
  const tuneSummary = [
    `- HMM/Viterbi selected on validation: \`${JSON.stringify(tuning.hmm.selected.parameters)}\` (mean trajectory accuracy ${fmt(tuning.hmm.selected.validation_mean_trajectory_accuracy)}).`,
    `- Adaptive Radius selected on validation: \`${JSON.stringify(tuning.adaptive.selected.parameters)}\` (mean trajectory accuracy ${fmt(tuning.adaptive.selected.validation_mean_trajectory_accuracy)}).`,
  ].join('\n');
  return `# Phase 3 — Map Matching Algorithm Development and Experimental Comparison\n\n` +
    `## Scope and reproducibility\n\n` +
    `Implemented and evaluated three matchers on the Phase 2 synthetic v2 benchmark. The inputs are synthetic and generator-defined; these results are not real-world validation. No raw or synthetic_v2 data was modified. All methods received the same ${eligibleCount.toLocaleString()} eligible test observation keys and the same scenario-specific network. The radius-label files were not loaded by the matcher or tuning pipeline.\n\n` +
    `Source raw records profiled: ${rawCount.toLocaleString()}. Scenario split counts: train ${splitCounts.train}, validation ${splitCounts.validation}, test ${splitCounts.test}; all sampling variants remain in their scenario split. Parameter selection used validation only; the test set was opened only after selection. No model fitting was needed because these are deterministic algorithms.\n\n` +
    `## Test-set overall results\n\n` +
    `| Algorithm | Edge accuracy | Macro precision | Macro recall | Macro F1 | Mean obs-to-edge distance (m) | Mean trajectory accuracy | Candidates/observation | ms/observation |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n${rowsOverall}\n\n` +
    `Micro precision, recall and F1 are also written to the CSV/JSON; for single-label multiclass classification they equal edge accuracy. Mean matching distance is the noisy GPS observation's distance to the selected edge, not error to the true coordinate.\n\n` +
    `## Context and GPS-noise results\n\n` +
    `| Road context | Test edge accuracy by algorithm |\n|---|---|\n${contextRows}\n\n` +
    `| GPS noise tier | Test edge accuracy (macro F1) by algorithm |\n|---|---|\n${noiseRows}\n\n` +
    `Adaptive Radius improves over Nearest Road in urban canyon and complex intersection, and is lower in open highway and airport/multilevel. HMM/Viterbi has the highest edge accuracy in all four context groups. The detailed context/noise metrics are also in \`results/metrics/algorithm_comparison.csv\`.\n\n` +
    `## Sampling interval results\n\n` +
    `| Sampling interval | Test edge accuracy by algorithm |\n|---|---|\n${intervalRows}\n\n` +
    `The lowest Adaptive Radius edge accuracy is at ${challenge?.sampling_interval_s}s (${fmt(challenge?.metrics.edge_accuracy ?? 0)}).\n\n` +
    `## Validation-only parameter selection\n\n${tuneSummary}\n\n` +
    `Nearest Road selects the globally nearest segment and has no tuned parameters. The HMM uses a Gaussian distance emission and a transition likelihood based on disagreement between network-route distance and observed displacement; unreachable transitions receive a penalty and the algorithm falls back to nearest candidates when a search returns none. Adaptive Radius computes \`r = max(d1 + 0.5, z·sigma + wd·sigma·log2(1+n100) + wl·sigma·max(0,L−1))\`. If the two nearest edges are near-parallel (bearing difference ≤20°) and their distance gap is ≤\`tau·sigma\`, the radius is expanded to include the second edge. Selected validation parameters were \`z=${fmt(parameters.adaptive.sigmaMultiplier,2)}\`, \`wd=${fmt(parameters.adaptive.densityWeight,2)}\`, \`wl=${fmt(parameters.adaptive.layerWeight,2)}\`, \`tau=${fmt(parameters.adaptive.parallelAmbiguitySigma,2)}\`; candidate scores then use distance, heading and one-step route continuity. Full implementation is in \`src/adaptive_radius.mjs\`.\n\n` +
    `## Interpretation and limits\n\n` +
    `Adaptive Radius ${adaptive.metrics.edge_accuracy > nearest.metrics.edge_accuracy && adaptive.metrics.edge_accuracy > hmm.metrics.edge_accuracy ? 'outperformed both baselines on overall test edge accuracy' : 'did not outperform both baselines on overall test edge accuracy'}: accuracy ${fmt(adaptive.metrics.edge_accuracy)} versus Nearest Road ${fmt(nearest.metrics.edge_accuracy)} and HMM/Viterbi ${fmt(hmm.metrics.edge_accuracy)}. ${testRows.toLocaleString()} prediction rows were evaluated per algorithm. Results vary by context and interval; consult the CSV for precision/recall/F1 and timing details. No performance target was imposed during generation or tuning.\n\n` +
    `Synthetic maps have controlled topology, a local metric projection and a known noise process. Processing times are local Node.js measurements, not deployment benchmarks. The HMM uses a bounded candidate list (8 maximum); results depend on the validation-selected candidate and transition parameters. The generated adaptive radius labels were not used as inputs or to tune this experiment. The training split was not used because no statistical model was fitted; HMM and adaptive configuration selection used validation only.\n\n` +
    `## Outputs\n\n` +
    `- \`results/metrics/algorithm_comparison.csv\`\n- \`results/metrics/detailed_metrics.json\`\n- \`results/figures/algorithm_comparison.png\`\n- \`results/figures/context_wise_comparison.png\`\n- \`results/figures/sampling_interval_comparison.png\`\n- \`reports/phase3_experiment_report.md\`\n`;
}

function main() {
  fs.mkdirSync(METRICS_DIR, { recursive: true });
  fs.mkdirSync(FIGURES_DIR, { recursive: true });
  const summary = JSON.parse(fs.readFileSync(path.join(ROOT, 'reports', 'preprocessing_summary.json'), 'utf8'));
  const manifestRows = readCsv(path.join(DATA, 'scenario_manifest.csv'));
  const sourceProfile = readCsv(path.join(ROOT, 'reports', 'source_profile.csv'));
  const rawCount = sourceProfile.reduce((n, r) => n + Number(r.records), 0);

  // Only validation observations/features and their separate truth labels are loaded during tuning.
  const validationIds = new Set(manifestRows.filter(r => r.split === 'validation').map(r => r.scenario_id));
  const validationNetworks = loadScenarioNetworks(DATA, validationIds);
  const validation = loadSplitTracks(DATA, 'validation', INTERVALS, validationNetworks);
  const hmmGrid = [1.96, 2.5, 3, 4].flatMap(candidateSigmaMultiplier => [1, 2, 4].map(transitionBetaMultiplier => ({
    ...DEFAULT_HMM_PARAMETERS, candidateSigmaMultiplier, transitionBetaMultiplier,
  })));
  const adaptiveGrid = [1.5, 1.96, 2.3].flatMap(sigmaMultiplier => [0.15, 0.35, 0.55].flatMap(densityWeight => [1, 1.5, 2].map(parallelAmbiguitySigma => ({
    ...DEFAULT_ADAPTIVE_PARAMETERS, sigmaMultiplier, densityWeight, parallelAmbiguitySigma,
  }))));
  const hmmTuning = tuneParameters(validation.tracks, 'HMM/Viterbi', hmmGrid);
  const adaptiveTuning = tuneParameters(validation.tracks, 'Adaptive Radius', adaptiveGrid);
  const selected = { hmm: hmmTuning.selected.parameters, adaptive: adaptiveTuning.selected.parameters };

  // Test observations and labels are loaded only after validation parameter selection is complete.
  const testIds = new Set(manifestRows.filter(r => r.split === 'test').map(r => r.scenario_id));
  const testNetworks = loadScenarioNetworks(DATA, testIds);
  const test = loadSplitTracks(DATA, 'test', INTERVALS, testNetworks);
  const testRows = [];
  const testStarted = performance.now();
  for (const track of test.tracks) {
    const runs = [
      ['Nearest Road', runMatcher(track, 'Nearest Road')],
      ['HMM/Viterbi', runMatcher(track, 'HMM/Viterbi', selected.hmm)],
      ['Adaptive Radius', runMatcher(track, 'Adaptive Radius', selected.adaptive)],
    ];
    for (const [algorithm, run] of runs) for (const match of run.matches) {
      if (!track.network.edgeById.has(match.edge_id)) throw new Error(`${algorithm} returned unknown edge ${match.edge_id} in ${track.scenario.scenario_id}`);
    }
    for (const [algorithm, run] of runs) testRows.push(...evaluationRows(track, run, algorithm, run.elapsedMs));
  }
  const testElapsed = performance.now() - testStarted;
  const eligibleCount = ensureSameKeys(testRows);
  const perAlgorithm = Object.fromEntries(ALGORITHMS.map(a => [a, testRows.filter(r => r.algorithm === a).length]));
  if (Object.values(perAlgorithm).some(count => count !== eligibleCount)) throw new Error('Algorithm test observation counts differ');
  const metricTables = buildMetricTables(testRows, ALGORITHMS);
  const outputColumns = ['algorithm', 'scope', 'context', 'noise_tier', 'sampling_interval_s', 'observations', 'trajectories', 'edge_accuracy', 'edge_precision_micro', 'edge_recall_micro', 'edge_f1_micro', 'edge_precision_macro', 'edge_recall_macro', 'edge_f1_macro', 'mean_matching_distance_m', 'trajectory_accuracy_mean', 'trajectory_exact_match_rate', 'candidate_count_mean', 'processing_ms_per_observation'];
  writeCsv(path.join(METRICS_DIR, 'algorithm_comparison.csv'), metricTables.records, outputColumns);
  const detailed = {
    project: 'GeoMotion AI', phase: 'Phase 3', generated_at_utc: new Date().toISOString(),
    benchmark: 'data/processed/synthetic_v2', benchmark_type: 'synthetic; generator-defined labels',
    evaluation_split: 'test', validation_split: 'validation', training_split: 'train',
    training_used_for_fitting: false, reason_no_fitting: 'All three methods are deterministic; parameters were selected only on validation.',
    test_tuning_or_selection_used: false, test_observation_keys_identical_for_all_algorithms: true,
    test_observations_labels_and_networks_loaded_after_validation_selection: true,
    eligible_test_observations_per_algorithm: perAlgorithm, unique_eligible_test_keys: eligibleCount,
    radius_label_files_loaded_by_matchers: false,
    radius_target_diagnostic_used_for_tuning: false,
    test_elapsed_wall_ms_all_methods: testElapsed,
    validation_tuning: {
      score: 'mean trajectory edge accuracy, equally weighted by scenario/interval trajectory',
      hmm: hmmTuning,
      adaptive_radius: adaptiveTuning,
    },
    metric_definitions: {
      micro_precision_recall_f1: 'Single-label multiclass micro scores; equal to edge accuracy.',
      macro_scores: 'Unweighted mean of one-vs-rest per-edge precision/recall/F1 over the union of true and predicted edge IDs.',
      mean_matching_distance_m: 'Metric distance from the noisy observation to the predicted road edge, not distance from the true coordinate.',
      trajectory_accuracy_mean: 'Mean of per-trajectory pointwise edge accuracy, giving each scenario/interval track equal weight.',
      trajectory_exact_match_rate: 'Fraction of scenario/interval tracks whose every point edge is correct.',
      candidate_count_mean: 'Mean candidates retained/considered by the algorithm at each observation; Nearest Road checks every edge in that scenario map.',
      processing_ms_per_observation: 'Matcher wall time divided by observations in each track; excludes CSV loading and network construction.',
    },
    test_results: metricTables.detailed,
    validation_tuning_summary: { hmm_selected: hmmTuning.selected, adaptive_selected: adaptiveTuning.selected },
    data_integrity: {
      raw_dataset_rows_profiled: rawCount,
      benchmark_scenario_split_counts: Object.fromEntries(['train', 'validation', 'test'].map(s => [s, manifestRows.filter(r => r.split === s).length])),
      sampling_intervals_s: INTERVALS,
      eligible_labels_resolve: summary.generated_benchmark.generated_ground_truth_unresolved_edges === 0,
      no_data_files_modified: true,
    },
  };
  fs.writeFileSync(path.join(METRICS_DIR, 'detailed_metrics.json'), JSON.stringify(detailed, null, 2) + '\n', 'utf8');
  fs.writeFileSync(REPORT, buildReport({ rawCount, eligibleCount, testRows: eligibleCount, metrics: metricTables, tuning: { hmm: hmmTuning, adaptive: adaptiveTuning }, testElapsed, parameters: selected }), 'utf8');

  const render = spawnSync('pwsh.exe', ['-NoProfile', '-File', path.join(ROOT, 'scripts', 'render_experiment_figures.ps1'), '-MetricsPath', path.join(METRICS_DIR, 'detailed_metrics.json'), '-OutputDir', FIGURES_DIR], { cwd: ROOT, encoding: 'utf8' });
  if (render.status !== 0) throw new Error(`Figure rendering failed: ${render.stderr || render.stdout}`);
  console.log(JSON.stringify({
    raw_records_profiled: rawCount, test_observations_per_algorithm: perAlgorithm,
    selected_hmm_parameters: hmmTuning.selected,
    selected_adaptive_parameters: adaptiveTuning.selected,
    overall_metrics: metricTables.detailed.overall.map(x => ({ algorithm: x.algorithm, metrics: x.metrics })),
    output_files: [path.join(METRICS_DIR, 'algorithm_comparison.csv'), path.join(METRICS_DIR, 'detailed_metrics.json'), path.join(FIGURES_DIR, 'algorithm_comparison.png'), path.join(FIGURES_DIR, 'context_wise_comparison.png'), path.join(FIGURES_DIR, 'sampling_interval_comparison.png'), REPORT],
  }, null, 2));
}

main();
