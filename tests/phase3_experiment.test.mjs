import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseCsv } from '../src/csv_utils.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const METRICS = path.join(ROOT, 'results', 'metrics');
const FIGURES = path.join(ROOT, 'results', 'figures');
const names = ['Nearest Road', 'HMM/Viterbi', 'Adaptive Radius'];

test('experiment outputs include all requested files and valid PNGs', () => {
  const files = [
    path.join(METRICS, 'algorithm_comparison.csv'), path.join(METRICS, 'detailed_metrics.json'),
    path.join(FIGURES, 'algorithm_comparison.png'), path.join(FIGURES, 'context_wise_comparison.png'),
    path.join(FIGURES, 'sampling_interval_comparison.png'), path.join(ROOT, 'reports', 'phase3_experiment_report.md'),
  ];
  for (const file of files) assert.ok(fs.existsSync(file) && fs.statSync(file).size > 0, `missing/empty ${file}`);
  for (const file of files.filter(f => f.endsWith('.png'))) {
    const buf = fs.readFileSync(file);
    assert.deepEqual([...buf.subarray(0, 8)], [137, 80, 78, 71, 13, 10, 26, 10]);
  }
});

test('all algorithms evaluate identical eligible test observations without test tuning', () => {
  const detailed = JSON.parse(fs.readFileSync(path.join(METRICS, 'detailed_metrics.json'), 'utf8'));
  assert.equal(detailed.evaluation_split, 'test');
  assert.equal(detailed.validation_split, 'validation');
  assert.equal(detailed.test_tuning_or_selection_used, false);
  assert.equal(detailed.test_observation_keys_identical_for_all_algorithms, true);
  assert.equal(detailed.test_observations_labels_and_networks_loaded_after_validation_selection, true);
  assert.equal(detailed.radius_label_files_loaded_by_matchers, false);
  assert.equal(detailed.radius_target_diagnostic_used_for_tuning, false);
  assert.equal(detailed.eligible_test_observations_per_algorithm['Nearest Road'], 12944);
  assert.equal(detailed.eligible_test_observations_per_algorithm['HMM/Viterbi'], 12944);
  assert.equal(detailed.eligible_test_observations_per_algorithm['Adaptive Radius'], 12944);
  assert.deepEqual(detailed.data_integrity.sampling_intervals_s, [1, 5, 15, 30, 45, 60, 120]);
  assert.equal(detailed.validation_tuning.hmm.selected.parameters.candidateSigmaMultiplier, 4);
  assert.ok(detailed.validation_tuning.adaptive_radius.selected.parameters.sigmaMultiplier > 0);
});

test('test metrics include requested scopes and finite quality/runtime metrics', () => {
  const detailed = JSON.parse(fs.readFileSync(path.join(METRICS, 'detailed_metrics.json'), 'utf8'));
  const result = detailed.test_results;
  assert.deepEqual(result.overall.map(r => r.algorithm), names);
  assert.equal(result.by_context.length, 12);
  assert.equal(result.by_noise.length, 6);
  assert.equal(result.by_sampling_interval.length, 21);
  for (const collection of [result.overall, result.by_context, result.by_noise, result.by_sampling_interval]) for (const row of collection) {
    for (const key of ['edge_accuracy', 'edge_precision_micro', 'edge_recall_micro', 'edge_f1_micro', 'edge_precision_macro', 'edge_recall_macro', 'edge_f1_macro', 'mean_matching_distance_m', 'trajectory_accuracy_mean', 'trajectory_exact_match_rate', 'candidate_count_mean', 'processing_ms_per_observation']) {
      assert.ok(Number.isFinite(row.metrics[key]), `${row.algorithm}/${key} is not finite`);
    }
  }
  const csv = parseCsv(fs.readFileSync(path.join(METRICS, 'algorithm_comparison.csv'), 'utf8'));
  assert.equal(csv.length, 42);
});

