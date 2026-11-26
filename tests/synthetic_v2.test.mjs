import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUT = path.join(ROOT, 'data', 'processed', 'synthetic_v2');
const INTERVALS = [1, 5, 15, 30, 45, 60, 120];
const SPLITS = ['train', 'validation', 'test'];

function parseCsv(text) {
  const rows = [];
  let row = [], field = '', quote = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quote) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (c === '"') quote = false;
      else field += c;
    } else if (c === '"') quote = true;
    else if (c === ',') { row.push(field); field = ''; }
    else if (c === '\n') { row.push(field.replace(/\r$/, '')); rows.push(row); row = []; field = ''; }
    else field += c;
  }
  if (field.length || row.length) { row.push(field.replace(/\r$/, '')); rows.push(row); }
  const headers = rows.shift() ?? [];
  return rows.filter(r => !(r.length === 1 && r[0] === '')).map(v => Object.fromEntries(headers.map((h, i) => [h, v[i] ?? ''])));
}

function load(file) { return parseCsv(fs.readFileSync(file, 'utf8')); }
function pointFile(split, role, interval) {
  const prefix = role === 'observations' ? 'traj' : role;
  return path.join(OUT, split, role, `${prefix}_dt_${interval}s.csv`);
}

test('all required outputs and sampling intervals exist', () => {
  for (const file of [
    'road_network.csv', 'road_context_features.csv', 'scenario_manifest.csv', 'feature_allowlist.json',
    'source_manifest.json', 'coordinate_reference.json', 'legacy_network_metadata.json',
  ]) assert.ok(fs.existsSync(path.join(OUT, file)), `missing ${file}`);
  for (const split of SPLITS) for (const interval of INTERVALS) {
    const expectedRows = 16 * Math.ceil(600 / interval);
    for (const role of ['observations', 'features', 'ground_truth', 'radius_labels']) {
      const file = pointFile(split, role, interval);
      assert.ok(fs.existsSync(file), `missing ${split}/${role}/dt_${interval}s`);
      assert.ok(fs.statSync(file).size > 0, `empty ${file}`);
      assert.equal(load(file).length, expectedRows, `${split}/${role}/${interval}s row count`);
    }
  }
});

test('required preprocessing reports and source profiling are present and consistent', () => {
  for (const file of [
    'preprocessing_report.md', 'preprocessing_summary.json', 'cleaning_log.csv', 'before_after_comparison.csv',
    'feature_engineering_report.md', 'dataset_generation_methodology.md', 'scenario_distribution.csv',
    'scenario_distribution.md', 'source_profile.csv', 'dataset_audit.md', 'dataset_audit.json',
  ]) assert.ok(fs.existsSync(path.join(ROOT, 'reports', file)), `missing reports/${file}`);
  const summary = JSON.parse(fs.readFileSync(path.join(ROOT, 'reports', 'preprocessing_summary.json'), 'utf8'));
  const audit = JSON.parse(fs.readFileSync(path.join(ROOT, 'reports', 'dataset_audit.json'), 'utf8'));
  assert.equal(summary.raw.total_records, 25708);
  assert.equal(audit.inventory.total_records, 25708);
  assert.equal(summary.generated_benchmark.unique_paired_observation_points, 38832);
  assert.equal(summary.generated_benchmark.core_artifact_rows_total, 161280);
  assert.equal(summary.readiness.synthetic_v2_benchmark, 'ready_for_fair_algorithm_comparison_on_generated_scenarios');
  assert.equal(load(path.join(ROOT, 'reports', 'source_profile.csv')).length, 12);
});

test('every generated ground-truth composite edge key resolves', () => {
  const edges = new Set(load(path.join(OUT, 'road_network.csv')).map(r => `${r.scenario_id}|${r.edge_id}`));
  for (const split of SPLITS) for (const interval of INTERVALS) {
    for (const gt of load(pointFile(split, 'ground_truth', interval))) {
      assert.ok(edges.has(`${gt.scenario_id}|${gt.edge_id_true}`), `${gt.scenario_id} ${gt.edge_id_true} missing at seq ${gt.seq}`);
    }
  }
  const legacy = load(path.join(OUT, 'legacy_augmented_road_network.csv'));
  for (const id of ['E220', 'E355', 'E821']) assert.ok(legacy.some(r => r.edge_id === id));
  const benchmark = load(path.join(OUT, 'road_network.csv'));
  const expectedContext = { E220: 'complex_intersection', E355: 'complex_intersection', E821: 'airport_multilevel' };
  for (const [id, context] of Object.entries(expectedContext)) assert.ok(benchmark.some(r => r.edge_id === id && r.scenario_type === context), `${id} generated in wrong scenario context`);
  for (const addedId of ['E220', 'E355', 'E821']) {
    const added = legacy.find(r => r.edge_id === addedId);
    assert.ok(legacy.some(r => r.edge_id !== addedId && r.from_node === added.from_node && r.to_node === added.to_node), `${addedId} endpoints are not shared with an existing road`);
  }
});

test('the three added legacy IDs resolve to connected nodes without rewriting raw labels', () => {
  const legacy = load(path.join(OUT, 'legacy_augmented_road_network.csv'));
  const ids = new Set(legacy.map(r => r.edge_id));
  for (const gtFile of ['delhi_ground_truth_90km.csv', 'adaptive_radius_ground_truth.csv']) {
    const rows = load(path.join(ROOT, 'data', 'raw', gtFile));
    const col = gtFile.startsWith('adaptive') ? 'edge_id_true' : 'edge_id';
    assert.ok(rows.every(r => ids.has(r[col])), `${gtFile} has a label ID absent after augmentation`);
  }
  for (const id of ['E220', 'E355', 'E821']) {
    const added = legacy.find(r => r.edge_id === id);
    assert.equal(added.scenario_id, 'LEGACY');
    assert.ok(legacy.some(r => r.edge_id !== id && r.from_node === added.from_node && r.to_node === added.to_node));
    assert.match(added.geometry_wkt, /^LINESTRING \(/);
    assert.equal(added.geometry_verified_against_raw_labels, 'false');
    assert.ok(Number(added.length_m) > 0);
    const coords = added.geometry_wkt.match(/^LINESTRING \((.+)\)$/)[1].split(',').map(p => p.trim().split(/\s+/).map(Number));
    assert.equal(coords.length, 3);
    for (const [lon, lat] of coords) {
      assert.ok(lat >= 28.63125 && lat <= 29.172393);
      assert.ok(lon >= 77.219564 && lon <= 77.879641);
    }
  }
  const adjacency = new Map();
  for (const e of legacy) {
    if (!adjacency.has(e.from_node)) adjacency.set(e.from_node, new Set());
    if (!adjacency.has(e.to_node)) adjacency.set(e.to_node, new Set());
    adjacency.get(e.from_node).add(e.to_node); adjacency.get(e.to_node).add(e.from_node);
  }
  const first = adjacency.keys().next().value, seen = new Set([first]), queue = [first];
  for (let i = 0; i < queue.length; i++) for (const node of adjacency.get(queue[i])) if (!seen.has(node)) { seen.add(node); queue.push(node); }
  assert.equal(seen.size, adjacency.size, 'legacy augmentation introduced a disconnected component');
});

test('adaptive radius intervals are ordered and contain each recommendation', () => {
  for (const split of SPLITS) for (const interval of INTERVALS) {
    for (const r of load(pointFile(split, 'radius_labels', interval))) {
      const min = Number(r.min_ok_radius_m), rec = Number(r.recommended_radius_m), max = Number(r.max_ok_radius_m);
      assert.ok(Number.isFinite(min) && Number.isFinite(rec) && Number.isFinite(max));
      assert.ok(min <= rec && rec <= max, `${r.scenario_id}/${interval}/${r.seq}: ${min} <= ${rec} <= ${max}`);
    }
  }
});

test('prediction feature tables contain no truth or radius target columns', () => {
  const allowlist = JSON.parse(fs.readFileSync(path.join(OUT, 'feature_allowlist.json'), 'utf8'));
  const banned = new Set(allowlist.prohibited_target_or_truth_columns);
  for (const split of SPLITS) for (const interval of INTERVALS) {
    const rows = load(pointFile(split, 'features', interval));
    const observations = load(pointFile(split, 'observations', interval));
    const columns = Object.keys(rows[0]);
    assert.ok(columns.every(c => !banned.has(c)), `${split} ${interval}s feature table leaks target/truth`);
    assert.ok(Object.keys(observations[0]).every(c => !banned.has(c)), `${split} ${interval}s observation table leaks target/truth`);
  }
});

test('scenario splits are disjoint and stratified; sampling variants stay grouped', () => {
  const scenarios = load(path.join(OUT, 'scenario_manifest.csv'));
  assert.equal(scenarios.length, 48);
  const idsBySplit = Object.fromEntries(SPLITS.map(split => [split, new Set(scenarios.filter(s => s.split === split).map(s => s.scenario_id))]));
  for (const split of SPLITS) assert.equal(idsBySplit[split].size, 16);
  for (let a = 0; a < SPLITS.length; a++) for (let b = a + 1; b < SPLITS.length; b++) {
    assert.equal([...idsBySplit[SPLITS[a]]].filter(id => idsBySplit[SPLITS[b]].has(id)).length, 0);
  }
  for (const split of SPLITS) for (const context of ['open_highway', 'urban_canyon', 'complex_intersection', 'airport_multilevel']) for (const noise of ['low', 'high']) {
    assert.equal(scenarios.filter(s => s.split === split && s.scenario_type === context && s.gps_noise_tier === noise).length, 2);
  }
});

test('each generated scenario road graph is connected', () => {
  const edges = load(path.join(OUT, 'road_network.csv')).filter(e => e.scenario_id.startsWith('SC'));
  const byScenario = new Map();
  for (const e of edges) {
    if (!byScenario.has(e.scenario_id)) byScenario.set(e.scenario_id, []);
    byScenario.get(e.scenario_id).push(e);
  }
  assert.equal(byScenario.size, 48);
  for (const [scenario, roads] of byScenario) {
    const adj = new Map();
    for (const e of roads) {
      if (!adj.has(e.from_node)) adj.set(e.from_node, new Set());
      if (!adj.has(e.to_node)) adj.set(e.to_node, new Set());
      adj.get(e.from_node).add(e.to_node); adj.get(e.to_node).add(e.from_node);
    }
    const first = adj.keys().next().value, seen = new Set([first]), q = [first];
    for (let i = 0; i < q.length; i++) for (const n of adj.get(q[i])) if (!seen.has(n)) { seen.add(n); q.push(n); }
    assert.equal(seen.size, adj.size, `${scenario} has disconnected nodes`);
  }
});

test('generated benchmark coordinates remain inside the source study-area extent', () => {
  const roads = load(path.join(OUT, 'road_network.csv')).filter(e => e.scenario_id.startsWith('SC'));
  for (const e of roads) for (const [lat, lon] of [[Number(e.from_lat), Number(e.from_lon)], [Number(e.to_lat), Number(e.to_lon)]]) {
    assert.ok(lat >= 28.63125 && lat <= 29.172393, `latitude outside source area: ${lat}`);
    assert.ok(lon >= 77.219564 && lon <= 77.879641, `longitude outside source area: ${lon}`);
  }
});

test('sampling variants share canonical truth and timestamps for corresponding sequence values', () => {
  for (const split of SPLITS) for (const scenario of load(path.join(OUT, 'scenario_manifest.csv')).filter(s => s.split === split)) {
    const reference = new Map(load(pointFile(split, 'ground_truth', 1)).filter(r => r.scenario_id === scenario.scenario_id).map(r => [r.seq, r]));
    for (const interval of INTERVALS) {
      const obs = load(pointFile(split, 'observations', interval)).filter(r => r.scenario_id === scenario.scenario_id);
      const gt = load(pointFile(split, 'ground_truth', interval)).filter(r => r.scenario_id === scenario.scenario_id);
      const features = load(pointFile(split, 'features', interval)).filter(r => r.scenario_id === scenario.scenario_id);
      const radius = load(pointFile(split, 'radius_labels', interval)).filter(r => r.scenario_id === scenario.scenario_id);
      assert.equal(obs.length, gt.length);
      assert.equal(obs.length, features.length);
      assert.equal(obs.length, radius.length);
      assert.equal(obs.length, Math.ceil(600 / interval), `${scenario.scenario_id} absent or duplicated in ${interval}s sampling`);
      for (let i = 0; i < obs.length; i++) {
        assert.equal(obs[i].seq, gt[i].seq);
        assert.equal(obs[i].timestamp, gt[i].timestamp);
        assert.equal(obs[i].seq, features[i].seq);
        assert.equal(obs[i].seq, radius[i].seq);
        assert.ok(Date.parse(obs[i].timestamp));
        const canonical = reference.get(obs[i].seq);
        assert.ok(canonical, `${scenario.scenario_id}: no canonical point ${obs[i].seq}`);
        assert.equal(gt[i].lat_true, canonical.lat_true);
        assert.equal(gt[i].lon_true, canonical.lon_true);
        assert.equal(gt[i].edge_id_true, canonical.edge_id_true);
        if (i > 0) {
          assert.equal(Number(obs[i].seq) - Number(obs[i - 1].seq), interval);
          assert.equal(Date.parse(obs[i].timestamp) - Date.parse(obs[i - 1].timestamp), interval * 1000);
        }
      }
    }
  }
});

test('raw source files still match their generation-time SHA-256 manifest', () => {
  const manifest = JSON.parse(fs.readFileSync(path.join(OUT, 'source_manifest.json'), 'utf8'));
  for (const f of manifest.raw_files) {
    const file = path.join(ROOT, 'data', 'raw', f.file);
    const hash = crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
    assert.equal(hash, f.sha256, `${f.file} changed after profiling`);
    assert.equal(fs.statSync(file).size, f.bytes, `${f.file} byte size changed`);
  }
});

test('fixed seed regenerates identical core benchmark tables', () => {
  const files = [
    'road_network.csv', 'road_context_features.csv', 'scenario_manifest.csv', 'sampling_distribution.csv',
    'train/observations/traj_dt_1s.csv', 'train/features/features_dt_1s.csv',
    'validation/ground_truth/ground_truth_dt_45s.csv', 'test/radius_labels/radius_labels_dt_120s.csv',
  ];
  const hash = file => crypto.createHash('sha256').update(fs.readFileSync(path.join(OUT, file))).digest('hex');
  const before = new Map(files.map(f => [f, hash(f)]));
  const run = spawnSync(process.execPath, ['scripts/build_synthetic_v2.mjs'], { cwd: ROOT, encoding: 'utf8' });
  assert.equal(run.status, 0, run.stderr || run.stdout);
  for (const file of files) assert.equal(hash(file), before.get(file), `${file} is not reproducible with the fixed seed`);
});

