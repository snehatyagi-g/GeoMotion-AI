import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const RAW = path.join(ROOT, 'data', 'raw');
const OUT = path.join(ROOT, 'data', 'processed', 'synthetic_v2');
const REPORTS = path.join(ROOT, 'reports');
const SEED = 20261103;
const INTERVALS = [1, 5, 15, 30, 45, 60, 120];
const CONTEXTS = ['open_highway', 'urban_canyon', 'complex_intersection', 'airport_multilevel'];
const NOISES = ['low', 'high'];
const SPLITS = ['train', 'validation', 'test'];
const EXPECTED_RAW = [
  'adaptive_radius_ground_truth.csv', 'delhi_ground_truth_90km.csv', 'delhi_road_network.csv',
  'road_context_flags.csv', 'segment_summary.csv', 'traj_dt_1s.csv', 'traj_dt_5s.csv',
  'traj_dt_15s.csv', 'traj_dt_30s.csv', 'traj_dt_45s.csv', 'traj_dt_60s.csv', 'traj_dt_120s.csv',
];
const OUTPUT_POINT_DIRS = ['observations', 'features', 'ground_truth', 'radius_labels'];

function csvParse(text) {
  const rows = [];
  let row = [], field = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (c === '"') quoted = false;
      else field += c;
    } else if (c === '"') quoted = true;
    else if (c === ',') { row.push(field); field = ''; }
    else if (c === '\n') { row.push(field.replace(/\r$/, '')); rows.push(row); row = []; field = ''; }
    else field += c;
  }
  if (field.length || row.length) { row.push(field.replace(/\r$/, '')); rows.push(row); }
  const headers = rows.shift() ?? [];
  return rows.filter(r => !(r.length === 1 && r[0] === '')).map(values => Object.fromEntries(headers.map((h, i) => [h, values[i] ?? ''])));
}

function csvEscape(value) {
  const s = value === null || value === undefined ? '' : String(value);
  return /[",\r\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
}

function writeCsv(file, rows, columns = null) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const cols = columns ?? [...new Set(rows.flatMap(r => Object.keys(r)))];
  const body = [cols.join(','), ...rows.map(r => cols.map(c => csvEscape(r[c])).join(','))].join('\n') + '\n';
  fs.writeFileSync(file, body, 'utf8');
}

function readCsv(file) { return csvParse(fs.readFileSync(file, 'utf8')); }
function sha256(file) { return crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex'); }
function round(n, digits = 3) { return Number(n.toFixed(digits)); }
function clamp(n, lo, hi) { return Math.max(lo, Math.min(hi, n)); }

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function normal(rng) {
  const u = Math.max(Number.EPSILON, rng());
  const v = Math.max(Number.EPSILON, rng());
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function seedFrom(...parts) {
  const h = crypto.createHash('sha256').update(parts.join('|')).digest();
  return h.readUInt32LE(0) ^ SEED;
}

function haversineM(aLat, aLon, bLat, bLon) {
  const R = 6371008.8, rad = Math.PI / 180;
  const p1 = aLat * rad, p2 = bLat * rad;
  const dp = (bLat - aLat) * rad, dl = (bLon - aLon) * rad;
  const q = Math.sin(dp / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(q));
}

function localToGeo(originLat, originLon, x, y) {
  const lat = originLat + y / 110540;
  const lon = originLon + x / (111320 * Math.cos(originLat * Math.PI / 180));
  return { lat: round(lat, 7), lon: round(lon, 7) };
}

function localMetric(point, originLat, originLon) {
  return {
    x: (point.lon - originLon) * 111320 * Math.cos(originLat * Math.PI / 180),
    y: (point.lat - originLat) * 110540,
  };
}

function pointSegmentDistance(p, a, b) {
  const dx = b.x - a.x, dy = b.y - a.y;
  const den = dx * dx + dy * dy;
  const t = den ? clamp(((p.x - a.x) * dx + (p.y - a.y) * dy) / den, 0, 1) : 0;
  return Math.hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy));
}

function geometryWkt(points) {
  return `LINESTRING (${points.map(p => `${p.lon} ${p.lat}`).join(', ')})`;
}

function profileRaw() {
  const profile = [];
  for (const name of EXPECTED_RAW) {
    const file = path.join(RAW, name);
    if (!fs.existsSync(file)) throw new Error(`Required raw dataset missing: ${file}`);
    const stat = fs.statSync(file);
    const rows = readCsv(file);
    const columns = Object.keys(rows[0] ?? {});
    const missingCells = rows.reduce((n, row) => n + Object.values(row).filter(v => v === '').length, 0);
    const inferredTypes = Object.fromEntries(columns.map(column => {
      const values = rows.map(r => r[column]).filter(v => v !== '');
      const numeric = values.length > 0 && values.every(v => Number.isFinite(Number(v)));
      return [column, numeric ? 'numeric-like' : 'string/category'];
    }));
    const hashes = new Map();
    for (const row of rows) { const k = JSON.stringify(row); hashes.set(k, (hashes.get(k) ?? 0) + 1); }
    const duplicateExcessRows = [...hashes.values()].reduce((n, count) => n + Math.max(0, count - 1), 0);
    profile.push({ file: name, records: rows.length, column_count: columns.length, column_names: columns.join('|'), inferred_types_json: JSON.stringify(inferredTypes), missing_cells: missingCells, duplicate_excess_rows: duplicateExcessRows, bytes: stat.size, sha256: sha256(file) });
  }
  return profile;
}

function makeLegacyAugmentation(rawRoad, rawContext, gt) {
  const added = [];
  const specs = [
    { id: 'E220', base: 'E219', gtId: 'E220', layer: 0, context: 'complex_intersection' },
    { id: 'E355', base: 'E354', gtId: 'E355', layer: 0, context: 'complex_intersection' },
    { id: 'E821', base: 'E820', gtId: 'E821', layer: 1, context: 'airport_multilevel' },
  ];
  for (const [idx, spec] of specs.entries()) {
    const base = rawRoad.find(r => r.edge_id === spec.base);
    if (!base) throw new Error(`Cannot create connected synthetic counterpart for ${spec.id}; base ${spec.base} absent`);
    const from = { lat: Number(base.from_lat), lon: Number(base.from_lon) };
    const to = { lat: Number(base.to_lat), lon: Number(base.to_lon) };
    const originLat = (from.lat + to.lat) / 2, originLon = (from.lon + to.lon) / 2;
    const a = localMetric(from, originLat, originLon), b = localMetric(to, originLat, originLon);
    const dx = b.x - a.x, dy = b.y - a.y, norm = Math.hypot(dx, dy);
    const offset = spec.id === 'E821' ? 24 : spec.id === 'E220' ? 14 : -14;
    const mid = localToGeo(originLat, originLon, (a.x + b.x) / 2 - dy / norm * offset, (a.y + b.y) / 2 + dx / norm * offset);
    const length = haversineM(from.lat, from.lon, mid.lat, mid.lon) + haversineM(mid.lat, mid.lon, to.lat, to.lon);
    added.push({
      edge_id: spec.id, from_node: base.from_node, to_node: base.to_node,
      from_lat: base.from_lat, from_lon: base.from_lon, to_lat: base.to_lat, to_lon: base.to_lon,
      highway: base.highway, length_m: Math.round(length), speed_limit_kmh: base.speed_limit_kmh,
      oneway: base.oneway, bridge: spec.layer ? '1' : base.bridge, tunnel: base.tunnel, layer: String(spec.layer),
      scenario_id: 'LEGACY', source_kind: 'synthetic_parallel_counterpart',
      geometry_wkt: geometryWkt([from, mid, to]), supersedes_unresolved_id: spec.gtId,
      geometry_verified_against_raw_labels: 'false',
    });
  }
  const baseCols = Object.keys(rawRoad[0]);
  const legacy = rawRoad.map(r => ({ ...r, scenario_id: 'LEGACY', source_kind: 'raw_network', geometry_wkt: geometryWkt([
    { lat: Number(r.from_lat), lon: Number(r.from_lon) }, { lat: Number(r.to_lat), lon: Number(r.to_lon) },
  ]), supersedes_unresolved_id: '', geometry_verified_against_raw_labels: 'not_applicable' }));
  const all = [...legacy, ...added];
  const ctxRows = rawContext.map(r => ({ ...r, scenario_id: 'LEGACY', source_kind: 'raw_context' }));
  const newFlags = specs.map((s, i) => {
    const source = rawContext.find(r => r.edge_id === s.base) ?? {};
    const byId = gt.filter(r => r.edge_id === s.gtId);
    const inferredSignals = s.context === 'complex_intersection' ? '2' : '1';
    return {
      edge_id: s.id, highway: source.highway ?? rawRoad.find(r => r.edge_id === s.base).highway,
      urban_canyon: s.context === 'urban_canyon' ? '1' : '0', complex_intersection: s.context === 'complex_intersection' ? '1' : '0',
      road_density_class: i < 2 ? 'medium' : 'high', avg_nearby_edges_within_100m: String(Math.max(2, Number(source.avg_nearby_edges_within_100m ?? 1) + 1)),
      signal_count: inferredSignals, scenario_id: 'LEGACY', source_kind: 'synthetic_context',
      context_label_source_rows: String(byId.length),
    };
  });
  return { network: all, contexts: [...ctxRows, ...newFlags], additions: added };
}

function addNode(graph, key, x, y, layer = 0) {
  if (!graph.nodes.has(key)) graph.nodes.set(key, { node_id: key, x, y, layer });
  return key;
}

function buildScenarioGraph(context, scenarioId, originLat, originLon, specialEdgeId = null) {
  const graph = { nodes: new Map(), edges: [], routeEdgeIds: [] };
  const xs = Array.from({ length: 13 }, (_, i) => -1500 + i * 250);
  const addEdge = (from, to, highway = 'primary', layer = 0, bridge = 0, tunnel = 0, route = false) => {
    const A = graph.nodes.get(from), B = graph.nodes.get(to);
    const n = graph.edges.length + 1;
    const edgeId = route && specialEdgeId && graph.routeEdgeIds.length === 5 ? specialEdgeId : `${scenarioId}_E${String(n).padStart(4, '0')}`;
    const p1 = localToGeo(originLat, originLon, A.x, A.y), p2 = localToGeo(originLat, originLon, B.x, B.y);
    const localLen = Math.hypot(B.x - A.x, B.y - A.y);
    const edge = {
      scenario_id: scenarioId, edge_id: edgeId, from_node: from, to_node: to,
      from_lat: p1.lat, from_lon: p1.lon, to_lat: p2.lat, to_lon: p2.lon,
      highway, length_m: Math.round(localLen), speed_limit_kmh: highway === 'motorway' ? 80 : highway === 'primary' ? 50 : 30,
      oneway: 1, bridge, tunnel, layer, source_kind: specialEdgeId && edgeId === specialEdgeId ? 'synthetic_missing_id_recreated' : 'synthetic_scenario_graph',
      scenario_type: context, geometry_wkt: geometryWkt([p1, p2]),
      _a: A, _b: B,
    };
    graph.edges.push(edge);
    if (route) graph.routeEdgeIds.push(edgeId);
    return edge;
  };
  const horizontalRows = context === 'open_highway' ? [-300, 0, 300]
    : context === 'urban_canyon' ? [-80, 0, 80]
      : context === 'complex_intersection' ? [-160, -80, 0, 80, 160] : [-120, 0, 120];
  const isRouteRow = y => y === 0;
  for (const [ri, y] of horizontalRows.entries()) {
    const highway = context === 'open_highway' ? 'motorway' : context === 'complex_intersection' ? 'primary' : 'secondary';
    for (let i = 0; i < 12; i++) {
      const u = addNode(graph, `${scenarioId}_N_L${ri}_${i}`, xs[i], y, 0);
      const v = addNode(graph, `${scenarioId}_N_L${ri}_${i + 1}`, xs[i + 1], y, 0);
      addEdge(u, v, highway, 0, 0, 0, isRouteRow(y));
    }
  }
  if (context === 'open_highway') {
    for (const xi of [0, 12]) for (let r = 0; r < 2; r++) {
      addEdge(`${scenarioId}_N_L${r}_${xi}`, `${scenarioId}_N_L${r + 1}_${xi}`, 'secondary');
    }
  } else if (context === 'urban_canyon') {
    for (const xi of [0, 3, 6, 9, 12]) for (let r = 0; r < 2; r++) {
      addEdge(`${scenarioId}_N_L${r}_${xi}`, `${scenarioId}_N_L${r + 1}_${xi}`, 'secondary');
    }
  } else if (context === 'complex_intersection') {
    for (let xi = 0; xi <= 12; xi++) for (let r = 0; r < horizontalRows.length - 1; r++) {
      addEdge(`${scenarioId}_N_L${r}_${xi}`, `${scenarioId}_N_L${r + 1}_${xi}`, 'secondary');
    }
  } else {
    for (const xi of [0, 6, 12]) for (let r = 0; r < 2; r++) {
      addEdge(`${scenarioId}_N_L${r}_${xi}`, `${scenarioId}_N_L${r + 1}_${xi}`, 'secondary');
    }
    // A distinct layer crosses the surface route at the centre. Its own nodes do not merge at the crossing.
    const flyNodes = [];
    for (let r = 0; r <= 4; r++) {
      const y = -200 + r * 100;
      flyNodes.push(addNode(graph, `${scenarioId}_N_FLY_${r}`, 0, y, 1));
      if (r > 0) addEdge(flyNodes[r - 1], flyNodes[r], 'primary', 1, 1, 0);
    }
    addEdge(`${scenarioId}_N_L1_6`, flyNodes[0], 'secondary', 0, 1, 0);
    addEdge(flyNodes.at(-1), `${scenarioId}_N_L1_6`, 'secondary', 0, 1, 0);
  }
  if (graph.routeEdgeIds.length !== 12) throw new Error(`${scenarioId}: expected 12 route edges, got ${graph.routeEdgeIds.length}`);
  return graph;
}

function makeScenarioSpecs() {
  const specs = [];
  let n = 1;
  for (const context of CONTEXTS) for (const noise of NOISES) {
    for (const split of SPLITS) for (let replicate = 0; replicate < 2; replicate++) {
      specs.push({ scenario_id: `SC${String(n++).padStart(3, '0')}`, context, noise, split, replicate });
    }
  }
  return specs;
}

function pointAtDistance(s) {
  const d = clamp(s, 0, 3000 - 1e-6);
  const edgeIndex = Math.floor(d / 250);
  const offset = d - edgeIndex * 250;
  return { x: -1500 + edgeIndex * 250 + offset, y: 0, edgeIndex };
}

function lineProjectionFeatures(obs, graph, originLat, originLon, sigma) {
  const p = localMetric(obs, originLat, originLon);
  const distances = graph.edges.map(e => ({ edge: e, d: pointSegmentDistance(p, e._a, e._b) })).sort((a, b) => a.d - b.d);
  const nearby = distances.filter(x => x.d <= 100);
  const layers = new Set(nearby.map(x => x.edge.layer));
  const densityClass = nearby.length <= 3 ? 'low' : nearby.length <= 8 ? 'medium' : 'high';
  const nearest = distances[0]?.d ?? 0;
  const second = distances[1]?.d ?? nearest + 100;
  const logDensity = Math.log2(1 + nearby.length);
  const layerAmbiguity = Math.max(0, layers.size - 1);
  const rec = Math.max(nearest + 0.5, 1.96 * sigma + 0.35 * sigma * logDensity + 0.6 * sigma * layerAmbiguity);
  const min = Math.max(nearest + 0.25, 1.28 * sigma);
  const max = rec + Math.max(0.5 * sigma, Math.min(0.25 * Math.max(0, second - nearest), 2 * sigma));
  return {
    p, nearest, second,
    feature: {
      sigma_m: round(sigma, 3), nearby_edge_count_100m: nearby.length,
      road_density_class: densityClass, layer_count_100m: layers.size,
      nearest_candidate_distance_m: round(nearest), second_candidate_distance_m: round(second),
      local_complexity_score: round(logDensity + 0.75 * layerAmbiguity),
    },
    radius: { min: round(min), recommended: round(rec), max: round(Math.max(rec, max)) },
  };
}

function timeAt(secondFromStart, scenarioIndex) {
  const start = Date.UTC(2025, 3, 12, 2, 30, 0) + scenarioIndex * 10 * 60 * 1000; // 08:00 +05:30
  const d = new Date(start + secondFromStart * 1000 + 330 * 60 * 1000);
  const z = n => String(n).padStart(2, '0');
  return `${d.getUTCFullYear()}-${z(d.getUTCMonth() + 1)}-${z(d.getUTCDate())}T${z(d.getUTCHours())}:${z(d.getUTCMinutes())}:${z(d.getUTCSeconds())}+05:30`;
}

function scenarioOrigin(index) {
  const col = index % 8, row = Math.floor(index / 8);
  return { lat: 28.665 + row * 0.062, lon: 77.245 + col * 0.073 };
}

function main() {
  fs.mkdirSync(OUT, { recursive: true });
  fs.mkdirSync(REPORTS, { recursive: true });
  for (const split of SPLITS) for (const dir of OUTPUT_POINT_DIRS) fs.mkdirSync(path.join(OUT, split, dir), { recursive: true });

  const rawProfile = profileRaw();
  writeCsv(path.join(REPORTS, 'source_profile.csv'), rawProfile);
  const rawRoad = readCsv(path.join(RAW, 'delhi_road_network.csv'));
  const rawContext = readCsv(path.join(RAW, 'road_context_flags.csv'));
  const rawGt = readCsv(path.join(RAW, 'delhi_ground_truth_90km.csv'));
  const rawAdaptive = readCsv(path.join(RAW, 'adaptive_radius_ground_truth.csv'));
  const legacy = makeLegacyAugmentation(rawRoad, rawContext, rawGt);
  const legacyNetFile = path.join(OUT, 'legacy_augmented_road_network.csv');
  const legacyCtxFile = path.join(OUT, 'legacy_augmented_road_context_flags.csv');
  writeCsv(legacyNetFile, legacy.network);
  writeCsv(legacyCtxFile, legacy.contexts);

  const specs = makeScenarioSpecs();
  const networkRows = [], contextRows = [], manifests = [];
  const aggregatedRows = Object.fromEntries(SPLITS.map(split => [split, Object.fromEntries(OUTPUT_POINT_DIRS.map(dir => [dir, Object.fromEntries(INTERVALS.map(s => [s, []]))]))]));
  const usedSpecial = new Map([
    ['SC025', 'E220'], ['SC032', 'E355'], ['SC046', 'E821'],
  ]);
  let observationTotal = 0;
  const rowCountsByInterval = Object.fromEntries(INTERVALS.map(s => [s, 0]));
  const rowCountsBySplitInterval = Object.fromEntries(SPLITS.map(split => [split, Object.fromEntries(INTERVALS.map(s => [s, 0]))]));
  const splitScenarioCounts = Object.fromEntries(SPLITS.map(s => [s, 0]));
  const contextCounts = Object.fromEntries(CONTEXTS.map(c => [c, 0]));
  const noiseCounts = Object.fromEntries(NOISES.map(c => [c, 0]));

  for (const [scenarioIndex, spec] of specs.entries()) {
    const origin = scenarioOrigin(scenarioIndex);
    const specialId = usedSpecial.get(spec.scenario_id) ?? null;
    const graph = buildScenarioGraph(spec.context, spec.scenario_id, origin.lat, origin.lon, specialId);
    const sigmaRng = mulberry32(seedFrom(spec.scenario_id, 'sigma'));
    const sigma = spec.noise === 'low' ? 3.5 + sigmaRng() * 4 : 18 + sigmaRng() * 18;
    const scenarioManifest = {
      scenario_id: spec.scenario_id, split: spec.split, scenario_type: spec.context, gps_noise_tier: spec.noise,
      replicate: spec.replicate, sampling_intervals_s: INTERVALS.join('|'), duration_s: 600,
      nominal_sigma_m: round(sigma), seed: seedFrom(spec.scenario_id, 'scenario'),
      route_edge_count: 12, map_edge_count: graph.edges.length,
      generation_status: 'generated_independent_scenario',
    };
    manifests.push(scenarioManifest);
    splitScenarioCounts[spec.split]++;
    contextCounts[spec.context]++;
    noiseCounts[spec.noise]++;

    for (const e of graph.edges) {
      const row = { ...e };
      delete row._a; delete row._b;
      networkRows.push(row);
      const mid = localToGeo(origin.lat, origin.lon, (e._a.x + e._b.x) / 2, (e._a.y + e._b.y) / 2);
      const m = localMetric(mid, origin.lat, origin.lon);
      const count = graph.edges.filter(other => pointSegmentDistance(m, other._a, other._b) <= 100).length;
      contextRows.push({
        scenario_id: spec.scenario_id, edge_id: e.edge_id, highway: e.highway,
        urban_canyon: spec.context === 'urban_canyon' ? 1 : 0,
        complex_intersection: spec.context === 'complex_intersection' ? 1 : 0,
        road_density_class: count <= 3 ? 'low' : count <= 8 ? 'medium' : 'high',
        avg_nearby_edges_within_100m: count, signal_count: spec.context === 'complex_intersection' ? Math.max(1, Math.round(count / 10)) : 0,
        layer_count_at_edge: new Set(graph.edges.filter(other => pointSegmentDistance(m, other._a, other._b) <= 100).map(other => other.layer)).size,
        source_kind: 'derived_from_synthetic_scenario_network',
      });
    }

    const canonical = [];
    let distanceAlongCanonical = 0;
    for (let sec = 0; sec < 600; sec++) {
      const speedMs = 3.25 + 0.55 * Math.sin(sec / 71 + scenarioIndex) + 0.2 * Math.sin(sec / 19);
      if (sec > 0) distanceAlongCanonical += 3.25 + 0.55 * Math.sin((sec - 1) / 71 + scenarioIndex) + 0.2 * Math.sin((sec - 1) / 19);
      canonical.push({ speedMs, point: pointAtDistance(distanceAlongCanonical) });
    }
    const radiusRows = Object.fromEntries(INTERVALS.map(s => [s, []]));
    const featureRows = Object.fromEntries(INTERVALS.map(s => [s, []]));
    const truthRows = Object.fromEntries(INTERVALS.map(s => [s, []]));
    const observationRows = Object.fromEntries(INTERVALS.map(s => [s, []]));
    for (const interval of INTERVALS) {
      const rng = mulberry32(seedFrom(spec.scenario_id, 'sampling', interval));
      let ex = 0, ey = 0;
      const phi = 0.35;
      for (let sec = 0; sec < 600; sec += interval) {
        const { speedMs, point: truthLocal } = canonical[sec];
        const trueEdgeId = graph.routeEdgeIds[truthLocal.edgeIndex];
        const truthGeo = localToGeo(origin.lat, origin.lon, truthLocal.x, truthLocal.y);
        const noiseScale = sigma * Math.sqrt(1 - phi * phi);
        ex = phi * ex + normal(rng) * noiseScale;
        ey = phi * ey + normal(rng) * noiseScale;
        if (rng() < 0.012) { ex += normal(rng) * sigma * 1.8; ey += normal(rng) * sigma * 1.8; }
        const noisyGeo = localToGeo(origin.lat, origin.lon, truthLocal.x + ex, truthLocal.y + ey);
        const seq = sec + 1;
        const timestamp = timeAt(sec, scenarioIndex);
        const key = { scenario_id: spec.scenario_id, trajectory_id: `${spec.scenario_id}_T${interval}`, split: spec.split, sampling_interval_s: interval, seq, timestamp };
        const obs = { ...key, lat: noisyGeo.lat, lon: noisyGeo.lon, sigma_m: round(sigma), speed_kmh: round(speedMs * 3.6, 2), heading_deg: 90 };
        const f = lineProjectionFeatures(noisyGeo, graph, origin.lat, origin.lon, sigma);
        observationRows[interval].push(obs);
        featureRows[interval].push({ ...key, ...f.feature });
        truthRows[interval].push({ ...key, lat_true: truthGeo.lat, lon_true: truthGeo.lon, edge_id_true: trueEdgeId, altitude_m: 216 });
        radiusRows[interval].push({ ...key, min_ok_radius_m: f.radius.min, recommended_radius_m: f.radius.recommended, max_ok_radius_m: f.radius.max, label_method: 'noise_and_map_geometry_v1' });
      }
    }
    for (const interval of INTERVALS) {
      aggregatedRows[spec.split].observations[interval].push(...observationRows[interval]);
      aggregatedRows[spec.split].features[interval].push(...featureRows[interval]);
      aggregatedRows[spec.split].ground_truth[interval].push(...truthRows[interval]);
      aggregatedRows[spec.split].radius_labels[interval].push(...radiusRows[interval]);
      const count = observationRows[interval].length;
      rowCountsByInterval[interval] += count;
      rowCountsBySplitInterval[spec.split][interval] += count;
      observationTotal += count;
    }
  }

  for (const split of SPLITS) for (const interval of INTERVALS) {
    for (const role of OUTPUT_POINT_DIRS) {
      const data = aggregatedRows[split][role][interval];
      const dest = path.join(OUT, split, role, `${role === 'observations' ? 'traj' : role}_dt_${interval}s.csv`);
      writeCsv(dest, data);
      if (!fs.existsSync(dest)) throw new Error(`Output file was not created: ${dest}`);
    }
  }

  // Keep the EPSG:4326 benchmark maps separate from the raw-CRS legacy augmentation.
  const benchmarkNetwork = networkRows;
  const benchmarkContext = contextRows;
  writeCsv(path.join(OUT, 'road_network.csv'), benchmarkNetwork);
  writeCsv(path.join(OUT, 'road_context_features.csv'), benchmarkContext);
  writeCsv(path.join(OUT, 'scenario_manifest.csv'), manifests);

  const scenarioDistribution = [];
  for (const split of SPLITS) for (const context of CONTEXTS) for (const noise of NOISES) {
    const n = specs.filter(s => s.split === split && s.context === context && s.noise === noise).length;
    scenarioDistribution.push({ split, scenario_type: context, gps_noise_tier: noise, scenarios: n });
  }
  writeCsv(path.join(REPORTS, 'scenario_distribution.csv'), scenarioDistribution);
  const samplingDistribution = INTERVALS.map(s => ({ sampling_interval_s: s, records: rowCountsByInterval[s], scenarios: specs.length }));
  writeCsv(path.join(OUT, 'sampling_distribution.csv'), samplingDistribution);
  writeCsv(path.join(OUT, 'split_sampling_distribution.csv'), SPLITS.flatMap(split => INTERVALS.map(s => ({ split, sampling_interval_s: s, records: rowCountsBySplitInterval[split][s], scenarios: splitScenarioCounts[split] }))));

  const beforeRefMissing = rawGt.filter(r => !rawRoad.some(e => e.edge_id === r.edge_id)).length;
  const beforeAdaptiveMissing = rawAdaptive.filter(r => !rawRoad.some(e => e.edge_id === r.edge_id_true)).length;
  const beforeRadiusOutside = rawAdaptive.filter(r => Number(r.recommended_radius_m) < Number(r.min_ok_radius_m) || Number(r.recommended_radius_m) > Number(r.max_ok_radius_m)).length;
  const beforeRadiusInverted = rawAdaptive.filter(r => Number(r.min_ok_radius_m) > Number(r.max_ok_radius_m)).length;
  const rawDuplicateExcess = rawProfile.reduce((n, r) => n + r.duplicate_excess_rows, 0);
  const rawMissingCells = rawProfile.reduce((n, r) => n + r.missing_cells, 0);
  const generatedGt = [];
  const generatedRadius = [];
  const generatedObs = [];
  const generatedFeatures = [];
  for (const split of SPLITS) for (const interval of INTERVALS) {
    for (const [dir, arr] of [['ground_truth', generatedGt], ['radius_labels', generatedRadius], ['observations', generatedObs], ['features', generatedFeatures]]) {
      arr.push(...readCsv(path.join(OUT, split, dir, `${dir === 'observations' ? 'traj' : dir}_dt_${interval}s.csv`)));
    }
  }
  const edgeKeys = new Set(benchmarkNetwork.map(e => `${e.scenario_id}|${e.edge_id}`));
  const generatedUnresolved = generatedGt.filter(r => !edgeKeys.has(`${r.scenario_id}|${r.edge_id_true}`)).length;
  const radiusInvalid = generatedRadius.filter(r => Number(r.min_ok_radius_m) > Number(r.max_ok_radius_m) || Number(r.recommended_radius_m) < Number(r.min_ok_radius_m) || Number(r.recommended_radius_m) > Number(r.max_ok_radius_m)).length;
  const pointRoleRows = generatedObs.length + generatedFeatures.length + generatedGt.length + generatedRadius.length;
  const coreArtifactRows = pointRoleRows + benchmarkNetwork.length + benchmarkContext.length + manifests.length;

  const sourceManifest = {
    generated_at_utc: new Date().toISOString(), seed: SEED,
    raw_directory: 'data/raw', raw_files: rawProfile,
    preservation_check: 'Hashes are rechecked by automated validation tests after generation.',
    generation_code: 'scripts/build_synthetic_v2.mjs',
  };
  fs.writeFileSync(path.join(OUT, 'source_manifest.json'), JSON.stringify(sourceManifest, null, 2) + '\n', 'utf8');

  const featureAllowlist = {
    row_keys_metadata_only_not_predictors: ['scenario_id', 'trajectory_id', 'split', 'sampling_interval_s', 'seq', 'timestamp'],
    observation_predictor_allowlist: ['lat', 'lon', 'sigma_m', 'speed_kmh', 'heading_deg'],
    network_predictor_allowlist: ['geometry_wkt', 'length_m', 'highway', 'speed_limit_kmh', 'oneway', 'bridge', 'tunnel', 'layer', 'nearby_edge_count_100m', 'road_density_class', 'layer_count_100m', 'nearest_candidate_distance_m', 'second_candidate_distance_m', 'local_complexity_score'],
    network_identifiers_and_generation_metadata_not_predictors: ['scenario_id', 'edge_id', 'from_node', 'to_node', 'source_kind', 'scenario_type'],
    prohibited_target_or_truth_columns: ['lat_true', 'lon_true', 'edge_id_true', 'min_ok_radius_m', 'recommended_radius_m', 'max_ok_radius_m', 'evaluation_metrics'],
    scenario_manifest_not_predictors: ['scenario_type', 'gps_noise_tier', 'replicate', 'split', 'seed'],
    feature_provenance_rule: 'Map features are derived from the noisy observation and scenario-specific network only; generation code does not pass truth rows to feature derivation. Keys and scenario metadata are never model predictors.',
  };
  fs.writeFileSync(path.join(OUT, 'feature_allowlist.json'), JSON.stringify(featureAllowlist, null, 2) + '\n', 'utf8');
  fs.writeFileSync(path.join(OUT, 'coordinate_reference.json'), JSON.stringify({
    generated_coordinate_crs: 'EPSG:4326', axis_order: ['latitude', 'longitude'],
    metric_generation_method: 'Local east/north tangent approximation per scenario origin; latitude scale 110540 m/degree and longitude scale 111320*cos(origin_latitude) m/degree.',
    distance_unit: 'metre', raw_source_crs: 'unknown; raw coordinates were not transformed or used to assert a CRS.',
    scenario_origin_method: 'Grid of scenario origins within the numeric extent of the supplied study area; each synthetic map is a separate local world.',
  }, null, 2) + '\n', 'utf8');
  fs.writeFileSync(path.join(OUT, 'legacy_network_metadata.json'), JSON.stringify({
    coordinate_reference: 'Inherited numeric coordinate frame from data/raw/delhi_road_network.csv; formal CRS remains unknown.',
    note: 'This file is an audit-preserving augmented legacy network and is not the benchmark road network. Three synthetic counterparts share connected endpoint nodes but are not verified against original ground-truth geometry.',
    edge_ids_added: legacy.additions.map(e => e.edge_id), raw_network_edges: rawRoad.length, augmented_edges: legacy.network.length,
  }, null, 2) + '\n', 'utf8');

  const cleaningLog = [
    { issue: 'Missing raw network edge IDs E220/E355/E821', records_affected: beforeRefMissing + beforeAdaptiveMissing, correction_method: 'Added three synthetic connected counterpart edges to the legacy augmented network; separately regenerated benchmark maps and labels from connected graphs.', reason: 'Provide valid synthetic map coverage without claiming raw label recovery.', records_corrected: 0, records_resolved_by_id: beforeRefMissing + beforeAdaptiveMissing, synthetic_v2_records_generated: 3, records_unresolved: beforeRefMissing + beforeAdaptiveMissing, evidence: `${beforeRefMissing} rows in delhi_ground_truth_90km.csv and ${beforeAdaptiveMissing} rows in adaptive_radius_ground_truth.csv; ID joins resolve after augmentation, spatial alignment remains unverified.` },
    { issue: 'Contradictory raw adaptive-radius bounds', records_affected: rawAdaptive.length, correction_method: 'Generated new labels from noisy GPS uncertainty and map-derived candidate geometry/density; raw labels retained unchanged and unresolved.', reason: 'Ensure v2 benchmark intervals are valid while preserving original source.', records_corrected: 0, synthetic_v2_records_generated: generatedRadius.length, records_unresolved: rawAdaptive.length, evidence: `${beforeRadiusInverted} raw intervals inverted; ${beforeRadiusOutside} recommendations outside bounds.` },
    { issue: 'Exact duplicate full rows', records_affected: rawDuplicateExcess, correction_method: 'No deduplication applied because source profiling found no exact duplicate excess.', reason: 'Avoid unnecessary loss of source rows.', records_corrected: 0, synthetic_v2_records_generated: 0, records_unresolved: 0, evidence: 'Source row-level duplicate hash check.' },
    { issue: 'Blank/missing raw CSV cells', records_affected: rawMissingCells, correction_method: 'No imputation or deletion applied; source values are preserved.', reason: 'Avoid inventing values when profiling found no blanks.', records_corrected: 0, synthetic_v2_records_generated: 0, records_unresolved: rawMissingCells, evidence: 'Cell-level blank check across all 12 raw CSV files.' },
    { issue: 'CRS and distance methodology undocumented in raw files', records_affected: rawProfile.reduce((n, f) => n + f.records, 0), correction_method: 'Raw coordinates left untouched; new synthetic maps generated in local metre coordinates and exported as EPSG:4326 with explicit metadata.', reason: 'Avoid guessing the CRS of legacy files; create a known metric benchmark.', records_corrected: 0, synthetic_v2_records_generated: generatedObs.length, records_unresolved: rawProfile.reduce((n, f) => n + f.records, 0), evidence: 'No CRS declaration in raw CSV schemas; v2 generator defines local tangent plane anchored to EPSG:4326 origins.' },
    { issue: 'Only one apparent source trajectory and no independent split', records_affected: rawGt.length, correction_method: 'Generated 48 independent route/network scenarios; split by scenario before sampling variants.', reason: 'Prevent point-level and sampling-variant cross-split leakage.', records_corrected: 0, synthetic_v2_records_generated: specs.length, records_unresolved: 0, evidence: '16 scenarios per split, balanced across four road contexts and two noise tiers.' },
  ];
  writeCsv(path.join(REPORTS, 'cleaning_log.csv'), cleaningLog);

  const beforeAfter = [
    { measure: 'CSV data records in all 12 raw files', before: rawProfile.reduce((n, f) => n + f.records, 0), after: '', unit: 'rows', note: 'Raw inputs preserved; generated v2 is a new benchmark and not a row-wise cleaned copy.' },
    { measure: 'Blank/missing cells in raw CSVs', before: rawMissingCells, after: rawMissingCells, unit: 'cells', note: 'No imputations performed; source values retained.' },
    { measure: 'Raw edge-label references missing from original network', before: beforeRefMissing + beforeAdaptiveMissing, after: 0, unit: 'references across two GT files', note: `${beforeRefMissing} in reference GT and ${beforeAdaptiveMissing} in adaptive GT; edge IDs now resolve by name after synthetic augmentation.` },
    { measure: 'Raw label rows with synthetic geometry not independently verified', before: beforeRefMissing + beforeAdaptiveMissing, after: beforeRefMissing + beforeAdaptiveMissing, unit: 'references across two GT files', note: 'All 140 remain ineligible for legacy evaluation until label-to-geometry agreement is validated.' },
    { measure: 'Adaptive rows with invalid stated intervals', before: beforeRadiusInverted, after: radiusInvalid, unit: 'rows', note: 'After value is generated v2 radius-label rows; raw 7,200 labels remain unresolved.' },
    { measure: 'Adaptive recommendations outside stated interval', before: beforeRadiusOutside, after: radiusInvalid, unit: 'rows', note: 'Generated v2 validation checks ordering and membership.' },
    { measure: 'Synthetic paired observation points', before: '', after: observationTotal, unit: 'points', note: 'Each point appears in four separate role-specific tables.' },
    { measure: 'Synthetic observation rows across seven sampling intervals', before: '', after: generatedObs.length, unit: 'rows', note: 'All scenarios retain split assignment across interval variants.' },
    { measure: 'Synthetic feature rows', before: '', after: generatedFeatures.length, unit: 'rows', note: 'Inference-time-only derived columns.' },
    { measure: 'Synthetic ground-truth rows', before: '', after: generatedGt.length, unit: 'rows', note: 'Evaluation-only table.' },
    { measure: 'Synthetic adaptive-radius label rows', before: '', after: generatedRadius.length, unit: 'rows', note: 'Separate target table.' },
    { measure: 'Synthetic point rows across four separated roles', before: '', after: pointRoleRows, unit: 'rows', note: 'Four tables × one row per paired observation key.' },
    { measure: 'Scenario-specific synthetic network edges', before: '', after: networkRows.length, unit: 'edges', note: 'All scenario components are connected; scenario_id scopes each independent map.' },
    { measure: 'Legacy augmented network edges', before: rawRoad.length, after: legacy.network.length, unit: 'edges', note: 'Three generated counterparts are connected to existing endpoint nodes; not claimed to verify raw GT geometry.' },
    { measure: 'Core output artifact rows (point-role tables + network + context + manifest)', before: '', after: coreArtifactRows, unit: 'rows', note: 'Excludes reporting CSVs and distribution summaries.' },
  ];
  writeCsv(path.join(REPORTS, 'before_after_comparison.csv'), beforeAfter);

  const summary = {
    project: 'GeoMotion AI', phase: 'Phase 2 — Synthetic Dataset Correction and Benchmark Preparation',
    generated_at_utc: sourceManifest.generated_at_utc, seed: SEED,
    raw: {
      directory: 'data/raw', files: rawProfile,
      total_records: rawProfile.reduce((n, f) => n + f.records, 0), missing_cells: rawMissingCells, exact_duplicate_excess_rows: rawDuplicateExcess,
      ground_truth_missing_ids_against_original_network_reference_file: beforeRefMissing,
      ground_truth_missing_ids_against_original_network_adaptive_file: beforeAdaptiveMissing,
      ground_truth_missing_ids_after_legacy_augmentation: 0,
      legacy_label_geometry_unverified_references: beforeRefMissing + beforeAdaptiveMissing,
      adaptive_radius_rows: rawAdaptive.length, invalid_interval_order_rows: beforeRadiusInverted,
      recommended_radius_outside_interval_rows: beforeRadiusOutside,
      untouched: true,
    },
    generated_benchmark: {
      scenario_count: specs.length, scenario_counts_by_split: splitScenarioCounts,
      scenario_counts_by_type: contextCounts, scenario_counts_by_noise: noiseCounts,
      sampling_intervals_s: INTERVALS, observation_points: observationTotal,
      rows_per_interval: rowCountsByInterval, rows_per_split_interval: rowCountsBySplitInterval,
      paired_points_by_split_across_intervals: Object.fromEntries(SPLITS.map(split => [split, Object.values(rowCountsBySplitInterval[split]).reduce((a, n) => a + n, 0)])),
      observations_rows: generatedObs.length, features_rows: generatedFeatures.length,
      ground_truth_rows: generatedGt.length, adaptive_radius_label_rows: generatedRadius.length,
      unique_paired_observation_points: observationTotal, point_role_table_rows_total: pointRoleRows,
      connected_scenario_network_edges: networkRows.length, legacy_network_before_edges: rawRoad.length,
      legacy_network_after_edges: legacy.network.length, named_missing_ids_added: legacy.additions.map(e => e.edge_id),
      benchmark_network_rows: benchmarkNetwork.length, benchmark_network_context_rows: benchmarkContext.length,
      scenario_manifest_rows: manifests.length, core_artifact_rows_total: coreArtifactRows,
      generated_ground_truth_unresolved_edges: generatedUnresolved,
      invalid_generated_radius_rows: radiusInvalid,
    },
    decisions: {
      raw_ground_truth_repaired: false,
      benchmark_is_independently_generated: true,
      missing_edge_policy: 'Synthetic connected counterparts are added; raw labels remain flagged until geometry is independently verified.',
      crs: 'Generated coordinates are EPSG:4326; local metric generation uses a per-scenario tangent approximation. Raw CRS is not inferred.',
      split_policy: 'Scenario is the split unit; all sampling variants remain in the same split.',
      metrics_claimed: false,
      algorithm_training_or_dashboard: false,
    },
    validation: { generated_ground_truth_unresolved_edge_rows: generatedUnresolved, invalid_radius_label_rows: radiusInvalid },
    readiness: {
      synthetic_v2_benchmark: generatedUnresolved === 0 && radiusInvalid === 0 ? 'ready_for_fair_algorithm_comparison_on_generated_scenarios' : 'not_ready',
      legacy_raw_trajectory: 'not_ready_for_fair_evaluation_until_140_reference_rows_are_spatially_verified',
      performance_claims: 'none',
    },
    paths: {
      processed_root: 'data/processed/synthetic_v2',
      network: 'data/processed/synthetic_v2/road_network.csv',
      manifest: 'data/processed/synthetic_v2/scenario_manifest.csv',
      feature_allowlist: 'data/processed/synthetic_v2/feature_allowlist.json',
      preprocessing_report: 'reports/preprocessing_report.md',
      cleaning_log: 'reports/cleaning_log.csv',
      before_after: 'reports/before_after_comparison.csv',
      methodology: 'reports/dataset_generation_methodology.md',
    },
  };
  fs.writeFileSync(path.join(REPORTS, 'preprocessing_summary.json'), JSON.stringify(summary, null, 2) + '\n', 'utf8');

  const report = `# GeoMotion AI — Synthetic v2 Preprocessing Report\n\n` +
    `Generated ${summary.generated_at_utc} with fixed seed ${SEED}. Raw CSVs in \`data/raw/\` were not modified.\n\n` +
    `## Source audit and handling\n\n` +
    `- Raw CSV records across 12 files: **${summary.raw.total_records}**. The Phase 1 summary had an arithmetic undercount; [dataset_audit.md](dataset_audit.md) and its JSON now report the corrected total of 25,708.\n` +
    `- Exact duplicate row excess: **${rawDuplicateExcess}**; no source rows were removed.\n` +
    `- Raw ground-truth references missing from the original network: **${beforeRefMissing} in the reference file and ${beforeAdaptiveMissing} in the adaptive file**. All 140 IDs resolve by name in the augmented network after adding E220/E355/E821; the synthetic geometries are not independently verified against the raw labels, so those 140 rows remain unresolved for legacy evaluation.\n` +
    `- Raw adaptive labels: ${beforeRadiusInverted} inverted intervals and ${beforeRadiusOutside} recommendations outside their stated bounds. Raw values were not edited.\n\n` +
    `## Generated benchmark\n\n` +
    `- **${specs.length}** independently generated route scenarios: ${SPLITS.map(s => `${s} ${splitScenarioCounts[s]}`).join(', ')}. Each split has four road-context classes and two GPS-noise tiers, with two replicates per combination. Each split represents ${Object.values(rowCountsBySplitInterval.train).reduce((a, n) => a + n, 0)} paired point/interval observations.\n` +
    `- **${observationTotal}** paired observation points across seven sampling intervals; the separated observations, feature, ground-truth and radius-label tables contain ${generatedObs.length}, ${generatedFeatures.length}, ${generatedGt.length} and ${generatedRadius.length} rows respectively (${pointRoleRows} point-role table rows total).\n` +
    `- Scenario road edges: **${networkRows.length}**, grouped into connected scenario-specific graphs. Legacy augmented network: ${rawRoad.length} original edges plus three connected synthetic counterparts. Core output artifacts contain ${coreArtifactRows} rows across point-role tables, network, context and scenario manifest.\n` +
    `- Generated ground-truth references unresolved in the scenario network: **${generatedUnresolved}**. Generated radius-label rows failing interval checks: **${radiusInvalid}**.\n\n` +
    `## Readiness\n\n` +
    `The synthetic v2 benchmark is ready for a later fair algorithm comparison on its generated scenarios: Nearest Road, HMM/Viterbi and Adaptive Radius can use the same scenario-specific network and observation keys, the splits are scenario-disjoint, all generated labels resolve, and all generated radius intervals pass validation. This does not guarantee that Adaptive Radius wins or make the benchmark real-world evidence. The legacy raw trajectory/labels are not evaluation-ready while the spatial alignment of the 140 originally unmatched label references remains unverified.\n\n` +
    `## Separation and limits\n\n` +
    `No algorithm was trained, no dashboard was built and no performance metrics were calculated. The v2 benchmark is synthetic and its labels are generator-defined rather than empirical road truth. The three legacy edge IDs are present as connected synthetic geometry in the augmented legacy network, but this is not proof that those geometries reproduce the original raw labels.\n\n` +
    `See [source_profile.csv](source_profile.csv), [dataset_generation_methodology.md](dataset_generation_methodology.md), [feature_engineering_report.md](feature_engineering_report.md), [cleaning_log.csv](cleaning_log.csv), [before_after_comparison.csv](before_after_comparison.csv), and [scenario_distribution.csv](scenario_distribution.csv). Automated validation: \`node --test\`.\n`;
  fs.writeFileSync(path.join(REPORTS, 'preprocessing_report.md'), report, 'utf8');

  const featureReport = `# Feature Engineering and Leakage Controls\n\n` +
    `## Prediction-time inputs\n\n` +
    `The benchmark stores noisy observations, map-derived features, ground truth and radius labels in separate files and directories. The feature table is keyed by scenario, trajectory, interval, sequence and timestamp. It includes only \`sigma_m\`, nearby edge count within 100 m, road-density class, layer count, first/second candidate distances and a local complexity score. Those values are computed from the noisy point and the scenario map; the feature function does not receive true coordinates or true edge IDs.\n\n` +
    `Observation predictors are noisy \`lat/lon\`, \`sigma_m\`, speed and heading. Network-derived predictors are nearby edge count, density class, layer count, nearest/second candidate distance and local complexity score. Scenario/trajectory IDs, split, time and sequence are keys/metadata, not model predictors. Network geometry and road attributes are available by the same \`scenario_id\`.\n\n` +
    `## Targets and evaluation-only fields\n\n` +
    `Ground-truth \`lat_true/lon_true/edge_id_true\` and radius \`min_ok_radius_m/recommended_radius_m/max_ok_radius_m\` are in separate files. They are never emitted into the feature table. Evaluation joins occur only after a prediction has been produced. Raw trajectory radius values and old true labels are not copied into synthetic v2 features.\n\n` +
    `## Radius target construction\n\n` +
    `For each noisy point, the map feature routine computes distance to the nearest and second-nearest edge, count of edges within 100 m, and layer count. With uncertainty \`sigma\`, the benchmark defines \`complexity = log2(1 + nearby_edge_count) + 0.75 * max(0, layer_count - 1)\`; \`min_ok = max(nearest_distance + 0.25, 1.28 * sigma)\`; \`recommended = max(nearest_distance + 0.5, 1.96 * sigma + 0.35 * sigma * complexity)\`; and \`max_ok = recommended + max(0.5 * sigma, min(0.25 * (second_distance - nearest_distance), 2 * sigma))\`. Values are rounded to 0.001 m. By construction the interval is ordered and contains the recommendation.\n\n` +
    `This is a transparent synthetic benchmark target rule, not an empirically optimal radius or a claim about real GPS. It uses available observation/map quantities only; it does not use the true edge, true coordinate, or matching outcome. Scenarios where uncertainty is larger than road separation remain ambiguous; adaptive radius is not guaranteed to improve matching.\n\n` +
    `## Allowlist\n\n` +
    `The machine-readable allowlist is [feature_allowlist.json](../data/processed/synthetic_v2/feature_allowlist.json). Keep truth/target files outside feature joins and keep every sampling variant of a scenario in its assigned split. Generated coordinates and local metric units are described in [coordinate_reference.json](../data/processed/synthetic_v2/coordinate_reference.json).\n`;
  fs.writeFileSync(path.join(REPORTS, 'feature_engineering_report.md'), featureReport, 'utf8');

  const methodology = `# Synthetic v2 Dataset Generation Methodology\n\n` +
    `## Purpose and provenance\n\n` +
    `This creates a controlled synthetic benchmark for later comparisons of Nearest Road, HMM/Viterbi and Adaptive Radius. It is not real-world validation and does not report algorithm performance. The source CSVs are profiled and retained unchanged; the benchmark trajectories, map scenarios, noise and labels are generated anew with seed ${SEED}.\n\n` +
    `## Pipeline stages\n\n` +
    `1. **Load/profile:** read all 12 named CSVs, record file hashes, row counts, columns and bytes in \`source_manifest.json\`.\n` +
    `2. **Clean/profile:** check duplicate full rows and issue counts. No raw rows are edited or dropped; exact raw duplicate excess observed was ${rawDuplicateExcess}.\n` +
    `3. **Missing-edge coverage:** add E220/E355/E821 as explicit synthetic parallel counterparts to connected legacy edges E219/E354/E820, sharing existing endpoint nodes. They carry \`source_kind=synthetic_parallel_counterpart\` and are not described as recovered authoritative geometry. The raw edge labels remain unverified.\n` +
    `4. **Ground-truth policy:** do not silently rewrite legacy edge labels. Instead generate independent v2 routes on connected synthetic graphs and derive each v2 edge label from the graph segment used to generate the true position.\n` +
    `5. **Radius labels:** use the documented inference-time uncertainty/map formula in the feature report. Bounds are internally valid by construction; no values from raw radius labels are reused.\n` +
    `6. **Coordinates:** generate each scenario in a local east/north metre plane anchored to a declared EPSG:4326 origin, then export decimal-degree coordinates. Raw CRS is not inferred or transformed. The local conversion uses 110,540 m/degree latitude and latitude-adjusted 111,320 m/degree longitude.\n` +
    `7. **Feature engineering:** calculate candidate distances and density from noisy observations and the scenario map only. True coordinates and edge IDs are kept in separate ground-truth files.\n` +
    `8. **Sampling:** create a 600-second route and sample the same scenario at 1, 5, 15, 30, 45, 60 and 120 seconds. Each interval receives deterministic independent observation noise; sequence and time identify the same underlying scenario positions.\n` +
    `9. **Validation:** test edge joins, radius bounds, feature allowlist, graph connectivity within each scenario, split exclusivity, interval coverage and raw input hashes.\n` +
    `10. **Splits:** create 48 scenario groups, stratified by four road-context designs and two noise tiers; assign two replicates for each combination to each split (16 train, 16 validation, 16 test). Sampling variants remain grouped by scenario. Fixed seed ${SEED}; no point-level split or test tuning.\n\n` +
    `## Scenario design\n\n` +
    `Each split includes open-highway sparse parallel corridors, urban-canyon close parallel corridors, dense connected grids for complex intersections, and an airport-style map with a layer-separated crossing plus connected ramps. Low-noise sigma is sampled from 3.5–7.5 m; high-noise sigma from 18–36 m. Correlated Gaussian position error is generated per interval with occasional seeded outliers. All generated scenario graphs are connected internally; they are intentionally separate map components keyed by scenario because they are independent benchmark worlds.\n\n` +
    `## Synthetic assumptions and limitations\n\n` +
    `The map layouts, noise model, context classes, target formula and timestamp schedule are controlled design choices. The labels express generator-defined truth, not independent survey truth. The benchmark offers varied ambiguity and sampling conditions but cannot establish real-world accuracy. Nearest road and HMM receive exactly the same observation and network for an eligible scenario as adaptive radius; no method is guaranteed to win.\n`;
  fs.writeFileSync(path.join(REPORTS, 'dataset_generation_methodology.md'), methodology, 'utf8');

  const distributionMd = `# Scenario Distribution\n\n` +
    `Fixed seed ${SEED}. The scenario, not individual GPS points, is the split unit. Each sampling interval for a scenario remains in that same split.\n\n` +
    `| Split | Scenarios | Per context × noise combination |\n|---|---:|---:|\n` +
    SPLITS.map(s => `| ${s} | ${splitScenarioCounts[s]} | 2 |`).join('\n') +
    `\n\nFour context designs and two noise tiers have equal scenario counts. See [scenario_distribution.csv](scenario_distribution.csv) for the full cross-tab and [scenario_manifest.csv](../data/processed/synthetic_v2/scenario_manifest.csv) for IDs and fixed seeds.\n`;
  fs.writeFileSync(path.join(REPORTS, 'scenario_distribution.md'), distributionMd, 'utf8');

  console.log(JSON.stringify({
    raw_records: summary.raw.total_records,
    generated_points: observationTotal,
    rows_per_interval: rowCountsByInterval,
    rows_per_split_interval: rowCountsBySplitInterval,
    scenarios: splitScenarioCounts,
    scenario_network_edges: networkRows.length,
    legacy_augmented_edges: legacy.network.length,
    missing_ids_added: legacy.additions.map(e => e.edge_id),
    generated_unresolved_edges: generatedUnresolved,
    invalid_radius_rows: radiusInvalid,
    output: OUT,
  }, null, 2));
}

main();
