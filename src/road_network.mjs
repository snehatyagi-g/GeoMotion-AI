const METERS_PER_DEG_LAT = 110540;
const METERS_PER_DEG_LON = 111320;

export function haversineMeters(a, b) {
  const R = 6371008.8, rad = Math.PI / 180;
  const p1 = Number(a.lat) * rad, p2 = Number(b.lat) * rad;
  const dp = (Number(b.lat) - Number(a.lat)) * rad;
  const dl = (Number(b.lon) - Number(a.lon)) * rad;
  const q = Math.sin(dp / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(q));
}

function parseWkt(text) {
  const match = String(text ?? '').match(/^LINESTRING\s*\((.+)\)$/i);
  if (!match) return null;
  return match[1].split(',').map(pair => {
    const [lon, lat] = pair.trim().split(/\s+/).map(Number);
    if (!Number.isFinite(lon) || !Number.isFinite(lat)) throw new Error(`Invalid LINESTRING coordinate: ${pair}`);
    return { lat, lon };
  });
}

class MinHeap {
  items = [];
  push(item) {
    const a = this.items; a.push(item); let i = a.length - 1;
    while (i > 0) { const p = (i - 1) >> 1; if (a[p].priority <= item.priority) break; a[i] = a[p]; i = p; }
    a[i] = item;
  }
  pop() {
    const a = this.items;
    if (!a.length) return null;
    const root = a[0], last = a.pop();
    if (a.length) {
      let i = 0;
      while (true) {
        let c = i * 2 + 1;
        if (c >= a.length) break;
        if (c + 1 < a.length && a[c + 1].priority < a[c].priority) c++;
        if (a[c].priority >= last.priority) break;
        a[i] = a[c]; i = c;
      }
      a[i] = last;
    }
    return root;
  }
  get length() { return this.items.length; }
}

export class RoadNetwork {
  constructor(scenarioId, rows) {
    if (!rows.length) throw new Error(`No network edges for scenario ${scenarioId}`);
    this.scenarioId = scenarioId;
    const lats = [], lons = [];
    for (const r of rows) for (const [lat, lon] of [[Number(r.from_lat), Number(r.from_lon)], [Number(r.to_lat), Number(r.to_lon)]]) {
      lats.push(lat); lons.push(lon);
    }
    this.originLat = lats.reduce((a, b) => a + b, 0) / lats.length;
    this.originLon = lons.reduce((a, b) => a + b, 0) / lons.length;
    this.edges = rows.map(r => {
      const geo = parseWkt(r.geometry_wkt) ?? [
        { lat: Number(r.from_lat), lon: Number(r.from_lon) },
        { lat: Number(r.to_lat), lon: Number(r.to_lon) },
      ];
      const points = geo.map(p => this.project(p));
      const segments = [];
      let length = 0;
      for (let i = 1; i < points.length; i++) {
        const a = points[i - 1], b = points[i];
        const len = Math.hypot(b.x - a.x, b.y - a.y);
        segments.push({ a, b, len, offset: length });
        length += len;
      }
      if (!segments.length || length <= 0) throw new Error(`Invalid geometry for ${scenarioId}/${r.edge_id}`);
      return {
        ...r, edgeId: String(r.edge_id), fromNode: String(r.from_node), toNode: String(r.to_node),
        layer: Number(r.layer ?? 0), lengthMeters: length, segments,
      };
    });
    const ids = new Set();
    for (const edge of this.edges) {
      if (ids.has(edge.edgeId)) throw new Error(`Duplicate edge ID in ${scenarioId}: ${edge.edgeId}`);
      ids.add(edge.edgeId);
    }
    this.edgeById = new Map(this.edges.map(e => [e.edgeId, e]));
    this.adjacency = new Map();
    for (const edge of this.edges) {
      this.addArc(edge.fromNode, edge.toNode, edge.lengthMeters);
      if (Number(edge.oneway) === 0 || String(edge.oneway).toLowerCase() === 'false') this.addArc(edge.toNode, edge.fromNode, edge.lengthMeters);
    }
    this.shortestCache = new Map();
    this.transitionCache = new Map();
  }

  project(point) {
    return {
      x: (Number(point.lon) - this.originLon) * METERS_PER_DEG_LON * Math.cos(this.originLat * Math.PI / 180),
      y: (Number(point.lat) - this.originLat) * METERS_PER_DEG_LAT,
    };
  }

  addArc(from, to, weight) {
    if (!this.adjacency.has(from)) this.adjacency.set(from, []);
    this.adjacency.get(from).push({ to, weight });
  }

  distanceToEdge(point, edge) {
    const p = this.project(point);
    let best = null;
    for (const seg of edge.segments) {
      const dx = seg.b.x - seg.a.x, dy = seg.b.y - seg.a.y;
      const den = dx * dx + dy * dy;
      const t = den ? Math.max(0, Math.min(1, ((p.x - seg.a.x) * dx + (p.y - seg.a.y) * dy) / den)) : 0;
      const x = seg.a.x + t * dx, y = seg.a.y + t * dy;
      const distance = Math.hypot(p.x - x, p.y - y);
      if (!best || distance < best.distanceMeters) {
        const bearing = (Math.atan2(dx, dy) * 180 / Math.PI + 360) % 360;
        best = { edge, edgeId: edge.edgeId, distanceMeters: distance, alongMeters: seg.offset + t * seg.len, headingDeg: bearing };
      }
    }
    return best;
  }

  candidates(point) {
    return this.edges.map(edge => this.distanceToEdge(point, edge)).sort((a, b) => a.distanceMeters - b.distanceMeters);
  }

  shortestFrom(source) {
    if (this.shortestCache.has(source)) return this.shortestCache.get(source);
    const distances = new Map([[source, 0]]), heap = new MinHeap();
    heap.push({ node: source, priority: 0 });
    while (heap.length) {
      const cur = heap.pop();
      if (cur.priority !== distances.get(cur.node)) continue;
      for (const arc of this.adjacency.get(cur.node) ?? []) {
        const nd = cur.priority + arc.weight;
        if (nd < (distances.get(arc.to) ?? Infinity)) { distances.set(arc.to, nd); heap.push({ node: arc.to, priority: nd }); }
      }
    }
    this.shortestCache.set(source, distances);
    return distances;
  }

  transitionDistance(fromCandidate, toCandidate) {
    const a = fromCandidate.edge, b = toCandidate.edge;
    if (a.edgeId === b.edgeId) {
      const delta = toCandidate.alongMeters - fromCandidate.alongMeters;
      return { distanceMeters: Math.abs(delta), connected: true, reverse: delta < 0 };
    }
    const cacheKey = `${a.edgeId}|${b.edgeId}`;
    let middle = this.transitionCache.get(cacheKey);
    if (middle === undefined) {
      middle = this.shortestFrom(a.toNode).get(b.fromNode) ?? null;
      this.transitionCache.set(cacheKey, middle);
    }
    if (middle === null) return { distanceMeters: null, connected: false, reverse: false };
    const remaining = Math.max(0, a.lengthMeters - fromCandidate.alongMeters);
    return { distanceMeters: remaining + middle + Math.max(0, toCandidate.alongMeters), connected: true, reverse: false };
  }
}

export function circularDifferenceDeg(a, b) {
  const d = Math.abs(((Number(a) - Number(b) + 180) % 360 + 360) % 360 - 180);
  return d;
}

