"""Scenario-local metric road geometry and directed graph helpers.

Matches the reference implementation's local east/north tangent approximation.
All edge and point-to-segment distances returned by this module are metres.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache

import networkx as nx

METERS_PER_DEG_LAT = 110540.0
METERS_PER_DEG_LON = 111320.0


def haversine_meters(a, b) -> float:
    radius = 6371008.8
    p1, p2 = math.radians(float(a["lat"])), math.radians(float(b["lat"]))
    dp = p2 - p1
    dl = math.radians(float(b["lon"]) - float(a["lon"]))
    q = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(q))


def circular_difference_deg(a, b) -> float:
    return abs((float(a) - float(b) + 180) % 360 - 180)


def _parse_wkt(text):
    if not text:
        return None
    match = re.match(r"^LINESTRING\s*\((.+)\)$", str(text).strip(), re.I)
    if not match:
        return None
    points = []
    for pair in match.group(1).split(","):
        lon, lat = (float(part) for part in pair.strip().split()[:2])
        if not (math.isfinite(lon) and math.isfinite(lat)):
            raise ValueError(f"Invalid LINESTRING coordinate: {pair}")
        points.append({"lat": lat, "lon": lon})
    return points


class RoadNetwork:
    def __init__(self, scenario_id: str, rows: list[dict]):
        if not rows:
            raise ValueError(f"No network edges for scenario {scenario_id}")
        self.scenario_id = scenario_id
        coords = [(float(r[k]), float(r[j])) for r in rows for k, j in (("from_lat", "from_lon"), ("to_lat", "to_lon"))]
        self.origin_lat = sum(p[0] for p in coords) / len(coords)
        self.origin_lon = sum(p[1] for p in coords) / len(coords)
        self.edges = []
        self.edge_by_id = {}
        # MultiDiGraph preserves parallel road arcs; weighted Dijkstra selects the
        # least-cost arc, matching the reference adjacency-list behavior.
        self.graph = nx.MultiDiGraph()
        for row in rows:
            edge_id = str(row["edge_id"])
            if edge_id in self.edge_by_id:
                raise ValueError(f"Duplicate edge ID in {scenario_id}: {edge_id}")
            geo = _parse_wkt(row.get("geometry_wkt")) or [
                {"lat": float(row["from_lat"]), "lon": float(row["from_lon"])},
                {"lat": float(row["to_lat"]), "lon": float(row["to_lon"])},
            ]
            points = [self.project(p) for p in geo]
            segments, length = [], 0.0
            for a, b in zip(points, points[1:]):
                seg_len = math.hypot(b[0] - a[0], b[1] - a[1])
                segments.append({"a": a, "b": b, "len": seg_len, "offset": length})
                length += seg_len
            if not segments or length <= 0:
                raise ValueError(f"Invalid geometry for {scenario_id}/{edge_id}")
            edge = dict(row, edgeId=edge_id, fromNode=str(row["from_node"]), toNode=str(row["to_node"]),
                        layer=int(float(row.get("layer") or 0)), lengthMeters=length, segments=segments)
            self.edges.append(edge)
            self.edge_by_id[edge_id] = edge
            self.graph.add_edge(edge["fromNode"], edge["toNode"], weight=length)
            if str(row.get("oneway", "1")).lower() in ("0", "false"):
                self.graph.add_edge(edge["toNode"], edge["fromNode"], weight=length)
        self._shortest_cache = {}
        self._transition_cache = {}

    def project(self, point):
        return ((float(point["lon"]) - self.origin_lon) * METERS_PER_DEG_LON * math.cos(math.radians(self.origin_lat)),
                (float(point["lat"]) - self.origin_lat) * METERS_PER_DEG_LAT)

    def distance_to_edge(self, point, edge):
        px, py = self.project(point)
        best = None
        for seg in edge["segments"]:
            ax, ay = seg["a"]
            dx, dy = seg["b"][0] - ax, seg["b"][1] - ay
            den = dx * dx + dy * dy
            t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / den)) if den else 0.0
            distance = math.hypot(px - (ax + t * dx), py - (ay + t * dy))
            if best is None or distance < best["distanceMeters"]:
                bearing = (math.degrees(math.atan2(dx, dy)) + 360) % 360
                best = {"edge": edge, "edgeId": edge["edgeId"], "distanceMeters": distance,
                        "alongMeters": seg["offset"] + t * seg["len"], "headingDeg": bearing}
        return best

    def candidates(self, point):
        return sorted((self.distance_to_edge(point, edge) for edge in self.edges), key=lambda c: c["distanceMeters"])

    def _shortest_from(self, source):
        if source not in self._shortest_cache:
            try:
                self._shortest_cache[source] = nx.single_source_dijkstra_path_length(self.graph, source, weight="weight")
            except nx.NodeNotFound:
                self._shortest_cache[source] = {}
        return self._shortest_cache[source]

    def transition_distance(self, from_candidate, to_candidate):
        a, b = from_candidate["edge"], to_candidate["edge"]
        if a["edgeId"] == b["edgeId"]:
            delta = to_candidate["alongMeters"] - from_candidate["alongMeters"]
            return {"distanceMeters": abs(delta), "connected": True, "reverse": delta < 0}
        key = f'{a["edgeId"]}|{b["edgeId"]}'
        if key not in self._transition_cache:
            self._transition_cache[key] = self._shortest_from(a["toNode"]).get(b["fromNode"])
        middle = self._transition_cache[key]
        if middle is None:
            return {"distanceMeters": None, "connected": False, "reverse": False}
        remaining = max(0.0, a["lengthMeters"] - from_candidate["alongMeters"])
        return {"distanceMeters": remaining + middle + max(0.0, to_candidate["alongMeters"]), "connected": True, "reverse": False}
