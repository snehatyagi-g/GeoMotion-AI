"""Metrics and aligned trajectory loading for Phase 3 experiments."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .csv_utils import read_csv
from .road_network import RoadNetwork

KEY_FIELDS = ("scenario_id", "trajectory_id", "sampling_interval_s", "seq", "timestamp")


def _key(row):
    return "|".join(str(row.get(k, "")) for k in KEY_FIELDS)


def _index_unique(rows, name):
    result = {}
    for row in rows:
        key = _key(row)
        if key in result:
            raise ValueError(f"Duplicate row key in {name}: {key}")
        result[key] = row
    return result


def load_scenario_networks(processed_root, allowed_scenario_ids=None):
    grouped = defaultdict(list)
    for row in read_csv(Path(processed_root) / "road_network.csv"):
        scenario_id = row["scenario_id"]
        if not scenario_id.startswith("SC"):
            continue
        if allowed_scenario_ids is not None and scenario_id not in allowed_scenario_ids:
            continue
        grouped[scenario_id].append(row)
    return {scenario_id: RoadNetwork(scenario_id, edges) for scenario_id, edges in grouped.items()}


def load_split_tracks(processed_root, split, intervals, network_by_scenario, require_labels=True):
    processed_root = Path(processed_root)
    manifest_rows = [r for r in read_csv(processed_root / "scenario_manifest.csv") if r["split"] == split]
    manifest = {r["scenario_id"]: r for r in manifest_rows}
    tracks, eligible_keys = [], set()
    for interval in intervals:
        obs_rows = read_csv(processed_root / split / "observations" / f"traj_dt_{interval}s.csv")
        feat_rows = read_csv(processed_root / split / "features" / f"features_dt_{interval}s.csv")
        gt_rows = read_csv(processed_root / split / "ground_truth" / f"ground_truth_dt_{interval}s.csv") if require_labels else []
        obs_by_key = _index_unique(obs_rows, f"observations/{split}/{interval}s")
        feat_by_key = _index_unique(feat_rows, f"features/{split}/{interval}s")
        gt_by_key = _index_unique(gt_rows, f"ground_truth/{split}/{interval}s") if require_labels else {}
        if len(obs_by_key) != len(feat_by_key) or (require_labels and len(obs_by_key) != len(gt_by_key)):
            raise ValueError(f"Table row count mismatch for {split}/{interval}s")
        grouped = {}
        for key, obs in obs_by_key.items():
            feature, truth = feat_by_key.get(key), gt_by_key.get(key)
            if feature is None or (require_labels and truth is None):
                raise ValueError(f"Missing aligned row for key {key}")
            if obs["split"] != split or feature["split"] != split or (truth and truth["split"] != split):
                raise ValueError(f"Split mismatch for {key}")
            scenario = manifest.get(obs["scenario_id"])
            network = network_by_scenario.get(obs["scenario_id"])
            if scenario is None or network is None:
                raise ValueError(f"Scenario/network missing for {obs['scenario_id']}")
            if truth and truth["edge_id_true"] not in network.edge_by_id:
                raise ValueError(f"Unresolved truth edge {truth['edge_id_true']} for {key}")
            track = grouped.setdefault(obs["trajectory_id"], {"scenario": scenario, "network": network,
                "interval": interval, "trajectoryId": obs["trajectory_id"], "observations": [], "features": [], "groundTruth": []})
            track["observations"].append(obs)
            track["features"].append(feature)
            if truth:
                track["groundTruth"].append(truth)
            eligible_keys.add(key)
        for track in grouped.values():
            order = sorted(range(len(track["observations"])), key=lambda i: int(track["observations"][i]["seq"]))
            for field in ("observations", "features", "groundTruth"):
                if track[field]:
                    track[field] = [track[field][i] for i in order]
            tracks.append(track)
    return {"tracks": tracks, "eligibleKeys": eligible_keys}


def evaluation_rows(track, prediction, algorithm, elapsed_ms):
    matches = prediction["matches"]
    if len(track["groundTruth"]) != len(matches):
        raise ValueError(f"Prediction/GT length mismatch: {track['trajectoryId']}")
    per_obs = elapsed_ms / len(track["observations"]) if track["observations"] else 0.0
    output = []
    for i, match in enumerate(matches):
        obs = track["observations"][i]
        output.append({"algorithm": algorithm, "scenario_id": track["scenario"]["scenario_id"],
            "trajectory_id": track["trajectoryId"], "split": track["scenario"]["split"],
            "scenario_type": track["scenario"]["scenario_type"], "gps_noise_tier": track["scenario"]["gps_noise_tier"],
            "sampling_interval_s": int(track["interval"]), "seq": int(obs["seq"]), "timestamp": obs["timestamp"],
            "edge_id_true": track["groundTruth"][i]["edge_id_true"], "edge_id_pred": match["edge_id"],
            "matching_distance_m": match["matching_distance_m"], "candidate_count": match["candidate_count"],
            "search_radius_m": match["search_radius_m"], "processing_ms_per_observation": per_obs})
    return output


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def trajectory_accuracy_score(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["trajectory_id"]].append(float(row["edge_id_pred"] == row["edge_id_true"]))
    return _mean([_mean(values) for values in groups.values()])


def calculate_metrics(rows):
    # Preserve first-seen class order, as in the JavaScript reference Set.
    classes, seen_classes = [], set()
    by_true, by_pred, true_pred = defaultdict(int), defaultdict(int), defaultdict(int)
    correct = 0
    for row in rows:
        true, pred = row["edge_id_true"], row["edge_id_pred"]
        for edge_id in (true, pred):
            if edge_id not in seen_classes:
                classes.append(edge_id)
                seen_classes.add(edge_id)
        correct += true == pred
        by_true[true] += 1
        by_pred[pred] += 1
        true_pred[(true, pred)] += 1
    macro_p, macro_r, macro_f = [], [], []
    for edge_id in classes:
        tp = true_pred[(edge_id, edge_id)]
        p = tp / by_pred[edge_id] if by_pred[edge_id] else 0.0
        r = tp / by_true[edge_id] if by_true[edge_id] else 0.0
        macro_p.append(p); macro_r.append(r); macro_f.append(2 * p * r / (p + r) if p + r else 0.0)
    tracks = defaultdict(list)
    for row in rows:
        tracks[row["trajectory_id"]].append(row)
    acc = correct / len(rows) if rows else 0.0
    return {"observations": len(rows), "trajectories": len(tracks), "edge_accuracy": acc,
        "edge_precision_micro": acc, "edge_recall_micro": acc, "edge_f1_micro": acc,
        "edge_precision_macro": _mean(macro_p), "edge_recall_macro": _mean(macro_r), "edge_f1_macro": _mean(macro_f),
        "mean_matching_distance_m": _mean([float(r["matching_distance_m"]) for r in rows]),
        "trajectory_accuracy_mean": _mean([_mean([float(x["edge_id_pred"] == x["edge_id_true"]) for x in track]) for track in tracks.values()]),
        "trajectory_exact_match_rate": sum(all(r["edge_id_pred"] == r["edge_id_true"] for r in track) for track in tracks.values()) / len(tracks) if tracks else 0.0,
        "candidate_count_mean": _mean([float(r["candidate_count"]) for r in rows]),
        "processing_ms_per_observation": _mean([float(r["processing_ms_per_observation"]) for r in rows])}


def build_metric_tables(rows, algorithms):
    records, detailed = [], {"overall": [], "by_context": [], "by_noise": [], "by_sampling_interval": []}
    scopes = (("road_context", "scenario_type", "context", "by_context"),
              ("gps_noise", "gps_noise_tier", "noise_tier", "by_noise"),
              ("sampling_interval", "sampling_interval_s", "sampling_interval_s", "by_sampling_interval"))
    for algorithm in algorithms:
        selected = [r for r in rows if r["algorithm"] == algorithm]
        metrics = calculate_metrics(selected)
        records.append({"algorithm": algorithm, "scope": "overall", "context": "", "noise_tier": "", "sampling_interval_s": "", **metrics})
        detailed["overall"].append({"algorithm": algorithm, "metrics": metrics})
        for scope, field, output_field, detail_key in scopes:
            values = sorted({r[field] for r in rows}, key=lambda x: float(x) if field == "sampling_interval_s" else str(x))
            for value in values:
                subset = [r for r in selected if str(r[field]) == str(value)]
                m = calculate_metrics(subset)
                record = {"algorithm": algorithm, "scope": scope, "context": "", "noise_tier": "", "sampling_interval_s": ""}
                if output_field == "context": record["context"] = value
                elif output_field == "noise_tier": record["noise_tier"] = value
                else: record["sampling_interval_s"] = int(value)
                records.append({**record, **m})
                item = {"algorithm": algorithm, "metrics": m, output_field: int(value) if output_field == "sampling_interval_s" else value}
                detailed[detail_key].append(item)
    return {"records": records, "detailed": detailed}
