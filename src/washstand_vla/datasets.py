"""Read-only LeRobot metadata inspection and episode-group split planning.

This module deliberately does not decode video or load parquet. A successful
metadata inspection is not proof of temporal alignment or robot readiness.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any


def inspect_dataset(root: Path) -> dict[str, Any]:
    """Inspect available LeRobot metadata using only the standard library.

    ``info.json`` and ``stats.json`` are required for this project's preflight.
    JSONL files are parsed when present. For v3, parquet files are discovered,
    not decoded: their rows, schema and episode coverage remain unverified.
    """
    root = Path(root).expanduser().resolve()
    checks: list[dict[str, str]] = []
    errors: list[str] = []
    warnings: list[str] = []
    summary: dict[str, Any] = {
        "root": str(root),
        "format": "unknown",
        "version": None,
        "features": {},
        "action_dim": None,
        "state_dim": None,
        "image_features": [],
        "top_image_features": [],
        "wrist_image_features": [],
        "quantile_keys": {},
        "episode_metadata_files": [],
        "task_metadata_files": [],
        "jsonl_episode_count": None,
        "jsonl_task_count": None,
        "declared_total_episodes": None,
        "declared_fps": None,
        "video_decoded": False,
        "temporal_alignment_verified": False,
        "parquet_rows_verified": False,
    }

    def record(name: str, status: str, message: str) -> None:
        checks.append({"name": name, "status": status, "message": message})
        if status == "error":
            errors.append(message)
        elif status == "warning":
            warnings.append(message)

    def read_object(path: Path, name: str) -> dict[str, Any] | None:
        if not path.is_file():
            record(name, "error", f"Required metadata file missing: {path}")
            return None
        try:
            with path.open(encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            record(name, "error", f"Cannot parse {path}: {exc}")
            return None
        if not isinstance(value, dict):
            record(name, "error", f"Expected a JSON object in {path}.")
            return None
        record(name, "pass", f"Parsed metadata: {path.relative_to(root)}")
        return value

    def read_jsonl(path: Path, name: str) -> int | None:
        count = 0
        try:
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as exc:
                        record(name, "error", f"Malformed JSON in {path}, line {line_number}: {exc}")
                        return None
                    if not isinstance(value, dict):
                        record(name, "error", f"Expected an object in {path}, line {line_number}.")
                        return None
                    count += 1
        except (OSError, UnicodeError) as exc:
            record(name, "error", f"Cannot read {path}: {exc}")
            return None
        record(name, "pass", f"Parsed {count} metadata records from {path.relative_to(root)}.")
        return count

    if not root.is_dir():
        record("root", "error", f"Dataset directory does not exist: {root}")
    info = read_object(root / "meta" / "info.json", "info")
    stats = read_object(root / "meta" / "stats.json", "stats")
    features: dict[str, Any] = {}
    feature_dimensions: dict[str, int | None] = {}
    if info is not None:
        summary["version"] = info.get("codebase_version", info.get("version"))
        summary["declared_total_episodes"] = info.get("total_episodes")
        summary["declared_fps"] = info.get("fps")
        if isinstance(info.get("features"), dict):
            features = info["features"]
            summary["features"] = features
        else:
            record("features", "error", "meta/info.json must contain a features object.")

        for kind, candidates in (("action", ("action",)), ("state", ("observation.state", "state"))):
            feature_name = next((name for name in candidates if name in features), None)
            if feature_name is None:
                record(f"{kind}_dim", "error", f"Missing {kind} feature; expected one of {list(candidates)}.")
                continue
            feature = features[feature_name]
            shape = feature.get("shape") if isinstance(feature, dict) else None
            if not (isinstance(shape, list) and len(shape) == 1
                    and isinstance(shape[0], int) and not isinstance(shape[0], bool)
                    and shape[0] > 0):
                record(f"{kind}_dim", "error", f"{feature_name} must declare a one-dimensional vector shape, got {shape!r}.")
                feature_dimensions[feature_name] = None
                continue
            dimension = shape[0]
            feature_dimensions[feature_name] = dimension
            summary[f"{kind}_dim"] = dimension
            if dimension != 6:
                record(f"{kind}_dim", "error", f"{feature_name} has dimension {dimension}; this SO101 project expects 6.")
            else:
                record(f"{kind}_dim", "pass", f"{feature_name} declares 6 values; their units, order and meaning remain unverified.")

        image_names = sorted(name for name, feature in features.items()
                             if isinstance(feature, dict) and feature.get("dtype") in ("image", "video"))
        summary["image_features"] = image_names
        summary["top_image_features"] = [name for name in image_names
                                          if any(part in name.lower() for part in ("top", "overhead"))]
        summary["wrist_image_features"] = [name for name in image_names if "wrist" in name.lower()]
        for camera in ("top", "wrist"):
            matches = summary[f"{camera}_image_features"]
            if matches:
                record(f"{camera}_camera", "pass", f"Found {camera} image metadata: {', '.join(matches)}.")
            else:
                record(f"{camera}_camera", "warning", f"No image/video feature named for the {camera} camera; check the camera mapping manually.")

    if stats is not None:
        for feature_name, dimension in feature_dimensions.items():
            feature_stats = stats.get(feature_name)
            presence = {key: isinstance(feature_stats, dict) and key in feature_stats for key in ("q01", "q99")}
            summary["quantile_keys"][feature_name] = presence
            missing = [key for key, present in presence.items() if not present]
            if missing:
                record(f"quantiles:{feature_name}", "warning", f"Missing {', '.join(missing)} statistics for {feature_name}; check the chosen normalization mode.")
            for key, present in presence.items():
                if not present:
                    continue
                values = feature_stats[key]
                if not isinstance(values, list) or (dimension is not None and len(values) != dimension):
                    record(f"quantiles:{feature_name}:{key}", "error", f"{feature_name}.{key} must be a list matching vector dimension {dimension}.")
                elif any(isinstance(value, bool) or not isinstance(value, (int, float))
                         or (isinstance(value, float) and not math.isfinite(value)) for value in values):
                    record(f"quantiles:{feature_name}:{key}", "error", f"{feature_name}.{key} must contain finite numbers.")
                else:
                    record(f"quantiles:{feature_name}:{key}", "pass", f"Found {feature_name}.{key} with {len(values)} values.")

    for kind, count_key in (("episodes", "jsonl_episode_count"), ("tasks", "jsonl_task_count")):
        path = root / "meta" / f"{kind}.jsonl"
        if path.exists():
            summary[f"{'episode' if kind == 'episodes' else 'task'}_metadata_files"].append(str(path.relative_to(root)))
            summary[count_key] = read_jsonl(path, kind)

    meta = root / "meta"
    try:
        episode_parquet = sorted(path for path in (meta / "episodes").rglob("*.parquet") if path.is_file())
        task_parquet = [meta / "tasks.parquet"] if (meta / "tasks.parquet").is_file() else []
        for path in episode_parquet:
            summary["episode_metadata_files"].append(str(path.relative_to(root)))
        for path in task_parquet:
            summary["task_metadata_files"].append(str(path.relative_to(root)))
        if episode_parquet or task_parquet:
            record("parquet", "warning", "Parquet metadata files were discovered only; their schema, rows and episode coverage were not read.")
    except OSError as exc:
        record("parquet", "error", f"Cannot discover parquet metadata files: {exc}")
        episode_parquet = []

    version = str(summary["version"] or "")
    if version.startswith("v3") or version.startswith("3") or episode_parquet:
        summary["format"] = "lerobot-v3"
    elif version.startswith("v2") or version.startswith("2"):
        summary["format"] = "lerobot-v2"
    if not summary["episode_metadata_files"]:
        record("episodes", "warning", "No meta/episodes.jsonl or meta/episodes/**/*.parquet found; episode metadata was not inspected.")
    if not summary["task_metadata_files"]:
        record("tasks", "warning", "No meta/tasks.jsonl or meta/tasks.parquet found; task metadata was not inspected.")
    record("inspection_scope", "warning", "Metadata inspection only: images/videos were not decoded; timestamps, image/action alignment, physical action semantics and task success were not verified.")
    return {
        "status": "error" if errors else "warning" if warnings else "ok",
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "summary": summary,
    }


def make_episode_split(
    episodes: list[dict], validation_fraction: float, seed: int,
    group_key: str = "session_id",
) -> dict[str, Any]:
    """Plan a deterministic split of complete groups and complete episodes.

    Each input row must provide ``episode_id`` and ``group_key`` as string or
    integer values. Group selection targets a fraction of *groups*, so the
    achieved fraction of episodes can differ. No frames or instructions are
    automatically balanced. The function returns a manifest and writes no files.
    """
    if (isinstance(validation_fraction, bool)
            or not isinstance(validation_fraction, (int, float))
            or not 0 < validation_fraction < 1
            or not math.isfinite(validation_fraction)):
        raise ValueError("validation_fraction must be a finite number strictly between 0 and 1.")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer.")
    if not isinstance(group_key, str) or not group_key.strip():
        raise ValueError("group_key must be a non-empty string.")
    if not isinstance(episodes, list):
        raise ValueError("episodes must be a list of metadata objects.")

    def identity(value: Any, field: str, index: int) -> tuple[str, Any]:
        if (isinstance(value, bool) or not isinstance(value, (str, int))
                or (isinstance(value, str) and not value.strip())):
            raise ValueError(f"Episode row {index}: {field} must be a non-empty string or integer.")
        return type(value).__name__, value

    groups: dict[tuple[str, Any], list[Any]] = {}
    seen: set[tuple[str, Any]] = set()
    for index, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            raise ValueError(f"Episode row {index} must be a metadata object.")
        if "episode_id" not in episode:
            raise ValueError(f"Episode row {index} is missing episode_id.")
        if group_key not in episode:
            raise ValueError(f"Episode row {index} is missing group field {group_key!r}.")
        episode_id = identity(episode["episode_id"], "episode_id", index)
        group_id = identity(episode[group_key], group_key, index)
        if episode_id in seen:
            raise ValueError(f"Duplicate episode_id: {episode['episode_id']!r}.")
        seen.add(episode_id)
        groups.setdefault(group_id, []).append(episode["episode_id"])
    if len(groups) < 2:
        raise ValueError("At least two distinct groups are required for a leakage-free split.")

    def sort_key(value: Any) -> tuple[str, str]:
        return type(value).__name__, str(value)

    ordered_groups = sorted(groups, key=lambda value: sort_key(value[1]))
    shuffled = ordered_groups.copy()
    random.Random(seed).shuffle(shuffled)
    validation_count = min(len(groups) - 1, max(1, int(len(groups) * validation_fraction + 0.5)))
    validation_groups = set(shuffled[:validation_count])
    split_groups = {
        "train": [group for group in ordered_groups if group not in validation_groups],
        "validation": [group for group in ordered_groups if group in validation_groups],
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "strategy": "whole_group",
        "seed": seed,
        "validation_fraction": validation_fraction,
        "group_key": group_key,
        "groups": {},
        "group_assignments": [],
        "notes": [
            "Whole groups and whole episodes are assigned; frames are never split.",
            "Validation fraction targets groups; episode counts may differ from the requested fraction.",
            "Instruction balance and unrecorded layout/session overlap are not checked automatically.",
        ],
    }
    for split, group_ids in split_groups.items():
        result[split] = sorted([episode_id for group in group_ids for episode_id in groups[group]], key=sort_key)
        result["groups"][split] = [group[1] for group in group_ids]
        for group in group_ids:
            result["group_assignments"].append({
                "group": group[1], "split": split,
                "episode_ids": sorted(groups[group], key=sort_key),
            })
    result["counts"] = {"episodes": len(episodes), "groups": len(groups),
                         "train": len(result["train"]), "validation": len(result["validation"])}
    result["actual_validation_fraction"] = len(result["validation"]) / len(episodes)
    return result
