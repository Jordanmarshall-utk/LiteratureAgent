"""Check the unchanged-target paired PCE comparison from saved predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis/pce"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(actual, prediction)),
        "rmse": float(mean_squared_error(actual, prediction) ** 0.5),
        "mae": float(mean_absolute_error(actual, prediction)),
    }


def reproduce(iterations: int = 2000, seed: int = 42) -> list[dict[str, float | str]]:
    original = rows(DATA / "literatureagent_124_original_only_predictions.csv")
    expansion = rows(DATA / "literatureagent_124_expansion_predictions.csv")
    expansion_by_id = {row["row_key"]: row for row in expansion}
    identities = [row["row_key"] for row in original]
    if len(original) != 124 or len(set(identities)) != 124:
        raise ValueError("Expected 124 unique unchanged-target record identities.")
    if set(identities) != set(expansion_by_id) or len(expansion_by_id) != len(expansion):
        raise ValueError("Paired prediction populations differ.")
    if any(float(row["actual"]) != float(expansion_by_id[row["row_key"]]["actual"]) for row in original):
        raise ValueError("A paired target differs between prediction tables.")
    actual = np.array([float(row["actual"]) for row in original])
    baseline = np.array([float(row["predicted"]) for row in original])
    revised = np.array([float(expansion_by_id[key]["predicted"]) for key in identities])
    groups = np.array([row["group"] or "missing_group" for row in original])
    unique_groups = np.array(list(dict.fromkeys(groups)))
    if len(unique_groups) != 79:
        raise ValueError("Expected 79 normalized publication groups.")
    positions = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in ("r2", "rmse", "mae")}
    for _ in range(iterations):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([positions[group] for group in sampled])
        before = metrics(actual[indices], baseline[indices])
        after = metrics(actual[indices], revised[indices])
        for metric in draws:
            draws[metric].append(after[metric] - before[metric])
    result = []
    for metric, values in draws.items():
        result.append({
            "metric": metric,
            "bootstrap_median_delta": float(np.median(values)),
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Compare to the retained 2,000-replicate, seed-42 estimates")
    parser.add_argument("--output", type=Path, help="Optional CSV destination for recomputed estimates")
    args = parser.parse_args()
    result = reproduce()
    if args.check:
        retained = {row["metric"]: row for row in rows(DATA / "literatureagent_124_unchanged_target_paired_deltas.csv")}
        for row in result:
            for key in ("bootstrap_median_delta", "ci_low", "ci_high"):
                if not np.isclose(row[key], float(retained[row["metric"]][key]), rtol=0, atol=1e-12):
                    raise ValueError(f"Retained result mismatch: {row['metric']}/{key}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(result)
    print(json.dumps({"status": "PASS", "records": 124, "publication_groups": 79, "iterations": 2000, "seed": 42, "estimates": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
