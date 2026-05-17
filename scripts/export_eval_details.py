"""
Export enriched evaluation details from eval results.

This script reads:
- eval_results.json
- evaluation_dataset.json
- data/shl_catalog.json

For each scenario, it outputs:
- query/scenario id
- description
- full input messages
- expected_ids with full catalog records
- recommended_ids with full catalog records
- hit_ids with full catalog records

Usage:
    python scripts/export_eval_details.py
    python scripts/export_eval_details.py --output detailed_eval_results.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_EVAL_RESULTS = Path("eval_results.json")
DEFAULT_DATASET = Path("evaluation_dataset.json")
DEFAULT_CATALOG = Path("data/shl_catalog.json")
DEFAULT_OUTPUT = Path("detailed_eval_results.json")


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_catalog_index(catalog_items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {item["entity_id"]: item for item in catalog_items}


def build_dataset_index(dataset_items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {item["id"]: item for item in dataset_items}


def expand_ids(
    ids: List[str],
    catalog_by_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    expanded: List[Dict[str, Any]] = []
    for entity_id in ids:
        item = catalog_by_id.get(entity_id)
        if item is None:
            expanded.append(
                {
                    "entity_id": entity_id,
                    "missing_in_catalog": True,
                }
            )
            continue
        expanded.append(item)
    return expanded


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export eval results with full catalog details and input messages."
    )
    parser.add_argument(
        "--eval-results",
        type=Path,
        default=DEFAULT_EVAL_RESULTS,
        help=f"Path to eval_results.json (default: {DEFAULT_EVAL_RESULTS})",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Path to evaluation_dataset.json (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG,
        help=f"Path to shl_catalog.json (default: {DEFAULT_CATALOG})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON file (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    eval_results = load_json(args.eval_results)
    dataset = load_json(args.dataset)
    catalog = load_json(args.catalog)

    dataset_by_id = build_dataset_index(dataset)
    catalog_by_id = build_catalog_index(catalog)

    enriched_results: List[Dict[str, Any]] = []

    for result in eval_results.get("results", []):
        scenario_id = result["scenario_id"]
        dataset_item = dataset_by_id.get(scenario_id, {})

        expected_ids = result.get("expected_ids", [])
        recommended_ids = result.get("recommended_ids", [])
        hit_ids = result.get("hit_ids", [])

        enriched_results.append(
            {
                "scenario_id": scenario_id,
                "description": result.get("description", dataset_item.get("description", "")),
                "messages": dataset_item.get("messages", []),
                "expected_ids": expected_ids,
                "expected_records": expand_ids(expected_ids, catalog_by_id),
                "recommended_ids": recommended_ids,
                "recommended_records": expand_ids(recommended_ids, catalog_by_id),
                "hit_ids": hit_ids,
                "hit_records": expand_ids(hit_ids, catalog_by_id),
                "hit_names": result.get("hit_names", []),
                "recall": result.get("recall", 0.0),
            }
        )

    output = {
        "source_files": {
            "eval_results": str(args.eval_results),
            "dataset": str(args.dataset),
            "catalog": str(args.catalog),
        },
        "summary": {
            "mean_recall_at_10": eval_results.get("mean_recall_at_10"),
            "k": eval_results.get("k"),
            "num_scenarios": eval_results.get("num_scenarios"),
            "num_enriched_results": len(enriched_results),
        },
        "results": enriched_results,
    }

    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print(f"Detailed export written to: {args.output}")
    print(f"Scenarios exported: {len(enriched_results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
