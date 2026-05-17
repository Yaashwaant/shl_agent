"""
eval_recall.py — Recall@10 evaluation on the evaluation_dataset.json

Runs each scenario through the /chat API, extracts the recommended IDs from
the JSON block in the reply, and computes Recall@10 against ground-truth
expected_assessments (matched by name → entity_id).

Usage:
    python scripts/eval_recall.py [--base-url http://localhost:8000] [--catalog data/shl_catalog.json]
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import httpx

CATALOG_PATH = Path("data/shl_catalog.json")
DATASET_PATH = Path("evaluation_dataset.json")
DEFAULT_BASE_URL = "http://localhost:8000"
K = 10


def load_catalog(catalog_path: Path) -> tuple:
    """Build name → entity_id and entity_id → item mappings from the catalog JSON."""
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    name_to_id = {item["name"]: item["entity_id"] for item in catalog}
    id_to_name = {item["entity_id"]: item["name"] for item in catalog}
    return name_to_id, id_to_name


def load_dataset(dataset_path: Path) -> List[dict]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_match(rec_name: str, expected_names: List[str]) -> bool:
    """Check if recommended name matches any expected name via substring or fuzzy match."""
    rec_lower = rec_name.lower()
    for exp in expected_names:
        exp_lower = exp.lower()
        if exp_lower in rec_lower or rec_lower in exp_lower:
            return True
        if rec_lower.split()[0] == exp_lower.split()[0] and len(rec_name) < len(exp) + 15:
            return True
    return False


def extract_recommended_ids(reply: str) -> List[str]:
    """Extract recommended_ids from the JSON block in the agent's reply."""
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", reply, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            return parsed.get("recommended_ids", [])
        except json.JSONDecodeError:
            pass
    return []


def run_chat(
    base_url: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0,
) -> Optional[List[Dict]]:
    """Send a conversation to /chat and return recommendation objects."""
    try:
        with httpx.Client(base_url=base_url, timeout=timeout) as client:
            response = client.post(
                "/chat",
                json={"messages": messages},
            )
            response.raise_for_status()
            data = response.json()
            return data.get("recommendations")
    except Exception as e:
        print(f"      [WARN] /chat call failed: {e}")
        return None


def recall_at_k(
    recommended_ids: List[str],
    relevant_ids: Set[str],
    k: int = 10,
) -> float:
    """
    Recall@K = (Number of relevant assessments in top K) / (Total relevant assessments)
    """
    top_k = set(recommended_ids[:k])
    if not relevant_ids:
        return 1.0 if not recommended_ids else 0.0
    return len(top_k & relevant_ids) / len(relevant_ids)


def main():
    parser = argparse.ArgumentParser(description="Recall@10 evaluation for SHL Agent")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Base URL of the deployed API (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=CATALOG_PATH,
        help=f"Path to shl_catalog.json (default: {CATALOG_PATH})",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help=f"Path to evaluation_dataset.json (default: {DATASET_PATH})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Timeout per /chat call in seconds (default: 120)",
    )
    args = parser.parse_args()

    catalog, id_to_name = load_catalog(args.catalog)
    dataset = load_dataset(args.dataset)

    print(f"{'='*70}")
    print(f"Recall@10 Evaluation — SHL Assessment Advisor")
    print(f"{'='*70}")
    print(f"Base URL   : {args.base_url}")
    print(f"Catalog    : {args.catalog}")
    print(f"Dataset    : {args.dataset} ({len(dataset)} scenarios)")
    print(f"K          : {K}")
    print(f"{'='*70}\n")

    catalog, id_to_name = load_catalog(args.catalog)

    results: List[dict] = []
    recall_scores: List[float] = []

    for i, scenario in enumerate(dataset, 1):
        scenario_id = scenario["id"]
        description = scenario["description"]
        messages = scenario["messages"]
        expected_ids = set(scenario.get("expected_assessment_ids", []))
        expected_names = [id_to_name.get(eid, "") for eid in expected_ids]

        print(f"[{i:02d}] {scenario_id}: {description[:60]}...")
        print(f"       Expected ({len(expected_ids)}): {expected_names[:3]}{'...' if len(expected_names) > 3 else ''}")

        recommendations = run_chat(args.base_url, messages, timeout=args.timeout)

        if recommendations is None:
            recall = 0.0
            hit_names = []
            recommended_ids = []
            recommended_names = []
            hit_ids = []
        else:
            recommended_ids = []
            recommended_names = []
            hit_ids = []
            for rec in recommendations[:K]:
                name = rec.get("name", "")
                recommended_names.append(name)
                entity_id = catalog.get(name)
                if entity_id:
                    recommended_ids.append(entity_id)
                    if entity_id in expected_ids:
                        hit_ids.append(entity_id)
            recall = len(hit_ids) / len(expected_ids) if expected_ids else 1.0
            hit_names = [name for name in recommended_names if catalog.get(name) in expected_ids]

        recall_scores.append(recall)
        results.append({
            "scenario_id": scenario_id,
            "description": description,
            "expected_ids": list(expected_ids),
            "recommended_ids": recommended_ids[:K],
            "hit_ids": hit_ids,
            "hit_names": hit_names,
            "recall": recall,
        })

        print(f"       Recommended: {recommended_names[:3] if recommended_names else 'N/A'}{'...' if len(recommended_names) > 3 else ''}")
        print(f"       Hits: {hit_names}")
        print(f"       Recall@{K}: {recall:.3f}\n")

    mean_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0

    print(f"{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Scenario':<8} {'Recall@10':>10}  {'Hits':>6}  {'Expected':>10}  {'Status'}")
    print(f"{'-'*70}")
    for r in results:
        status = "✅ PASS" if r["recall"] >= 1.0 else "⚠️  PARTIAL" if r["recall"] > 0 else "❌ FAIL"
        print(
            f"{r['scenario_id']:<8} {r['recall']:>10.3f}  "
            f"{len(r['hit_names']):>6}  "
            f"{len(r['expected_ids']):>10}  "
            f"{status}"
        )
    print(f"{'-'*70}")
    print(f"{'Mean Recall@10':>30}: {mean_recall:.3f} ({mean_recall*100:.1f}%)")
    print(f"{'='*70}\n")

    output_path = Path("eval_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "mean_recall_at_10": mean_recall,
            "k": K,
            "num_scenarios": len(dataset),
            "results": results,
        }, f, indent=2)
    print(f"Detailed results saved to: {output_path}")

    passed = sum(1 for r in results if r["recall"] >= 1.0)
    partial = sum(1 for r in results if 0 < r["recall"] < 1.0)
    failed = sum(1 for r in results if r["recall"] == 0)
    print(f"\nPassed: {passed}/{len(results)}  |  Partial: {partial}  |  Failed: {failed}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
