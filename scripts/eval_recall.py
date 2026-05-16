"""
Recall@K Evaluation Script
===========================
Simulates multi-turn conversations against the live /chat endpoint
and computes Mean Recall@10 against the expected final shortlists
extracted from the 10 sample conversation files.

Usage:
    python scripts/eval_recall.py [--base-url http://localhost:8000] [--k 10]
"""

import re
import sys
import json
import argparse
import requests
from pathlib import Path
from typing import List, Dict, Tuple

# ── Expected final shortlists (manually extracted from C1–C10.md) ──────────
# Each entry: (conversation_id, description, user_turns, expected_names)
# expected_names = the FINAL confirmed shortlist from the last table in each .md

CONVERSATIONS = [
    {
        "id": "C1",
        "description": "Senior leadership / CXO executive assessment",
        "user_turns": [
            "We need a solution for senior leadership.",
            "The pool consists of CXOs, director-level positions; people with more than 15 years of experience.",
            "This is for selection into a newly created C-suite role. We want personality and leadership. No development reports.",
            "Yes, that works. Confirmed.",
        ],
        "expected": [
            "Occupational Personality Questionnaire OPQ32r",
        ],
    },
    {
        "id": "C2",
        "description": "Volume hiring for customer service agents",
        "user_turns": [
            "I'm hiring for a customer service role. High volume, entry level.",
            "We need remote testing. English speakers only.",
            "Add a personality measure too.",
            "Looks good. Confirm.",
        ],
        "expected": [
            "Occupational Personality Questionnaire OPQ32r",
            "SHL Verify Interactive G+",
        ],
    },
    {
        "id": "C3",
        "description": "Graduate management trainee scheme - cognitive, personality, SJT",
        "user_turns": [
            "We run a graduate management trainee scheme. We need a full battery — cognitive, personality, and situational judgement. All recent graduates.",
            "Drop the OPQ. Final list: Verify G+ and Graduate Scenarios.",
        ],
        "expected": [
            "SHL Verify Interactive G+",
            "Graduate Scenarios",
        ],
    },
    {
        "id": "C4",
        "description": "Sales representative hiring",
        "user_turns": [
            "Hiring sales reps. Mid-level. Need to assess drive, resilience, and customer focus.",
            "Add a cognitive test too.",
            "Confirmed.",
        ],
        "expected": [
            "Occupational Personality Questionnaire OPQ32r",
            "SHL Verify Interactive G+",
        ],
    },
    {
        "id": "C5",
        "description": "Senior Java backend engineer - microservices",
        "user_turns": [
            "I need to assess a senior backend engineer. Java, microservices, works with stakeholders.",
            "Senior IC level — owns end-to-end microservice delivery.",
            "Spring, SQL, AWS, Docker on top of core Java.",
            "Also need a cognitive and personality test.",
            "On Java — they'd be working on existing services, not greenfield. Is the Advanced level the right pick?",
            "Keep Verify G+. Locking it in.",
        ],
        "expected": [
            "Core Java (Advanced Level) (New)",
            "Spring (New)",
            "SQL (New)",
            "Amazon Web Services (AWS) Development (New)",
            "Docker (New)",
            "SHL Verify Interactive G+",
            "Occupational Personality Questionnaire OPQ32r",
        ],
    },
    {
        "id": "C6",
        "description": "Data analyst role",
        "user_turns": [
            "Need assessments for a data analyst position. Mid-level, 3 years experience.",
            "SQL and Python are core. Also needs to present findings to non-technical stakeholders.",
            "Confirm.",
        ],
        "expected": [
            "SQL (New)",
            "Python (New)",
            "Occupational Personality Questionnaire OPQ32r",
        ],
    },
    {
        "id": "C7",
        "description": "Safety-critical industrial role",
        "user_turns": [
            "We're hiring for a safety-critical role in a chemical plant. Operators.",
            "Entry level. Safety compliance is essential.",
            "Yes add a dependability/reliability measure.",
            "Confirmed.",
        ],
        "expected": [
            "Occupational Personality Questionnaire OPQ32r",
            "SHL Verify Interactive G+",
        ],
    },
    {
        "id": "C8",
        "description": "HR business partner role",
        "user_turns": [
            "Looking for assessments for an HR Business Partner.",
            "Mid-level. They need to influence stakeholders and manage change.",
            "Add a cognitive measure.",
            "Confirmed.",
        ],
        "expected": [
            "Occupational Personality Questionnaire OPQ32r",
            "SHL Verify Interactive G+",
        ],
    },
    {
        "id": "C9",
        "description": "Finance manager / FP&A",
        "user_turns": [
            "Hiring a finance manager. FP&A focus. Senior level.",
            "Numerical reasoning and Excel are important. Also stakeholder management.",
            "Add personality.",
            "Confirmed.",
        ],
        "expected": [
            "Occupational Personality Questionnaire OPQ32r",
            "SHL Verify Interactive G+",
        ],
    },
    {
        "id": "C10",
        "description": "Graduate trainee full battery (cognitive + SJT, OPQ dropped)",
        "user_turns": [
            "We run a graduate management trainee scheme. We need a full battery — cognitive, personality, and situational judgement. All recent graduates.",
            "But can you remove the OPQ32r and replace it with something shorter? Candidates complain it takes too long.",
            "Drop the OPQ. Final list: Verify G+ and Graduate Scenarios.",
        ],
        "expected": [
            "SHL Verify Interactive G+",
            "Graduate Scenarios",
        ],
    },
]


# ── Recall@K helpers ─────────────────────────────────────────────────────────

def normalise(name: str) -> str:
    """Lowercase + strip punctuation for fuzzy matching."""
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()


def recall_at_k(recommended: List[str], expected: List[str], k: int = 10) -> float:
    """
    Recall@K = |relevant ∩ top-K recommended| / |relevant|
    Matching is done by normalised name substring check.
    """
    if not expected:
        return 1.0  # nothing expected → trivially satisfied

    top_k = recommended[:k]
    top_k_norm = [normalise(n) for n in top_k]

    hits = 0
    for exp in expected:
        exp_norm = normalise(exp)
        # Match if any recommended name contains or is contained by the expected name
        if any(exp_norm in r or r in exp_norm for r in top_k_norm):
            hits += 1

    return hits / len(expected)


# ── Agent interaction ─────────────────────────────────────────────────────────

def run_conversation(base_url: str, user_turns: List[str], conv_id: str) -> Tuple[List[str], List[Dict]]:
    """
    Simulate a multi-turn conversation with the agent.
    Returns (recommended_names, full_recommendations).
    Stops when end_of_conversation=true or all turns exhausted.
    """
    messages = []
    final_recs = []
    final_names = []

    for turn_idx, user_text in enumerate(user_turns):
        messages.append({"role": "user", "content": user_text})

        try:
            resp = requests.post(
                f"{base_url}/chat",
                json={"messages": messages},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.Timeout:
            print(f"  [{conv_id}] Turn {turn_idx+1}: TIMEOUT")
            break
        except Exception as e:
            print(f"  [{conv_id}] Turn {turn_idx+1}: ERROR — {e}")
            break

        reply = data.get("reply", "")
        recs = data.get("recommendations") or []
        eoc = data.get("end_of_conversation", False)

        messages.append({"role": "assistant", "content": reply})

        if recs:
            final_recs = recs
            final_names = [r["name"] for r in recs]

        status = "✓ eoc" if eoc else f"{len(recs)} recs"
        print(f"  [{conv_id}] Turn {turn_idx+1}: {status} | reply: {reply[:80]}...")

        if eoc:
            break

    return final_names, final_recs


# ── Main evaluation loop ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Recall@K evaluator for SHL agent")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  SHL Agent — Recall@{args.k} Evaluation")
    print(f"  Endpoint: {args.base_url}")
    print(f"{'='*60}\n")

    # Check health first
    try:
        h = requests.get(f"{args.base_url}/health", timeout=10)
        print(f"Health: {h.json()}\n")
    except Exception as e:
        print(f"ERROR: Cannot reach {args.base_url}/health — {e}")
        sys.exit(1)

    results = []

    for conv in CONVERSATIONS:
        cid = conv["id"]
        desc = conv["description"]
        expected = conv["expected"]
        user_turns = conv["user_turns"]

        print(f"\n{'─'*60}")
        print(f"  {cid}: {desc}")
        print(f"  Expected ({len(expected)}): {', '.join(expected)}")
        print()

        recommended_names, full_recs = run_conversation(args.base_url, user_turns, cid)

        score = recall_at_k(recommended_names, expected, k=args.k)

        print(f"\n  Recommended ({len(recommended_names)}): {', '.join(recommended_names) or 'NONE'}")
        print(f"  Recall@{args.k}: {score:.2f}  ({'PASS' if score > 0 else 'FAIL'})")

        results.append({
            "id": cid,
            "description": desc,
            "expected": expected,
            "recommended": recommended_names,
            "recall": score,
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    mean_recall = sum(r["recall"] for r in results) / len(results)

    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    for r in results:
        bar = "█" * int(r["recall"] * 10) + "░" * (10 - int(r["recall"] * 10))
        print(f"  {r['id']:4s} [{bar}] {r['recall']:.2f}  {r['description'][:45]}")
    print(f"{'─'*60}")
    print(f"  Mean Recall@{args.k}: {mean_recall:.4f}  ({mean_recall*100:.1f}%)")
    print(f"{'='*60}\n")

    # Save to JSON
    output = {
        "mean_recall_at_k": mean_recall,
        "k": args.k,
        "conversations": results,
    }
    out_path = Path(__file__).parent.parent / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {out_path}\n")


if __name__ == "__main__":
    main()
