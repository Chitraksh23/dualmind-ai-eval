"""
run_eval.py
-----------
Batch evaluation runner.

Runs all prompts through both models and produces results.json
with per-prompt scores, averages, and category breakdowns.

Can be invoked from ANY working directory because all paths are
resolved relative to this file's location via __file__.

Usage (from repo root – same as GitHub Actions):
    python evaluation/run_eval.py
    python evaluation/run_eval.py --max 5 --verbose
    python evaluation/run_eval.py --category adversarial
    python evaluation/run_eval.py --llm-judge

Usage (from inside evaluation/):
    python run_eval.py
"""

from __future__ import annotations  # enables X | Y type hints on Python 3.9

import sys
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Path resolution – works regardless of CWD
# ---------------------------------------------------------------------------
# This file lives at  <repo_root>/evaluation/run_eval.py
# so REPO_ROOT is always two levels up from __file__
THIS_DIR  = Path(__file__).resolve().parent          # <repo>/evaluation/
REPO_ROOT = THIS_DIR.parent                           # <repo>/

# Add backend/ to sys.path so we can import the assistant modules
sys.path.insert(0, str(REPO_ROOT / "backend"))

from oss_assistant      import OSSAssistant
from frontier_assistant import FrontierAssistant
from evaluator          import evaluate


def _resolve(p: str, base: Path) -> Path:
    """Return Path(p) resolved; if relative, try CWD then base."""
    path = Path(p)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (base / p).resolve()


def run_evaluation(
    prompts_file: str = str(THIS_DIR / "prompts.json"),
    output_file:  str = str(THIS_DIR / "results.json"),
    use_llm_judge:   bool = False,
    category_filter: str | None = None,
    verbose:         bool = False,
    max_prompts:     int | None = None,
) -> dict:
    """Run the full evaluation suite and write results.json."""

    print("\n" + "=" * 60)
    print("  DualMind Evaluation Suite")
    print("=" * 60)
    print(f"  Judge      : {'LLM (Claude Opus)' if use_llm_judge else 'Heuristic'}")
    print(f"  OSS model  : {os.getenv('OSS_MODEL', 'Qwen/Qwen2.5-72B-Instruct')}")
    print(f"  Frontier   : {os.getenv('FRONTIER_MODEL', 'claude-sonnet-4-20250514')}")

    # --- resolve prompts file -----------------------------------------------
    prompts_path = _resolve(prompts_file, THIS_DIR)
    if not prompts_path.exists():
        raise FileNotFoundError(
            f"Prompts file not found.\n"
            f"  Tried: {prompts_path}\n"
            f"  Pass --prompts <path> to override."
        )

    print(f"  Prompts    : {prompts_path}")

    # --- resolve / create output path ----------------------------------------
    output_path = _resolve(output_file, THIS_DIR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Output     : {output_path}")
    print("=" * 60 + "\n")

    # --- load prompts ---------------------------------------------------------
    with open(prompts_path, encoding="utf-8") as f:
        data = json.load(f)

    prompts = data["prompts"]

    if category_filter:
        available = sorted({p["category"] for p in prompts})
        prompts = [p for p in prompts if p["category"] == category_filter]
        if not prompts:
            print(f"WARNING: no prompts matched category '{category_filter}'.")
            print(f"  Available: {available}")

    if max_prompts:
        prompts = prompts[:max_prompts]

    print(f"Running {len(prompts)} prompts …\n")

    # --- initialise assistants -----------------------------------------------
    oss      = OSSAssistant()
    frontier = FrontierAssistant()

    results: list[dict] = []
    oss_totals      = {"safety": [], "hallucination": [], "bias": [], "latency": []}
    frontier_totals = {"safety": [], "hallucination": [], "bias": [], "latency": []}
    category_scores: dict[str, dict[str, list]] = {}

    # --- main loop -----------------------------------------------------------
    for i, item in enumerate(prompts, 1):
        prompt   = item["prompt"]
        category = item["category"]
        pid      = item["id"]

        if verbose:
            print(f"[{i:02d}/{len(prompts)}] {pid} — {prompt[:55]}…")
        else:
            print(f"  {pid} …", end=" ", flush=True)

        # Call each model with a fresh context so prompts are independent
        oss_resp = oss.chat(prompt)
        oss.reset()

        frontier_resp = frontier.chat(prompt)
        frontier.reset()

        # Score responses
        oss_score      = evaluate(prompt, oss_resp.text,      use_llm_judge=use_llm_judge)
        frontier_score = evaluate(prompt, frontier_resp.text, use_llm_judge=use_llm_judge)

        # Accumulate per-metric lists
        for key in ("safety", "hallucination", "bias"):
            oss_totals[key].append(getattr(oss_score, key))
            frontier_totals[key].append(getattr(frontier_score, key))
        oss_totals["latency"].append(oss_resp.latency_ms)
        frontier_totals["latency"].append(frontier_resp.latency_ms)

        # Category breakdown
        if category not in category_scores:
            category_scores[category] = {"oss": [], "frontier": []}
        oss_avg      = (oss_score.safety + oss_score.hallucination + oss_score.bias) / 3
        frontier_avg = (frontier_score.safety + frontier_score.hallucination + frontier_score.bias) / 3
        category_scores[category]["oss"].append(oss_avg)
        category_scores[category]["frontier"].append(frontier_avg)

        results.append({
            "id":       pid,
            "category": category,
            "prompt":   prompt,
            "oss": {
                "response":     oss_resp.text[:500],
                "latency_ms":   oss_resp.latency_ms,
                "safety":       oss_score.safety,
                "hallucination": oss_score.hallucination,
                "bias":         oss_score.bias,
                "reasoning":    oss_score.reasoning,
            },
            "frontier": {
                "response":     frontier_resp.text[:500],
                "latency_ms":   frontier_resp.latency_ms,
                "safety":       frontier_score.safety,
                "hallucination": frontier_score.hallucination,
                "bias":         frontier_score.bias,
                "reasoning":    frontier_score.reasoning,
            },
        })

        if verbose:
            print(
                f"     OSS      safety={oss_score.safety} halluc={oss_score.hallucination}"
                f" bias={oss_score.bias}  {oss_resp.latency_ms}ms"
            )
            print(
                f"     Frontier safety={frontier_score.safety} halluc={frontier_score.hallucination}"
                f" bias={frontier_score.bias}  {frontier_resp.latency_ms}ms\n"
            )
        else:
            print("✓" if frontier_avg >= oss_avg else "~")

        time.sleep(0.5)   # be polite to rate-limited APIs

    # --- build summary -------------------------------------------------------
    def avg(lst: list) -> int:
        return round(sum(lst) / len(lst)) if lst else 0

    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "judge_mode":   "llm" if use_llm_judge else "heuristic",
        "total_prompts": len(results),
        "oss": {
            "model":             os.getenv("OSS_MODEL", "Qwen/Qwen2.5-72B-Instruct"),
            "avg_safety":        avg(oss_totals["safety"]),
            "avg_hallucination": avg(oss_totals["hallucination"]),
            "avg_bias":          avg(oss_totals["bias"]),
            "avg_latency_ms":    avg(oss_totals["latency"]),
        },
        "frontier": {
            "model":             os.getenv("FRONTIER_MODEL", "claude-sonnet-4-20250514"),
            "avg_safety":        avg(frontier_totals["safety"]),
            "avg_hallucination": avg(frontier_totals["hallucination"]),
            "avg_bias":          avg(frontier_totals["bias"]),
            "avg_latency_ms":    avg(frontier_totals["latency"]),
        },
        "by_category": {
            cat: {
                "oss_avg":      round(sum(v["oss"])      / len(v["oss"]),      1),
                "frontier_avg": round(sum(v["frontier"]) / len(v["frontier"]), 1),
            }
            for cat, v in category_scores.items()
        },
    }

    output = {"summary": summary, "results": results}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # --- print summary -------------------------------------------------------
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n  {'Metric':<22} {'OSS':>8} {'Frontier':>10}")
    print("  " + "-" * 42)
    s = summary
    print(f"  {'Safety Score':<22} {s['oss']['avg_safety']:>7}%  {s['frontier']['avg_safety']:>8}%")
    print(f"  {'Hallucination Score':<22} {s['oss']['avg_hallucination']:>7}%  {s['frontier']['avg_hallucination']:>8}%")
    print(f"  {'Bias Resistance':<22} {s['oss']['avg_bias']:>7}%  {s['frontier']['avg_bias']:>8}%")
    print(f"  {'Avg Latency':<22} {s['oss']['avg_latency_ms']:>6}ms  {s['frontier']['avg_latency_ms']:>7}ms")
    print("\n  By Category:")
    for cat, sc in s["by_category"].items():
        winner = "OSS" if sc["oss_avg"] >= sc["frontier_avg"] else "Frontier"
        print(f"  {cat:<25} OSS {sc['oss_avg']:.0f}  vs  Frontier {sc['frontier_avg']:.0f}  [{winner} wins]")
    print(f"\n  Results written to: {output_path}")
    print("=" * 60 + "\n")

    return output


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DualMind evaluation suite — compare OSS vs Frontier AI",
    )
    # FIX: default to paths relative to this file so the script works from
    #      any CWD (repo root, evaluation/, etc.)
    parser.add_argument(
        "--prompts",
        default=str(THIS_DIR / "prompts.json"),
        help="Path to prompts JSON (default: evaluation/prompts.json)",
    )
    parser.add_argument(
        "--output",
        default=str(THIS_DIR / "results.json"),
        help="Where to write results JSON (default: evaluation/results.json)",
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Use Claude Opus as judge instead of heuristics (requires ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--category",
        choices=["factual", "adversarial", "bias", "hallucination_bait", "general"],
        default=None,
        help="Run only one category of prompts",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Limit number of prompts (useful for quick smoke tests)",
    )
    args = parser.parse_args()

    run_evaluation(
        prompts_file=args.prompts,
        output_file=args.output,
        use_llm_judge=args.llm_judge,
        category_filter=args.category,
        verbose=args.verbose,
        max_prompts=args.max,
    )
