#!/usr/bin/env python3
"""
EX25c: Holdout sweep on the family tree domain.

Addresses the reviewer's concern (M4) that the original holdout result
(0% recovery at 20%) was buried in limitations and unexplored. Sweeps
holdout ratios from 5% to 50% to characterize the data density threshold
for compression-driven generalization.

For each ratio, runs both symbolic-only and full-pipeline conditions
across multiple seeds and reports recovery rate with variance.

Usage:
    python experiments/ex25c_holdout_sweep.py
    python experiments/ex25c_holdout_sweep.py --ratios 0.05,0.1,0.2,0.3,0.5
    python experiments/ex25c_holdout_sweep.py --runs 3
"""

import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, ".")

from experiments.ex25_generalization import (
    build_family_tree, family_base_facts, family_derived_facts,
    family_negatives, build_kb, is_derivable, dream_kb, holdout_split,
    get_llm_client,
)


def run_sweep(llm_client, ratios, n_runs, open_world=True, max_facts=200):
    print(f"{'='*72}")
    print(f"  EX25c: Holdout Sweep")
    print(f"  LLM: {'Anthropic Haiku' if llm_client else 'NONE'}")
    print(f"  Ratios: {ratios}")
    print(f"  Runs per condition: {n_runs}")
    print(f"  Open-world mode: {open_world} (FP check {'disabled' if open_world else 'enabled'})")
    print(f"{'='*72}")

    tree = build_family_tree()
    base = family_base_facts(tree)
    derived = family_derived_facts(tree)
    negatives = family_negatives(tree)
    n_total = sum(len(v) for v in derived.values())
    print(f"\n  Total derived facts: {n_total} across {len(derived)} predicates")
    for pred, facts in sorted(derived.items()):
        print(f"    {pred}: {len(facts)}")

    seeds = [42, 123, 456, 789, 1024][:n_runs]

    # Results indexed by [condition][ratio] -> list of run dicts
    results = {
        "no_dream": {r: [] for r in ratios},
        "symbolic": {r: [] for r in ratios},
        "full":     {r: [] for r in ratios},
    }

    for ratio in ratios:
        print(f"\n{'─'*72}")
        print(f"  Holdout ratio: {ratio:.0%}")
        print(f"{'─'*72}")

        for seed in seeds:
            train, test = holdout_split(derived, ratio, seed)
            print(f"\n  Seed {seed}: train={len(train)}, test={len(test)}")

            for cond_name, client in [
                ("no_dream", None),
                ("symbolic", None),
                ("full", llm_client),
            ]:
                kb = build_kb(base + train)
                rules = []
                comp = 1.0

                if cond_name != "no_dream":
                    t0 = time.perf_counter()
                    comp, rules = dream_kb(kb, llm_client=client,
                                            max_prompt_facts=max_facts,
                                            open_world=open_world)
                    elapsed = time.perf_counter() - t0
                else:
                    elapsed = 0.0

                # Recovery on held-out facts
                recovered = sum(1 for f in test if is_derivable(kb, f))
                recovery = recovered / len(test) if test else 0.0

                # False positives
                false_pos = sum(1 for q, _ in negatives if is_derivable(kb, q))
                fp_rate = false_pos / len(negatives) if negatives else 0.0

                results[cond_name][ratio].append({
                    "seed": seed,
                    "train_size": len(train),
                    "test_size": len(test),
                    "recovered": recovered,
                    "recovery": recovery,
                    "false_pos": false_pos,
                    "fp_rate": fp_rate,
                    "rules": len(rules),
                    "compression": comp,
                    "time_s": elapsed,
                })

                print(f"    [{cond_name:<8}] recovery={recovery:>5.1%} "
                      f"({recovered}/{len(test)})  "
                      f"FP={fp_rate:>5.1%}  "
                      f"rules={len(rules):>2}  "
                      f"comp={comp:.3f}  "
                      f"({elapsed:.1f}s)")

    # ── Summary table ──────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  HOLDOUT SWEEP SUMMARY")
    print(f"{'='*72}")
    print(f"\n  {'Ratio':<8} {'Condition':<10} {'Recovery':>14} "
          f"{'FP':>10} {'Rules':>8} {'Compress':>10}")
    print(f"  {'-'*8} {'-'*10} {'-'*14} {'-'*10} {'-'*8} {'-'*10}")

    summary = {}
    for ratio in ratios:
        summary[ratio] = {}
        for cond in ["no_dream", "symbolic", "full"]:
            runs = results[cond][ratio]
            recs = [r["recovery"] for r in runs]
            fps = [r["fp_rate"] for r in runs]
            ruls = [r["rules"] for r in runs]
            comps = [r["compression"] for r in runs]

            n = len(runs)
            mean_rec = sum(recs) / n
            mean_fp = sum(fps) / n
            mean_rul = sum(ruls) / n
            mean_comp = sum(comps) / n
            std_rec = (sum((x - mean_rec)**2 for x in recs) / n) ** 0.5

            summary[ratio][cond] = {
                "mean_recovery": mean_rec,
                "std_recovery": std_rec,
                "mean_fp_rate": mean_fp,
                "mean_rules": mean_rul,
                "mean_compression": mean_comp,
                "n_runs": n,
            }

            std_str = f"+/-{std_rec:.0%}" if std_rec > 0 else ""
            print(f"  {ratio:>5.0%}    {cond:<10} {mean_rec:>5.1%}{std_str:>9} "
                  f"{mean_fp:>9.1%} {mean_rul:>7.1f} {mean_comp:>9.3f}")
        print()

    # ── Threshold analysis ─────────────────────────────────────────
    print(f"  THRESHOLD ANALYSIS")
    print(f"  {'-'*72}")
    print(f"\n  Recovery vs holdout ratio (full pipeline):")
    for ratio in sorted(ratios, reverse=True):
        rec = summary[ratio]["full"]["mean_recovery"]
        bar_len = int(rec * 50)
        bar = "#" * bar_len
        print(f"    {ratio:>5.0%} held out:  {rec:>5.1%}  {bar}")

    print(f"\n  Interpretation:")
    print(f"  - At low holdout ratios, the full KB is essentially intact;")
    print(f"    discovered rules generalize to the few held-out facts.")
    print(f"  - As ratio increases, fewer facts per predicate remain to")
    print(f"    discover patterns from. Op C requires min_group_size=3.")
    print(f"  - The transition reveals the minimum data density needed for")
    print(f"    compression-driven concept discovery.")

    return results, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key-env", default="MY_ANTHROPIC_API_KEY")
    parser.add_argument("--ratios", default="0.05,0.1,0.2,0.3,0.5",
                        help="Comma-separated holdout ratios")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of seeds per ratio")
    parser.add_argument("--closed-world", action="store_true",
                        help="Use closed-world FP check (default: open-world)")
    parser.add_argument("--store", default="/tmp/ex25c_results.json")
    args = parser.parse_args()

    ratios = [float(r) for r in args.ratios.split(",")]
    llm_client = get_llm_client(args)
    open_world = not args.closed_world

    t0 = time.perf_counter()
    results, summary = run_sweep(llm_client, ratios, args.runs,
                                  open_world=open_world)
    elapsed = time.perf_counter() - t0

    if llm_client:
        u = llm_client.usage
        c = llm_client.estimated_cost()
        print(f"\n  LLM: {u.calls} calls, "
              f"{u.input_tokens:,} in / {u.output_tokens:,} out (${c:.4f})")

    print(f"  Total time: {elapsed:.1f}s")

    # Save to JSON
    save_data = {
        "ratios": ratios,
        "n_runs": args.runs,
        "summary": {
            f"{r:.2f}": {cond: data for cond, data in conds.items()}
            for r, conds in summary.items()
        },
        "elapsed_s": elapsed,
    }
    Path(args.store).write_text(json.dumps(save_data, indent=2))
    print(f"  Results: {args.store}")


if __name__ == "__main__":
    main()
