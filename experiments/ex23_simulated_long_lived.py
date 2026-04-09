#!/usr/bin/env python3
"""
EX23: Simulated long-lived knowledge accumulation.

Models a user building up knowledge over multiple "sessions" — asserting
facts, querying, and periodically dreaming. Tracks KB evolution to show
the learning-over-time story for the paper.

Scenario: A developer using DreamLog to track their project ecosystem.
Over 10 "sessions" they add facts about repos, languages, dependencies,
team members, and roles. The system should discover:
- role(X,Y) :- member(X,Y), skill(Y,lang) patterns
- dependency transitivity
- team-language correlations

Usage:
    python experiments/ex23_simulated_long_lived.py
"""

import sys
import time
import json
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, ".")

from integrations.mcp.knowledge_store import KnowledgeStore


# ── Session data: simulates a user building knowledge over time ──────

SESSIONS = [
    {
        "name": "Session 1: Team setup",
        "assertions": [
            "(person alice)", "(person bob)", "(person carol)",
            "(person dave)", "(person eve)",
            "(team backend alice)", "(team backend bob)",
            "(team frontend carol)", "(team frontend dave)",
            "(team devops eve)",
            "(role alice lead)", "(role bob senior)",
            "(role carol senior)", "(role dave junior)",
            "(role eve senior)",
        ],
        "queries": [
            "(team backend X)",
            "(role X lead)",
        ],
    },
    {
        "name": "Session 2: Languages and skills",
        "assertions": [
            "(language python)", "(language typescript)", "(language go)",
            "(language rust)", "(language sql)",
            "(knows alice python)", "(knows alice sql)",
            "(knows bob python)", "(knows bob go)",
            "(knows carol typescript)", "(knows carol python)",
            "(knows dave typescript)",
            "(knows eve go)", "(knows eve python)",
        ],
        "queries": [
            "(knows X python)",
            "(knows alice X)",
        ],
    },
    {
        "name": "Session 3: Repos and ownership",
        "assertions": [
            "(repo api_server)", "(repo web_app)", "(repo infra)",
            "(repo ml_pipeline)", "(repo shared_lib)",
            "(owns alice api_server)", "(owns bob api_server)",
            "(owns carol web_app)", "(owns dave web_app)",
            "(owns eve infra)",
            "(owns alice ml_pipeline)",
            "(written_in api_server python)", "(written_in web_app typescript)",
            "(written_in infra go)", "(written_in ml_pipeline python)",
            "(written_in shared_lib python)",
        ],
        "queries": [
            "(owns X api_server)",
            "(written_in X python)",
        ],
    },
    {
        "name": "Session 4: Dependencies",
        "assertions": [
            "(depends api_server shared_lib)",
            "(depends web_app api_server)",
            "(depends ml_pipeline shared_lib)",
            "(depends ml_pipeline api_server)",
            "(depends infra shared_lib)",
            # Transitive closure rule
            "(depends_on X Y) :- (depends X Y)",
            "(depends_on X Z) :- (depends X Y), (depends_on Y Z)",
        ],
        "queries": [
            "(depends_on web_app X)",
            "(depends_on X shared_lib)",
        ],
    },
    {
        "name": "Session 5: More team members and backend facts",
        "assertions": [
            "(person frank)", "(person grace)", "(person henry)",
            "(team backend frank)", "(team backend grace)",
            "(team frontend henry)",
            "(role frank junior)", "(role grace senior)",
            "(role henry junior)",
            "(knows frank python)", "(knows frank rust)",
            "(knows grace python)", "(knows grace sql)",
            "(knows henry typescript)",
            # backend_dev facts (derivable from team + knows python)
            "(backend_dev alice)", "(backend_dev bob)",
            "(backend_dev frank)", "(backend_dev grace)",
        ],
        "queries": [
            "(backend_dev X)",
            "(team backend X)",
        ],
    },
    {
        "name": "Session 6: Code review pairs",
        "assertions": [
            "(reviews alice bob)", "(reviews bob alice)",
            "(reviews carol dave)", "(reviews dave carol)",
            "(reviews frank grace)", "(reviews grace frank)",
            "(reviews alice frank)", "(reviews bob grace)",
            # Pair programming
            "(pair_programs alice bob)",
            "(pair_programs carol dave)",
            "(pair_programs frank grace)",
        ],
        "queries": [
            "(reviews alice X)",
            "(pair_programs X Y)",
        ],
    },
    {
        "name": "Session 7: Project status",
        "assertions": [
            "(status api_server production)",
            "(status web_app production)",
            "(status infra production)",
            "(status ml_pipeline staging)",
            "(status shared_lib production)",
            "(production_repo api_server)", "(production_repo web_app)",
            "(production_repo infra)", "(production_repo shared_lib)",
        ],
        "queries": [
            "(status X production)",
            "(production_repo X)",
        ],
    },
    {
        "name": "Session 8: Oncall and incident data",
        "assertions": [
            "(oncall alice backend)", "(oncall eve infra_oncall)",
            "(oncall bob backend)", "(oncall carol frontend_oncall)",
            "(incident inc001 api_server high)",
            "(incident inc002 web_app medium)",
            "(incident inc003 infra high)",
            "(assigned inc001 alice)", "(assigned inc002 carol)",
            "(assigned inc003 eve)",
        ],
        "queries": [
            "(incident X Y high)",
            "(assigned X alice)",
        ],
    },
    {
        "name": "Session 9: More dependency and ownership patterns",
        "assertions": [
            "(repo data_lake)", "(repo auth_service)",
            "(written_in data_lake python)",
            "(written_in auth_service go)",
            "(owns bob auth_service)", "(owns eve auth_service)",
            "(owns alice data_lake)", "(owns grace data_lake)",
            "(depends api_server auth_service)",
            "(depends data_lake shared_lib)",
            "(production_repo auth_service)",
            "(production_repo data_lake)",
            "(status auth_service production)",
            "(status data_lake staging)",
        ],
        "queries": [
            "(written_in X go)",
            "(depends_on api_server X)",
            "(production_repo X)",
        ],
    },
    {
        "name": "Session 10: Team leads pattern",
        "assertions": [
            "(leads alice backend)", "(leads carol frontend)",
            "(leads eve devops)",
            "(mentor alice frank)", "(mentor bob dave)",
            "(mentor carol henry)", "(mentor grace frank)",
        ],
        "queries": [
            "(leads X backend)",
            "(mentor X frank)",
            "(knows X python)",
        ],
    },
]


def run_simulation():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="anthropic")
    parser.add_argument("--api-key-env", default="MY_ANTHROPIC_API_KEY")
    parser.add_argument("--dream-every", type=int, default=3,
                        help="Dream every N sessions")
    parser.add_argument("--store", default=None,
                        help="Store path (default: temp)")
    args = parser.parse_args()

    import tempfile
    store_path = Path(args.store) if args.store else Path(tempfile.mktemp(suffix=".json"))

    store = KnowledgeStore(store_path, llm_budget_usd=2.0, dream_threshold=999)

    # Configure LLM if available
    if args.provider == "anthropic":
        from dreamlog.llm_client import LLMClient
        store.llm_client = LLMClient(
            provider="anthropic", api_key_env=args.api_key_env,
            temperature=0.3, max_tokens=800)

    print(f"{'='*70}")
    print(f"  EX23: Simulated Long-Lived Knowledge Accumulation")
    print(f"  Store: {store_path}")
    print(f"  Dream every {args.dream_every} sessions")
    print(f"{'='*70}")

    # Track evolution
    timeline = []

    for i, session in enumerate(SESSIONS, 1):
        print(f"\n  --- {session['name']} ---", flush=True)

        # Assert facts
        for fact in session["assertions"]:
            store.assert_fact(fact)

        # Run queries
        for query in session["queries"]:
            store.query(query)

        snapshot = {
            "session": i,
            "name": session["name"],
            "facts_added": len(session["assertions"]),
            "queries_run": len(session["queries"]),
            "total_clauses": len(store.kb),
            "facts": len(store.kb.facts),
            "rules": len(store.kb.rules),
            "dreams": store._dream_count,
        }

        # Dream periodically
        if i % args.dream_every == 0:
            print(f"    Dreaming...", flush=True)
            t0 = time.perf_counter()
            result = store.dream()
            elapsed = time.perf_counter() - t0

            snapshot["dream"] = {
                "before": result["before"],
                "after": result["after"],
                "removed": result["removed"],
                "ratio": round(result["ratio"], 3),
                "operations": result["operations"],
                "new_rules": result["new_rules"],
                "time": round(elapsed, 1),
                "llm_used": result["llm_used"],
            }

            if result["new_rules"]:
                print(f"    Discovered rules:", flush=True)
                for r in result["new_rules"]:
                    print(f"      + {r}", flush=True)
            print(f"    {result['before']} -> {result['after']} "
                  f"({result['ratio']:.2f}) in {elapsed:.1f}s", flush=True)

            # Update snapshot with post-dream state
            snapshot["total_clauses"] = len(store.kb)
            snapshot["facts"] = len(store.kb.facts)
            snapshot["rules"] = len(store.kb.rules)

        print(f"    KB: {snapshot['total_clauses']} clauses "
              f"({snapshot['facts']} facts, {snapshot['rules']} rules)",
              flush=True)

        timeline.append(snapshot)

    # ── Final dream ───────────────────────────────────────────────
    print(f"\n  --- Final dream ---", flush=True)
    t0 = time.perf_counter()
    final = store.dream()
    elapsed = time.perf_counter() - t0
    print(f"    {final['before']} -> {final['after']} in {elapsed:.1f}s",
          flush=True)
    if final["new_rules"]:
        print(f"    Discovered:", flush=True)
        for r in final["new_rules"]:
            print(f"      + {r}", flush=True)

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  EVOLUTION TIMELINE")
    print(f"{'='*70}")
    print(f"  {'Session':<30} {'Clauses':>8} {'Facts':>6} {'Rules':>6} {'Dream':>6}")
    print(f"  {'-'*30} {'-'*8} {'-'*6} {'-'*6} {'-'*6}")

    for s in timeline:
        dream_str = ""
        if "dream" in s:
            d = s["dream"]
            dream_str = f"-{d['removed']}"
        print(f"  {s['name']:<30} {s['total_clauses']:>8} {s['facts']:>6} "
              f"{s['rules']:>6} {dream_str:>6}")

    # Final state
    print(f"\n  Final KB: {len(store.kb)} clauses "
          f"({len(store.kb.facts)} facts, {len(store.kb.rules)} rules)")

    print(f"\n  Rules in final KB:")
    for rule in store.kb.rules:
        print(f"    {rule}")

    # Budget
    if store.llm_client:
        print(f"\n  LLM usage: {store.llm_client.usage}")

    # Verify key queries still work
    print(f"\n  Correctness spot-checks:")
    checks = [
        ("(team backend alice)", True),
        ("(knows bob python)", True),
        ("(depends_on web_app shared_lib)", True),
        ("(owns alice api_server)", True),
        ("(leads alice backend)", True),
        ("(team backend carol)", False),
    ]
    all_pass = True
    for query, expected in checks:
        result = store.query(query)
        got = result["count"] > 0
        status = "PASS" if got == expected else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"    {status}: {query} = {got}")
    print(f"  All pass: {all_pass}")

    # Save timeline for paper
    timeline_path = store_path.with_suffix(".timeline.json")
    with open(timeline_path, "w") as f:
        json.dump(timeline, f, indent=2)
    print(f"\n  Timeline saved: {timeline_path}")

    return timeline


if __name__ == "__main__":
    run_simulation()
