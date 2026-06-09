"""Proposal-rate probe: how often does Op G propose a rule structurally
equivalent to the domain's target rule, over N runs? Records full per-run
metadata (the proposed rules and the verdict)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ex25_generalization import build_kb
from dreamlog.kb_dreamer import KnowledgeBaseDreamer
from dreamlog.rule_equivalence import rules_structurally_equivalent


def proposal_rate(domain, llm_client, n_runs=30):
    hits = 0
    runs = []
    for i in range(n_runs):
        kb = build_kb(domain.base + domain.derived)
        dreamer = KnowledgeBaseDreamer(llm_client=llm_client)
        proposed = dreamer._llm_propose(kb)
        hit = any(rules_structurally_equivalent(r, domain.target_rule)
                  for r in proposed)
        hits += int(hit)
        runs.append({"run": i, "hit": hit,
                     "proposed": [str(r) for r in proposed]})
    # Record the target identity so a saved run log stays auditable even if the
    # domain definitions later change (the user requires rich, durable metadata).
    return {"hits": hits, "n": n_runs, "rate": hits / n_runs if n_runs else 0.0,
            "domain": domain.name, "target": str(domain.target_rule),
            "runs": runs}
