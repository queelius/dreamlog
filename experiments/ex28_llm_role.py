"""EX28: isolating the LLM's contribution by rule type, on qwen2.5:3b."""
import argparse, os, sys, pathlib, subprocess
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ex28_domains import all_domains
from ex28_probe import proposal_rate
from ex28_harness import run_units, summarize, unit_key
from ollama_helper import make_ollama_client
from ex25_generalization import build_kb, dream_kb
from ex25b_novel_generalization import evaluate_checks, run_raw_llm_check


class BudgetExceeded(Exception):
    """Raised to stop the run when cumulative LLM cost crosses the budget."""


def make_client(provider, model):
    """Build the LLM client. ollama (local, free) or anthropic (paid, tracked
    via estimated_cost). One client accumulates usage across all units, so the
    budget guard sees the running total."""
    if provider == "ollama":
        return make_ollama_client(model=model) if model else make_ollama_client()
    from dreamlog.llm_client import LLMClient
    key_env = "MY_ANTHROPIC_API_KEY" if provider == "anthropic" else None
    return LLMClient(provider=provider, model=model, api_key_env=key_env,
                     temperature=0.3, max_tokens=800, timeout=180)


# Which symbolic op each rule type owns (for the symbolic-only / LLM-only split)
SYMBOLIC_OP = {"within_predicate": "op_c", "recursive": "op_i",
               "cross_predicate": None}


def _dream_flags(rule_type, condition):
    """Return (discover_recursion, disable_op_c, use_llm) for a condition.

    In the llm_only condition we turn OFF the symbolic op that owns the rule
    type, so the LLM is isolated: Op I for recursive (discover=False), Op C for
    within_predicate (disable_op_c=True). cross_predicate needs no disable flag:
    no symbolic op can build a cross-functor rule from a fact-only KB (Op C only
    generalizes a single functor; Op D/E need pre-existing rules and run before
    Op G), so the symbolic pipeline is already inert there.
    """
    use_llm = condition in ("llm_only", "full")
    discover = (rule_type == "recursive") and condition in ("symbolic_only", "full")
    # llm_only: turn OFF the rule-type's owning symbolic op (see docstring)
    disable_c = (rule_type == "within_predicate") and condition == "llm_only"
    return discover, disable_c, use_llm


def run_one_unit(unit, domains_by_name, client, n_probe):
    dom = domains_by_name[unit["cell"]]
    cond = unit["condition"]
    if cond == "raw_llm":
        tp = tn = fp = fn = 0
        for q, exp, _ in dom.new_checks:
            got = run_raw_llm_check(client, dom.base + dom.derived, dom.new_base, q, "")
            tp += int(exp and got); fn += int(exp and not got)
            tn += int((not exp) and not got); fp += int((not exp) and got)
        return {"recovery": tp / (tp + fn) if (tp + fn) else 0.0,
                "precision": tp / (tp + fp) if (tp + fp) else 0.0,
                "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                "proposal_rate": None, "probe": None}
    discover, disable_c, use_llm = _dream_flags(dom.rule_type, cond)
    kb = build_kb(dom.base + dom.derived)
    dream_kb(kb, llm_client=client if use_llm else None,
             discover_recursion=discover, disable_op_c=disable_c, open_world=True)
    res = evaluate_checks(kb, dom.new_base, dom.new_checks)
    # proposal rate only where the LLM is the route under study; keep the full
    # probe result (per-run proposed rules) for provenance, not just the rate.
    probe = None
    pr = None
    if use_llm:
        probe = proposal_rate(dom, client, n_runs=n_probe)
        pr = probe["rate"]
    return {"recovery": res["recall"], "precision": res["precision"],
            "tp": res["tp"], "tn": res["tn"], "fp": res["fp"], "fn": res["fn"],
            "proposal_rate": pr, "probe": probe}


def main():
    ap = argparse.ArgumentParser(description="EX28 LLM-role ablation")
    ap.add_argument("--n-probe", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--provider", default="ollama",
                    help="ollama (local, free) or anthropic (paid)")
    ap.add_argument("--model", default=None,
                    help="model id; defaults to the provider default")
    ap.add_argument("--budget-usd", type=float, default=None,
                    help="stop before any unit once cumulative cost exceeds this")
    ap.add_argument("--store", default="experiments/data/ex28/results.jsonl")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()

    if args.summarize:
        for r in summarize(args.store):
            print(r["cell"], r["condition"], r.get("recovery"),
                  r.get("proposal_rate"), r.get("cost_usd"))
        return

    doms = {d.name: d for d in all_domains(seed=args.seed)}
    client = make_client(args.provider, args.model)
    conditions = ["symbolic_only", "llm_only", "full", "raw_llm"]
    units = [{"cell": name, "condition": c, "run": 0}
             for name in doms for c in conditions]
    gp = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True)
    git_sha = (gp.stdout.strip() if gp.returncode == 0 else "") or "unknown"
    store_dir = os.path.dirname(args.store) or "."
    os.makedirs(store_dir, exist_ok=True)

    def run_with_guard(u):
        if args.budget_usd is not None and client.estimated_cost() > args.budget_usd:
            raise BudgetExceeded(
                f"cumulative cost ${client.estimated_cost():.4f} exceeds budget "
                f"${args.budget_usd:.2f}; completed units saved, rerun to resume")
        res = run_one_unit(u, doms, client, args.n_probe)
        res["cost_usd"] = round(client.estimated_cost(), 4)
        return res

    try:
        run_units(units, run_with_guard, args.store, manifest_dir=store_dir,
                  git_sha=git_sha, fresh=args.fresh)
    except BudgetExceeded as e:
        print(f"BUDGET STOP: {e}")
        print(f"Cost so far: ${client.estimated_cost():.4f}, "
              f"{client.usage.calls} calls, model={client.model}")
        raise SystemExit(2)
    print(f"EX28 complete ({args.provider}/{client.model}). "
          f"cost=${client.estimated_cost():.4f}, {client.usage.calls} calls")


if __name__ == "__main__":
    main()
