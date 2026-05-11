"""
EX26: Popper ILP Baseline on EX25/EX25b Generalization Protocol.

Apples-to-apples comparison of Popper (Cropper & Morel 2021) against
DreamLog ablations on the canonical (family, EX25) and novel (crafting,
EX25b) domains. See experiments/experiment_registry.yaml::EX26 for the
full design (hypotheses, success criteria, pitfalls addressed).

Setup verified 2026-05-09 (Popper commit af39e522 pinned).

Run:
    python experiments/ex26_popper_baseline.py --domain both \\
        --systems all --seeds 42 43 44 --timeout 300

Per-cell timeout caps Popper search; failures (timeout, parse error,
crash) are recorded honestly as zero rules / N/A recall, not retried.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field, asdict
from io import StringIO
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dreamlog.knowledge import KnowledgeBase, Fact, Rule
from dreamlog.terms import Atom, Compound, Variable
from dreamlog.prefix_parser import parse_s_expression

from ex25_generalization import (  # type: ignore
    NEW_ENTITY_CHECKS,
    NEW_FAMILY_BASE,
    build_family_tree,
    build_kb,
    dream_kb,
    family_base_facts,
    family_derived_facts,
    get_llm_client,
    is_derivable,
)
from ex25b_novel_generalization import (  # type: ignore
    NEW_CRAFTING_BASE,
    NEW_CRAFTING_CHECKS,
    crafting_base_facts,
    crafting_derived_facts,
)


# ══════════════════════════════════════════════════════════════════════
# Domain bias declarations (committed before any results were observed)
# ══════════════════════════════════════════════════════════════════════

FAMILY_TARGETS = [
    "father", "mother", "grandparent", "grandfather", "grandmother",
    "great_grandparent", "ancestor",
]
# ancestor requires recursion. Popper hangs in clingo C-land (Python-level
# timeout doesn't interrupt) on this bias. EX25 also reports DreamLog full
# pipeline does not recover ancestor. Excluding from default comparison so
# both systems get a pass on the recursive case. Override with --include-ancestor.
FAMILY_TARGETS_DEFAULT = [t for t in FAMILY_TARGETS if t != "ancestor"]

# One bias template per target predicate. Body predicates listed are the
# natural primitives a learner would consider for that target.
FAMILY_BIAS_BY_TARGET: Dict[str, str] = {
    "father": """\
max_vars(3).
max_body(2).
max_clauses(1).
head_pred(father,2).
body_pred(parent,2).
body_pred(male,1).
body_pred(female,1).
type(father,(person,person)).
type(parent,(person,person)).
type(male,(person,)).
type(female,(person,)).
direction(father,(in,out)).
direction(parent,(in,out)).
direction(male,(in,)).
direction(female,(in,)).
""",
    "mother": """\
max_vars(3).
max_body(2).
max_clauses(1).
head_pred(mother,2).
body_pred(parent,2).
body_pred(male,1).
body_pred(female,1).
type(mother,(person,person)).
type(parent,(person,person)).
type(male,(person,)).
type(female,(person,)).
direction(mother,(in,out)).
direction(parent,(in,out)).
direction(male,(in,)).
direction(female,(in,)).
""",
    "grandparent": """\
max_vars(3).
max_body(2).
max_clauses(1).
head_pred(grandparent,2).
body_pred(parent,2).
type(grandparent,(person,person)).
type(parent,(person,person)).
direction(grandparent,(in,out)).
direction(parent,(in,out)).
""",
    "grandfather": """\
max_vars(4).
max_body(3).
max_clauses(1).
head_pred(grandfather,2).
body_pred(parent,2).
body_pred(male,1).
type(grandfather,(person,person)).
type(parent,(person,person)).
type(male,(person,)).
direction(grandfather,(in,out)).
direction(parent,(in,out)).
direction(male,(in,)).
""",
    "grandmother": """\
max_vars(4).
max_body(3).
max_clauses(1).
head_pred(grandmother,2).
body_pred(parent,2).
body_pred(female,1).
type(grandmother,(person,person)).
type(parent,(person,person)).
type(female,(person,)).
direction(grandmother,(in,out)).
direction(parent,(in,out)).
direction(female,(in,)).
""",
    "great_grandparent": """\
max_vars(4).
max_body(3).
max_clauses(1).
head_pred(great_grandparent,2).
body_pred(parent,2).
type(great_grandparent,(person,person)).
type(parent,(person,person)).
direction(great_grandparent,(in,out)).
direction(parent,(in,out)).
""",
    # ancestor is recursive - allow recursion + 2 clauses (base + step)
    "ancestor": """\
max_vars(3).
max_body(2).
max_clauses(2).
head_pred(ancestor,2).
body_pred(parent,2).
body_pred(ancestor,2).
type(ancestor,(person,person)).
type(parent,(person,person)).
direction(ancestor,(in,out)).
direction(parent,(in,out)).
""",
}

CRAFTING_TARGETS = [
    "hazardous_recipe", "safe_recipe", "same_phase_recipe",
    "metallic_alloy", "can_craft", "master_artisan",
]

CRAFTING_BIAS_BY_TARGET: Dict[str, str] = {
    "hazardous_recipe": """\
max_vars(4).
max_body(3).
max_clauses(2).
head_pred(hazardous_recipe,1).
body_pred(recipe,3).
body_pred(hazardous,1).
type(hazardous_recipe,(product,)).
type(recipe,(product,material,material)).
type(hazardous,(material,)).
direction(hazardous_recipe,(in,)).
direction(recipe,(in,out,out)).
direction(hazardous,(in,)).
""",
    "safe_recipe": """\
max_vars(4).
max_body(3).
max_clauses(1).
head_pred(safe_recipe,1).
body_pred(product,1).
body_pred(recipe,3).
body_pred(hazardous,1).
type(safe_recipe,(product,)).
type(product,(product,)).
type(recipe,(product,material,material)).
type(hazardous,(material,)).
direction(safe_recipe,(in,)).
direction(product,(in,)).
direction(recipe,(in,out,out)).
direction(hazardous,(in,)).
""",
    "same_phase_recipe": """\
max_vars(5).
max_body(3).
max_clauses(1).
head_pred(same_phase_recipe,1).
body_pred(recipe,3).
body_pred(phase,2).
type(same_phase_recipe,(product,)).
type(recipe,(product,material,material)).
type(phase,(material,phase_v)).
direction(same_phase_recipe,(in,)).
direction(recipe,(in,out,out)).
direction(phase,(in,out)).
""",
    "metallic_alloy": """\
max_vars(5).
max_body(3).
max_clauses(1).
head_pred(metallic_alloy,1).
body_pred(recipe,3).
body_pred(material_class,2).
type(metallic_alloy,(product,)).
type(recipe,(product,material,material)).
type(material_class,(material,class_v)).
direction(metallic_alloy,(in,)).
direction(recipe,(in,out,out)).
direction(material_class,(in,out)).
""",
    "can_craft": """\
max_vars(5).
max_body(4).
max_clauses(1).
head_pred(can_craft,2).
body_pred(skill,2).
body_pred(recipe,3).
body_pred(requires_skill,2).
type(can_craft,(artisan,product)).
type(skill,(artisan,skill_v)).
type(recipe,(product,material,material)).
type(requires_skill,(prefix_v,skill_v)).
direction(can_craft,(in,out)).
direction(skill,(in,out)).
direction(recipe,(in,out,out)).
direction(requires_skill,(in,out)).
""",
    "master_artisan": """\
max_vars(4).
max_body(3).
max_clauses(1).
head_pred(master_artisan,1).
body_pred(artisan,1).
body_pred(skill,2).
type(master_artisan,(artisan,)).
type(artisan,(artisan,)).
type(skill,(artisan,skill_v)).
direction(master_artisan,(in,)).
direction(artisan,(in,)).
direction(skill,(in,out)).
""",
}


@dataclass
class DomainConfig:
    name: str
    base_facts_fn: Callable[[], List[str]]
    derived_facts_fn: Callable[[], Dict[str, List[str]]]
    targets: List[str]
    bias_by_target: Dict[str, str]
    new_base: List[str]
    new_checks: List[Tuple[str, bool, str]]


def family_domain(include_ancestor: bool = False) -> DomainConfig:
    tree = build_family_tree()
    targets = FAMILY_TARGETS if include_ancestor else FAMILY_TARGETS_DEFAULT
    return DomainConfig(
        name="family",
        base_facts_fn=lambda: family_base_facts(tree),
        derived_facts_fn=lambda: family_derived_facts(tree),
        targets=targets,
        bias_by_target=FAMILY_BIAS_BY_TARGET,
        new_base=NEW_FAMILY_BASE,
        new_checks=NEW_ENTITY_CHECKS,
    )


def crafting_domain() -> DomainConfig:
    return DomainConfig(
        name="crafting",
        base_facts_fn=crafting_base_facts,
        derived_facts_fn=crafting_derived_facts,
        targets=CRAFTING_TARGETS,
        bias_by_target=CRAFTING_BIAS_BY_TARGET,
        new_base=NEW_CRAFTING_BASE,
        new_checks=NEW_CRAFTING_CHECKS,
    )


# ══════════════════════════════════════════════════════════════════════
# DreamLog ↔ Popper conversion
# ══════════════════════════════════════════════════════════════════════

def sexpr_to_prolog(s: str) -> str:
    """(parent john mary) -> parent(john, mary). No nested terms in our KBs."""
    s = s.strip()
    if not (s.startswith("(") and s.endswith(")")):
        raise ValueError(f"not an s-expression: {s!r}")
    parts = s[1:-1].split()
    return f"{parts[0]}({', '.join(parts[1:])})"


def fact_strs_to_prolog(facts: List[str]) -> str:
    """Render base facts as Prolog clauses, one per line."""
    return "\n".join(f"{sexpr_to_prolog(f)}." for f in facts)


def all_terms_in_facts(facts: List[str], position_predicate: str = None,
                       position: int = None) -> List[str]:
    """Collect all unique constants. If position_predicate given, only from
    that argument position of that predicate."""
    terms = set()
    for f in facts:
        s = f.strip()[1:-1].split()
        functor, args = s[0], s[1:]
        if position_predicate is not None:
            if functor == position_predicate and position is not None and position < len(args):
                terms.add(args[position])
        else:
            for a in args:
                terms.add(a)
    return sorted(terms)


def universe_for_target(target_pred: str, arity: int,
                        base_facts: List[str]) -> List[Tuple[str, ...]]:
    """All possible ground tuples for target_pred(_,...) given the constants
    appearing in base_facts. Used as the closed-world universe for negative
    sampling."""
    # Universe of terms = everything that ever appears as an argument to any
    # base predicate. This is the closed-world domain.
    domain = all_terms_in_facts(base_facts)
    if arity == 1:
        return [(t,) for t in domain]
    elif arity == 2:
        return [(a, b) for a in domain for b in domain]
    elif arity == 3:
        return [(a, b, c) for a in domain for b in domain for c in domain]
    else:
        raise ValueError(f"arity {arity} not supported")


def sample_negatives(target_pred: str, arity: int, positives: List[str],
                     base_facts: List[str], rng: random.Random,
                     n: Optional[int] = None) -> List[str]:
    """Closed-world random sample of ground tuples not in positives.
    |E-| = |E+| by default."""
    if n is None:
        n = len(positives)
    pos_set = set(positives)
    universe = universe_for_target(target_pred, arity, base_facts)
    candidates = [
        f"({target_pred} {' '.join(t)})"
        for t in universe
        if f"({target_pred} {' '.join(t)})" not in pos_set
    ]
    if len(candidates) <= n:
        return candidates
    return rng.sample(candidates, n)


def write_popper_inputs(out_dir: Path, target_pred: str,
                        bk_facts: List[str], pos: List[str],
                        neg: List[str], bias: str) -> None:
    """Write bk.pl, exs.pl, bias.pl into out_dir for one Popper call."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bk.pl").write_text(fact_strs_to_prolog(bk_facts) + "\n")
    pos_lines = [f"pos({sexpr_to_prolog(p)})." for p in pos]
    neg_lines = [f"neg({sexpr_to_prolog(n)})." for n in neg]
    (out_dir / "exs.pl").write_text("\n".join(pos_lines + neg_lines) + "\n")
    (out_dir / "bias.pl").write_text(bias)


# ══════════════════════════════════════════════════════════════════════
# Popper invocation + output parsing
# ══════════════════════════════════════════════════════════════════════

# Popper returns prog as frozenset[ (head_literal, body_frozenset) ].
# Each Literal has .predicate (str) and .arguments (tuple of ints,
# where each int is a variable index — V0, V1, ... in canonical form).
def popper_literal_to_term(lit) -> Compound:
    args = [Variable(f"V{i}") for i in lit.arguments]
    return Compound(lit.predicate, args)


def popper_program_to_rules(prog) -> List[Rule]:
    """Convert Popper's hypothesis (frozenset of (head, body) tuples) to a
    list of DreamLog Rules. Returns [] if prog is None."""
    if prog is None:
        return []
    rules: List[Rule] = []
    for clause in prog:
        head_lit, body_set = clause
        head = popper_literal_to_term(head_lit)
        body = [popper_literal_to_term(lit) for lit in body_set]
        rules.append(Rule(head, body))
    return rules


def format_rule(rule: Rule) -> str:
    """Render a Rule as a Prolog-ish string for logging."""
    head_args = ",".join(str(a) for a in rule.head.args)
    body_parts = []
    for goal in rule.body:
        if isinstance(goal, Compound):
            body_parts.append(f"{goal.functor}({','.join(str(a) for a in goal.args)})")
        else:
            body_parts.append(str(goal))
    head_str = f"{rule.head.functor}({head_args})"
    if body_parts:
        return f"{head_str} :- {', '.join(body_parts)}."
    return f"{head_str}."


def run_popper_one_target(domain: str, target_pred: str, bk_facts: List[str],
                          positives: List[str], bias: str,
                          rng: random.Random, timeout: int,
                          input_persist_dir: Optional[Path] = None,
                          ) -> Dict:
    """Run Popper for one target predicate. Returns dict with rules, score,
    elapsed, and a status string ('ok', 'no_rule', 'timeout', 'error')."""
    if not positives:
        return {"target": target_pred, "rules": [], "score": None,
                "elapsed": 0.0, "status": "no_positives"}

    # Determine arity from a positive example.
    arity = len(positives[0].strip()[1:-1].split()) - 1
    negatives = sample_negatives(target_pred, arity, positives, bk_facts, rng)

    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        write_popper_inputs(tmp, target_pred, bk_facts, positives, negatives, bias)

        if input_persist_dir is not None:
            input_persist_dir.mkdir(parents=True, exist_ok=True)
            for fn in ("bk.pl", "exs.pl", "bias.pl"):
                (input_persist_dir / fn).write_text((tmp / fn).read_text())

        start = time.time()
        try:
            from popper.loop import popper as popper_main
            from popper.util import Settings
            settings = Settings(
                bk_file=str(tmp / "bk.pl"),
                ex_file=str(tmp / "exs.pl"),
                bias_file=str(tmp / "bias.pl"),
                timeout=timeout,
                verbosity=0,
            )
            buf_out, buf_err = StringIO(), StringIO()
            with redirect_stdout(buf_out), redirect_stderr(buf_err):
                prog, score = popper_main(settings)
            elapsed = time.time() - start
        except Exception as e:
            return {"target": target_pred, "rules": [], "score": None,
                    "elapsed": time.time() - start, "status": "error",
                    "error": f"{type(e).__name__}: {e}"}

        if prog is None:
            return {"target": target_pred, "rules": [], "score": score,
                    "elapsed": elapsed, "status": "no_rule"}

        rules = popper_program_to_rules(prog)
        rule_strs = [format_rule(r) for r in rules]
        return {"target": target_pred, "rules": rules,
                "rule_strs": rule_strs,
                "score": list(score) if score else None,
                "elapsed": elapsed, "status": "ok"}


# ══════════════════════════════════════════════════════════════════════
# System runners (return list of Rule objects to layer onto the test KB)
# ══════════════════════════════════════════════════════════════════════

def run_popper(domain: DomainConfig, train_derived: Dict[str, List[str]],
               base_facts: List[str], seed: int, timeout: int,
               persist_root: Optional[Path] = None,
               ) -> Tuple[List[Rule], Dict]:
    rng = random.Random(seed)
    all_rules: List[Rule] = []
    per_target = []
    for target in domain.targets:
        bias = domain.bias_by_target[target]
        # Background = base facts + other targets' positives (so multi-target
        # rules can reference each other when bias permits).
        bk = list(base_facts)
        for other_target, other_facts in train_derived.items():
            if other_target != target:
                bk.extend(other_facts)
        positives = train_derived.get(target, [])
        persist = (persist_root / target) if persist_root else None
        result = run_popper_one_target(
            domain.name, target, bk, positives, bias,
            rng, timeout, input_persist_dir=persist,
        )
        per_target.append({k: v for k, v in result.items() if k != "rules"})
        all_rules.extend(result.get("rules", []))
    return all_rules, {"per_target": per_target}


def run_dreamlog(mode: str, domain: DomainConfig,
                 train_derived: Dict[str, List[str]], base_facts: List[str],
                 llm_client) -> Tuple[List[Rule], Dict]:
    """mode in {'no-dream', 'symbolic', 'full'}.

    no-dream: pure base KB, no rule discovery. Baseline showing what
              the unaugmented KB derives.
    symbolic: dream() without LLM. Symbolic ops (A-F, H) only.
    full:     dream() with LLM. Symbolic ops + Op G (LLM-assisted).
    """
    if mode == "no-dream":
        return [], {"mode": "no-dream", "rule_strs": []}

    from dreamlog.kb_dreamer import KnowledgeBaseDreamer
    train_facts = base_facts + [f for facts in train_derived.values() for f in facts]
    kb = build_kb(train_facts)
    client = llm_client if mode == "full" else None
    start = time.time()
    dreamer = KnowledgeBaseDreamer(llm_client=client, max_prompt_facts=200)
    session = dreamer.dream(kb, verify=True)
    elapsed = time.time() - start
    discovered: List[Rule] = []
    for op in session.operations:
        for c in op.new_clauses:
            if isinstance(c, Rule):
                discovered.append(c)
    return discovered, {"mode": mode,
                        "compression_ratio": session.compression_ratio,
                        "elapsed": elapsed,
                        "rule_strs": [str(r) for r in discovered]}


# ══════════════════════════════════════════════════════════════════════
# Evaluation on held-out new entities
# ══════════════════════════════════════════════════════════════════════

@dataclass
class EvalResult:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    per_check: List[Dict] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total else 0.0


def evaluate_on_new_entities(rules: List[Rule], base_facts: List[str],
                             new_base: List[str],
                             new_checks: List[Tuple[str, bool, str]],
                             ) -> EvalResult:
    """Build a KB from base + new entity facts + discovered rules, then check
    each (query, expected) against derivability."""
    kb = build_kb(base_facts + new_base)
    for r in rules:
        kb.add_rule(r)
    res = EvalResult()
    for query, expected, desc in new_checks:
        derived = is_derivable(kb, query, max_calls=20000)
        if expected and derived:
            res.tp += 1
            outcome = "TP"
        elif expected and not derived:
            res.fn += 1
            outcome = "FN"
        elif not expected and derived:
            res.fp += 1
            outcome = "FP"
        else:
            res.tn += 1
            outcome = "TN"
        res.per_check.append({"query": query, "expected": expected,
                              "derived": derived, "outcome": outcome,
                              "desc": desc})
    return res


# ══════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════

DOMAIN_BUILDERS: Dict[str, Callable[..., DomainConfig]] = {
    "family": family_domain, "crafting": crafting_domain,
}

SYSTEMS = ["dreamlog-no-dream", "dreamlog-symbolic", "dreamlog-full", "popper"]


def run_cell(system: str, domain: DomainConfig, seed: int,
             timeout: int, llm_client, persist_root: Optional[Path]) -> Dict:
    base_facts = domain.base_facts_fn()
    derived = domain.derived_facts_fn()  # full set; no holdout for now (ratio=0)
    cell_id = f"{system}_{domain.name}_seed{seed}"
    print(f"\n  ── {cell_id} ──")

    start = time.time()
    if system == "popper":
        persist = (persist_root / cell_id) if persist_root else None
        rules, meta = run_popper(domain, derived, base_facts, seed, timeout, persist)
    elif system.startswith("dreamlog"):
        mode = system.split("-", 1)[1]
        rules, meta = run_dreamlog(mode, domain, derived, base_facts, llm_client)
    else:
        raise ValueError(f"unknown system: {system}")
    elapsed = time.time() - start

    eval_res = evaluate_on_new_entities(
        rules, base_facts, domain.new_base, domain.new_checks,
    )
    print(f"     rules={len(rules)} recall={eval_res.recall:.0%} "
          f"precision={eval_res.precision:.0%} accuracy={eval_res.accuracy:.0%} "
          f"elapsed={elapsed:.1f}s")
    return {
        "system": system,
        "domain": domain.name,
        "seed": seed,
        "elapsed": elapsed,
        "n_rules": len(rules),
        "rule_strs": [str(r) for r in rules],
        "tp": eval_res.tp, "fp": eval_res.fp,
        "tn": eval_res.tn, "fn": eval_res.fn,
        "recall": eval_res.recall,
        "precision": eval_res.precision,
        "accuracy": eval_res.accuracy,
        "per_check": eval_res.per_check,
        "meta": meta,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["family", "crafting", "both"],
                    default="both")
    ap.add_argument("--systems", nargs="+", default=SYSTEMS,
                    help=f"Subset of {SYSTEMS}")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--timeout", type=int, default=300,
                    help="Per-Popper-call seconds")
    ap.add_argument("--api-key-env", default="MY_ANTHROPIC_API_KEY")
    ap.add_argument("--out", default="experiments/data/popper/results.jsonl")
    ap.add_argument("--persist-inputs", action="store_true",
                    help="Save Popper bk/exs/bias files for reproducibility")
    ap.add_argument("--pilot", action="store_true",
                    help="Single cell only: popper + family + seed 42")
    ap.add_argument("--include-ancestor", action="store_true",
                    help="Include the recursive ancestor target (default: excluded; "
                         "Popper hangs in C-land on this bias and EX25 also fails)")
    args = ap.parse_args()

    if args.pilot:
        args.systems = ["popper"]
        args.domain = "family"
        args.seeds = [42]

    domains = ["family", "crafting"] if args.domain == "both" else [args.domain]
    llm_client = get_llm_client(args) if "dreamlog-full" in args.systems else None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    persist_root = (out_path.parent / "cells") if args.persist_inputs else None

    print(f"EX26 Popper baseline run starting")
    print(f"  systems  : {args.systems}")
    print(f"  domains  : {domains}")
    print(f"  seeds    : {args.seeds}")
    print(f"  timeout  : {args.timeout}s per Popper call")
    print(f"  out      : {out_path}")

    rows = []
    for dname in domains:
        if dname == "family":
            domain = DOMAIN_BUILDERS[dname](include_ancestor=args.include_ancestor)
        else:
            domain = DOMAIN_BUILDERS[dname]()
        for system in args.systems:
            for seed in args.seeds:
                row = run_cell(system, domain, seed, args.timeout,
                               llm_client, persist_root)
                rows.append(row)
                with out_path.open("a") as f:
                    f.write(json.dumps(row, default=str) + "\n")

    print(f"\nDone. {len(rows)} cells written to {out_path}.")


if __name__ == "__main__":
    main()
