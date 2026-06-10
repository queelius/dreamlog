"""LLM proposal stage for Operation G (LLM-assisted compression).

Pure proposal: build the prompt, call the LLM, parse the response, apply
Phase 1 structural validation and Phase 2 cyclic filtering. Mutates nothing
(reads the KB only). The acceptance battery (Phase 4/5) lives in the gate +
LlmPolicy; this module stops at returning validated, cycle-filtered rules.
"""
import json
from typing import List, Optional

from ...terms import Atom, Variable, Compound
from ...knowledge import KnowledgeBase, Rule
from ..util import (_strip_llm_noise, _filter_cyclic_rules,
                    _collect_user_functors)


def build_op_g_prompt(kb: KnowledgeBase, max_prompt_facts: int) -> Optional[str]:
    """Build the Operation G prompt string from a knowledge base.

    Samples facts round-robin by predicate, computes predicate fact
    counts, and assembles the full prompt. Returns None when there are
    no facts to prompt with. Reads ``max_prompt_facts`` and ``kb``
    only; mutates nothing.
    """
    # Sample facts for the prompt, using Prolog notation
    # Sample facts ensuring every predicate is represented
    max_facts = max_prompt_facts
    facts_by_pred: dict = {}
    for fact in kb.facts:
        if isinstance(fact.term, Compound):
            facts_by_pred.setdefault(fact.term.functor, []).append(fact)
    sampled: list = []
    # Round-robin: take up to max_facts / n_preds from each
    per_pred = max(2, max_facts // max(len(facts_by_pred), 1))
    for fn in sorted(facts_by_pred):
        sampled.extend(facts_by_pred[fn][:per_pred])
    sampled = sampled[:max_facts]

    fact_lines = []
    for fact in sampled:
        term = fact.term
        if isinstance(term, Compound):
            args = ", ".join(
                str(a.value) if isinstance(a, Atom) else str(a)
                for a in term.args)
            fact_lines.append(f"{term.functor}({args}).")
        else:
            fact_lines.append(f"{term}.")
    if not fact_lines:
        return None

    # Compute predicate fact counts to guide directionality
    pred_counts: dict = {}
    for fact in kb.facts:
        if isinstance(fact.term, Compound):
            pred_counts[fact.term.functor] = (
                pred_counts.get(fact.term.functor, 0) + 1)
    count_lines = "\n".join(
        f"  {fn}: {cnt} facts" for fn, cnt in
        sorted(pred_counts.items(), key=lambda x: -x[1]))

    prompt = (
        "Given these facts from a knowledge base:\n\n"
        + "\n".join(fact_lines)
        + f"\n\nPredicate fact counts:\n{count_lines}\n\n"
        "Propose rules that derive SPECIFIC predicates from MORE GENERAL ones. "
        "A rule should EXPLAIN why a fact is true using simpler building blocks.\n\n"
        "IMPORTANT constraints:\n"
        "- Rules must go in ONE direction only: specific <- general.\n"
        "- NEVER propose reverse/inverse rules (if father <- parent+male, "
        "do NOT also propose parent <- father).\n"
        "- Body predicates should have MORE facts than the head predicate.\n"
        "- Each rule must derive at least 2 existing facts.\n"
        "- For 'all X satisfy P' patterns (e.g., vegan_recipe requires ALL "
        "ingredients to be vegan), use a helper predicate with not/1:\n"
        '  ["rule", ["has_non_vegan", "X"], [["uses", "X", "Y"], ["vegan", "Y", "false"]]]\n'
        '  ["rule", ["vegan_recipe", "X"], [["recipe", "X"], ["not", ["has_non_vegan", "X"]]]]\n\n'
        "Example format:\n"
        '  ["rule", ["father", "X", "Y"], [["parent", "X", "Y"], ["male", "X"]]]\n'
        '  For not/1: ["not", ["predicate", "X"]] as a body goal.\n\n'
        "If a relation appears to be the transitive closure of another "
        "relation (its facts are exactly the reachable pairs over a base "
        "relation), propose BOTH a base rule and a right-recursive rule, "
        "for example:\n"
        "  ancestor(X, Y) :- parent(X, Y).\n"
        "  ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).\n"
        "Always include the base case. Never write a left-recursive body "
        "(do not put the recursive call first).\n\n"
        "Reply with ONLY a JSON array of rules. "
        "No explanation, no markdown.\n\n"
        "Rules:"
    )
    return prompt


def parse_llm_rules(llm_client, prompt: str, parse_llm_response) -> list:
    """Send prompt to LLM and parse the response into raw rule data."""
    try:
        response = _strip_llm_noise(llm_client.complete(prompt))
        try:
            parsed_json = json.loads(response)
            if isinstance(parsed_json, list):
                return parsed_json
        except (json.JSONDecodeError, ValueError):
            pass
        # Try line-by-line extraction for partially valid JSON
        raw_rules = []
        for line in response.split("\n"):
            line = line.strip().rstrip(",")
            if line.startswith("[") and "rule" in line:
                try:
                    raw_rules.append(json.loads(line))
                except (json.JSONDecodeError, ValueError):
                    continue
        if raw_rules:
            return raw_rules
        # Fall back to the structured parser
        try:
            parsed_resp, _ = parse_llm_response(response)
            return parsed_resp.rules if parsed_resp else []
        except Exception:
            return []
    except Exception:
        return []


def build_rule_from_parsed(rule_data):
    """Build a Rule from parsed LLM output (raw list format)."""
    try:
        if not isinstance(rule_data, (list, tuple)) or len(rule_data) < 3:
            return None
        head_data = rule_data[1] if isinstance(rule_data[1], list) else rule_data
        body_data = rule_data[2] if len(rule_data) > 2 else []

        def make_term(data):
            if not isinstance(data, list) or len(data) == 0:
                return None
            functor = data[0]
            if not isinstance(functor, str) or len(functor) == 0:
                return None
            args = []
            for a in data[1:]:
                if isinstance(a, str) and len(a) > 0:
                    if a[0].isupper():
                        args.append(Variable(a))
                    else:
                        args.append(Atom(a))
                elif isinstance(a, list):
                    # Nested term (e.g., not(inner_goal))
                    inner = make_term(a)
                    if inner is None:
                        return None
                    args.append(inner)
                else:
                    return None
            return Compound(functor, args)

        head = make_term(head_data)
        if head is None:
            return None
        body = []
        for b in body_data:
            bt = make_term(b)
            if bt is None:
                return None
            body.append(bt)
        if not body:
            return None
        return Rule(head, body)
    except Exception:
        return None


def propose_rules(kb: KnowledgeBase, llm_client,
                  max_prompt_facts: int) -> List[Rule]:
    """Operation G, proposal stage: prompt + parse + validate.

    Builds the prompt, calls the LLM, parses the response, then applies
    Phase 1 structural validation and Phase 2 cyclic filtering. Returns
    the proposed, validated, cycle-filtered rules. Mutates nothing (reads
    ``kb`` only).
    """
    if llm_client is None:
        return []

    from ...llm_response_parser import parse_llm_response

    prompt = build_op_g_prompt(kb, max_prompt_facts)
    if prompt is None:
        return []

    raw_rules = parse_llm_rules(llm_client, prompt, parse_llm_response)
    if not raw_rules:
        return []

    # Collect user-defined (non-system) functors once for validation
    kb_functors = _collect_user_functors(kb)

    # Phase 1: Parse and structurally validate all proposed rules
    parsed_rules: List[Rule] = []
    for rule_data in raw_rules:
        try:
            if isinstance(rule_data, Rule):
                rule = rule_data
            else:
                rule = build_rule_from_parsed(rule_data)
                if rule is None:
                    continue
            if not isinstance(rule.head, Compound) or not rule.head.functor:
                continue
            if not rule.body:
                continue
            if any(not isinstance(g, Compound) or not g.functor
                   for g in rule.body):
                continue
            # Body functors must be in KB or be builtins (not, call).
            # Also allow functors that appear as heads of OTHER
            # proposed rules (helper predicates like has_non_vegan).
            _BUILTIN_FUNCTORS = {"not", "call"}
            if any(g.functor not in kb_functors
                   and g.functor not in _BUILTIN_FUNCTORS
                   for g in rule.body):
                continue
            # Reject unstratified negation: head functor inside not/1
            # (e.g., pet(X) :- ..., not(pet(X))) creates a fixed-point
            # paradox where the rule defines pet in terms of not(pet).
            head_fn = rule.head.functor
            if any(g.functor == "not" and g.arity == 1
                   and isinstance(g.args[0], Compound)
                   and g.args[0].functor == head_fn
                   for g in rule.body):
                continue
            parsed_rules.append(rule)
        except Exception:
            continue

    # Phase 2: Filter out rules that create cross-functor cycles
    # (e.g., parent<-father + father<-parent). Self-recursion is fine.
    parsed_rules = _filter_cyclic_rules(parsed_rules)

    return parsed_rules
