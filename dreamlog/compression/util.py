"""Shared helpers for the compression package, moved from kb_dreamer."""
import re
from typing import Dict, List, Set

from ..knowledge import KnowledgeBase, Rule
from ..terms import Compound

# Prefixes used by system-generated predicates.
_SYSTEM_PREFIXES = ("_invented_", "_extracted_", "exception_")


def _is_system_predicate(functor: str) -> bool:
    """Check if a functor name is system-generated."""
    return any(functor.startswith(p) for p in _SYSTEM_PREFIXES)


def _next_generated_name(kb: KnowledgeBase, prefix: str) -> str:
    """Find the next available name with the given prefix (e.g. '_invented_')."""
    max_n = -1
    for rule in kb.rules:
        if isinstance(rule.head, Compound) and rule.head.functor.startswith(prefix):
            try:
                n = int(rule.head.functor[len(prefix):])
                max_n = max(max_n, n)
            except (ValueError, IndexError):
                pass
    return f"{prefix}{max_n + 1}"


def _strip_llm_noise(text: str) -> str:
    """Remove thinking tags and markdown code fences from LLM output."""
    text = text.strip()
    if "<think>" in text:
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines)
    return text


def _filter_cyclic_rules(rules: List[Rule]) -> List[Rule]:
    """Remove proposed rules that would create cycles in the functor dependency graph.

    Self-recursive rules (head functor in own body) are allowed since the
    evaluator handles those via depth limits. Only cross-functor cycles
    (A<-B and B<-A) are rejected, as they create combinatorial explosion.
    """
    from collections import defaultdict
    graph: Dict[str, Set[str]] = defaultdict(set)
    result = []

    for rule in rules:
        head_fn = rule.head.functor
        body_fns = {g.functor for g in rule.body
                    if isinstance(g, Compound)} - {head_fn}

        old_edges = graph[head_fn].copy()
        graph[head_fn] |= body_fns

        # DFS cycle check
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = defaultdict(int)
        has_cycle = False

        def dfs(node):
            nonlocal has_cycle
            color[node] = GRAY
            for nbr in graph.get(node, set()):
                if color[nbr] == GRAY:
                    has_cycle = True
                    return
                if color[nbr] == WHITE:
                    dfs(nbr)
                    if has_cycle:
                        return
            color[node] = BLACK

        for node in graph:
            if color[node] == WHITE:
                dfs(node)
                if has_cycle:
                    break

        if has_cycle:
            graph[head_fn] = old_edges  # rollback
        else:
            result.append(rule)

    return result


def _collect_user_functors(kb: KnowledgeBase) -> Set[str]:
    """Collect functor names of user-defined (non-system) predicates."""
    functors: Set[str] = set()
    for fact in kb.facts:
        if isinstance(fact.term, Compound) and not _is_system_predicate(fact.term.functor):
            functors.add(fact.term.functor)
    for rule in kb.rules:
        if isinstance(rule.head, Compound) and not _is_system_predicate(rule.head.functor):
            functors.add(rule.head.functor)
    return functors
