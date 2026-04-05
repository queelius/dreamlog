"""
Persistent, self-compressing knowledge store for agentic use.

Wraps DreamLogEngine with:
- Disk persistence (atomic writes)
- LLM budget tracking
- Dream-readiness advisory
- Session metadata (assertion/query counts, dream history)
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dreamlog.knowledge import KnowledgeBase, Fact, Rule
from dreamlog.terms import Compound
from dreamlog.factories import atom, var, compound
from dreamlog.prefix_parser import parse_s_expression
from dreamlog.evaluator import PrologEvaluator
from dreamlog.kb_dreamer import KnowledgeBaseDreamer
from dreamlog.llm_client import LLMClient, LLMUsage


class KnowledgeStore:
    """Persistent knowledge base with compression and LLM budget tracking."""

    def __init__(self, store_path: Path,
                 llm_budget_usd: float = 0.50,
                 dream_threshold: int = 50):
        self.store_path = Path(store_path).expanduser()
        self.llm_budget_usd = llm_budget_usd
        self.dream_threshold = dream_threshold

        self.kb = KnowledgeBase()
        self.kb.enable_derivation_tracking()
        self.llm_client: Optional[LLMClient] = None

        # Session counters
        self._facts_since_dream = 0
        self._queries_since_dream = 0
        self._dream_count = 0
        self._total_assertions = 0
        self._total_queries = 0
        self._llm_usage = LLMUsage()
        self._dirty = False

        self._init_llm()
        self._load()

    def _init_llm(self):
        """Initialize LLM client if API key is available."""
        for env_var in ("MY_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"):
            if os.getenv(env_var):
                self.llm_client = LLMClient(
                    provider="anthropic",
                    api_key_env=env_var,
                    temperature=0.3,
                    max_tokens=800,
                )
                break

    # ── Persistence ───────────────────────────────────────────────

    def _load(self):
        if not self.store_path.exists():
            return
        try:
            data = json.loads(self.store_path.read_text())
            if isinstance(data, dict) and "kb" in data:
                # Envelope format
                self.kb.from_prefix(json.dumps(data["kb"]))
                meta = data.get("metadata", {})
                self._dream_count = meta.get("dream_count", 0)
                self._total_assertions = meta.get("total_assertions", 0)
                self._total_queries = meta.get("total_queries", 0)
                usage = meta.get("llm_usage", {})
                self._llm_usage = LLMUsage(
                    calls=usage.get("calls", 0),
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                )
                if self.llm_client:
                    self.llm_client.usage = self._llm_usage
            elif isinstance(data, list):
                # Legacy raw prefix format
                self.kb.from_prefix(json.dumps(data))
        except (json.JSONDecodeError, ValueError, KeyError):
            pass  # Start fresh on corrupt data

    def save(self):
        if not self._dirty:
            return
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        usage = self.llm_client.usage if self.llm_client else self._llm_usage
        envelope = {
            "version": 1,
            "kb": json.loads(self.kb.to_prefix()),
            "metadata": {
                "dream_count": self._dream_count,
                "total_assertions": self._total_assertions,
                "total_queries": self._total_queries,
                "facts_since_dream": self._facts_since_dream,
                "queries_since_dream": self._queries_since_dream,
                "llm_usage": {
                    "calls": usage.calls,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                },
            },
        }
        tmp = self.store_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(envelope, indent=2))
        tmp.rename(self.store_path)
        self._dirty = False

    # ── Core operations ───────────────────────────────────────────

    def assert_fact(self, sexpr: str) -> Dict[str, Any]:
        """Add a fact or rule from S-expression string."""
        if ":-" in sexpr:
            parts = sexpr.split(":-")
            head = parse_s_expression(parts[0].strip())
            body = [parse_s_expression(b.strip())
                    for b in parts[1].split(",")]
            rule = Rule(head, body)
            self.kb.add_rule(rule)
            clause_str = str(rule)
        else:
            term = parse_s_expression(sexpr)
            self.kb.add_fact(Fact(term))
            clause_str = str(term)

        self._facts_since_dream += 1
        self._total_assertions += 1
        self._dirty = True
        self.save()

        return {
            "added": clause_str,
            "kb_size": len(self.kb),
            "dream_recommended": self.should_dream(),
        }

    def query(self, sexpr: str, limit: int = 10) -> Dict[str, Any]:
        """Query the KB. Returns bindings for variables."""
        term = parse_s_expression(sexpr)
        ev = PrologEvaluator(self.kb)
        solutions = []
        for sol in ev.query([term]):
            bindings = {}
            for vname, val in sol.get_ground_bindings().items():
                bindings[vname] = str(val)
            solutions.append(bindings)
            if len(solutions) >= limit:
                break

        self._queries_since_dream += 1
        self._total_queries += 1

        return {
            "query": sexpr,
            "solutions": solutions,
            "count": len(solutions),
            "has_more": len(solutions) == limit,
            "dream_recommended": self.should_dream(),
        }

    def dream(self, dry_run: bool = False) -> Dict[str, Any]:
        """Run the sleep/compression cycle."""
        before = len(self.kb)
        before_facts = len(self.kb.facts)
        before_rules = len(self.kb.rules)

        use_llm = (self.llm_client is not None
                    and self.llm_client.estimated_cost() < self.llm_budget_usd)
        dreamer = KnowledgeBaseDreamer(
            llm_client=self.llm_client if use_llm else None)

        if dry_run:
            kb_copy = self.kb.copy()
            session = dreamer.dream(kb_copy, verify=True)
            after = len(kb_copy)
        else:
            session = dreamer.dream(self.kb, verify=True)
            after = len(self.kb)
            self._facts_since_dream = 0
            self._queries_since_dream = 0
            self._dream_count += 1
            self._dirty = True
            self.save()

        ops_summary = {}
        new_rules = []
        for op in session.operations:
            ops_summary[op.operation] = ops_summary.get(op.operation, 0) + 1
            for c in op.new_clauses:
                if isinstance(c, Rule):
                    new_rules.append(str(c))

        return {
            "compressed": session.compressed,
            "before": before,
            "after": after,
            "removed": before - after,
            "ratio": session.compression_ratio,
            "operations": ops_summary,
            "new_rules": new_rules,
            "dry_run": dry_run,
            "llm_used": use_llm,
        }

    def explain(self, sexpr: str) -> Dict[str, Any]:
        """Explain how a query resolves."""
        term = parse_s_expression(sexpr)
        ev = PrologEvaluator(self.kb)
        derivable = ev.has_solution(term)

        # Find matching facts
        matching_facts = []
        if isinstance(term, Compound):
            for fact in self.kb.get_matching_facts(term):
                matching_facts.append(str(fact.term))

        # Find matching rules
        matching_rules = []
        if isinstance(term, Compound):
            for rule in self.kb.get_matching_rules(term):
                matching_rules.append(str(rule))

        return {
            "query": sexpr,
            "derivable": derivable,
            "matching_facts": matching_facts,
            "matching_rules": matching_rules,
        }

    def status(self) -> Dict[str, Any]:
        """Current store status."""
        functors = set()
        for f in self.kb.facts:
            if isinstance(f.term, Compound):
                functors.add(f.term.functor)

        usage = self.llm_client.usage if self.llm_client else self._llm_usage
        budget = self.budget_remaining()

        return {
            "facts": len(self.kb.facts),
            "rules": len(self.kb.rules),
            "total_clauses": len(self.kb),
            "functors": sorted(functors),
            "dreams": self._dream_count,
            "total_assertions": self._total_assertions,
            "total_queries": self._total_queries,
            "facts_since_dream": self._facts_since_dream,
            "queries_since_dream": self._queries_since_dream,
            "dream_recommended": self.should_dream(),
            "llm": budget,
            "store_path": str(self.store_path),
        }

    # ── Helpers ───────────────────────────────────────────────────

    def should_dream(self) -> bool:
        return (self._facts_since_dream + self._queries_since_dream
                >= self.dream_threshold)

    def budget_remaining(self) -> Dict[str, Any]:
        if not self.llm_client:
            return {"enabled": False}
        spent = self.llm_client.estimated_cost()
        return {
            "enabled": True,
            "budget_usd": self.llm_budget_usd,
            "spent_usd": round(spent, 4),
            "remaining_usd": round(max(0, self.llm_budget_usd - spent), 4),
            "calls": self.llm_client.usage.calls,
            "over_budget": spent >= self.llm_budget_usd,
        }
