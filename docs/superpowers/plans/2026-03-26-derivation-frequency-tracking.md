# Derivation Frequency Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Track per-clause usage frequency during wake phase, prune dead clauses in the sleep cycle, and prioritize compression of heavily-used patterns.

**Architecture:** Usage counters on `KnowledgeBase` (dict mapping clause hash to count), incremented by the evaluator on each successful match. New Operation F prunes 0-usage clauses. Frequency scores used as tiebreaker in compression candidate selection.

**Tech Stack:** Python 3.8+, pytest, no new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-26-derivation-frequency-tracking-design.md`

**API note:** `compound("f", atom("a"))` is variadic. `Compound("f", [atom("a")])` takes a list.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `dreamlog/knowledge.py` | **Edit** | Add `_usage_counts`, `record_usage`, `get_usage`, `reset_usage`, `total_queries_tracked`; update `copy`, `restore_from` |
| `dreamlog/evaluator.py` | **Edit** | Add `record_usage` calls on fact/rule match |
| `dreamlog/kb_dreamer.py` | **Edit** | Add Operation F, `_frequency_score`, frequency-weighted selection |
| `tests/test_sleep_cycle.py` | **Edit** | Add usage counter, Operation F, and frequency tests |

---

### Task 1: Usage counters on KnowledgeBase

**Files:**
- Modify: `dreamlog/knowledge.py`
- Modify: `tests/test_sleep_cycle.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_sleep_cycle.py`:

```python
class TestUsageTracking:
    def test_record_and_get_usage(self):
        kb = KnowledgeBase()
        f = Fact(compound("a", atom("x")))
        kb.add_fact(f)
        assert kb.get_usage(f) == 0
        kb.record_usage(f)
        assert kb.get_usage(f) == 1
        kb.record_usage(f)
        assert kb.get_usage(f) == 2

    def test_get_usage_unknown_clause(self):
        kb = KnowledgeBase()
        f = Fact(compound("a", atom("x")))
        assert kb.get_usage(f) == 0

    def test_reset_usage(self):
        kb = KnowledgeBase()
        f = Fact(compound("a", atom("x")))
        kb.add_fact(f)
        kb.record_usage(f)
        kb.record_usage(f)
        assert kb.get_usage(f) == 2
        kb.reset_usage()
        assert kb.get_usage(f) == 0

    def test_total_queries_tracked(self):
        kb = KnowledgeBase()
        f1 = Fact(compound("a", atom("x")))
        f2 = Fact(compound("b", atom("y")))
        kb.add_fact(f1)
        kb.add_fact(f2)
        kb.record_usage(f1)
        kb.record_usage(f1)
        kb.record_usage(f2)
        assert kb.total_queries_tracked() == 3

    def test_copy_preserves_usage(self):
        kb = KnowledgeBase()
        f = Fact(compound("a", atom("x")))
        kb.add_fact(f)
        kb.record_usage(f)
        kb.record_usage(f)
        copy = kb.copy()
        assert copy.get_usage(f) == 2

    def test_restore_from_restores_usage(self):
        kb = KnowledgeBase()
        f = Fact(compound("a", atom("x")))
        kb.add_fact(f)
        kb.record_usage(f)
        snapshot = kb.copy()
        kb.record_usage(f)
        kb.record_usage(f)
        assert kb.get_usage(f) == 3
        kb.restore_from(snapshot)
        assert kb.get_usage(f) == 1

    def test_rule_usage(self):
        kb = KnowledgeBase()
        r = Rule(compound("a", var("X")), [compound("b", var("X"))])
        kb.add_rule(r)
        kb.record_usage(r)
        assert kb.get_usage(r) == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_sleep_cycle.py::TestUsageTracking -v`
Expected: FAIL

- [ ] **Step 3: Implement usage counter methods**

Add to `KnowledgeBase.__init__` (after the index dicts):
```python
        self._usage_counts: Dict[int, int] = {}
```

Add these methods to `KnowledgeBase`:
```python
    def record_usage(self, clause: Union[Fact, Rule]) -> None:
        """Increment usage counter for a clause."""
        key = hash(clause)
        self._usage_counts[key] = self._usage_counts.get(key, 0) + 1

    def get_usage(self, clause: Union[Fact, Rule]) -> int:
        """Get usage count for a clause (0 if never used)."""
        return self._usage_counts.get(hash(clause), 0)

    def reset_usage(self) -> None:
        """Clear all usage counters."""
        self._usage_counts.clear()

    def total_queries_tracked(self) -> int:
        """Total usage events recorded."""
        return sum(self._usage_counts.values())
```

Update `copy()` to preserve usage counts:
```python
    def copy(self) -> 'KnowledgeBase':
        """Deep copy for rollback."""
        new_kb = KnowledgeBase()
        for fact in self._facts:
            new_kb.add_fact(fact)
        for rule in self._rules:
            new_kb.add_rule(rule)
        new_kb._usage_counts = dict(self._usage_counts)
        return new_kb
```

Update `restore_from()` to restore usage counts:
```python
    def restore_from(self, other: 'KnowledgeBase') -> None:
        """Replace contents with another KB's contents."""
        self.clear()
        for fact in other._facts:
            self.add_fact(fact)
        for rule in other._rules:
            self.add_rule(rule)
        self._usage_counts = dict(other._usage_counts)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_sleep_cycle.py::TestUsageTracking -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add dreamlog/knowledge.py tests/test_sleep_cycle.py
git commit -m "Add usage counters to KnowledgeBase (record_usage, get_usage, reset_usage)"
```

---

### Task 2: Evaluator records usage on match

**Files:**
- Modify: `dreamlog/evaluator.py`
- Modify: `tests/test_sleep_cycle.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_sleep_cycle.py`:

```python
class TestEvaluatorUsageRecording:
    def test_fact_usage_recorded(self):
        kb = KnowledgeBase()
        kb.add_fact(compound("parent", atom("john"), atom("mary")))
        ev = PrologEvaluator(kb)
        list(ev.query([compound("parent", atom("john"), atom("mary"))]))
        f = kb.facts[0]
        assert kb.get_usage(f) >= 1

    def test_rule_usage_recorded(self):
        kb = KnowledgeBase()
        kb.add_fact(compound("parent", atom("john"), atom("mary")))
        r = Rule(compound("anc", var("X"), var("Y")),
                 [compound("parent", var("X"), var("Y"))])
        kb.add_rule(r)
        ev = PrologEvaluator(kb)
        list(ev.query([compound("anc", atom("john"), atom("mary"))]))
        assert kb.get_usage(r) >= 1

    def test_usage_accumulates_across_queries(self):
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        ev = PrologEvaluator(kb)
        list(ev.query([compound("a", atom("x"))]))
        list(ev.query([compound("a", atom("x"))]))
        list(ev.query([compound("a", atom("x"))]))
        assert kb.get_usage(kb.facts[0]) >= 3

    def test_unused_fact_has_zero_usage(self):
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        kb.add_fact(compound("b", atom("y")))
        ev = PrologEvaluator(kb)
        list(ev.query([compound("a", atom("x"))]))
        assert kb.get_usage(kb.facts[0]) >= 1  # a(x) used
        assert kb.get_usage(kb.facts[1]) == 0   # b(y) never queried
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_sleep_cycle.py::TestEvaluatorUsageRecording -v`
Expected: FAIL

- [ ] **Step 3: Add record_usage calls to evaluator**

In `dreamlog/evaluator.py`, in `_solve_goals`:

After line 199 (`solutions_found = True` in the fact match block), add:
```python
                        self.kb.record_usage(fact)
```

After line 217 (`solutions_found = True` in the rule match block), add:
```python
                        self.kb.record_usage(rule)
```

Note: we record the ORIGINAL fact/rule (not the renamed copy), because that's what's stored in the KB and what the dreamer will look up.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_sleep_cycle.py::TestEvaluatorUsageRecording -v`
Expected: All PASS

- [ ] **Step 5: Run full suite**

Run: `python -m pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add dreamlog/evaluator.py tests/test_sleep_cycle.py
git commit -m "Evaluator records usage on fact and rule match"
```

---

### Task 3: Operation F (dead clause pruning) and frequency-weighted prioritization

**Files:**
- Modify: `dreamlog/kb_dreamer.py`
- Modify: `tests/test_sleep_cycle.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_sleep_cycle.py`:

```python
class TestOperationF:
    def test_dead_fact_removed(self):
        """Fact with 0 usage after queries is pruned."""
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        kb.add_fact(compound("b", atom("y")))  # never queried
        # Simulate wake phase: query a(x) enough times
        from dreamlog.evaluator import PrologEvaluator
        ev = PrologEvaluator(kb)
        for _ in range(15):
            list(ev.query([compound("a", atom("x"))]))
        # b(y) has 0 usage, a(x) has 15
        assert kb.get_usage(kb.facts[1]) == 0
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=False)
        remaining = [f.term for f in kb.facts]
        assert compound("a", atom("x")) in remaining
        assert compound("b", atom("y")) not in remaining

    def test_dead_rule_removed(self):
        """Rule with 0 usage after queries is pruned."""
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        r_used = Rule(compound("b", var("X")), [compound("a", var("X"))])
        r_dead = Rule(compound("c", var("X")), [compound("d", var("X"))])
        kb.add_rule(r_used)
        kb.add_rule(r_dead)
        from dreamlog.evaluator import PrologEvaluator
        ev = PrologEvaluator(kb)
        for _ in range(15):
            list(ev.query([compound("b", atom("x"))]))
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=False)
        assert len(kb.rules) == 1  # dead rule removed

    def test_used_clauses_preserved(self):
        """Clauses with usage > 0 are kept."""
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        kb.add_fact(compound("b", atom("y")))
        from dreamlog.evaluator import PrologEvaluator
        ev = PrologEvaluator(kb)
        for _ in range(10):
            list(ev.query([compound("a", atom("x"))]))
            list(ev.query([compound("b", atom("y"))]))
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=False)
        assert len(kb.facts) == 2  # both used, both kept

    def test_threshold_prevents_premature_pruning(self):
        """Not enough queries -> skip dead clause pruning."""
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        kb.add_fact(compound("b", atom("y")))
        from dreamlog.evaluator import PrologEvaluator
        ev = PrologEvaluator(kb)
        list(ev.query([compound("a", atom("x"))]))  # only 1 query
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=False)
        assert len(kb.facts) == 2  # not enough data, keep both

    def test_generated_predicates_not_pruned(self):
        """_invented_, _extracted_, exception_ predicates skipped even if 0 usage."""
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        # Simulate generated predicates
        from dreamlog.terms import Compound, Atom, Variable
        kb.add_rule(Rule(Compound("_invented_0", [Variable("R"), Variable("X")]),
                         [Compound("call", [Variable("R"), Variable("X")])]))
        kb.add_fact(Fact(Compound("exception_likes_person", [Atom("dave")])))
        from dreamlog.evaluator import PrologEvaluator
        ev = PrologEvaluator(kb)
        for _ in range(15):
            list(ev.query([compound("a", atom("x"))]))
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=False)
        # Generated predicates should survive even with 0 usage
        inv_rules = [r for r in kb.rules
                     if isinstance(r.head, Compound)
                     and r.head.functor.startswith("_invented_")]
        assert len(inv_rules) == 1
        exc_facts = [f for f in kb.facts
                     if isinstance(f.term, Compound)
                     and f.term.functor.startswith("exception_")]
        assert len(exc_facts) == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_sleep_cycle.py::TestOperationF -v`
Expected: FAIL

- [ ] **Step 3: Implement Operation F and frequency scoring**

Add to `KnowledgeBaseDreamer` in `dreamlog/kb_dreamer.py`:

```python
    def _prune_dead_clauses(self, kb: KnowledgeBase,
                            min_query_threshold: int = 10
                            ) -> List[CompressionCandidate]:
        """Operation F: Remove clauses with 0 usage after sufficient queries."""
        ops = []

        if kb.total_queries_tracked() < min_query_threshold:
            return ops

        # Find dead facts
        dead_facts = []
        for fact in kb.facts:
            if isinstance(fact.term, Compound):
                f = fact.term.functor
                if f.startswith("exception_") or f.startswith("_extracted_"):
                    continue
            if kb.get_usage(fact) == 0:
                dead_facts.append(fact)

        # Find dead rules
        dead_rules = []
        for rule in kb.rules:
            if isinstance(rule.head, Compound):
                f = rule.head.functor
                if (f.startswith("_invented_") or f.startswith("_extracted_")
                        or f.startswith("exception_")):
                    continue
            if kb.get_usage(rule) == 0:
                dead_rules.append(rule)

        # Remove dead facts
        for fact in dead_facts:
            kb.remove_fact_by_value(fact)
            ops.append(CompressionCandidate(
                operation="dead_clause", original_clauses=[fact]))

        # Remove dead rules
        for rule in dead_rules:
            kb.remove_rule_by_value(rule)
            ops.append(CompressionCandidate(
                operation="dead_clause", original_clauses=[rule]))

        return ops

    def _frequency_score(self, kb: KnowledgeBase,
                         clauses: List[Union[Fact, Rule]]) -> float:
        """Compute frequency-weighted score for a set of clauses."""
        import math
        total = sum(kb.get_usage(c) for c in clauses)
        return 1.0 + math.log2(total + 1)
```

Wire Operation F into `dream()`. After the line `ops.extend(self._extract_body_patterns(kb, suite))`, add:

```python
        ops.extend(self._prune_dead_clauses(kb))
```

Also update the help text in the dream handler. In the TUI `_cmd_dream`, update the operation label dict to include:
```python
"dead_clause": "Dead clause pruned",
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_sleep_cycle.py::TestOperationF -v`
Expected: All PASS

- [ ] **Step 5: Run full suite**

Run: `python -m pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 6: End-to-end smoke test**

```python
from dreamlog.knowledge import KnowledgeBase, Fact, Rule
from dreamlog.factories import atom, var, compound
from dreamlog.evaluator import PrologEvaluator
from dreamlog.kb_dreamer import KnowledgeBaseDreamer

kb = KnowledgeBase()
kb.add_fact(compound("parent", atom("john"), atom("mary")))
kb.add_fact(compound("parent", atom("mary"), atom("alice")))
kb.add_fact(compound("stale", atom("unused")))  # never queried
kb.add_rule(Rule(compound("anc", var("X"), var("Y")),
                 [compound("parent", var("X"), var("Y"))]))
kb.add_rule(Rule(compound("anc", var("X"), var("Z")),
                 [compound("parent", var("X"), var("Y")),
                  compound("anc", var("Y"), var("Z"))]))
kb.add_rule(Rule(compound("dead_rule", var("X")),
                 [compound("nonexistent", var("X"))]))  # never fires

# Wake phase: run queries
ev = PrologEvaluator(kb)
for _ in range(20):
    list(ev.query([compound("anc", atom("john"), atom("alice"))]))

print(f"Before dream: {len(kb)} clauses")
print(f"Usage: stale={kb.get_usage(kb.facts[2])}, dead_rule={kb.get_usage(kb.rules[2])}")

dreamer = KnowledgeBaseDreamer()
session = dreamer.dream(kb, verify=False)

print(f"After dream: {len(kb)} clauses")
print(f"Operations: {[(op.operation, op.mdl_delta) for op in session.operations]}")
for f in kb.facts:
    print(f"  fact: {f.term} (usage: {kb.get_usage(f)})")
for r in kb.rules:
    print(f"  rule: {r} (usage: {kb.get_usage(r)})")
```

Expected: `stale(unused)` and `dead_rule` removed (0 usage). Parent facts and ancestor rules kept (used 20+ times).

- [ ] **Step 7: Commit**

```bash
git add dreamlog/kb_dreamer.py dreamlog/tui.py tests/test_sleep_cycle.py
git commit -m "Add Operation F (dead clause pruning) and frequency scoring"
```
