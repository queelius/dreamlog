# Rule Body Pattern Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Operation E to the sleep cycle: extract common contiguous sub-goal sequences from rule bodies as named predicates.

**Architecture:** Single method `_extract_body_patterns` added to `KnowledgeBaseDreamer`. Indexes all contiguous sub-sequences from rule bodies, groups by structural key, selects best candidate by MDL, computes interface variables, builds extracted predicate, rewrites affected rules, verifies, and re-scans for cascading patterns.

**Tech Stack:** Python 3.8+, pytest, no new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-25-body-pattern-extraction-design.md`

**API note:** `compound("f", atom("a"))` is variadic. `Compound("f", [atom("a")])` takes a list. `Rule(head, [body_goals])`.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `dreamlog/kb_dreamer.py` | **Edit** | Add `_extract_body_patterns`, `_next_extracted_name`, wire into `dream()` |
| `tests/test_sleep_cycle.py` | **Edit** | Add `TestOperationE` class |

---

### Task 1: Operation E implementation and tests

**Files:**
- Modify: `dreamlog/kb_dreamer.py`
- Modify: `tests/test_sleep_cycle.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_sleep_cycle.py`:

```python
class TestOperationE:
    def test_three_rules_shared_prefix(self):
        """3 rules sharing a 2-goal prefix: extracted (K=2, N=3, savings=1)."""
        kb = KnowledgeBase()
        # grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
        kb.add_rule(Rule(compound("grandparent", var("X"), var("Z")),
                         [compound("parent", var("X"), var("Y")),
                          compound("parent", var("Y"), var("Z"))]))
        # great_gp(X, W) :- parent(X, Y), parent(Y, Z), parent(Z, W).
        kb.add_rule(Rule(compound("great_gp", var("X"), var("W")),
                         [compound("parent", var("X"), var("Y")),
                          compound("parent", var("Y"), var("Z")),
                          compound("parent", var("Z"), var("W"))]))
        # great_uncle(X, W) :- parent(X, Y), parent(Y, Z), brother(Z, W).
        kb.add_rule(Rule(compound("great_uncle", var("X"), var("W")),
                         [compound("parent", var("X"), var("Y")),
                          compound("parent", var("Y"), var("Z")),
                          compound("brother", var("Z"), var("W"))]))
        # Add facts for verification
        kb.add_fact(compound("parent", atom("john"), atom("mary")))
        kb.add_fact(compound("parent", atom("mary"), atom("alice")))
        kb.add_fact(compound("parent", atom("alice"), atom("bob")))
        kb.add_fact(compound("brother", atom("alice"), atom("charlie")))

        dreamer = KnowledgeBaseDreamer()
        session = dreamer.dream(kb, verify=True)

        # Should have extracted the common prefix
        extracted_rules = [r for r in kb.rules
                           if isinstance(r.head, Compound)
                           and r.head.functor.startswith("_extracted_")]
        assert len(extracted_rules) >= 1

        # Verify queries still work
        from dreamlog.evaluator import PrologEvaluator
        ev = PrologEvaluator(kb)
        assert ev.has_solution(compound("grandparent", atom("john"), atom("alice")))
        assert ev.has_solution(compound("great_gp", atom("john"), atom("bob")))
        assert ev.has_solution(compound("great_uncle", atom("john"), atom("charlie")))

    def test_two_rules_k3_extracted(self):
        """2 rules sharing a 3-goal sub-sequence: extracted (K=3, N=2, savings=1)."""
        kb = KnowledgeBase()
        kb.add_rule(Rule(compound("f", var("X"), var("W")),
                         [compound("a", var("X"), var("Y")),
                          compound("b", var("Y"), var("Z")),
                          compound("c", var("Z"), var("W"))]))
        kb.add_rule(Rule(compound("g", var("X"), var("W")),
                         [compound("a", var("X"), var("Y")),
                          compound("b", var("Y"), var("Z")),
                          compound("c", var("Z"), var("W"))]))
        dreamer = KnowledgeBaseDreamer()
        session = dreamer.dream(kb, verify=False)
        extracted = [r for r in kb.rules
                     if isinstance(r.head, Compound)
                     and r.head.functor.startswith("_extracted_")]
        assert len(extracted) >= 1

    def test_two_rules_k2_rejected(self):
        """2 rules sharing a 2-goal sub-sequence: rejected (K=2, N=2, savings=0)."""
        kb = KnowledgeBase()
        kb.add_rule(Rule(compound("f", var("X"), var("Z")),
                         [compound("a", var("X"), var("Y")),
                          compound("b", var("Y"), var("Z"))]))
        kb.add_rule(Rule(compound("g", var("X"), var("Z")),
                         [compound("a", var("X"), var("Y")),
                          compound("b", var("Y"), var("Z"))]))
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=False)
        extracted = [r for r in kb.rules
                     if isinstance(r.head, Compound)
                     and r.head.functor.startswith("_extracted_")]
        assert len(extracted) == 0

    def test_interface_variables(self):
        """Extracted predicate exposes only interface vars, hides internal ones."""
        kb = KnowledgeBase()
        # f(X, W) :- a(X, Y), b(Y, Z), c(Z, W) -- Y and Z are internal to prefix a,b
        # g(X, W) :- a(X, Y), b(Y, Z), d(Z, W)
        # h(X, W) :- a(X, Y), b(Y, Z), e(Z, W)
        for head, tail_f in [("f", "c"), ("g", "d"), ("h", "e")]:
            kb.add_rule(Rule(compound(head, var("X"), var("W")),
                             [compound("a", var("X"), var("Y")),
                              compound("b", var("Y"), var("Z")),
                              compound(tail_f, var("Z"), var("W"))]))
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=False)
        extracted = [r for r in kb.rules
                     if isinstance(r.head, Compound)
                     and r.head.functor.startswith("_extracted_")]
        assert len(extracted) == 1
        # Interface should be (X, Z) -- X from head, Z connects to remaining body
        # Y is internal to the sub-sequence
        assert extracted[0].head.arity == 2

    def test_subsequence_at_end(self):
        """Sub-sequence at end of body (not just prefix) is detected."""
        kb = KnowledgeBase()
        # f(X, W) :- start(X, Y), common(Y, Z), common2(Z, W)
        # g(X, W) :- other(X, Y), common(Y, Z), common2(Z, W)
        # h(X, W) :- third(X, Y), common(Y, Z), common2(Z, W)
        for head, start_f in [("f", "start"), ("g", "other"), ("h", "third")]:
            kb.add_rule(Rule(compound(head, var("X"), var("W")),
                             [compound(start_f, var("X"), var("Y")),
                              compound("common", var("Y"), var("Z")),
                              compound("common2", var("Z"), var("W"))]))
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=False)
        extracted = [r for r in kb.rules
                     if isinstance(r.head, Compound)
                     and r.head.functor.startswith("_extracted_")]
        assert len(extracted) >= 1

    def test_generated_predicates_excluded(self):
        """_extracted_, _invented_, exception_ predicates are skipped."""
        kb = KnowledgeBase()
        for name in ["alice", "bob", "carol", "dave"]:
            kb.add_fact(compound("person", atom(name)))
        for name in ["alice", "bob", "carol"]:
            kb.add_fact(compound("likes", atom(name), atom("chocolate")))
        dreamer = KnowledgeBaseDreamer(min_group_size=3)
        dreamer.dream(kb, verify=True)
        size_after = len(kb)
        dreamer.dream(kb, verify=True)
        assert len(kb) == size_after

    def test_idempotent(self):
        kb = KnowledgeBase()
        kb.add_rule(Rule(compound("grandparent", var("X"), var("Z")),
                         [compound("parent", var("X"), var("Y")),
                          compound("parent", var("Y"), var("Z"))]))
        kb.add_rule(Rule(compound("great_gp", var("X"), var("W")),
                         [compound("parent", var("X"), var("Y")),
                          compound("parent", var("Y"), var("Z")),
                          compound("parent", var("Z"), var("W"))]))
        kb.add_rule(Rule(compound("great_uncle", var("X"), var("W")),
                         [compound("parent", var("X"), var("Y")),
                          compound("parent", var("Y"), var("Z")),
                          compound("brother", var("Z"), var("W"))]))
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=False)
        size_first = len(kb)
        dreamer.dream(kb, verify=False)
        assert len(kb) == size_first
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_sleep_cycle.py::TestOperationE -v`
Expected: FAIL

- [ ] **Step 3: Implement Operation E**

Add to `KnowledgeBaseDreamer` class in `dreamlog/kb_dreamer.py`:

```python
    def _extract_body_patterns(self, kb: KnowledgeBase,
                               suite: Optional['VerificationSuite'] = None,
                               max_rounds: int = 10
                               ) -> List[CompressionCandidate]:
        """Operation E: Extract common contiguous sub-goal sequences from rule bodies."""
        from .evaluator import PrologEvaluator

        all_ops = []

        for _round in range(max_rounds):
            candidate = self._find_best_body_pattern(kb)
            if candidate is None:
                break

            subseq, occurrences = candidate

            # Compute interface variables for each occurrence
            extracted_name = self._next_extracted_name(kb)
            subseq_len = len(subseq)

            # Use first occurrence to determine interface vars
            # (all occurrences have same structure, so same interface)
            first_rule, first_start = occurrences[0]
            interface_vars = self._compute_interface_vars(
                first_rule, first_start, subseq_len)

            # Build extracted predicate
            # Use normalized variable names from the sub-sequence
            template_rule = first_rule
            template_body = list(template_rule.body)
            subseq_goals = template_body[first_start:first_start + subseq_len]

            # Map interface vars to positional names for the extracted predicate head
            extracted_head = Compound(extracted_name,
                                     [Variable(v) for v in interface_vars])
            extracted_rule = Rule(extracted_head, subseq_goals)

            # Verify before applying
            if suite is not None:
                test_kb = kb.copy()
                test_kb.add_rule(extracted_rule)
                for rule, start in occurrences:
                    body = list(rule.body)
                    # Map this occurrence's vars to the extracted predicate's interface
                    call_args = self._map_interface_vars(
                        rule, start, subseq_len, interface_vars, first_rule, first_start)
                    new_body = (body[:start]
                                + [Compound(extracted_name, call_args)]
                                + body[start + subseq_len:])
                    new_rule = Rule(rule.head, new_body)
                    test_kb.remove_rule_by_value(rule)
                    test_kb.add_rule(new_rule)
                result = suite.verify(test_kb, lambda k: PrologEvaluator(k))
                if not result.passed:
                    break

            # Apply
            kb.add_rule(extracted_rule)
            original_rules = []
            new_rules = []
            for rule, start in occurrences:
                body = list(rule.body)
                call_args = self._map_interface_vars(
                    rule, start, subseq_len, interface_vars, first_rule, first_start)
                new_body = (body[:start]
                            + [Compound(extracted_name, call_args)]
                            + body[start + subseq_len:])
                new_rule = Rule(rule.head, new_body)
                kb.remove_rule_by_value(rule)
                kb.add_rule(new_rule)
                original_rules.append(rule)
                new_rules.append(new_rule)

            all_ops.append(CompressionCandidate(
                operation="extraction",
                original_clauses=original_rules,
                new_clauses=[extracted_rule] + new_rules))

        return all_ops

    def _find_best_body_pattern(self, kb: KnowledgeBase):
        """Find the best common contiguous sub-sequence across rule bodies."""
        # Collect rules to scan (skip generated predicates)
        rules = []
        for rule in kb.rules:
            if isinstance(rule.head, Compound):
                f = rule.head.functor
                if (f.startswith("_extracted_") or f.startswith("_invented_")
                        or f.startswith("exception_")):
                    continue
            if len(rule.body) >= 2:
                rules.append(rule)

        if not rules:
            return None

        # Index all sub-sequences by structural key
        subseq_index: Dict[tuple, list] = {}
        for rule in rules:
            body = list(rule.body)
            for length in range(2, len(body) + 1):
                for start in range(len(body) - length + 1):
                    subseq = body[start:start + length]
                    key = self._subseq_structural_key(subseq)
                    subseq_index.setdefault(key, []).append((rule, start))

        # Find best candidate: highest savings
        best = None
        best_savings = 0
        for key, occurrences in subseq_index.items():
            # Deduplicate: one occurrence per rule
            seen_rules = set()
            unique_occs = []
            for rule, start in occurrences:
                rule_id = id(rule)
                if rule_id not in seen_rules:
                    seen_rules.add(rule_id)
                    unique_occs.append((rule, start))

            n = len(unique_occs)
            k = key[0]  # length stored as first element of key
            if n < 2 or k < 2:
                continue
            savings = (k - 1) * (n - 1) - 1
            if savings > best_savings:
                best_savings = savings
                best = (key, unique_occs)

        if best is None:
            return None

        key, occurrences = best
        # Recover the actual sub-sequence goals from first occurrence
        first_rule, first_start = occurrences[0]
        subseq_len = key[0]
        subseq = list(first_rule.body)[first_start:first_start + subseq_len]
        return (subseq, occurrences)

    def _subseq_structural_key(self, subseq: list) -> tuple:
        """Compute structural key for a contiguous sub-sequence of body goals.

        Captures: length, functor names, arities, and normalized variable connectivity.
        """
        length = len(subseq)
        functors = []
        for goal in subseq:
            if isinstance(goal, Compound):
                functors.append((goal.functor, goal.arity))
            else:
                functors.append((str(goal), 0))

        # Normalize variables within the sub-sequence
        var_order = []
        for goal in subseq:
            for name in self._iter_var_names(goal):
                if name not in var_order:
                    var_order.append(name)

        var_rename = {old: f"_S{i}" for i, old in enumerate(var_order)}

        # Build variable connectivity
        var_positions: Dict[str, list] = {}
        for g_idx, goal in enumerate(subseq):
            if isinstance(goal, Compound):
                for a_idx, arg in enumerate(goal.args):
                    if isinstance(arg, Variable):
                        renamed = var_rename.get(arg.name, arg.name)
                        var_positions.setdefault(renamed, []).append((g_idx, a_idx))

        sorted_vars = sorted(var_positions.keys())
        var_map = tuple(tuple(sorted(var_positions[v])) for v in sorted_vars)

        return (length, tuple(functors), var_map)

    def _iter_var_names(self, term: Term):
        """Yield variable names left-to-right in a term."""
        if isinstance(term, Variable):
            yield term.name
        elif isinstance(term, Compound):
            for arg in term.args:
                yield from self._iter_var_names(arg)

    def _compute_interface_vars(self, rule: Rule, start: int,
                                length: int) -> List[str]:
        """Compute interface variables for a sub-sequence extraction.

        Interface vars = variables appearing in both the sub-sequence AND
        the rest of the rule (head + remaining body goals).
        """
        body = list(rule.body)
        subseq_goals = body[start:start + length]
        rest_goals = [rule.head] + body[:start] + body[start + length:]

        subseq_vars = set()
        for goal in subseq_goals:
            subseq_vars.update(goal.get_variables())

        rest_vars = set()
        for goal in rest_goals:
            rest_vars.update(goal.get_variables())

        interface = subseq_vars & rest_vars

        # Order by first appearance in sub-sequence
        ordered = []
        for goal in subseq_goals:
            for name in self._iter_var_names(goal):
                if name in interface and name not in ordered:
                    ordered.append(name)

        return ordered

    def _map_interface_vars(self, rule: Rule, start: int, length: int,
                            template_interface: List[str],
                            template_rule: Rule, template_start: int
                            ) -> list:
        """Map this occurrence's variables to the extracted predicate's interface."""
        body = list(rule.body)
        subseq = body[start:start + length]
        template_body = list(template_rule.body)
        template_subseq = template_body[template_start:template_start + length]

        # Build mapping from template var names to this occurrence's var names
        var_map = {}
        for t_goal, o_goal in zip(template_subseq, subseq):
            if isinstance(t_goal, Compound) and isinstance(o_goal, Compound):
                for t_arg, o_arg in zip(t_goal.args, o_goal.args):
                    if isinstance(t_arg, Variable) and isinstance(o_arg, Variable):
                        var_map[t_arg.name] = o_arg.name

        return [Variable(var_map.get(v, v)) for v in template_interface]

    def _next_extracted_name(self, kb: KnowledgeBase) -> str:
        """Find the next available _extracted_N name."""
        max_n = -1
        for rule in kb.rules:
            if isinstance(rule.head, Compound) and rule.head.functor.startswith("_extracted_"):
                try:
                    n = int(rule.head.functor.split("_extracted_")[1])
                    max_n = max(max_n, n)
                except (ValueError, IndexError):
                    pass
        return f"_extracted_{max_n + 1}"
```

Wire into `dream()`. After line 206 (`ops.extend(self._invent_predicates(kb, suite))`), add:

```python
        ops.extend(self._extract_body_patterns(kb, suite))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_sleep_cycle.py::TestOperationE -v`
Expected: All PASS

- [ ] **Step 5: Run full suite**

Run: `python -m pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 6: End-to-end smoke test**

```python
from dreamlog.knowledge import KnowledgeBase, Rule
from dreamlog.factories import atom, var, compound
from dreamlog.kb_dreamer import KnowledgeBaseDreamer

kb = KnowledgeBase()
kb.add_rule(Rule(compound("grandparent", var("X"), var("Z")),
                 [compound("parent", var("X"), var("Y")),
                  compound("parent", var("Y"), var("Z"))]))
kb.add_rule(Rule(compound("great_gp", var("X"), var("W")),
                 [compound("parent", var("X"), var("Y")),
                  compound("parent", var("Y"), var("Z")),
                  compound("parent", var("Z"), var("W"))]))
kb.add_rule(Rule(compound("great_uncle", var("X"), var("W")),
                 [compound("parent", var("X"), var("Y")),
                  compound("parent", var("Y"), var("Z")),
                  compound("brother", var("Z"), var("W"))]))
kb.add_fact(compound("parent", atom("john"), atom("mary")))
kb.add_fact(compound("parent", atom("mary"), atom("alice")))
kb.add_fact(compound("parent", atom("alice"), atom("bob")))
kb.add_fact(compound("brother", atom("alice"), atom("charlie")))

print(f"Before: {len(kb)} rules")
for r in kb.rules: print(f"  {r}")

dreamer = KnowledgeBaseDreamer()
session = dreamer.dream(kb)

print(f"\nAfter: {len(kb.rules)} rules")
for r in kb.rules: print(f"  {r}")
print(f"Compressed: {session.compressed}")

from dreamlog.evaluator import PrologEvaluator
ev = PrologEvaluator(kb)
assert ev.has_solution(compound("grandparent", atom("john"), atom("alice")))
assert ev.has_solution(compound("great_gp", atom("john"), atom("bob")))
assert ev.has_solution(compound("great_uncle", atom("john"), atom("charlie")))
print("All assertions passed!")
```

- [ ] **Step 7: Commit**

```bash
git add dreamlog/kb_dreamer.py tests/test_sleep_cycle.py
git commit -m "Add Operation E (rule body pattern extraction) to sleep cycle"
```
