# call/N + Predicate Invention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `call/N` meta-predicate to the evaluator and Operation D (cross-functor predicate invention via identical rule-set detection) to the sleep cycle.

**Architecture:** `call/N` is a 15-line evaluator intercept (same pattern as `not/1`). Skeleton extraction is a new module that normalizes rule sets into hashable structural fingerprints. Operation D groups predicates by skeleton, checks MDL, builds parameterized invented predicates with `call/N` dispatch, and verifies correctness.

**Tech Stack:** Python 3.8+, pytest, no new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-25-call-n-predicate-invention-design.md`

**API note:** `compound("f", atom("a"))` is variadic. `Compound("f", [atom("a")])` takes a list. `Rule(head, [body_goals])`.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `dreamlog/evaluator.py` | **Edit** | Add `call/N` handler and `InstantiationError` |
| `dreamlog/skeleton.py` | **Create** | `extract_skeleton`, `RuleSkeleton`, `RuleSetSkeleton` |
| `dreamlog/kb_dreamer.py` | **Edit** | Add Operation D, extend verification for rule-derived queries |
| `tests/test_call.py` | **Create** | `call/N` evaluator tests |
| `tests/test_skeleton.py` | **Create** | Skeleton extraction and comparison tests |
| `tests/test_sleep_cycle.py` | **Edit** | Add Operation D integration tests |

---

### Task 1: `call/N` in the evaluator

**Files:**
- Modify: `dreamlog/evaluator.py`
- Create: `tests/test_call.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_call.py
import pytest
from dreamlog.factories import atom, var, compound
from dreamlog.knowledge import KnowledgeBase, Fact, Rule
from dreamlog.evaluator import PrologEvaluator, InstantiationError


class TestCallN:
    def test_basic_call(self):
        """call(parent, john, mary) resolves like parent(john, mary)."""
        kb = KnowledgeBase()
        kb.add_fact(compound("parent", atom("john"), atom("mary")))
        ev = PrologEvaluator(kb)
        sols = list(ev.query([compound("call", atom("parent"), atom("john"), atom("mary"))]))
        assert len(sols) == 1

    def test_call_with_variable(self):
        """call(parent, X, mary) binds X."""
        kb = KnowledgeBase()
        kb.add_fact(compound("parent", atom("john"), atom("mary")))
        ev = PrologEvaluator(kb)
        sols = list(ev.query([compound("call", atom("parent"), var("X"), atom("mary"))]))
        assert len(sols) == 1
        assert sols[0].bindings.get("X") == atom("john")

    def test_call_functor_bound_via_unification(self):
        """call(F, john, mary) with F bound to parent via a rule."""
        kb = KnowledgeBase()
        kb.add_fact(compound("parent", atom("john"), atom("mary")))
        kb.add_fact(compound("rel", atom("parent")))
        # test_rel(X, Y) :- rel(F), call(F, X, Y)
        kb.add_rule(Rule(
            compound("test_rel", var("X"), var("Y")),
            [compound("rel", var("F")),
             compound("call", var("F"), var("X"), var("Y"))]))
        ev = PrologEvaluator(kb)
        sols = list(ev.query([compound("test_rel", var("X"), var("Y"))]))
        assert len(sols) == 1

    def test_call_unbound_functor_raises(self):
        """call with unbound first arg raises InstantiationError."""
        kb = KnowledgeBase()
        ev = PrologEvaluator(kb)
        with pytest.raises(InstantiationError):
            list(ev.query([compound("call", var("F"), atom("x"))]))

    def test_call_compound_functor_raises(self):
        """call with Compound first arg raises InstantiationError."""
        kb = KnowledgeBase()
        ev = PrologEvaluator(kb)
        with pytest.raises(InstantiationError):
            list(ev.query([compound("call", compound("f", atom("a")), atom("x"))]))

    def test_call_arity_1_raises(self):
        """call/1 (zero additional args) raises InstantiationError."""
        kb = KnowledgeBase()
        ev = PrologEvaluator(kb)
        with pytest.raises(InstantiationError):
            list(ev.query([compound("call", atom("foo"))]))

    def test_call_in_rule_body(self):
        """call/N works inside a rule body (the predicate invention pattern)."""
        kb = KnowledgeBase()
        kb.add_fact(compound("parent", atom("john"), atom("mary")))
        kb.add_fact(compound("parent", atom("mary"), atom("alice")))
        # closure(R, X, Y) :- call(R, X, Y)
        kb.add_rule(Rule(
            compound("closure", var("R"), var("X"), var("Y")),
            [compound("call", var("R"), var("X"), var("Y"))]))
        # closure(R, X, Z) :- call(R, X, Y), closure(R, Y, Z)
        kb.add_rule(Rule(
            compound("closure", var("R"), var("X"), var("Z")),
            [compound("call", var("R"), var("X"), var("Y")),
             compound("closure", var("R"), var("Y"), var("Z"))]))
        ev = PrologEvaluator(kb)
        # Direct: closure(parent, john, mary) should work
        sols = list(ev.query([compound("closure", atom("parent"), atom("john"), atom("mary"))]))
        assert len(sols) >= 1
        # Transitive: closure(parent, john, alice) should work (2 hops)
        sols = list(ev.query([compound("closure", atom("parent"), atom("john"), atom("alice"))]))
        assert len(sols) >= 1

    def test_call_triggers_unknown_hook(self):
        """call/N triggers the unknown hook when predicate is undefined."""
        hook_called = []
        def hook(term, evaluator):
            hook_called.append(str(term))
        kb = KnowledgeBase()
        ev = PrologEvaluator(kb, unknown_hook=hook)
        list(ev.query([compound("call", atom("undefined"), atom("x"))]))
        assert len(hook_called) > 0

    def test_call_inside_not(self):
        """not(call(pred, X)) works correctly."""
        kb = KnowledgeBase()
        kb.add_fact(compound("bird", atom("tweety")))
        ev = PrologEvaluator(kb)
        # not(call(bird, fido)) should succeed
        sols = list(ev.query([compound("not", compound("call", atom("bird"), atom("fido")))]))
        assert len(sols) == 1
        # not(call(bird, tweety)) should fail
        sols = list(ev.query([compound("not", compound("call", atom("bird"), atom("tweety")))]))
        assert len(sols) == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_call.py -v`
Expected: FAIL (InstantiationError not defined)

- [ ] **Step 3: Implement `call/N` and `InstantiationError`**

Add after `FlounderingError` in `dreamlog/evaluator.py` (after line 17):

```python
class InstantiationError(Exception):
    """Raised when call/N receives a non-ground or invalid functor."""
```

In `_solve_goals`, insert after the NAF handler `return` (after line 164) and before `# Try to solve the current goal` (line 166):

```python
                # Handle call/N meta-predicate
                if (isinstance(current_goal.term, Compound)
                        and current_goal.term.functor == "call"):
                    if current_goal.term.arity < 2:
                        raise InstantiationError(
                            "call/N requires at least 2 arguments: call(Functor, Arg1, ...)")
                    functor_arg = current_goal.term.args[0].substitute(global_bindings)
                    if isinstance(functor_arg, Variable):
                        raise InstantiationError(
                            f"call/N: first argument is unbound variable: {functor_arg}")
                    if not isinstance(functor_arg, Atom):
                        raise InstantiationError(
                            f"call/N: first argument must be an atom, got: {functor_arg}")
                    remaining_args = list(current_goal.term.args[1:])
                    constructed = Compound(functor_arg.value, remaining_args)
                    # Replace current goal with constructed term and continue resolution
                    new_goal = Goal(constructed, current_goal.bindings)
                    new_goals = [new_goal] + list(remaining_goals)
                    yield from self._solve_goals(new_goals, global_bindings)
                    return
```

Also add `Atom` to the imports from `.terms` on line 10 (currently imports `Term, Variable, Compound`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_call.py -v`
Expected: All PASS

- [ ] **Step 5: Run full suite for regressions**

Run: `python -m pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add dreamlog/evaluator.py tests/test_call.py
git commit -m "Add call/N meta-predicate and InstantiationError to evaluator"
```

---

### Task 2: Skeleton extraction module

**Files:**
- Create: `dreamlog/skeleton.py`
- Create: `tests/test_skeleton.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_skeleton.py
import pytest
from dreamlog.factories import atom, var, compound
from dreamlog.knowledge import Rule
from dreamlog.skeleton import extract_skeleton, RuleSkeleton, RuleSetSkeleton


class TestSkeletonExtraction:
    def test_single_non_recursive_rule(self):
        """ancestor(X, Y) :- parent(X, Y) -> skeleton with PARAM_0."""
        rules = [Rule(compound("ancestor", var("X"), var("Y")),
                      [compound("parent", var("X"), var("Y"))])]
        skeleton, fmap = extract_skeleton("ancestor", rules)
        assert skeleton.param_count == 1
        assert fmap["PARAM_0"] == "parent"
        assert len(skeleton.rules) == 1

    def test_recursive_rule_self_detected(self):
        """ancestor(X,Z) :- parent(X,Y), ancestor(Y,Z) -> SELF in body."""
        rules = [
            Rule(compound("anc", var("X"), var("Y")),
                 [compound("par", var("X"), var("Y"))]),
            Rule(compound("anc", var("X"), var("Z")),
                 [compound("par", var("X"), var("Y")),
                  compound("anc", var("Y"), var("Z"))]),
        ]
        skeleton, fmap = extract_skeleton("anc", rules)
        assert skeleton.param_count == 1
        assert fmap["PARAM_0"] == "par"
        # Body of recursive rule should have SELF
        recursive_skel = [r for r in skeleton.rules if len(r.body) == 2][0]
        body_roles = [role for role, _ in recursive_skel.body]
        assert "SELF" in body_roles
        assert "PARAM_0" in body_roles

    def test_identical_rule_sets_same_skeleton(self):
        """ancestor/parent and reachable/edge produce same skeleton."""
        rules_a = [
            Rule(compound("ancestor", var("X"), var("Y")),
                 [compound("parent", var("X"), var("Y"))]),
            Rule(compound("ancestor", var("X"), var("Z")),
                 [compound("parent", var("X"), var("Y")),
                  compound("ancestor", var("Y"), var("Z"))]),
        ]
        rules_b = [
            Rule(compound("reachable", var("A"), var("B")),
                 [compound("edge", var("A"), var("B"))]),
            Rule(compound("reachable", var("A"), var("C")),
                 [compound("edge", var("A"), var("B")),
                  compound("reachable", var("B"), var("C"))]),
        ]
        skel_a, fmap_a = extract_skeleton("ancestor", rules_a)
        skel_b, fmap_b = extract_skeleton("reachable", rules_b)
        assert skel_a == skel_b
        assert fmap_a["PARAM_0"] == "parent"
        assert fmap_b["PARAM_0"] == "edge"

    def test_different_arity_different_skeleton(self):
        rules_a = [Rule(compound("f", var("X")), [compound("g", var("X"))])]
        rules_b = [Rule(compound("f", var("X"), var("Y")),
                        [compound("g", var("X"), var("Y"))])]
        skel_a, _ = extract_skeleton("f", rules_a)
        skel_b, _ = extract_skeleton("f", rules_b)
        assert skel_a != skel_b

    def test_different_body_length_different_skeleton(self):
        rules_a = [Rule(compound("f", var("X")), [compound("g", var("X"))])]
        rules_b = [Rule(compound("f", var("X")),
                        [compound("g", var("X")), compound("h", var("X"))])]
        skel_a, _ = extract_skeleton("f", rules_a)
        skel_b, _ = extract_skeleton("f", rules_b)
        assert skel_a != skel_b

    def test_different_variable_connectivity(self):
        """f(X,Y) :- g(X,Y) vs f(X,Y) :- g(Y,X) -> different skeletons."""
        rules_a = [Rule(compound("f", var("X"), var("Y")),
                        [compound("g", var("X"), var("Y"))])]
        rules_b = [Rule(compound("f", var("X"), var("Y")),
                        [compound("g", var("Y"), var("X"))])]
        skel_a, _ = extract_skeleton("f", rules_a)
        skel_b, _ = extract_skeleton("f", rules_b)
        assert skel_a != skel_b

    def test_rule_order_independent(self):
        """Rules defined in different order produce same skeleton."""
        base = Rule(compound("f", var("X"), var("Y")),
                    [compound("g", var("X"), var("Y"))])
        recursive = Rule(compound("f", var("X"), var("Z")),
                         [compound("g", var("X"), var("Y")),
                          compound("f", var("Y"), var("Z"))])
        skel_a, _ = extract_skeleton("f", [base, recursive])
        skel_b, _ = extract_skeleton("f", [recursive, base])
        assert skel_a == skel_b

    def test_same_param_functor_in_multiple_body_goals(self):
        """related(X,Z) :- friend(X,Y), friend(Y,Z) -> PARAM_0 twice."""
        rules = [Rule(compound("related", var("X"), var("Z")),
                      [compound("friend", var("X"), var("Y")),
                       compound("friend", var("Y"), var("Z"))])]
        skeleton, fmap = extract_skeleton("related", rules)
        assert fmap["PARAM_0"] == "friend"
        # Both body goals should reference PARAM_0
        skel_rule = skeleton.rules[0]
        assert all(role == "PARAM_0" for role, _ in skel_rule.body)

    def test_not_in_body_is_opaque(self):
        """Rules with not/1 treat it as opaque functor."""
        rules = [Rule(compound("f", var("X")),
                      [compound("g", var("X")),
                       compound("not", compound("h", var("X")))])]
        skeleton, fmap = extract_skeleton("f", rules)
        # not should appear as a body functor, not decomposed
        skel_rule = skeleton.rules[0]
        assert any(role == "PARAM_0" and arity == 1
                   for role, arity in skel_rule.body) or \
               any(role == "PARAM_1" and arity == 1
                   for role, arity in skel_rule.body)

    def test_skeleton_is_hashable(self):
        """Skeletons can be used as dict keys."""
        rules = [Rule(compound("f", var("X")), [compound("g", var("X"))])]
        skel, _ = extract_skeleton("f", rules)
        d = {skel: "test"}
        assert d[skel] == "test"
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_skeleton.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement skeleton module**

```python
# dreamlog/skeleton.py
"""
Rule-set skeleton extraction for structural comparison.

A skeleton captures the structure of a predicate's rule set while abstracting
away functor names. Two predicates with identical skeletons are structurally
identical and candidates for predicate invention.
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
from .terms import Term, Atom, Variable, Compound
from .knowledge import Rule


@dataclass(frozen=True)
class RuleSkeleton:
    """Skeleton of a single rule."""
    head_arity: int
    body: tuple  # tuple of (functor_role: str, arity: int)
    variable_map: tuple  # tuple of (tuple of (goal_idx, arg_idx), ...)

@dataclass(frozen=True)
class RuleSetSkeleton:
    """Skeleton of a complete rule set."""
    rules: tuple  # tuple of RuleSkeleton, sorted
    param_count: int


def extract_skeleton(
    predicate: str, rules: List[Rule]
) -> Tuple[RuleSetSkeleton, Dict[str, str]]:
    """Extract skeleton and functor mapping from a rule set.

    Returns:
        skeleton: Hashable structural fingerprint
        functor_map: Maps PARAM_0, PARAM_1, ... to actual functor names
    """
    # Build functor -> role mapping
    param_counter = [0]
    functor_to_role: Dict[str, str] = {}
    functor_map: Dict[str, str] = {}  # role -> actual functor

    def _get_role(functor_name: str) -> str:
        if functor_name == predicate:
            return "SELF"
        if functor_name in functor_to_role:
            return functor_to_role[functor_name]
        role = f"PARAM_{param_counter[0]}"
        param_counter[0] += 1
        functor_to_role[functor_name] = role
        functor_map[role] = functor_name
        return role

    # First pass: discover all functor roles across all rules
    # (order: rules sorted by body length for determinism)
    sorted_rules = sorted(rules, key=lambda r: (len(r.body), str(r)))
    for rule in sorted_rules:
        for goal in rule.body:
            if isinstance(goal, Compound):
                _get_role(goal.functor)

    # Second pass: build skeletons
    rule_skeletons = []
    for rule in sorted_rules:
        head_arity = rule.head.arity if isinstance(rule.head, Compound) else 0

        # Normalize variables
        var_positions = _collect_variable_positions(rule)
        var_order = _determine_variable_order(rule)
        var_rename = {old: f"_V{i}" for i, old in enumerate(var_order)}
        var_map = _build_variable_map(var_positions, var_rename)

        # Build body skeleton
        body_parts = []
        for goal in rule.body:
            if isinstance(goal, Compound):
                role = _get_role(goal.functor)
                body_parts.append((role, goal.arity))
            else:
                body_parts.append(("UNKNOWN", 0))

        rule_skeletons.append(RuleSkeleton(
            head_arity=head_arity,
            body=tuple(body_parts),
            variable_map=var_map,
        ))

    # Sort skeletons for order independence
    rule_skeletons.sort(key=lambda s: (len(s.body), s.body, s.variable_map))

    return (
        RuleSetSkeleton(
            rules=tuple(rule_skeletons),
            param_count=param_counter[0],
        ),
        functor_map,
    )


def _determine_variable_order(rule: Rule) -> List[str]:
    """Determine variable ordering: left-to-right through head then body."""
    seen = []
    for var_name in _iter_variables_in_term(rule.head):
        if var_name not in seen:
            seen.append(var_name)
    for goal in rule.body:
        for var_name in _iter_variables_in_term(goal):
            if var_name not in seen:
                seen.append(var_name)
    return seen


def _iter_variables_in_term(term: Term):
    """Yield variable names left-to-right in a term."""
    if isinstance(term, Variable):
        yield term.name
    elif isinstance(term, Compound):
        for arg in term.args:
            yield from _iter_variables_in_term(arg)


def _collect_variable_positions(rule: Rule) -> Dict[str, List[Tuple[int, int]]]:
    """Collect all (goal_index, arg_index) positions for each variable.

    goal_index = -1 for the head.
    """
    positions: Dict[str, List[Tuple[int, int]]] = {}

    def _record(term: Term, goal_idx: int, arg_idx: int):
        if isinstance(term, Variable):
            positions.setdefault(term.name, []).append((goal_idx, arg_idx))
        elif isinstance(term, Compound):
            for i, arg in enumerate(term.args):
                _record(arg, goal_idx, i)

    if isinstance(rule.head, Compound):
        for i, arg in enumerate(rule.head.args):
            _record(arg, -1, i)

    for g_idx, goal in enumerate(rule.body):
        if isinstance(goal, Compound):
            for i, arg in enumerate(goal.args):
                _record(arg, g_idx, i)

    return positions


def _build_variable_map(
    positions: Dict[str, List[Tuple[int, int]]],
    var_rename: Dict[str, str],
) -> tuple:
    """Build the normalized variable connectivity map."""
    # Order by renamed variable index
    renamed_positions = {}
    for old_name, pos_list in positions.items():
        new_name = var_rename.get(old_name, old_name)
        renamed_positions[new_name] = tuple(sorted(pos_list))

    # Sort by variable name (_V0, _V1, ...)
    sorted_vars = sorted(renamed_positions.keys())
    return tuple(renamed_positions[v] for v in sorted_vars)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_skeleton.py -v`
Expected: All PASS

- [ ] **Step 5: Run full suite**

Run: `python -m pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add dreamlog/skeleton.py tests/test_skeleton.py
git commit -m "Add skeleton extraction module for structural rule-set comparison"
```

---

### Task 3: Extended verification for rule-derived queries

**Files:**
- Modify: `dreamlog/kb_dreamer.py`
- Modify: `tests/test_sleep_cycle.py`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_sleep_cycle.py in TestVerification class
    def test_rule_derived_queries_in_suite(self):
        """Verification suite includes queries derived from rules, not just facts."""
        from dreamlog.kb_dreamer import build_verification_suite, extend_verification_for_rules
        from dreamlog.evaluator import PrologEvaluator
        kb = KnowledgeBase()
        kb.add_fact(compound("parent", atom("john"), atom("mary")))
        kb.add_fact(compound("parent", atom("mary"), atom("alice")))
        kb.add_rule(Rule(compound("anc", var("X"), var("Y")),
                         [compound("parent", var("X"), var("Y"))]))
        kb.add_rule(Rule(compound("anc", var("X"), var("Z")),
                         [compound("parent", var("X"), var("Y")),
                          compound("anc", var("Y"), var("Z"))]))
        suite = build_verification_suite(kb)
        extend_verification_for_rules(suite, kb)
        # Should have positive queries for derived facts like anc(john, mary)
        ev = PrologEvaluator(kb)
        assert any(ev.has_solution(q) for q in suite.positive_queries
                   if isinstance(q, Compound) and q.functor == "anc")
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_sleep_cycle.py::TestVerification::test_rule_derived_queries_in_suite -v`
Expected: FAIL

- [ ] **Step 3: Implement `extend_verification_for_rules`**

Add to `dreamlog/kb_dreamer.py` after `build_verification_suite`:

```python
def extend_verification_for_rules(suite: VerificationSuite,
                                  kb: KnowledgeBase,
                                  max_queries: int = 50) -> None:
    """Extend verification suite with rule-derived positive/negative queries.

    For each rule-defined predicate, generate ground queries using atom values
    from the KB and test which are derivable. Adds results to the suite in-place.
    """
    from .evaluator import PrologEvaluator

    ev = PrologEvaluator(kb)

    # Collect atom pool
    atom_values = set()
    for fact in kb.facts:
        if isinstance(fact.term, Compound):
            for arg in fact.term.args:
                if isinstance(arg, Atom):
                    atom_values.add(arg.value)

    if not atom_values:
        return

    atoms = sorted(atom_values)

    # Find rule-defined predicates
    rule_preds = {}
    for rule in kb.rules:
        if isinstance(rule.head, Compound):
            key = (rule.head.functor, rule.head.arity)
            rule_preds[key] = True

    added = 0
    for (functor, arity) in rule_preds:
        if added >= max_queries:
            break
        if functor.startswith("_invented_") or functor.startswith("exception_"):
            continue
        # Generate candidate ground queries from atom pool
        # For arity 1: try each atom. For arity 2: try pairs.
        if arity == 1:
            candidates = [Compound(functor, [Atom(a)]) for a in atoms[:10]]
        elif arity == 2:
            candidates = [Compound(functor, [Atom(a), Atom(b)])
                          for a in atoms[:8] for b in atoms[:8]]
        else:
            continue  # Skip higher arities for now

        for candidate in candidates:
            if added >= max_queries:
                break
            if ev.has_solution(candidate):
                if candidate not in suite.positive_queries:
                    suite.positive_queries.append(candidate)
                    added += 1
            else:
                if candidate not in suite.negative_queries:
                    suite.negative_queries.append(candidate)
                    added += 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sleep_cycle.py::TestVerification -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add dreamlog/kb_dreamer.py tests/test_sleep_cycle.py
git commit -m "Add extend_verification_for_rules for Operation D verification"
```

---

### Task 4: Operation D (predicate invention)

**Files:**
- Modify: `dreamlog/kb_dreamer.py`
- Modify: `tests/test_sleep_cycle.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/test_sleep_cycle.py

class TestOperationD:
    def _make_transitive_closure_kb(self, n_predicates=3):
        """Create KB with N structurally identical transitive closure predicates."""
        kb = KnowledgeBase()
        preds = [("ancestor", "parent"), ("reachable", "edge"), ("connected", "link")]
        for head, base in preds[:n_predicates]:
            kb.add_rule(Rule(
                compound(head, var("X"), var("Y")),
                [compound(base, var("X"), var("Y"))]))
            kb.add_rule(Rule(
                compound(head, var("X"), var("Z")),
                [compound(base, var("X"), var("Y")),
                 compound(head, var("Y"), var("Z"))]))
        # Add some facts for the base predicates
        kb.add_fact(compound("parent", atom("john"), atom("mary")))
        kb.add_fact(compound("parent", atom("mary"), atom("alice")))
        kb.add_fact(compound("edge", atom("a"), atom("b")))
        kb.add_fact(compound("edge", atom("b"), atom("c")))
        kb.add_fact(compound("link", atom("x"), atom("y")))
        return kb

    def test_three_predicates_compressed(self):
        """3 identical rule sets (K=2 each) compress to invented + wrappers."""
        kb = self._make_transitive_closure_kb(3)
        assert len(kb.rules) == 6
        dreamer = KnowledgeBaseDreamer()
        session = dreamer.dream(kb, verify=True)
        # 6 rules -> 2 invented + 3 wrappers = 5
        assert len(kb.rules) == 5
        assert session.compressed is True

    def test_two_predicates_k2_rejected(self):
        """2 predicates with K=2: MDL rejects (4 -> 4, not less)."""
        kb = self._make_transitive_closure_kb(2)
        assert len(kb.rules) == 4
        dreamer = KnowledgeBaseDreamer()
        session = dreamer.dream(kb, verify=False)
        assert len(kb.rules) == 4  # unchanged

    def test_two_predicates_k3_accepted(self):
        """2 predicates with K=3 each: MDL accepts (6 -> 5)."""
        kb = KnowledgeBase()
        for head, base in [("f", "g"), ("p", "q")]:
            kb.add_rule(Rule(compound(head, var("X")),
                             [compound(base, var("X"))]))
            kb.add_rule(Rule(compound(head, var("X")),
                             [compound(base, var("Y")),
                              compound(head, var("Y"))]))  # different pattern
            kb.add_rule(Rule(compound(head, var("X")),
                             [compound(base, var("X")),
                              compound(base, var("Y"))]))  # third rule
        dreamer = KnowledgeBaseDreamer()
        original_count = len(kb.rules)
        session = dreamer.dream(kb, verify=False)
        assert len(kb.rules) < original_count

    def test_invented_excluded_from_future_detection(self):
        """_invented_ predicates from previous dream are skipped."""
        kb = self._make_transitive_closure_kb(3)
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=True)
        size_after = len(kb.rules)
        dreamer.dream(kb, verify=True)
        assert len(kb.rules) == size_after

    def test_different_skeletons_not_grouped(self):
        """Predicates with different structure are not grouped."""
        kb = KnowledgeBase()
        # f(X) :- g(X) -- arity 1, 1 body goal
        kb.add_rule(Rule(compound("f", var("X")),
                         [compound("g", var("X"))]))
        # h(X, Y) :- j(X, Y) -- arity 2, 1 body goal
        kb.add_rule(Rule(compound("h", var("X"), var("Y")),
                         [compound("j", var("X"), var("Y"))]))
        dreamer = KnowledgeBaseDreamer()
        session = dreamer.dream(kb, verify=False)
        assert len(kb.rules) == 2  # unchanged, different arities

    def test_single_rule_predicates_skipped(self):
        """K=1 predicates are never compressed (no MDL gain)."""
        kb = KnowledgeBase()
        for head, base in [("f", "g"), ("h", "j"), ("k", "m")]:
            kb.add_rule(Rule(compound(head, var("X")),
                             [compound(base, var("X"))]))
        dreamer = KnowledgeBaseDreamer()
        session = dreamer.dream(kb, verify=False)
        assert len(kb.rules) == 3  # K=1, N=3: K+N=4, N*K=3, 4 > 3, skip

    def test_derived_queries_still_work(self):
        """After invention, multi-hop derived queries still resolve."""
        kb = self._make_transitive_closure_kb(3)
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=True)
        from dreamlog.evaluator import PrologEvaluator
        ev = PrologEvaluator(kb)
        # ancestor(john, alice) via parent(john,mary), parent(mary,alice)
        assert ev.has_solution(compound("ancestor", atom("john"), atom("alice")))
        # reachable(a, c) via edge(a,b), edge(b,c)
        assert ev.has_solution(compound("reachable", atom("a"), atom("c")))
        # Direct still works
        assert ev.has_solution(compound("ancestor", atom("john"), atom("mary")))

    def test_facts_preserved(self):
        """Facts for a predicate are preserved when its rules are transformed."""
        kb = KnowledgeBase()
        kb.add_fact(compound("ancestor", atom("adam"), atom("eve")))
        for head, base in [("ancestor", "parent"), ("reachable", "edge"),
                           ("connected", "link")]:
            kb.add_rule(Rule(compound(head, var("X"), var("Y")),
                             [compound(base, var("X"), var("Y"))]))
            kb.add_rule(Rule(compound(head, var("X"), var("Z")),
                             [compound(base, var("X"), var("Y")),
                              compound(head, var("Y"), var("Z"))]))
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=False)
        # The explicit ancestor fact should still be there
        anc_facts = [f for f in kb.facts
                     if isinstance(f.term, Compound) and f.term.functor == "ancestor"]
        assert len(anc_facts) == 1

    def test_idempotent(self):
        kb = self._make_transitive_closure_kb(3)
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=True)
        size_first = len(kb)
        dreamer.dream(kb, verify=True)
        assert len(kb) == size_first
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_sleep_cycle.py::TestOperationD -v`
Expected: FAIL

- [ ] **Step 3: Implement Operation D**

Add `_invent_predicates` method to `KnowledgeBaseDreamer` in `dreamlog/kb_dreamer.py`:

```python
    def _invent_predicates(self, kb: KnowledgeBase,
                           suite: Optional['VerificationSuite'] = None
                           ) -> List[CompressionCandidate]:
        """Operation D: Invent predicates from structurally identical rule sets."""
        from .skeleton import extract_skeleton
        from .evaluator import PrologEvaluator

        ops = []

        # Group rules by head functor
        pred_rules: Dict[str, List[Rule]] = {}
        for rule in kb.rules:
            if isinstance(rule.head, Compound):
                f = rule.head.functor
                if f.startswith("_invented_") or f.startswith("exception_"):
                    continue
                pred_rules.setdefault(f, []).append(rule)

        # Extract skeletons and group by skeleton
        skeleton_groups: Dict = {}
        for pred, rules in pred_rules.items():
            if len(rules) < 1:
                continue
            skeleton, fmap = extract_skeleton(pred, rules)
            if skeleton.param_count != 1:
                continue  # Only single-param for this pass
            key = skeleton
            skeleton_groups.setdefault(key, []).append((pred, rules, fmap))

        # For each group, check MDL and build invented predicate
        for skeleton, members in skeleton_groups.items():
            n = len(members)
            k = len(skeleton.rules)
            if k <= 1:
                continue  # K=1 never compresses
            if k + n >= n * k:
                continue  # MDL check

            # Determine next invented predicate name
            invented_name = self._next_invented_name(kb)

            # Build invented predicate from first member as template
            template_pred, template_rules, template_fmap = members[0]
            param_functor = template_fmap["PARAM_0"]

            invented_rules = []
            for rule in sorted(template_rules, key=lambda r: len(r.body)):
                param_var = Variable("R")
                # Build new head: add R as first argument
                old_head = rule.head
                new_head_args = [param_var] + list(old_head.args)
                new_head = Compound(invented_name, new_head_args)

                # Transform body
                new_body = []
                for goal in rule.body:
                    if isinstance(goal, Compound):
                        if goal.functor == template_pred:
                            # Recursive SELF call -> invented predicate call
                            new_args = [param_var] + list(goal.args)
                            new_body.append(Compound(invented_name, new_args))
                        elif goal.functor == param_functor:
                            # PARAM_0 -> call(R, args...)
                            call_args = [param_var] + list(goal.args)
                            new_body.append(Compound("call", call_args))
                        else:
                            new_body.append(goal)
                    else:
                        new_body.append(goal)

                # Normalize variable names in invented rules
                invented_rules.append(Rule(new_head, new_body))

            # Build wrapper rules for each member
            wrapper_rules = []
            for pred, rules, fmap in members:
                actual_functor = fmap["PARAM_0"]
                # head arity from original
                head_arity = rules[0].head.arity if isinstance(rules[0].head, Compound) else 0
                wrapper_vars = [Variable(f"_W{i}") for i in range(head_arity)]
                wrapper_head = Compound(pred, wrapper_vars)
                wrapper_body_args = [Atom(actual_functor)] + wrapper_vars
                wrapper_body = [Compound(invented_name, wrapper_body_args)]
                wrapper_rules.append(Rule(wrapper_head, wrapper_body))

            # Collect all original rules to remove
            all_original = []
            for pred, rules, fmap in members:
                all_original.extend(rules)
            all_new = invented_rules + wrapper_rules

            # Verify
            if suite is not None:
                test_kb = kb.copy()
                for rule in all_original:
                    test_kb.remove_rule_by_value(rule)
                for rule in all_new:
                    test_kb.add_rule(rule)
                result = suite.verify(test_kb, lambda k: PrologEvaluator(k))
                if not result.passed:
                    continue

            # Apply
            for rule in all_original:
                kb.remove_rule_by_value(rule)
            for rule in all_new:
                kb.add_rule(rule)

            ops.append(CompressionCandidate(
                operation="invention",
                original_clauses=all_original,
                new_clauses=all_new))

        return ops

    def _next_invented_name(self, kb: KnowledgeBase) -> str:
        """Find the next available _invented_N name."""
        max_n = -1
        for rule in kb.rules:
            if isinstance(rule.head, Compound) and rule.head.functor.startswith("_invented_"):
                try:
                    n = int(rule.head.functor.split("_invented_")[1])
                    max_n = max(max_n, n)
                except (ValueError, IndexError):
                    pass
        return f"_invented_{max_n + 1}"
```

Wire into `dream()` by adding after line 144 (`ops.extend(self._generalize_facts(kb, suite))`):

```python
        # Extend verification for rule-derived queries before Operation D
        if verify and suite:
            extend_verification_for_rules(suite, kb)
        ops.extend(self._invent_predicates(kb, suite))
```

Also add `Dict` to the typing imports at the top if not present.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_sleep_cycle.py::TestOperationD -v`
Expected: All PASS

- [ ] **Step 5: Run full suite**

Run: `python -m pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add dreamlog/kb_dreamer.py tests/test_sleep_cycle.py
git commit -m "Add Operation D (predicate invention via identical rule-set detection)"
```

---

### Task 5: Public exports and end-to-end smoke test

**Files:**
- Modify: `dreamlog/__init__.py`

- [ ] **Step 1: Add exports**

Add to `dreamlog/__init__.py`:
```python
from .evaluator import InstantiationError
from .skeleton import extract_skeleton, RuleSetSkeleton
```
Add `InstantiationError`, `extract_skeleton`, `RuleSetSkeleton` to `__all__`.

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest tests/ -v --cov=dreamlog --cov-report=term-missing`
Expected: All pass

- [ ] **Step 3: End-to-end smoke test**

```python
from dreamlog.knowledge import KnowledgeBase, Rule
from dreamlog.factories import atom, var, compound
from dreamlog.kb_dreamer import KnowledgeBaseDreamer

kb = KnowledgeBase()
# Three transitive closure predicates
for head, base in [("ancestor","parent"), ("reachable","edge"), ("connected","link")]:
    kb.add_rule(Rule(compound(head, var("X"), var("Y")),
                     [compound(base, var("X"), var("Y"))]))
    kb.add_rule(Rule(compound(head, var("X"), var("Z")),
                     [compound(base, var("X"), var("Y")),
                      compound(head, var("Y"), var("Z"))]))
# Facts
kb.add_fact(compound("parent", atom("john"), atom("mary")))
kb.add_fact(compound("parent", atom("mary"), atom("alice")))
kb.add_fact(compound("edge", atom("a"), atom("b")))
kb.add_fact(compound("edge", atom("b"), atom("c")))

print(f"Before: {len(kb)} clauses ({len(kb.rules)} rules)")
print(kb)

dreamer = KnowledgeBaseDreamer()
session = dreamer.dream(kb)
print(f"\nAfter: {len(kb)} clauses ({len(kb.rules)} rules)")
print(kb)
print(f"Compressed: {session.compressed}, Removed: {session.clauses_removed}")

from dreamlog.evaluator import PrologEvaluator
ev = PrologEvaluator(kb)
assert ev.has_solution(compound("ancestor", atom("john"), atom("alice")))
assert ev.has_solution(compound("reachable", atom("a"), atom("c")))
print("All assertions passed!")
```

Expected: 10 clauses (6 rules + 4 facts) -> 9 clauses (5 rules + 4 facts). 6 rules become 2 invented + 3 wrappers = 5.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "Complete call/N and predicate invention: exports and integration"
```
