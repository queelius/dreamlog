# Sleep Cycle First Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the first working sleep/dream cycle that compresses a knowledge base via anti-unification, subsumption elimination, redundant fact pruning, and fact generalization with exceptions.

**Architecture:** Three new/rewritten modules: `anti_unification.py` (Plotkin's algorithm), enhanced `evaluator.py` (NAF + `has_solution`), and rewritten `kb_dreamer.py` (three compression operations + verification). Small additions to `knowledge.py` (copy/restore/remove-by-value) and `unification.py` (`clause_subsumes`). TDD throughout.

**Tech Stack:** Python 3.8+, pytest, no new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-24-sleep-cycle-first-pass-design.md`

**Important API note:** The `compound()` factory uses variadic args: `compound("f", atom("a"), atom("b"))`. The `Compound()` constructor takes a list: `Compound("f", [atom("a"), atom("b")])`. Do not confuse them.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `dreamlog/anti_unification.py` | **Create** | `anti_unify`, `anti_unify_many`, `node_count`, `AntiUnificationResult` |
| `dreamlog/evaluator.py` | **Edit** | Add `not/1` NAF handler, `has_solution`, `FlounderingError` |
| `dreamlog/knowledge.py` | **Edit** | Add `copy`, `restore_from`, `remove_fact_by_value`, `remove_rule_by_value`, `replace_facts` |
| `dreamlog/unification.py` | **Edit** | Add `clause_subsumes` |
| `dreamlog/kb_dreamer.py` | **Rewrite** | Sleep cycle: Op A, Op B, Op C, verification, `DreamSession` |
| `tests/test_anti_unification.py` | **Create** | Unit tests for anti-unification |
| `tests/test_naf.py` | **Create** | Unit tests for negation as failure |
| `tests/test_kb_methods.py` | **Create** | Tests for new KB methods |
| `tests/test_clause_subsumes.py` | **Create** | Tests for clause-level subsumption |
| `tests/test_sleep_cycle.py` | **Create** | Integration tests for the full dream loop |

---

### Task 1: Anti-unification core algorithm

**Files:**
- Create: `dreamlog/anti_unification.py`
- Create: `tests/test_anti_unification.py`

- [ ] **Step 1: Write failing tests for `node_count`**

```python
# tests/test_anti_unification.py
import pytest
from dreamlog.terms import Compound, Atom, Variable
from dreamlog.factories import atom, var, compound
from dreamlog.anti_unification import node_count


class TestNodeCount:
    def test_atom(self):
        assert node_count(atom("a")) == 1

    def test_variable(self):
        assert node_count(var("X")) == 1

    def test_compound_no_args(self):
        assert node_count(Compound("f", [])) == 1

    def test_compound_with_args(self):
        # f(a, b) = 1 + 1 + 1 = 3
        assert node_count(compound("f", atom("a"), atom("b"))) == 3

    def test_nested_compound(self):
        # f(g(a), b) = 1 + (1 + 1) + 1 = 4
        inner = compound("g", atom("a"))
        assert node_count(compound("f", inner, atom("b"))) == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_anti_unification.py::TestNodeCount -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement `node_count` and module skeleton**

```python
# dreamlog/anti_unification.py
"""
Anti-unification (least general generalization) for DreamLog terms.

Implements Plotkin's (1970) algorithm: the dual of unification.
Where unification finds the most general common instance,
anti-unification finds the most specific common generalization.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from .terms import Term, Atom, Variable, Compound


@dataclass
class AntiUnificationResult:
    """Result of anti-unifying two or more terms."""
    generalized: Term
    substitutions: List[Dict[str, Term]]
    variables_introduced: int
    shared_structure: float


def node_count(term: Term) -> int:
    """Count nodes in a term tree.

    Atom or Variable = 1 node. Compound = 1 + sum of arg node counts.
    """
    if isinstance(term, (Atom, Variable)):
        return 1
    if isinstance(term, Compound):
        return 1 + sum(node_count(arg) for arg in term.args)
    raise TypeError(f"Unknown term type: {type(term)}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_anti_unification.py::TestNodeCount -v`
Expected: All 5 PASS

- [ ] **Step 5: Write failing tests for `anti_unify` (two-term)**

```python
# Append to tests/test_anti_unification.py
from dreamlog.anti_unification import anti_unify, AntiUnificationResult


class TestAntiUnifyTwoTerms:
    def test_identical_atoms(self):
        result = anti_unify(atom("a"), atom("a"))
        assert result.generalized == atom("a")
        assert result.variables_introduced == 0
        assert result.shared_structure == 1.0

    def test_different_atoms(self):
        result = anti_unify(atom("a"), atom("b"))
        assert isinstance(result.generalized, Variable)
        assert result.generalized.name.startswith("_G")
        assert result.variables_introduced == 1
        assert result.shared_structure == 0.0

    def test_same_functor_same_arity(self):
        # f(a, b) vs f(a, c) -> f(a, _G0)
        t1 = compound("f", atom("a"), atom("b"))
        t2 = compound("f", atom("a"), atom("c"))
        result = anti_unify(t1, t2)
        assert isinstance(result.generalized, Compound)
        assert result.generalized.functor == "f"
        assert result.generalized.args[0] == atom("a")
        assert isinstance(result.generalized.args[1], Variable)
        assert result.variables_introduced == 1

    def test_same_pair_consistency(self):
        # f(a, a) vs f(b, b) -> f(_G0, _G0)
        t1 = compound("f", atom("a"), atom("a"))
        t2 = compound("f", atom("b"), atom("b"))
        result = anti_unify(t1, t2)
        g = result.generalized
        assert isinstance(g, Compound)
        assert g.args[0] == g.args[1]  # same variable reused
        assert result.variables_introduced == 1

    def test_different_pairs_distinct(self):
        # f(a, b) vs f(b, a) -> f(_G0, _G1)
        t1 = compound("f", atom("a"), atom("b"))
        t2 = compound("f", atom("b"), atom("a"))
        result = anti_unify(t1, t2)
        g = result.generalized
        assert isinstance(g, Compound)
        assert g.args[0] != g.args[1]
        assert result.variables_introduced == 2

    def test_nested_compound(self):
        # f(g(a), h(b)) vs f(g(c), h(d)) -> f(g(_G0), h(_G1))
        t1 = compound("f", compound("g", atom("a")), compound("h", atom("b")))
        t2 = compound("f", compound("g", atom("c")), compound("h", atom("d")))
        result = anti_unify(t1, t2)
        g = result.generalized
        assert g.functor == "f"
        assert g.args[0].functor == "g"
        assert isinstance(g.args[0].args[0], Variable)
        assert g.args[1].functor == "h"
        assert isinstance(g.args[1].args[0], Variable)

    def test_mismatched_functors(self):
        t1 = compound("f", atom("a"))
        t2 = compound("g", atom("a"))
        result = anti_unify(t1, t2)
        assert isinstance(result.generalized, Variable)

    def test_mismatched_arities(self):
        t1 = compound("f", atom("a"), atom("b"))
        t2 = compound("f", atom("a"))
        result = anti_unify(t1, t2)
        assert isinstance(result.generalized, Variable)

    def test_substitution_recovery(self):
        t1 = compound("f", atom("a"), atom("b"))
        t2 = compound("f", atom("a"), atom("c"))
        result = anti_unify(t1, t2)
        recovered_1 = result.generalized.substitute(result.substitutions[0])
        recovered_2 = result.generalized.substitute(result.substitutions[1])
        assert recovered_1 == t1
        assert recovered_2 == t2

    def test_shared_structure_score(self):
        # f(a, b) vs f(a, c): generalized = f(a, _G0)
        # node_count = 3, shared = 2 (f, a), non-shared = 1 (_G0)
        t1 = compound("f", atom("a"), atom("b"))
        t2 = compound("f", atom("a"), atom("c"))
        result = anti_unify(t1, t2)
        assert abs(result.shared_structure - 2.0 / 3.0) < 0.01
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `python -m pytest tests/test_anti_unification.py::TestAntiUnifyTwoTerms -v`
Expected: FAIL (anti_unify not defined)

- [ ] **Step 7: Implement `anti_unify`**

Add to `dreamlog/anti_unification.py`:

```python
def anti_unify(term1: Term, term2: Term) -> AntiUnificationResult:
    """Compute the least general generalization of two terms."""
    seen_pairs: Dict[Tuple[Term, Term], Variable] = {}
    sub1: Dict[str, Term] = {}
    sub2: Dict[str, Term] = {}
    counter = [0]

    def _fresh_var() -> Variable:
        name = f"_G{counter[0]}"
        counter[0] += 1
        return Variable(name)

    def _anti_unify_rec(t1: Term, t2: Term) -> Term:
        if t1 == t2:
            return t1

        pair_key = (t1, t2)
        if pair_key in seen_pairs:
            return seen_pairs[pair_key]

        if (isinstance(t1, Compound) and isinstance(t2, Compound)
                and t1.functor == t2.functor and t1.arity == t2.arity):
            new_args = [_anti_unify_rec(a1, a2)
                        for a1, a2 in zip(t1.args, t2.args)]
            return Compound(t1.functor, new_args)

        v = _fresh_var()
        seen_pairs[pair_key] = v
        sub1[v.name] = t1
        sub2[v.name] = t2
        return v

    generalized = _anti_unify_rec(term1, term2)
    variables_introduced = counter[0]

    total_nodes = node_count(generalized)
    shared_nodes = total_nodes - variables_introduced
    shared_structure = shared_nodes / total_nodes if total_nodes > 0 else 1.0

    return AntiUnificationResult(
        generalized=generalized,
        substitutions=[sub1, sub2],
        variables_introduced=variables_introduced,
        shared_structure=shared_structure,
    )
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `python -m pytest tests/test_anti_unification.py -v`
Expected: All PASS

- [ ] **Step 9: Commit**

```bash
git add dreamlog/anti_unification.py tests/test_anti_unification.py
git commit -m "Add anti-unification algorithm with node_count and shared_structure"
```

---

### Task 2: Multi-term anti-unification

**Files:**
- Modify: `dreamlog/anti_unification.py`
- Modify: `tests/test_anti_unification.py`

- [ ] **Step 1: Write failing tests for `anti_unify_many`**

```python
# Append to tests/test_anti_unification.py
from dreamlog.anti_unification import anti_unify_many


class TestAntiUnifyMany:
    def test_single_term(self):
        t = compound("f", atom("a"))
        result = anti_unify_many([t])
        assert result.generalized == t
        assert len(result.substitutions) == 1
        assert result.variables_introduced == 0

    def test_two_terms(self):
        t1 = compound("f", atom("a"))
        t2 = compound("f", atom("b"))
        result = anti_unify_many([t1, t2])
        assert result.generalized.functor == "f"
        assert isinstance(result.generalized.args[0], Variable)
        assert len(result.substitutions) == 2

    def test_three_terms_preserves_shared(self):
        terms = [compound("f", atom("a"), atom(v)) for v in ["x", "y", "z"]]
        result = anti_unify_many(terms)
        assert result.generalized.functor == "f"
        assert result.generalized.args[0] == atom("a")
        assert isinstance(result.generalized.args[1], Variable)

    def test_substitution_recovery_many(self):
        terms = [compound("f", atom("a"), atom(v)) for v in ["x", "y", "z"]]
        result = anti_unify_many(terms)
        for i, t in enumerate(terms):
            recovered = result.generalized.substitute(result.substitutions[i])
            assert recovered == t

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            anti_unify_many([])
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_anti_unification.py::TestAntiUnifyMany -v`
Expected: FAIL

- [ ] **Step 3: Implement `anti_unify_many`**

Add to `dreamlog/anti_unification.py`:

```python
def anti_unify_many(terms: List[Term]) -> AntiUnificationResult:
    """Compute the lgg of multiple terms via pairwise folding."""
    if not terms:
        raise ValueError("Cannot anti-unify empty list")
    if len(terms) == 1:
        return AntiUnificationResult(
            generalized=terms[0], substitutions=[{}],
            variables_introduced=0, shared_structure=1.0)

    acc = anti_unify(terms[0], terms[1])
    all_subs = [acc.substitutions[0], acc.substitutions[1]]

    for i in range(2, len(terms)):
        prev_gen = acc.generalized
        acc = anti_unify(prev_gen, terms[i])
        bridge = acc.substitutions[0]
        new_all_subs = []
        for old_sub in all_subs:
            composed = {}
            for var_name, term in bridge.items():
                composed[var_name] = term.substitute(old_sub)
            new_all_subs.append(composed)
        new_all_subs.append(acc.substitutions[1])
        all_subs = new_all_subs

    return AntiUnificationResult(
        generalized=acc.generalized, substitutions=all_subs,
        variables_introduced=acc.variables_introduced,
        shared_structure=acc.shared_structure)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_anti_unification.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add dreamlog/anti_unification.py tests/test_anti_unification.py
git commit -m "Add multi-term anti-unification with substitution composition"
```

---

### Task 3: KnowledgeBase methods (copy, restore, remove-by-value)

**Files:**
- Modify: `dreamlog/knowledge.py`
- Create: `tests/test_kb_methods.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_kb_methods.py
import pytest
from dreamlog.factories import atom, compound
from dreamlog.knowledge import KnowledgeBase, Fact, Rule
from dreamlog.terms import Variable


class TestKBCopy:
    def test_copy_preserves_facts(self):
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        copy = kb.copy()
        assert len(copy.facts) == 1
        assert copy.facts[0] == kb.facts[0]

    def test_copy_is_independent(self):
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        copy = kb.copy()
        kb.add_fact(compound("b", atom("y")))
        assert len(copy.facts) == 1
        assert len(kb.facts) == 2


class TestKBRestoreFrom:
    def test_restore_reverts_changes(self):
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        snapshot = kb.copy()
        kb.add_fact(compound("b", atom("y")))
        kb.add_rule(Rule(compound("c", Variable("Z")), []))
        assert len(kb) == 3
        kb.restore_from(snapshot)
        assert len(kb) == 1


class TestKBRemoveByValue:
    def test_remove_fact_by_value(self):
        kb = KnowledgeBase()
        f = Fact(compound("a", atom("x")))
        kb.add_fact(f)
        kb.remove_fact_by_value(f)
        assert len(kb.facts) == 0

    def test_remove_fact_not_found_raises(self):
        kb = KnowledgeBase()
        f = Fact(compound("a", atom("x")))
        with pytest.raises(ValueError):
            kb.remove_fact_by_value(f)

    def test_remove_rule_by_value(self):
        kb = KnowledgeBase()
        r = Rule(compound("a", Variable("X")),
                 [compound("b", Variable("X"))])
        kb.add_rule(r)
        kb.remove_rule_by_value(r)
        assert len(kb.rules) == 0


class TestKBReplaceFacts:
    def test_replace_facts_with_rule(self):
        kb = KnowledgeBase()
        f1 = Fact(compound("likes", atom("a"), atom("choc")))
        f2 = Fact(compound("likes", atom("b"), atom("choc")))
        kb.add_fact(f1)
        kb.add_fact(f2)
        new_rule = Rule(compound("likes", Variable("X"), atom("choc")), [])
        kb.replace_facts([f1, f2], [new_rule])
        assert len(kb.facts) == 0
        assert len(kb.rules) == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_kb_methods.py -v`
Expected: FAIL

- [ ] **Step 3: Implement KB methods**

Add to `dreamlog/knowledge.py` imports at top:

```python
from typing import List, Dict, Any, Set, Optional, Iterator, Union
```

(`Union` must be present; verify the existing import line and add if missing.)

Add these methods to the `KnowledgeBase` class:

```python
def copy(self) -> 'KnowledgeBase':
    """Deep copy for rollback."""
    new_kb = KnowledgeBase()
    for fact in self._facts:
        new_kb.add_fact(fact)
    for rule in self._rules:
        new_kb.add_rule(rule)
    return new_kb

def restore_from(self, other: 'KnowledgeBase') -> None:
    """Replace contents with another KB's contents."""
    self.clear()
    for fact in other._facts:
        self.add_fact(fact)
    for rule in other._rules:
        self.add_rule(rule)

def remove_fact_by_value(self, fact: Fact) -> None:
    """Remove a fact by equality."""
    try:
        idx = self._facts.index(fact)
    except ValueError:
        raise ValueError(f"Fact not found: {fact}")
    self.remove_fact(idx)

def remove_rule_by_value(self, rule: Rule) -> None:
    """Remove a rule by equality."""
    try:
        idx = self._rules.index(rule)
    except ValueError:
        raise ValueError(f"Rule not found: {rule}")
    self.remove_rule(idx)

def replace_facts(self, old: List[Fact],
                  new: List[Union[Fact, Rule]]) -> None:
    """Atomic replacement: remove old facts, add new facts/rules."""
    for fact in old:
        self.remove_fact_by_value(fact)
    for item in new:
        if isinstance(item, Fact):
            self.add_fact(item)
        elif isinstance(item, Rule):
            self.add_rule(item)
        elif isinstance(item, Term):
            self.add_fact(item)
        else:
            raise TypeError(f"Expected Fact or Rule, got {type(item)}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_kb_methods.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add dreamlog/knowledge.py tests/test_kb_methods.py
git commit -m "Add KB copy, restore_from, remove-by-value, and replace_facts"
```

---

### Task 4: Negation as failure + `has_solution`

**Files:**
- Modify: `dreamlog/evaluator.py`
- Create: `tests/test_naf.py`

- [ ] **Step 1: Write failing tests for NAF**

```python
# tests/test_naf.py
import pytest
from dreamlog.factories import atom, var, compound
from dreamlog.knowledge import KnowledgeBase, Fact, Rule
from dreamlog.evaluator import PrologEvaluator, FlounderingError


class TestNAF:
    def _make_kb(self):
        kb = KnowledgeBase()
        kb.add_fact(compound("bird", atom("tweety")))
        kb.add_fact(compound("bird", atom("opus")))
        kb.add_fact(compound("penguin", atom("opus")))
        kb.add_rule(Rule(
            compound("flies", var("X")),
            [compound("bird", var("X")),
             compound("not", compound("penguin", var("X")))]))
        return kb

    def test_not_known_fact_fails(self):
        """not(bird(tweety)) should fail."""
        kb = self._make_kb()
        ev = PrologEvaluator(kb)
        sols = list(ev.query([compound("not", compound("bird", atom("tweety")))]))
        assert len(sols) == 0

    def test_not_unknown_fact_succeeds(self):
        """not(bird(fido)) should succeed."""
        kb = self._make_kb()
        ev = PrologEvaluator(kb)
        sols = list(ev.query([compound("not", compound("bird", atom("fido")))]))
        assert len(sols) == 1

    def test_not_derivable_from_rule_fails(self):
        """not(flies(tweety)) should fail since flies(tweety) is derivable."""
        kb = self._make_kb()
        ev = PrologEvaluator(kb)
        sols = list(ev.query([compound("not", compound("flies", atom("tweety")))]))
        assert len(sols) == 0

    def test_exception_clause_pattern(self):
        """flies(tweety) succeeds, flies(opus) fails."""
        kb = self._make_kb()
        ev = PrologEvaluator(kb)
        assert len(list(ev.query([compound("flies", atom("tweety"))]))) == 1
        assert len(list(ev.query([compound("flies", atom("opus"))]))) == 0

    def test_double_negation(self):
        """not(not(bird(tweety))) should succeed; not(not(bird(fido))) should fail."""
        kb = self._make_kb()
        ev = PrologEvaluator(kb)
        # not(not(bird(tweety))): inner not(bird(tweety)) fails, so outer not succeeds
        sols = list(ev.query([
            compound("not", compound("not", compound("bird", atom("tweety"))))]))
        assert len(sols) == 1
        # not(not(bird(fido))): inner not(bird(fido)) succeeds, so outer not fails
        sols = list(ev.query([
            compound("not", compound("not", compound("bird", atom("fido"))))]))
        assert len(sols) == 0

    def test_floundering_error(self):
        """not/1 on non-ground goal raises FlounderingError."""
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        ev = PrologEvaluator(kb)
        with pytest.raises(FlounderingError):
            list(ev.query([compound("not", compound("a", var("X")))]))

    def test_naf_suppresses_unknown_hook(self):
        """Unknown hook should NOT fire inside not/1."""
        hook_called = []
        def hook(term, evaluator):
            hook_called.append(term)
        kb = KnowledgeBase()
        ev = PrologEvaluator(kb, unknown_hook=hook)
        sols = list(ev.query([compound("not", compound("undefined", atom("x")))]))
        assert len(sols) == 1
        assert len(hook_called) == 0


class TestHasSolution:
    def test_true(self):
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        ev = PrologEvaluator(kb)
        assert ev.has_solution(compound("a", atom("x"))) is True

    def test_false(self):
        kb = KnowledgeBase()
        ev = PrologEvaluator(kb)
        assert ev.has_solution(compound("a", atom("x"))) is False

    def test_via_rule(self):
        kb = KnowledgeBase()
        kb.add_fact(compound("b", atom("x")))
        kb.add_rule(Rule(compound("a", var("X")), [compound("b", var("X"))]))
        ev = PrologEvaluator(kb)
        assert ev.has_solution(compound("a", atom("x"))) is True
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_naf.py -v`
Expected: FAIL

- [ ] **Step 3: Implement NAF and `has_solution`**

Add `FlounderingError` after the imports in `dreamlog/evaluator.py`:

```python
class FlounderingError(Exception):
    """Raised when not/1 is applied to a non-ground goal."""
```

Add `has_solution` method to `PrologEvaluator` (after `find_first_solution`):

```python
def has_solution(self, term: Term) -> bool:
    """Check if a term is derivable. Stops at first solution."""
    for _ in self.query([term]):
        return True
    return False
```

In `_solve_goals`, insert NAF handling after line 136 (`current_goal = current_goal.substitute(global_bindings)`) and before line 138 (`solutions_found = False`):

```python
                # Handle negation as failure: not/1
                if (isinstance(current_goal.term, Compound)
                        and current_goal.term.functor == "not"
                        and current_goal.term.arity == 1):
                    inner_term = current_goal.term.args[0]
                    # current_goal.term is already post-substitution
                    inner_resolved = inner_term.substitute(global_bindings)

                    # Floundering check
                    if inner_resolved.get_variables():
                        raise FlounderingError(
                            f"not/1 applied to non-ground goal: {inner_resolved}")

                    # Evaluate with hook suppressed
                    naf_evaluator = PrologEvaluator(self.kb, unknown_hook=None)
                    naf_evaluator._max_recursion_depth = self._max_recursion_depth
                    if not naf_evaluator.has_solution(inner_resolved):
                        # Inner goal failed -> not/1 succeeds
                        new_remaining = [g.substitute(global_bindings)
                                         for g in remaining_goals]
                        yield from self._solve_goals(
                            new_remaining, global_bindings)
                    return
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_naf.py -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite for regressions**

Run: `python -m pytest tests/ -x -q`
Expected: All existing tests pass

- [ ] **Step 6: Commit**

```bash
git add dreamlog/evaluator.py tests/test_naf.py
git commit -m "Add negation as failure (not/1) and has_solution to evaluator"
```

---

### Task 5: Clause-level subsumption

**Files:**
- Modify: `dreamlog/unification.py`
- Create: `tests/test_clause_subsumes.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_clause_subsumes.py
from dreamlog.factories import atom, var, compound
from dreamlog.knowledge import Rule
from dreamlog.unification import clause_subsumes


class TestClauseSubsumes:
    def test_general_rule_subsumes_specific(self):
        general = Rule(compound("anc", var("X"), var("Y")),
                       [compound("par", var("X"), var("Y"))])
        specific = Rule(compound("anc", atom("john"), var("Y")),
                        [compound("par", atom("john"), var("Y"))])
        assert clause_subsumes(general, specific) is True
        assert clause_subsumes(specific, general) is False

    def test_identical_rules(self):
        r = Rule(compound("a", var("X")), [compound("b", var("X"))])
        assert clause_subsumes(r, r) is True

    def test_different_body_length(self):
        r1 = Rule(compound("a", var("X")), [compound("b", var("X"))])
        r2 = Rule(compound("a", var("X")),
                  [compound("b", var("X")), compound("c", var("X"))])
        assert clause_subsumes(r1, r2) is False
        assert clause_subsumes(r2, r1) is False

    def test_bodyless_rules(self):
        general = Rule(compound("a", var("X")), [])
        specific = Rule(compound("a", atom("hello")), [])
        assert clause_subsumes(general, specific) is True

    def test_no_subsumption_different_head(self):
        r1 = Rule(compound("a", var("X")), [compound("b", var("X"))])
        r2 = Rule(compound("c", var("X")), [compound("b", var("X"))])
        assert clause_subsumes(r1, r2) is False
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_clause_subsumes.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `clause_subsumes`**

Add to `dreamlog/unification.py` after the `subsumes` function:

```python
def clause_subsumes(general: 'Rule', specific: 'Rule') -> bool:
    """Check if general clause subsumes specific (same-body-length only)."""
    from .knowledge import Rule

    if len(general.body) != len(specific.body):
        return False

    head_bindings = unify(general.head, specific.head,
                          mode=UnificationMode.SUBSUME)
    if head_bindings is None:
        return False

    for gen_goal, spec_goal in zip(general.body, specific.body):
        gen_substituted = gen_goal.substitute(head_bindings)
        goal_bindings = unify(gen_substituted, spec_goal,
                              mode=UnificationMode.SUBSUME)
        if goal_bindings is None:
            return False

    return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_clause_subsumes.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add dreamlog/unification.py tests/test_clause_subsumes.py
git commit -m "Add clause_subsumes for same-body-length rule subsumption"
```

---

### Task 6: Sleep cycle -- Operations A, B, verification, and dream loop

**Files:**
- Create: `dreamlog/kb_dreamer.py` (rewrite)
- Create: `tests/test_sleep_cycle.py`

This task writes the full dreamer module with Operations A, B, verification,
and the dream loop. Operation C is added in Task 7.

- [ ] **Step 1: Write failing tests for Op A, Op B, verification, and edge cases**

```python
# tests/test_sleep_cycle.py
import pytest
from dreamlog.factories import atom, var, compound
from dreamlog.knowledge import KnowledgeBase, Fact, Rule
from dreamlog.kb_dreamer import KnowledgeBaseDreamer, DreamSession


class TestOperationA:
    def test_specific_rule_removed(self):
        kb = KnowledgeBase()
        kb.add_rule(Rule(compound("anc", var("X"), var("Y")),
                         [compound("par", var("X"), var("Y"))]))
        kb.add_rule(Rule(compound("anc", atom("john"), var("Y")),
                         [compound("par", atom("john"), var("Y"))]))
        dreamer = KnowledgeBaseDreamer()
        session = dreamer.dream(kb, verify=False)
        assert len(kb.rules) == 1
        assert session.compressed is True

    def test_fact_subsumed_by_bodyless_rule(self):
        kb = KnowledgeBase()
        kb.add_rule(Rule(compound("a", var("X")), []))  # a(X). for all X
        kb.add_fact(compound("a", atom("hello")))
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=False)
        assert len(kb.facts) == 0
        assert len(kb.rules) == 1

    def test_different_body_lengths_kept(self):
        kb = KnowledgeBase()
        kb.add_rule(Rule(compound("a", var("X")),
                         [compound("b", var("X"))]))
        kb.add_rule(Rule(compound("a", var("X")),
                         [compound("b", var("X")), compound("c", var("X"))]))
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=False)
        assert len(kb.rules) == 2

    def test_rules_only_kb(self):
        kb = KnowledgeBase()
        kb.add_rule(Rule(compound("a", var("X")), [compound("b", var("X"))]))
        kb.add_rule(Rule(compound("a", atom("x")), [compound("b", atom("x"))]))
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=False)
        assert len(kb.rules) == 1

    def test_empty_kb(self):
        kb = KnowledgeBase()
        dreamer = KnowledgeBaseDreamer()
        session = dreamer.dream(kb)
        assert session.compressed is False
        assert session.compression_ratio == 1.0


class TestOperationB:
    def test_derivable_fact_removed(self):
        kb = KnowledgeBase()
        kb.add_fact(compound("parent", atom("john"), atom("mary")))
        kb.add_rule(Rule(compound("anc", var("X"), var("Y")),
                         [compound("parent", var("X"), var("Y"))]))
        kb.add_fact(compound("anc", atom("john"), atom("mary")))
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=False)
        anc_facts = [f for f in kb.facts
                     if hasattr(f.term, 'functor') and f.term.functor == "anc"]
        assert len(anc_facts) == 0

    def test_non_derivable_fact_kept(self):
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        kb.add_fact(compound("b", atom("y")))
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=False)
        assert len(kb.facts) == 2

    def test_mutual_dependency_fallback(self):
        """Two facts providing mutual derivability via circular rules."""
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        kb.add_fact(compound("b", atom("x")))
        kb.add_rule(Rule(compound("a", var("X")), [compound("b", var("X"))]))
        kb.add_rule(Rule(compound("b", var("X")), [compound("a", var("X"))]))
        # Both facts appear derivable individually (via circular rules + other fact)
        # but removing both makes neither derivable
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=False)
        # At least one fact must survive
        assert len(kb.facts) >= 1

    def test_facts_only_kb(self):
        """KB with only facts, no rules: nothing to prune."""
        kb = KnowledgeBase()
        for v in ["a", "b", "c"]:
            kb.add_fact(compound("x", atom(v)))
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=False)
        assert len(kb.facts) == 3


class TestVerification:
    def test_positive_queries_built(self):
        from dreamlog.kb_dreamer import build_verification_suite
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        kb.add_fact(compound("b", atom("y")))
        suite = build_verification_suite(kb)
        assert len(suite.positive_queries) >= 2

    def test_negative_queries_catch_overgeneration(self):
        from dreamlog.kb_dreamer import build_verification_suite
        from dreamlog.evaluator import PrologEvaluator
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        kb.add_fact(compound("a", atom("y")))
        kb.add_fact(compound("b", atom("z")))  # z enters atom pool
        suite = build_verification_suite(kb)
        result = suite.verify(kb, lambda k: PrologEvaluator(k))
        assert result.passed
        # Over-general KB
        bad_kb = KnowledgeBase()
        bad_kb.add_rule(Rule(compound("a", var("X")), []))
        bad_kb.add_fact(compound("b", atom("z")))
        result = suite.verify(bad_kb, lambda k: PrologEvaluator(k))
        assert not result.passed

    def test_rollback_preserves_kb(self):
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        original_size = len(kb)
        dreamer = KnowledgeBaseDreamer()
        session = dreamer.dream(kb, verify=True)
        assert len(kb) == original_size
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_sleep_cycle.py -v`
Expected: FAIL

- [ ] **Step 3: Implement the full dreamer with Op A, Op B, verification**

Write `dreamlog/kb_dreamer.py`:

```python
"""
Knowledge Base Dreamer - Sleep phase symbolic compression.

Implements the sleep/dream cycle via three operations:
A. Subsumption elimination
B. Redundant fact pruning
C. Fact generalization with exceptions (added in Task 7)

All operations are purely symbolic (no LLM). Compression is guided by
Minimum Description Length: compress only when the result is shorter.
"""

from typing import List, Optional, Union, Set
from dataclasses import dataclass, field
from .terms import Term, Atom, Variable, Compound
from .knowledge import KnowledgeBase, Fact, Rule
from .unification import clause_subsumes, subsumes


@dataclass
class CompressionCandidate:
    """A single compression operation that was applied."""
    operation: str
    original_clauses: List[Union[Fact, Rule]]
    new_clauses: List[Union[Fact, Rule]] = field(default_factory=list)

    @property
    def mdl_delta(self) -> int:
        return len(self.new_clauses) - len(self.original_clauses)

    @property
    def is_worth_it(self) -> bool:
        return self.mdl_delta < 0


@dataclass
class VerificationResult:
    """Result of verifying a compressed KB."""
    passed: bool
    failures: List[tuple]
    positive_count: int
    negative_count: int


@dataclass
class VerificationSuite:
    """Test suite for verifying KB compression preserves semantics."""
    positive_queries: List[Term]
    negative_queries: List[Term]

    def verify(self, kb: KnowledgeBase, evaluator_factory) -> VerificationResult:
        evaluator = evaluator_factory(kb)
        failures = []
        for q in self.positive_queries:
            if not evaluator.has_solution(q):
                failures.append(("false_negative", q))
        for q in self.negative_queries:
            if evaluator.has_solution(q):
                failures.append(("false_positive", q))
        return VerificationResult(
            passed=len(failures) == 0, failures=failures,
            positive_count=len(self.positive_queries),
            negative_count=len(self.negative_queries))


def build_verification_suite(kb: KnowledgeBase) -> VerificationSuite:
    """Build verification suite from current KB state."""
    positive = [fact.term for fact in kb.facts]

    atom_pool: Set[str] = set()
    fact_terms_by_key = {}
    for fact in kb.facts:
        term = fact.term
        if isinstance(term, Compound):
            key = (term.functor, term.arity)
            fact_terms_by_key.setdefault(key, []).append(term)
            for arg in term.args:
                if isinstance(arg, Atom):
                    atom_pool.add(arg.value)
        elif isinstance(term, Atom):
            atom_pool.add(term.value)

    negative: List[Term] = []
    max_negative = 2 * len(positive)
    for (functor, arity), terms in fact_terms_by_key.items():
        if len(negative) >= max_negative:
            break
        for pos in range(arity):
            existing_values = set()
            for t in terms:
                if isinstance(t.args[pos], Atom):
                    existing_values.add(t.args[pos].value)
            novel_values = atom_pool - existing_values
            if not novel_values:
                continue
            novel = sorted(novel_values)[0]
            base = terms[0]
            new_args = list(base.args)
            new_args[pos] = Atom(novel)
            neg_term = Compound(functor, new_args)
            if neg_term not in [f.term for f in kb.facts]:
                negative.append(neg_term)
            if len(negative) >= max_negative:
                break

    return VerificationSuite(positive_queries=positive,
                             negative_queries=negative)


@dataclass
class DreamSession:
    """Results from a dream cycle."""
    compressed: bool
    operations: List[CompressionCandidate]
    compression_ratio: float
    verification: Optional[VerificationResult]

    @property
    def clauses_removed(self) -> int:
        return sum(-op.mdl_delta for op in self.operations)


class KnowledgeBaseDreamer:
    """Symbolic sleep-phase compression via anti-unification and MDL."""

    def __init__(self, min_group_size: int = 3,
                 shared_structure_threshold: float = 0.1):
        self.min_group_size = min_group_size
        self.shared_structure_threshold = shared_structure_threshold

    def dream(self, kb: KnowledgeBase, verify: bool = True) -> DreamSession:
        original_size = len(kb)
        if original_size == 0:
            return DreamSession(compressed=False, operations=[],
                                compression_ratio=1.0, verification=None)

        snapshot = kb.copy()
        suite = build_verification_suite(kb) if verify else None
        result = None
        ops: List[CompressionCandidate] = []

        ops.extend(self._eliminate_subsumed(kb))
        ops.extend(self._prune_redundant_facts(kb))
        # Operation C added in Task 7

        if verify and suite:
            from .evaluator import PrologEvaluator
            result = suite.verify(kb, lambda k: PrologEvaluator(k))
            if not result.passed:
                kb.restore_from(snapshot)
                return DreamSession(compressed=False, operations=[],
                                    compression_ratio=1.0,
                                    verification=result)

        new_size = len(kb)
        return DreamSession(
            compressed=new_size < original_size, operations=ops,
            compression_ratio=new_size / original_size,
            verification=result)

    def _eliminate_subsumed(self, kb: KnowledgeBase) -> List[CompressionCandidate]:
        """Operation A: Remove clauses subsumed by more general clauses."""
        ops = []

        # A1: Rule-vs-rule subsumption (same body length)
        rules_to_remove = set()
        rules = kb.rules
        for i, rule_a in enumerate(rules):
            for j, rule_b in enumerate(rules):
                if i == j or j in rules_to_remove:
                    continue
                if clause_subsumes(rule_a, rule_b) and rule_a != rule_b:
                    rules_to_remove.add(j)

        for idx in sorted(rules_to_remove, reverse=True):
            removed = rules[idx]
            kb.remove_rule_by_value(removed)
            ops.append(CompressionCandidate(
                operation="subsumption", original_clauses=[removed]))

        # A2: Bodyless-rule-vs-fact subsumption
        facts_to_remove = []
        bodyless_rules = [r for r in kb.rules if len(r.body) == 0]
        for fact in kb.facts:
            for rule in bodyless_rules:
                if subsumes(rule.head, fact.term):
                    facts_to_remove.append(fact)
                    break

        for fact in facts_to_remove:
            kb.remove_fact_by_value(fact)
            ops.append(CompressionCandidate(
                operation="subsumption", original_clauses=[fact]))

        return ops

    def _prune_redundant_facts(self, kb: KnowledgeBase) -> List[CompressionCandidate]:
        """Operation B: Remove facts derivable from remaining KB."""
        from .evaluator import PrologEvaluator

        ops = []
        facts = kb.facts

        # Phase 1: Find independently redundant facts
        redundant = []
        for fact in facts:
            kb.remove_fact_by_value(fact)
            ev = PrologEvaluator(kb)
            if ev.has_solution(fact.term):
                redundant.append(fact)
            kb.add_fact(fact)

        if not redundant:
            return ops

        # Phase 2: Batch remove
        for fact in redundant:
            kb.remove_fact_by_value(fact)

        # Phase 3: Verify
        ev = PrologEvaluator(kb)
        still_ok = all(ev.has_solution(f.term) for f in redundant)

        if still_ok:
            for fact in redundant:
                ops.append(CompressionCandidate(
                    operation="pruning", original_clauses=[fact]))
            return ops

        # Phase 4: Fallback -- restore and remove one at a time
        for fact in redundant:
            kb.add_fact(fact)
        for fact in redundant:
            kb.remove_fact_by_value(fact)
            ev = PrologEvaluator(kb)
            if ev.has_solution(fact.term):
                ops.append(CompressionCandidate(
                    operation="pruning", original_clauses=[fact]))
            else:
                kb.add_fact(fact)

        return ops
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_sleep_cycle.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add dreamlog/kb_dreamer.py tests/test_sleep_cycle.py
git commit -m "Rewrite kb_dreamer with Op A, Op B, verification, and dream loop"
```

---

### Task 7: Sleep cycle -- Operation C (fact generalization with exceptions)

**Files:**
- Modify: `dreamlog/kb_dreamer.py`
- Modify: `tests/test_sleep_cycle.py`

- [ ] **Step 1: Write failing tests for Operation C**

```python
# Append to tests/test_sleep_cycle.py
class TestOperationC:
    def test_basic_generalization(self):
        kb = KnowledgeBase()
        for name in ["alice", "bob", "carol", "dave"]:
            kb.add_fact(compound("person", atom(name)))
        for name in ["alice", "bob", "carol"]:
            kb.add_fact(compound("likes", atom(name), atom("chocolate")))
        dreamer = KnowledgeBaseDreamer(min_group_size=3)
        session = dreamer.dream(kb, verify=True)
        assert session.compressed is True
        from dreamlog.evaluator import PrologEvaluator
        ev = PrologEvaluator(kb)
        for name in ["alice", "bob", "carol"]:
            assert ev.has_solution(compound("likes", atom(name), atom("chocolate")))
        assert not ev.has_solution(compound("likes", atom("dave"), atom("chocolate")))

    def test_mdl_rejects_small_group(self):
        kb = KnowledgeBase()
        for name in ["alice", "bob", "carol"]:
            kb.add_fact(compound("person", atom(name)))
        kb.add_fact(compound("likes", atom("alice"), atom("chocolate")))
        kb.add_fact(compound("likes", atom("bob"), atom("chocolate")))
        dreamer = KnowledgeBaseDreamer(min_group_size=3)
        dreamer.dream(kb, verify=False)
        likes_facts = [f for f in kb.facts
                       if hasattr(f.term, 'functor') and f.term.functor == "likes"]
        assert len(likes_facts) == 2

    def test_multi_variable_skipped(self):
        kb = KnowledgeBase()
        for a, b in [("a", "1"), ("b", "2"), ("c", "3")]:
            kb.add_fact(compound("f", atom(a), atom(b)))
        dreamer = KnowledgeBaseDreamer(min_group_size=3)
        dreamer.dream(kb, verify=False)
        assert len(kb.facts) == 3

    def test_no_guard_found_skipped(self):
        kb = KnowledgeBase()
        for name in ["alice", "bob", "carol"]:
            kb.add_fact(compound("likes", atom(name), atom("chocolate")))
        dreamer = KnowledgeBaseDreamer(min_group_size=3)
        dreamer.dream(kb, verify=False)
        likes_facts = [f for f in kb.facts
                       if hasattr(f.term, 'functor') and f.term.functor == "likes"]
        assert len(likes_facts) == 3

    def test_guard_selects_smallest_extension(self):
        """With two candidate guards, select the one with fewer exceptions."""
        kb = KnowledgeBase()
        # small_group has exactly the needed values (0 exceptions)
        for name in ["alice", "bob", "carol"]:
            kb.add_fact(compound("small_group", atom(name)))
        # big_group has extra values (more exceptions)
        for name in ["alice", "bob", "carol", "dave", "eve"]:
            kb.add_fact(compound("big_group", atom(name)))
        for name in ["alice", "bob", "carol"]:
            kb.add_fact(compound("likes", atom(name), atom("chocolate")))
        dreamer = KnowledgeBaseDreamer(min_group_size=3)
        dreamer.dream(kb, verify=False)
        # Should have used small_group (0 exceptions) not big_group (2 exceptions)
        exc_facts = [f for f in kb.facts
                     if hasattr(f.term, 'functor')
                     and f.term.functor.startswith("exception_")]
        assert len(exc_facts) == 0  # 0 exceptions with small_group

    def test_exception_predicates_excluded(self):
        """Exception predicates from previous cycle are not generalized."""
        kb = KnowledgeBase()
        for name in ["alice", "bob", "carol", "dave"]:
            kb.add_fact(compound("person", atom(name)))
        for name in ["alice", "bob", "carol"]:
            kb.add_fact(compound("likes", atom(name), atom("chocolate")))
        dreamer = KnowledgeBaseDreamer(min_group_size=3)
        dreamer.dream(kb, verify=True)
        # Run again -- exception predicates should not be touched
        size_after = len(kb)
        dreamer.dream(kb, verify=True)
        assert len(kb) == size_after

    def test_idempotent(self):
        kb = KnowledgeBase()
        for name in ["alice", "bob", "carol", "dave"]:
            kb.add_fact(compound("person", atom(name)))
        for name in ["alice", "bob", "carol"]:
            kb.add_fact(compound("likes", atom(name), atom("chocolate")))
        dreamer = KnowledgeBaseDreamer(min_group_size=3)
        dreamer.dream(kb, verify=True)
        size_after_first = len(kb)
        dreamer.dream(kb, verify=True)
        assert len(kb) == size_after_first
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_sleep_cycle.py::TestOperationC -v`
Expected: FAIL

- [ ] **Step 3: Implement Operation C**

Add `_generalize_facts` and `_find_guard` methods to `KnowledgeBaseDreamer`,
and wire into `dream()` by replacing the `# Operation C added in Task 7`
comment with:

```python
ops.extend(self._generalize_facts(kb, suite))
```

Methods to add:

```python
def _generalize_facts(self, kb: KnowledgeBase,
                      suite: Optional[VerificationSuite] = None
                      ) -> List[CompressionCandidate]:
    """Operation C: Generalize fact groups into rules with exceptions."""
    from .anti_unification import anti_unify_many
    from .evaluator import PrologEvaluator

    ops = []
    groups = {}
    for fact in kb.facts:
        term = fact.term
        if isinstance(term, Compound):
            key = (term.functor, term.arity)
            groups.setdefault(key, []).append(fact)

    for (functor, arity), facts in groups.items():
        if functor.startswith("exception_"):
            continue
        if len(facts) < self.min_group_size:
            continue

        terms = [f.term for f in facts]
        au_result = anti_unify_many(terms)

        if au_result.shared_structure < self.shared_structure_threshold:
            continue
        if au_result.variables_introduced != 1:
            continue

        gen = au_result.generalized
        if not isinstance(gen, Compound):
            continue
        var_pos = None
        for i, arg in enumerate(gen.args):
            if isinstance(arg, Variable):
                var_pos = i
                break
        if var_pos is None:
            continue

        fact_values = set()
        for term in terms:
            arg = term.args[var_pos]
            if isinstance(arg, Atom):
                fact_values.add(arg.value)

        guard = self._find_guard(kb, functor, fact_values)
        if guard is None:
            continue

        guard_functor, guard_values = guard
        exceptions = guard_values - fact_values
        cost_before = len(facts)
        cost_after = 1 + len(exceptions)
        if cost_after >= cost_before:
            continue

        exception_functor = f"exception_{functor}_{guard_functor}"
        new_clauses: List[Union[Fact, Rule]] = []

        rule_var = Variable("X")
        rule_args = list(gen.args)
        rule_args[var_pos] = rule_var
        rule_head = Compound(functor, rule_args)
        rule_body = [
            Compound(guard_functor, [rule_var]),
            Compound("not", [Compound(exception_functor, [rule_var])]),
        ]
        new_clauses.append(Rule(rule_head, rule_body))

        for exc_val in sorted(exceptions):
            new_clauses.append(
                Fact(Compound(exception_functor, [Atom(exc_val)])))

        if suite is not None:
            test_kb = kb.copy()
            test_kb.replace_facts(facts, new_clauses)
            result = suite.verify(test_kb, lambda k: PrologEvaluator(k))
            if not result.passed:
                continue

        kb.replace_facts(facts, new_clauses)
        ops.append(CompressionCandidate(
            operation="generalization",
            original_clauses=list(facts),
            new_clauses=list(new_clauses)))

    return ops

def _find_guard(self, kb: KnowledgeBase, functor: str,
                needed_values: set) -> Optional[tuple]:
    """Find a unary guard predicate whose extension covers needed_values."""
    candidates = []
    unary_groups: dict = {}
    for fact in kb.facts:
        term = fact.term
        if (isinstance(term, Compound) and term.arity == 1
                and isinstance(term.args[0], Atom)):
            f = term.functor
            if f == functor or f.startswith("exception_"):
                continue
            unary_groups.setdefault(f, set()).add(term.args[0].value)

    for guard_f, guard_vals in unary_groups.items():
        if needed_values.issubset(guard_vals):
            excess = len(guard_vals - needed_values)
            candidates.append((guard_f, guard_vals, excess))

    if not candidates:
        return None
    candidates.sort(key=lambda c: c[2])
    return (candidates[0][0], candidates[0][1])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_sleep_cycle.py -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -x -q`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add dreamlog/kb_dreamer.py tests/test_sleep_cycle.py
git commit -m "Add Operation C (fact generalization with exceptions) to sleep cycle"
```

---

### Task 8: Public exports and final integration

**Files:**
- Modify: `dreamlog/__init__.py`
- All files

- [ ] **Step 1: Add public exports**

Add to `dreamlog/__init__.py`:

```python
from .anti_unification import anti_unify, anti_unify_many, AntiUnificationResult
from .kb_dreamer import KnowledgeBaseDreamer, DreamSession
from .evaluator import FlounderingError
```

Add `anti_unify`, `anti_unify_many`, `AntiUnificationResult`,
`KnowledgeBaseDreamer`, `DreamSession`, `FlounderingError` to the `__all__` list.

- [ ] **Step 2: Run full test suite with coverage**

Run: `python -m pytest tests/ -v --cov=dreamlog --cov-report=term-missing`
Expected: All tests pass. Check coverage for `anti_unification.py`, `kb_dreamer.py`, and NAF code.

- [ ] **Step 3: Fix any coverage gaps**

Add targeted tests for uncovered branches if needed.

- [ ] **Step 4: End-to-end smoke test**

Run in Python:

```python
from dreamlog.knowledge import KnowledgeBase, Fact, Rule
from dreamlog.factories import atom, compound
from dreamlog.kb_dreamer import KnowledgeBaseDreamer

kb = KnowledgeBase()
for n in ["alice", "bob", "carol", "dave", "eve"]:
    kb.add_fact(compound("person", atom(n)))
for n in ["alice", "bob", "carol", "dave"]:
    kb.add_fact(compound("likes", atom(n), atom("chocolate")))
kb.add_fact(compound("likes", atom("eve"), atom("vanilla")))

print(f"Before: {len(kb)} clauses")
print(kb)

dreamer = KnowledgeBaseDreamer(min_group_size=3)
session = dreamer.dream(kb)
print(f"\nAfter: {len(kb)} clauses (ratio: {session.compression_ratio:.2f})")
print(kb)
print(f"Compressed: {session.compressed}, Removed: {session.clauses_removed}")
```

Expected: 11 clauses before, fewer after. The 4 `likes(..., chocolate)` facts
become 1 rule + 1 exception (`exception_likes_person(eve)`) = 2 clauses.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Complete sleep cycle first pass: anti-unification, NAF, compression"
```
