# Recursive Rule Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give DreamLog the ability to discover transitive-closure recursive rules (e.g. `ancestor`) from an unannotated fact base, and measure whether discovery is genuine (compression-driven) or LLM recall via an invented-vocabulary closure domain.

**Architecture:** A new symbolic sleep operation (Operation I) detects when a binary predicate `R` is the transitive closure of another binary predicate `B`, synthesizes the right-recursive definition, verifies it with the bounded evaluator, and replaces the closure facts with the two-clause rule. A flag gates it off by default (zero drift). A recursion-aware Op G prompt is the LLM route. EX27 compares both routes against a raw-LLM baseline on a canonical (`ancestor`) and an invented (`flux_reaches`) closure domain. The plan leads with a feasibility probe that is the go/no-go gate.

**Tech Stack:** Python 3.8+, pytest, DreamLog (`dreamlog/` package). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-06-03-recursive-rule-discovery-design.md`

---

## File Structure

| File | Responsibility | New/Modify |
|---|---|---|
| `tests/test_recursion_probe.py` | Go/no-go probe: evaluator + Op B handle right-recursion | Create (Task 1) |
| `dreamlog/recursive_discovery.py` | `transitive_closure()` helper + closure-pair detection | Create (Tasks 2, 3) |
| `tests/test_recursive_discovery.py` | Unit tests for the helper and Operation I | Create (Tasks 2, 3) |
| `dreamlog/kb_dreamer.py` | Add `_discover_recursion` method + flag wiring in `dream()` | Modify (Tasks 3, 4) |
| `tests/test_sleep_cycle.py` | Regression: flag off = no change; integration: flag on discovers recursion | Modify (Task 4) |
| `dreamlog/llm_prompt_templates.py` | Recursion-aware Op G prompt clause + toggle | Modify (Task 5) |
| `experiments/ex27_recursion.py` | Invented closure domain generator + 4-condition experiment | Create (Tasks 6, 7) |
| `experiments/experiment_registry.yaml` | Register EX27 | Modify (Task 7) |

**Design boundaries:** the pure closure math lives in `recursive_discovery.py` and is testable with plain sets (no KB). Operation I (the KB-mutating method) lives in `kb_dreamer.py` next to the other operations and calls the helper. The experiment and domain generator live entirely under `experiments/` and import the same shared helpers EX25b uses.

---

## Task 1: Feasibility probe (GO / NO-GO gate)

This task answers the one question that can kill the whole plan: does the SLD evaluator terminate on a right-recursive rule under the bounded evaluator, and does Operation B prune the closure facts? Build it first. If it fails, STOP and replan the evaluator before writing Operation I.

**Files:**
- Test: `tests/test_recursion_probe.py` (create)

- [ ] **Step 1: Write the probe test**

```python
# tests/test_recursion_probe.py
"""Feasibility probe for recursive rule discovery (go/no-go gate).

Confirms the evaluator terminates on a right-recursive transitive-closure
rule under the bounded evaluator, and that Operation B prunes the closure
facts once the recursive rule is present.
"""
from dreamlog.factories import atom, var, compound
from dreamlog.knowledge import Fact, Rule, KnowledgeBase
from dreamlog.evaluator import PrologEvaluator
from dreamlog.kb_dreamer import KnowledgeBaseDreamer


def _chain_kb():
    """parent(a,b), parent(b,c), parent(c,d) + the ancestor closure."""
    kb = KnowledgeBase()
    edges = [("a", "b"), ("b", "c"), ("c", "d")]
    for x, y in edges:
        kb.add_fact(Fact(compound("parent", atom(x), atom(y))))
    closure = [("a", "b"), ("a", "c"), ("a", "d"),
               ("b", "c"), ("b", "d"), ("c", "d")]
    for x, y in closure:
        kb.add_fact(Fact(compound("ancestor", atom(x), atom(y))))
    return kb


def _ancestor_rules():
    X, Y, Z = var("X"), var("Y"), var("Z")
    base = Rule(compound("ancestor", X, Y), [compound("parent", X, Y)])
    rec = Rule(compound("ancestor", X, Z),
               [compound("parent", X, Y), compound("ancestor", Y, Z)])
    return base, rec


def test_right_recursion_terminates_and_is_sound():
    kb = _chain_kb()
    # Remove the stored ancestor facts so derivation must use the rules
    for f in [f for f in kb.facts if f.term.functor == "ancestor"]:
        kb.remove_fact_by_value(f)
    base, rec = _ancestor_rules()
    kb.add_rule(base)
    kb.add_rule(rec)

    ev = PrologEvaluator(kb, max_total_calls=10000)
    # Reachable pair derivable, no RecursionError
    assert ev.has_solution(compound("ancestor", atom("a"), atom("d"))) is True
    # Non-reachable pair NOT derivable (sound, terminates)
    ev2 = PrologEvaluator(kb, max_total_calls=10000)
    assert ev2.has_solution(compound("ancestor", atom("d"), atom("a"))) is False


def test_op_b_prunes_closure_facts_under_recursion():
    kb = _chain_kb()
    base, rec = _ancestor_rules()
    kb.add_rule(base)
    kb.add_rule(rec)
    n_ancestor_before = sum(1 for f in kb.facts if f.term.functor == "ancestor")
    assert n_ancestor_before == 6

    dreamer = KnowledgeBaseDreamer()
    ops = dreamer._prune_redundant_facts(kb, max_calls=10000)

    n_ancestor_after = sum(1 for f in kb.facts if f.term.functor == "ancestor")
    assert n_ancestor_after == 0, "Op B should prune all ancestor facts"
    # Still derivable via the recursive rule
    ev = PrologEvaluator(kb, max_total_calls=10000)
    assert ev.has_solution(compound("ancestor", atom("a"), atom("d"))) is True
    assert len(ops) >= 1
```

- [ ] **Step 2: Run the probe**

Run: `python -m pytest tests/test_recursion_probe.py -v --no-cov`
Expected (GO): both tests PASS.
Expected (NO-GO): a `RecursionError` escapes, or `test_op_b_prunes...` leaves ancestor facts, or the non-reachable pair returns True. If NO-GO, STOP: the evaluator needs a recursion/depth guard (a separate spec). Record which assertion failed and how, then replan.

- [ ] **Step 3: Commit the probe**

```bash
git add tests/test_recursion_probe.py
git commit -m "test: ancestor recursion feasibility probe (go/no-go gate)"
```

---

## Task 2: `transitive_closure` helper

Pure set function, no KB. This is the math Operation I uses to decide whether `R == TC(B)`.

**Files:**
- Create: `dreamlog/recursive_discovery.py`
- Test: `tests/test_recursive_discovery.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_recursive_discovery.py
from dreamlog.recursive_discovery import transitive_closure


def test_tc_on_a_chain():
    edges = {("a", "b"), ("b", "c"), ("c", "d")}
    assert transitive_closure(edges) == {
        ("a", "b"), ("a", "c"), ("a", "d"),
        ("b", "c"), ("b", "d"), ("c", "d"),
    }


def test_tc_on_a_branch():
    edges = {("r", "x"), ("r", "y"), ("x", "z")}
    assert transitive_closure(edges) == {
        ("r", "x"), ("r", "y"), ("x", "z"), ("r", "z"),
    }


def test_tc_empty():
    assert transitive_closure(set()) == set()


def test_tc_is_irreflexive_on_a_dag():
    edges = {("a", "b"), ("b", "c")}
    tc = transitive_closure(edges)
    assert not any(x == y for x, y in tc)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_recursive_discovery.py -v --no-cov`
Expected: FAIL with `ModuleNotFoundError: No module named 'dreamlog.recursive_discovery'`.

- [ ] **Step 3: Write the helper**

```python
# dreamlog/recursive_discovery.py
"""Recursive rule discovery (Operation I): detect transitive-closure
relationships between binary predicates and synthesize right-recursive rules.

The pure closure math lives here so it can be tested without a knowledge base.
"""
from collections import defaultdict, deque
from typing import Set, Tuple, Hashable

Pair = Tuple[Hashable, Hashable]


def transitive_closure(edges: Set[Pair]) -> Set[Pair]:
    """Irreflexive transitive closure of a binary relation given as edge pairs.

    edges: set of (a, b) value pairs. Returns every (a, b) such that there is
    a non-empty directed path a -> ... -> b. On a DAG the result is
    irreflexive; on a cyclic relation a node reachable from itself yields (x, x).
    """
    adj = defaultdict(set)
    nodes = set()
    for a, b in edges:
        adj[a].add(b)
        nodes.add(a)
        nodes.add(b)

    closure: Set[Pair] = set()
    for start in nodes:
        seen = set()
        queue = deque(adj[start])
        while queue:
            node = queue.popleft()
            if node in seen:
                continue
            seen.add(node)
            closure.add((start, node))
            queue.extend(adj[node])
    return closure
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_recursive_discovery.py -v --no-cov`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add dreamlog/recursive_discovery.py tests/test_recursive_discovery.py
git commit -m "feat: transitive_closure helper for recursion discovery"
```

---

## Task 3: Operation I — closure detection, synthesis, verify, replace

The KB-mutating operation. It lives in `kb_dreamer.py` next to the other operations and follows the Op C idiom (build candidate, verify against the suite with a bounded evaluator, `replace_facts`).

**Files:**
- Modify: `dreamlog/kb_dreamer.py` (add a `min_base_facts` field to `__init__` near line 290; add the `_discover_recursion` method after `_extract_body_patterns`, near line 738)
- Test: `tests/test_recursive_discovery.py` (add)

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_recursive_discovery.py
from dreamlog.factories import atom, var, compound
from dreamlog.knowledge import Fact, KnowledgeBase
from dreamlog.kb_dreamer import KnowledgeBaseDreamer, build_verification_suite


def _closure_kb():
    kb = KnowledgeBase()
    edges = [("a", "b"), ("b", "c"), ("c", "d")]
    for x, y in edges:
        kb.add_fact(Fact(compound("parent", atom(x), atom(y))))
    for x, y in [("a", "b"), ("a", "c"), ("a", "d"),
                 ("b", "c"), ("b", "d"), ("c", "d")]:
        kb.add_fact(Fact(compound("ancestor", atom(x), atom(y))))
    return kb


def test_op_i_discovers_ancestor_closure():
    kb = _closure_kb()
    suite = build_verification_suite(kb)
    dreamer = KnowledgeBaseDreamer(min_base_facts=3)

    ops = dreamer._discover_recursion(kb, suite)

    assert len(ops) == 1
    assert ops[0].operation == "recursion"
    # The 6 ancestor facts were replaced by 2 rules
    assert ops[0].mdl_delta == 2 - 6
    # Two ancestor rules are now in the KB, no ancestor facts remain
    anc_rules = [r for r in kb.rules if r.head.functor == "ancestor"]
    assert len(anc_rules) == 2
    assert not any(f.term.functor == "ancestor" for f in kb.facts)


def test_op_i_no_false_discovery_when_not_a_closure():
    kb = KnowledgeBase()
    for x, y in [("a", "b"), ("b", "c"), ("c", "d")]:
        kb.add_fact(Fact(compound("parent", atom(x), atom(y))))
    # "likes" is unrelated, not a closure of parent
    for x, y in [("a", "c"), ("d", "a")]:
        kb.add_fact(Fact(compound("likes", atom(x), atom(y))))
    suite = build_verification_suite(kb)
    dreamer = KnowledgeBaseDreamer(min_base_facts=3)

    ops = dreamer._discover_recursion(kb, suite)
    assert ops == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_recursive_discovery.py -v --no-cov`
Expected: FAIL with `AttributeError: 'KnowledgeBaseDreamer' object has no attribute 'min_base_facts'` (or `_discover_recursion`).

- [ ] **Step 3: Add the `min_base_facts` field to `__init__`**

In `dreamlog/kb_dreamer.py`, the `KnowledgeBaseDreamer.__init__` signature (near line 290) currently ends with `open_world: bool = False`. Add two parameters and store them:

```python
    def __init__(self, min_group_size: int = 3,
                 shared_structure_threshold: float = 0.1,
                 llm_client=None,
                 max_prompt_facts: int = 50,
                 open_world: bool = False,
                 discover_recursion: bool = False,
                 min_base_facts: int = 3):
        self.min_group_size = min_group_size
        self.shared_structure_threshold = shared_structure_threshold
        self.llm_client = llm_client
        self.max_prompt_facts = max_prompt_facts
        self.open_world = open_world
        # Operation I (recursive closure discovery): off by default so the
        # standard compression pipeline is unchanged (zero drift).
        self.discover_recursion = discover_recursion
        self.min_base_facts = min_base_facts
```

- [ ] **Step 4: Add the `_discover_recursion` method**

Add this method to `KnowledgeBaseDreamer`, after `_extract_body_patterns` (which ends near line 738) and before `_find_best_body_pattern`. It imports the helper at the top of the method to avoid touching the module import block:

```python
    def _discover_recursion(self, kb: KnowledgeBase,
                            suite: Optional['VerificationSuite'] = None,
                            max_calls: int = 5000
                            ) -> List[CompressionCandidate]:
        """Operation I: discover transitive-closure recursive rules.

        For each binary predicate R whose ground extension equals the
        transitive closure of another binary predicate B, synthesize the
        right-recursive definition (base + recursive case), verify it against
        the suite with a bounded evaluator, and replace R's facts with the
        two rules. Returns at most one candidate per call (re-run on the next
        dream cycle to find further closures).
        """
        from .recursive_discovery import transitive_closure

        # Collect binary predicate extensions over Atom-only argument pairs.
        ext: dict = {}
        facts_by_pred: dict = {}
        for fact in kb.facts:
            t = fact.term
            if (isinstance(t, Compound) and t.arity == 2
                    and isinstance(t.args[0], Atom)
                    and isinstance(t.args[1], Atom)):
                ext.setdefault(t.functor, set()).add(
                    (t.args[0].value, t.args[1].value))
                facts_by_pred.setdefault(t.functor, []).append(fact)

        for R, r_ext in ext.items():
            if _is_system_predicate(R):
                continue
            for B, b_ext in ext.items():
                if R == B or len(b_ext) < self.min_base_facts:
                    continue
                if r_ext != transitive_closure(b_ext):
                    continue

                X, Y, Z = Variable("X"), Variable("Y"), Variable("Z")
                base_rule = Rule(Compound(R, [X, Y]), [Compound(B, [X, Y])])
                rec_rule = Rule(
                    Compound(R, [X, Z]),
                    [Compound(B, [X, Y]), Compound(R, [Y, Z])])
                r_facts = facts_by_pred[R]
                new_clauses: List[Union[Fact, Rule]] = [base_rule, rec_rule]

                # Verify with a bounded evaluator (recursion must terminate).
                if suite is not None:
                    test_kb = kb.copy()
                    test_kb.replace_facts(r_facts, new_clauses)
                    try:
                        result = suite.verify(
                            test_kb,
                            lambda k: PrologEvaluator(k, max_total_calls=max_calls))
                    except RecursionError:
                        continue
                    if not result.passed:
                        continue

                kb.replace_facts(r_facts, new_clauses)
                return [CompressionCandidate(
                    operation="recursion",
                    original_clauses=list(r_facts),
                    new_clauses=list(new_clauses))]

        return []
```

Note: `Variable`, `Compound`, `Atom` are already imported at the top of `kb_dreamer.py` (used by the existing operations). Confirm with `grep -n "from .terms import" dreamlog/kb_dreamer.py` before running; if `Variable` is missing from that import, add it.

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/test_recursive_discovery.py -v --no-cov`
Expected: all tests pass (the helper tests from Task 2 plus the two new Op I tests).

- [ ] **Step 6: Commit**

```bash
git add dreamlog/kb_dreamer.py tests/test_recursive_discovery.py
git commit -m "feat: Operation I closure detection and right-recursive synthesis"
```

---

## Task 4: Wire Operation I into `dream()` behind the flag

Operation I must run in the symbolic phase after Op E and before the LLM step, and only when `discover_recursion` is True (zero drift otherwise).

**Files:**
- Modify: `dreamlog/kb_dreamer.py` (`dream()` method, after the `_extract_body_patterns` call near line 348)
- Test: `tests/test_sleep_cycle.py` (add regression + integration tests)

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_sleep_cycle.py
from dreamlog.factories import atom, var, compound
from dreamlog.knowledge import Fact, KnowledgeBase
from dreamlog.kb_dreamer import KnowledgeBaseDreamer


def _ancestor_closure_kb():
    kb = KnowledgeBase()
    for x, y in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")]:
        kb.add_fact(Fact(compound("parent", atom(x), atom(y))))
    # full ancestor closure of the 4-edge chain
    nodes = ["a", "b", "c", "d", "e"]
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            kb.add_fact(Fact(compound("ancestor", atom(nodes[i]), atom(nodes[j]))))
    return kb


def test_dream_flag_off_does_not_discover_recursion():
    kb = _ancestor_closure_kb()
    n_before = len(kb)
    dreamer = KnowledgeBaseDreamer(discover_recursion=False)
    dreamer.dream(kb)
    # No ancestor RULE was created (recursion off)
    assert not any(len(r.body) > 0 and r.head.functor == "ancestor"
                   for r in kb.rules)


def test_dream_flag_on_discovers_recursion_and_compresses():
    kb = _ancestor_closure_kb()
    n_before = len(kb)
    dreamer = KnowledgeBaseDreamer(discover_recursion=True, min_base_facts=3)
    session = dreamer.dream(kb)
    # ancestor is now defined by rules, the closure facts are pruned
    anc_rules = [r for r in kb.rules
                 if r.head.functor == "ancestor" and len(r.body) > 0]
    assert len(anc_rules) == 2
    assert not any(f.term.functor == "ancestor" for f in kb.facts)
    assert len(kb) < n_before
```

- [ ] **Step 2: Run to verify the integration test fails**

Run: `python -m pytest tests/test_sleep_cycle.py -k recursion -v --no-cov`
Expected: `test_dream_flag_off...` PASSES already (flag does nothing yet), `test_dream_flag_on...` FAILS (no ancestor rules created).

- [ ] **Step 3: Add the wiring in `dream()`**

In `dreamlog/kb_dreamer.py`, in `dream()`, the symbolic phase currently reads (near lines 347 to 351):

```python
        ops.extend(self._invent_predicates(kb, suite))
        ops.extend(self._extract_body_patterns(kb, suite))

        # LLM-assisted naming (after symbolic ops, before final verify)
        self._name_invented_predicates(kb)
```

Insert the Operation I call between `_extract_body_patterns` and `_name_invented_predicates`:

```python
        ops.extend(self._invent_predicates(kb, suite))
        ops.extend(self._extract_body_patterns(kb, suite))

        # Operation I: recursive closure discovery (flag-gated, off by default).
        # Runs in the symbolic phase so symbolic-only and full-pipeline differ
        # only by Operation G.
        if self.discover_recursion:
            ops.extend(self._discover_recursion(kb, suite))

        # LLM-assisted naming (after symbolic ops, before final verify)
        self._name_invented_predicates(kb)
```

- [ ] **Step 4: Run to verify both tests pass**

Run: `python -m pytest tests/test_sleep_cycle.py -k recursion -v --no-cov`
Expected: both PASS.

- [ ] **Step 5: Run the full suite (regression)**

Run: `python -m pytest tests/ -q --no-cov`
Expected: 673 passed (671 prior + 2 new in sleep_cycle), plus the recursive_discovery and probe tests; 11 skipped. No prior test regresses (Op I is off by default).

- [ ] **Step 6: Commit**

```bash
git add dreamlog/kb_dreamer.py tests/test_sleep_cycle.py
git commit -m "feat: wire Operation I into dream() behind discover_recursion flag"
```

---

## Task 5: Recursion-aware Operation G prompt (the LLM route)

The symbolic route (Tasks 2 to 4) is the rigorous core; this adds the LLM proposer route for the full-pipeline condition. The cycle filter already permits self-recursion, so only the prompt text and a toggle are needed.

**Files:**
- Modify: `dreamlog/llm_prompt_templates.py` (locate the Op G prompt; the builder is invoked from `_llm_compress` at `dreamlog/kb_dreamer.py:1048`)
- Test: `tests/test_sleep_cycle.py` (add, using `MockLLMProvider` from `tests/mock_provider.py`)

- [ ] **Step 1: Read the Op G prompt builder**

Run: `grep -n "def .*prompt\|recursion\|transitive\|:- " dreamlog/llm_prompt_templates.py` and read `_llm_compress` at `dreamlog/kb_dreamer.py:1048-1207`. Identify the exact prompt string the LLM receives for rule proposal. This is required before editing so the recursion clause lands in the real template.

- [ ] **Step 2: Write the failing test**

```python
# add to tests/test_sleep_cycle.py
from tests.mock_provider import MockLLMProvider
from dreamlog.llm_client import LLMClient  # confirm import path during Step 1


def test_op_g_accepts_a_recursive_proposal():
    """A mock LLM that proposes a base+recursive ancestor pair gets accepted."""
    kb = _ancestor_closure_kb()  # defined in Task 4
    # Mock returns a recursive definition in the JSON shape Op G parses.
    # Confirm the exact response schema by reading _parse_llm_rules
    # (kb_dreamer.py:1270) during Step 1.
    mock = MockLLMProvider(responses=[
        '{"rules": ['
        '{"head": ["ancestor", "X", "Y"], "body": [["parent", "X", "Y"]]},'
        '{"head": ["ancestor", "X", "Z"], "body": ['
        '["parent", "X", "Y"], ["ancestor", "Y", "Z"]]}'
        ']}'
    ])
    client = LLMClient(provider=mock)
    dreamer = KnowledgeBaseDreamer(llm_client=client, discover_recursion=False)
    dreamer.dream(kb)
    anc_rules = [r for r in kb.rules
                 if r.head.functor == "ancestor" and len(r.body) > 0]
    assert len(anc_rules) >= 1
```

Note: the exact `MockLLMProvider` constructor and the JSON schema are confirmed in Step 1 by reading `tests/mock_provider.py` and `_parse_llm_rules`. Adjust the response string to match the real schema before running.

- [ ] **Step 3: Run to verify it fails**

Run: `python -m pytest tests/test_sleep_cycle.py -k recursive_proposal -v --no-cov`
Expected: FAIL (either the mock proposal is filtered, or the prompt never invites recursion so the schema check rejects it). If it already passes, the machinery accepts recursion as-is and only the prompt clause (Step 4) is needed to elicit it in practice.

- [ ] **Step 4: Add the recursion clause to the Op G prompt**

In the Op G prompt builder identified in Step 1, append one instruction inviting recursive definitions, with a right-recursive example. The exact insertion point is the prompt string; the added text is:

```text
If a relation appears to be the transitive closure of another relation
(its facts are exactly the reachable pairs over a base relation), propose
BOTH a base rule and a right-recursive rule, for example:
  ancestor(X, Y) :- parent(X, Y).
  ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
Always provide the base case. Never write a left-recursive body
(do not put the recursive call first).
```

Gate it behind the existing prompt path (no new flag needed: recursion proposals are always permitted by the cycle filter; this clause only improves elicitation). If a toggle is desired for ablation, add `recursion_prompt: bool = True` to `__init__` and branch on it where the clause is appended.

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/test_sleep_cycle.py -k recursive_proposal -v --no-cov`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add dreamlog/llm_prompt_templates.py tests/test_sleep_cycle.py
git commit -m "feat: recursion-aware Operation G prompt with right-recursion guidance"
```

---

## Task 6: Invented-vocabulary closure domain generator

Deterministic generator for the `flux_links` / `flux_reaches` domain, parallel to the crafting generator in `ex25b_novel_generalization.py`. Lives in the experiment file.

**Files:**
- Create: `experiments/ex27_recursion.py` (generator portion)
- Test: add a quick generator test to `tests/test_recursive_discovery.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_recursive_discovery.py
import importlib.util, pathlib

def _load_ex27():
    path = pathlib.Path(__file__).parent.parent / "experiments" / "ex27_recursion.py"
    spec = importlib.util.spec_from_file_location("ex27_recursion", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_flux_domain_reaches_is_closure_of_links():
    ex27 = _load_ex27()
    base, derived = ex27.flux_domain(n_nodes=8, seed=42)
    # base facts are flux_links, derived are flux_reaches
    links = {tuple(s.split()[1:]) for s in base}      # ("flux_links a b")
    reaches = {tuple(s.split()[1:]) for s in derived}
    from dreamlog.recursive_discovery import transitive_closure
    assert reaches == transitive_closure(links)
    assert len(links) >= 3
```

The fact-string format (`"flux_links qux vor"`) must match what `build_kb` (imported from `ex25_generalization.py`) parses. Confirm the exact string format by reading `crafting_base_facts()` in `ex25b_novel_generalization.py` during this step.

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_recursive_discovery.py -k flux_domain -v --no-cov`
Expected: FAIL (no `experiments/ex27_recursion.py`).

- [ ] **Step 3: Write the generator**

```python
# experiments/ex27_recursion.py  (generator portion)
"""EX27: recursive rule discovery on a canonical (ancestor) and an
invented-vocabulary (flux_reaches) transitive-closure domain."""
import random
from typing import List, Tuple

# Invented node names the LLM has not seen as graph nodes.
_INVENTED_NODES = [
    "qux", "vor", "zane", "plix", "drub", "yent", "kosh", "wimple",
    "fren", "glorb", "snee", "thock", "vung", "blee", "morx", "quail",
]


def flux_domain(n_nodes: int = 10, n_extra_edges: int = 3, seed: int = 42
                ) -> Tuple[List[str], List[str]]:
    """Return (base_facts, derived_facts) for the invented closure domain.

    base_facts: flux_links edges forming a random DAG over invented nodes.
    derived_facts: flux_reaches = transitive closure of flux_links.
    """
    from dreamlog.recursive_discovery import transitive_closure
    rng = random.Random(seed)
    nodes = _INVENTED_NODES[:n_nodes]

    # Random DAG: only edges from earlier to later in a shuffled order.
    order = nodes[:]
    rng.shuffle(order)
    edges = set()
    # a spanning chain guarantees a non-trivial closure
    for i in range(len(order) - 1):
        edges.add((order[i], order[i + 1]))
    # a few extra forward edges
    for _ in range(n_extra_edges):
        i = rng.randrange(0, len(order) - 1)
        j = rng.randrange(i + 1, len(order))
        edges.add((order[i], order[j]))

    closure = transitive_closure(edges)
    base = [f"flux_links {a} {b}" for a, b in sorted(edges)]
    derived = [f"flux_reaches {a} {b}" for a, b in sorted(closure)]
    return base, derived
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_recursive_discovery.py -k flux_domain -v --no-cov`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/ex27_recursion.py tests/test_recursive_discovery.py
git commit -m "feat: EX27 invented-vocabulary flux_reaches closure domain generator"
```

---

## Task 7: EX27 experiment (four conditions) + registry

Add the experiment driver to `experiments/ex27_recursion.py`, reusing the EX25 shared helpers (`build_kb`, `is_derivable`, `dream_kb`, `holdout_split`, `get_llm_client`) the same way `ex25b_novel_generalization.py` does, and the `run_raw_llm_check` helper for the raw-LLM condition.

**Files:**
- Modify: `experiments/ex27_recursion.py` (add the driver)
- Modify: `experiments/experiment_registry.yaml` (register EX27)

- [ ] **Step 1: Read the EX25b driver pattern**

Read `experiments/ex25b_novel_generalization.py` lines 354 to 460: the imports from `ex25_generalization`, `run_raw_llm_check`, `run_domain_test`, and the four-condition loop. The EX27 driver mirrors `run_domain_test` but: (a) dreams with `discover_recursion=True` for the symbolic and full conditions, (b) holds out a fraction of `flux_reaches` / `ancestor` facts and measures recovery (reuse the EX25c holdout helper), (c) reports recovery and precision per condition.

- [ ] **Step 2: Write the driver**

```python
# append to experiments/ex27_recursion.py
import argparse, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ex25_generalization import (  # noqa: E402
    build_kb, is_derivable, dream_kb, holdout_split, get_llm_client,
)
from ex25b_novel_generalization import run_raw_llm_check  # noqa: E402


def family_domain():
    """Canonical ancestor domain: returns (base_facts, derived_facts)."""
    # Reuse the EX25 family parent facts; ancestor closure as derived.
    # Confirm the exact family fact accessors in ex25_generalization.py
    # during Step 1 (e.g. family_base_facts(), family_ancestor_facts()).
    from ex25_generalization import family_base_facts
    base = [f for f in family_base_facts() if f.startswith("parent ")]
    links = {tuple(f.split()[1:]) for f in base}
    from dreamlog.recursive_discovery import transitive_closure
    derived = [f"ancestor {a} {b}" for a, b in sorted(transitive_closure(links))]
    return base, derived


def run_recursion_domain(name, base, derived, target_functor,
                         llm_client, holdout=0.2, seed=42):
    """Four conditions; measure recovery of held-out closure facts."""
    train_derived, held = holdout_split(derived, ratio=holdout, seed=seed)
    results = {}

    # no_dream
    kb = build_kb(base + train_derived)
    results["no_dream"] = _recovery(kb, held, max_calls=5000)

    # symbolic (Operation I, no LLM)
    kb = build_kb(base + train_derived)
    dream_kb(kb, llm_client=None, discover_recursion=True, open_world=True)
    results["symbolic"] = _recovery(kb, held, max_calls=5000)

    # full (Operation I + recursion-aware Op G)
    kb = build_kb(base + train_derived)
    dream_kb(kb, llm_client=llm_client, discover_recursion=True, open_world=True)
    results["full"] = _recovery(kb, held, max_calls=5000)

    # raw_llm baseline (LLM as inference engine on held-out queries)
    results["raw_llm"] = _raw_llm_recovery(llm_client, base + train_derived, held)
    return results


def _recovery(kb, held_facts, max_calls=5000):
    from dreamlog.evaluator import PrologEvaluator
    from dreamlog.prefix_parser import parse_s_expression  # confirm parser name
    recovered = 0
    for fact_str in held_facts:
        term = _term_from_fact_string(fact_str)
        ev = PrologEvaluator(kb, max_total_calls=max_calls)
        try:
            if ev.has_solution(term):
                recovered += 1
        except RecursionError:
            pass
    return {"recovered": recovered, "total": len(held_facts),
            "recall": recovered / len(held_facts) if held_facts else 0.0}
```

Note: `dream_kb` in `ex25_generalization.py` must accept and forward `discover_recursion` to `KnowledgeBaseDreamer`. If it does not, add the keyword (one-line passthrough) as part of this task and note it in the commit. `_term_from_fact_string` and `_raw_llm_recovery` are small helpers: confirm the fact-string-to-term path used by `build_kb` (likely `parse_s_expression` or `term_from_prefix`) in Step 1 and reuse it.

- [ ] **Step 3: Register EX27**

Add to `experiments/experiment_registry.yaml`, matching the existing entry shape (see the EX26 entry):

```yaml
EX27:
  title: "Recursive rule discovery (transitive closure) on canonical and invented domains"
  date: 2026-06-XX        # set to the run date
  script: experiments/ex27_recursion.py
  status: planned
  motivation: >
    Test whether DreamLog can discover recursive (transitive-closure) rules
    such as ancestor from an unannotated fact base, and whether the
    invented-vocabulary flux_reaches domain isolates genuine compression-driven
    discovery from LLM recall of textbook recursion.
  method: >
    Four conditions (no_dream, symbolic with Operation I, full with Operation I
    plus recursion-aware Op G, raw_llm). Hold out 20 percent of closure facts;
    measure recovery and precision on the family (ancestor) and invented
    (flux_reaches) domains. Open-world verification for the holdout, right-
    recursion only.
  key_result: "TBD after run"
  implications: "TBD after run"
  depends_on: [EX25, EX25b, EX25c]
```

- [ ] **Step 4: Verify the experiment imports and a dry run executes**

Run: `python experiments/ex27_recursion.py --help`
Expected: argparse help prints with no import errors. Fix any helper-name mismatches surfaced here against the real `ex25_generalization.py` API.

- [ ] **Step 5: Commit**

```bash
git add experiments/ex27_recursion.py experiments/experiment_registry.yaml
git commit -m "feat: EX27 four-condition recursion experiment + registry entry"
```

---

## Task 8: Run EX27, record results, characterize the outcome

**Files:**
- Modify: `experiments/experiment_registry.yaml` (fill `key_result`, `implications`, set `status: complete`)

- [ ] **Step 1: Run the experiment**

Run: `python experiments/ex27_recursion.py --runs 3` (use `MY_ANTHROPIC_API_KEY`; expected LLM cost is a few cents). Capture the per-condition recovery and precision for both domains.

- [ ] **Step 2: Characterize against the three discriminating outcomes**

Using the spec's framing, classify the result:
1. symbolic recovers `flux_reaches` on invented vocab with no LLM (strong "compression alone discovers recursion"); or
2. only full recovers it (LLM structural prior transfers, symbolic verification makes it sound); or
3. both. In all cases confirm the raw_llm baseline fails to recover held-out closure facts on the invented domain.

- [ ] **Step 3: Record the result in the registry**

Fill `key_result` and `implications` in the EX27 entry with the measured numbers and which outcome held. Set `status: complete` and the real run `date`.

- [ ] **Step 4: Run the full suite one last time**

Run: `python -m pytest tests/ -q --no-cov`
Expected: all green (no regression).

- [ ] **Step 5: Commit**

```bash
git add experiments/experiment_registry.yaml
git commit -m "experiments: EX27 complete - recursion discovery results recorded"
```

---

## Self-Review

**Spec coverage:**
- Operation I closure detection + right-recursive synthesis: Tasks 2, 3. Covered.
- Pipeline position (after Op E, before Op G, flag-gated, off by default): Task 4. Covered.
- Recursion-aware Op G prompt + right-recursion-only guidance: Task 5. Covered.
- Verification with the bounded evaluator and open-world holdout reuse: Tasks 3 (bounded verify), 7 (open_world holdout). Covered.
- Canonical (ancestor) + invented (flux_reaches) domains, four conditions, recovery + precision: Tasks 6, 7. Covered.
- The ancestor feasibility probe as go/no-go: Task 1. Covered.
- MDL via clause count (2 clauses replace N facts): asserted in Task 3 (`mdl_delta == 2 - 6`). Covered.
- Tests: unit (Tasks 2, 3, 6), verification (Tasks 1, 3), integration (Task 4), regression (Task 4 Step 5). Covered.

**Placeholder scan:** Tasks 1 to 4 and 6 contain complete code. Tasks 5, 7, 8 contain real code plus explicit "confirm against the real API in Step 1" notes for the three points that depend on files not yet read by the plan author (the Op G prompt builder, the `MockLLMProvider` schema, and the `ex25_generalization` helper signatures). These are scoped read-then-edit steps, not vague placeholders: each names the exact file, function, and the edit to make.

**Type consistency:** `discover_recursion` and `min_base_facts` are added to `__init__` (Task 3 Step 3) and used in `dream()` (Task 4) and `_discover_recursion` (Task 3 Step 4) consistently. `CompressionCandidate(operation=..., original_clauses=..., new_clauses=...)` matches the dataclass at `kb_dreamer.py:127`. `replace_facts(old, new)`, `PrologEvaluator(k, max_total_calls=...)`, `suite.verify(kb, factory)`, and `has_solution(term)` all match the signatures read from the source.

**Known follow-ups for the implementer (not blockers):** `dream_kb` in `ex25_generalization.py` may need a one-line passthrough of `discover_recursion` (flagged in Task 7 Step 2). The fact-string format and parser are confirmed in Task 6 Step 1 and Task 7 Step 1.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-03-recursive-rule-discovery.md`. Two execution options:

1. **Subagent-Driven (recommended)** - dispatch a fresh subagent per task, review between tasks, fast iteration. Task 1 (the probe) is a natural first checkpoint: if it is NO-GO, we replan before any further work.
2. **Inline Execution** - execute tasks in this session with checkpoints for review.

Which approach?
