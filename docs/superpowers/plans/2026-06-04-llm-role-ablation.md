# LLM-Role Ablation (EX28) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build EX28, a resumable, metadata-rich ablation that isolates the LLM's marginal contribution by rule type (within-predicate / recursive / cross-predicate) across canonical and fully-invented vocabulary, run on the local `qwen2.5:3b` model.

**Architecture:** Reuse DreamLog's dream pipeline and the EX27 new-entity protocol. Add: a tiny Ollama client helper (the client already supports it via `base_url`), a structural rule-equivalence checker, three new clean domain generators, an Op-C disable flag, a factored-out Op G proposal step, a proposal-rate probe, and a resumable work-unit harness with append-only JSONL metadata.

**Tech Stack:** Python 3.8+, pytest, DreamLog (`dreamlog/`), the OpenAI-compatible Ollama endpoint at `192.168.0.204:11434`.

**Spec:** `docs/superpowers/specs/2026-06-04-llm-role-ablation-design.md`

**Branch:** `llm-role-ablation` (already checked out, off `master`).

---

## File Structure

| File | Responsibility | New/Modify |
|---|---|---|
| `experiments/ollama_helper.py` | build an `LLMClient` for the remote Ollama model | Create (Task 1) |
| `tests/test_ollama_helper.py` | connectivity probe (network-marked) | Create (Task 1) |
| `dreamlog/rule_equivalence.py` | `rules_structurally_equivalent` (variant up to renaming + body reorder) | Create (Task 2) |
| `tests/test_rule_equivalence.py` | equivalence unit tests | Create (Task 2) |
| `experiments/ex28_domains.py` | `Domain` dataclass + birds/glonk/wibble generators (+ reuse flux/ancestor/father) | Create (Task 3) |
| `tests/test_ex28_domains.py` | domain validity tests | Create (Task 3) |
| `dreamlog/kb_dreamer.py` | Op-C disable flag; factor out `_llm_propose` | Modify (Tasks 4, 5) |
| `tests/test_sleep_cycle.py` | Op-C-off regression; `_llm_propose` test | Modify (Tasks 4, 5) |
| `experiments/ex28_probe.py` | `proposal_rate(domain, client, n)` | Create (Task 6) |
| `tests/test_ex28_probe.py` | probe test with MockLLMProvider | Create (Task 6) |
| `experiments/ex28_harness.py` | resumable work-unit runner + JSONL metadata + manifest | Create (Task 7) |
| `tests/test_ex28_harness.py` | resume/skip-completed logic test | Create (Task 7) |
| `experiments/ex28_llm_role.py` | the EX28 experiment: 4 conditions x 6 cells, CLI | Create (Task 8) |
| `experiments/experiment_registry.yaml` | register EX28 | Modify (Task 8) |

---

## Task 1: Ollama client helper + connectivity probe (foundation gate)

Everything depends on reaching `qwen2.5:3b`. Build this first; if the probe cannot reach the model, STOP and resolve connectivity before continuing.

**Files:**
- Create: `experiments/ollama_helper.py`
- Test: `tests/test_ollama_helper.py`

- [ ] **Step 1: Write the helper**

```python
# experiments/ollama_helper.py
"""Build an LLMClient pointed at the remote Ollama model.

The Ollama server exposes an OpenAI-compatible endpoint at /v1, which
dreamlog.llm_client.LLMClient already supports via base_url. The timeout is
generous because the GPU is shared with another session, so calls can be slow.
"""
from dreamlog.llm_client import LLMClient

OLLAMA_HOST = "192.168.0.204"
OLLAMA_PORT = 11434
OLLAMA_MODEL = "qwen2.5:3b"


def make_ollama_client(model: str = OLLAMA_MODEL, host: str = OLLAMA_HOST,
                       port: int = OLLAMA_PORT, temperature: float = 0.3,
                       max_tokens: int = 800, timeout: int = 180) -> LLMClient:
    """LLMClient for the remote Ollama model. Only qwen2.5:3b by default, so the
    shared GPU keeps it warm (never request another model from this helper)."""
    return LLMClient(provider="ollama",
                     base_url=f"http://{host}:{port}/v1",
                     model=model, temperature=temperature,
                     max_tokens=max_tokens, timeout=timeout)
```

- [ ] **Step 2: Write the connectivity probe test**

```python
# tests/test_ollama_helper.py
import pytest
from experiments.ollama_helper import make_ollama_client


@pytest.mark.integration
def test_ollama_reachable_and_responds():
    client = make_ollama_client(timeout=60)
    try:
        resp = client.complete("Reply with exactly one word: pong")
    except Exception as e:
        pytest.skip(f"Ollama host not reachable: {e}")
    assert "pong" in resp.lower()
    assert client.usage.calls == 1
```

- [ ] **Step 3: Run the probe**

Run: `python -m pytest tests/test_ollama_helper.py -v --no-cov`
Expected: PASS if the host is up (`pong` in the reply). If it SKIPS (host unreachable), STOP and report: the rest of the plan needs the model. If it FAILS on the assertion (model replied but not `pong`), that is fine to note but not a blocker (qwen2.5:3b is chatty); relax to `len(resp) > 0`.

- [ ] **Step 4: Commit**

```bash
git add experiments/ollama_helper.py tests/test_ollama_helper.py
git commit -m "feat: EX28 Ollama client helper + connectivity probe"
```

---

## Task 2: Structural rule-equivalence checker

The decisive proposal-rate metric needs to decide whether Op G proposed a rule *structurally equivalent* to the target: identical up to variable renaming and body-literal reordering, but connectivity-sensitive (so left- and right-recursion are NOT equivalent). The existing `clause_subsumes` is positional/body-order-sensitive, so we write a dedicated checker.

**Files:**
- Create: `dreamlog/rule_equivalence.py`
- Test: `tests/test_rule_equivalence.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_rule_equivalence.py
from dreamlog.factories import atom, var, compound
from dreamlog.knowledge import Rule
from dreamlog.rule_equivalence import rules_structurally_equivalent

X, Y, Z = var("X"), var("Y"), var("Z")
A, B, C = var("A"), var("B"), var("C")


def test_equivalent_under_renaming_and_reorder():
    r1 = Rule(compound("father", X, Y), [compound("parent", X, Y), compound("male", X)])
    # renamed vars + reordered body
    r2 = Rule(compound("father", A, B), [compound("male", A), compound("parent", A, B)])
    assert rules_structurally_equivalent(r1, r2)


def test_right_vs_left_recursion_not_equivalent():
    right = Rule(compound("anc", X, Z), [compound("par", X, Y), compound("anc", Y, Z)])
    left = Rule(compound("anc", X, Z), [compound("anc", X, Y), compound("par", Y, Z)])
    assert not rules_structurally_equivalent(right, left)


def test_different_body_predicate_not_equivalent():
    r1 = Rule(compound("father", X, Y), [compound("parent", X, Y), compound("male", X)])
    r2 = Rule(compound("father", X, Y), [compound("parent", X, Y), compound("female", X)])
    assert not rules_structurally_equivalent(r1, r2)


def test_different_body_length_not_equivalent():
    r1 = Rule(compound("p", X), [compound("q", X)])
    r2 = Rule(compound("p", X), [compound("q", X), compound("r", X)])
    assert not rules_structurally_equivalent(r1, r2)


def test_atom_constants_must_match():
    r1 = Rule(compound("p", X), [compound("q", X, atom("a"))])
    r2 = Rule(compound("p", X), [compound("q", X, atom("b"))])
    assert not rules_structurally_equivalent(r1, r2)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_rule_equivalence.py -v --no-cov`
Expected: FAIL with `ModuleNotFoundError: No module named 'dreamlog.rule_equivalence'`.

- [ ] **Step 3: Write the checker**

```python
# dreamlog/rule_equivalence.py
"""Structural equivalence of Horn clauses: identical up to variable renaming
and body-literal reordering, but connectivity-sensitive (a consistent variable
bijection must exist), so left- and right-recursion are distinguished."""
from itertools import permutations
from typing import Optional, Dict
from .terms import Term, Variable, Atom, Compound
from .knowledge import Rule


def _match(a: Term, b: Term, vmap: Dict[str, str]) -> Optional[Dict[str, str]]:
    """Extend vmap (a-var name -> b-var name) so a matches b; None on failure.
    The bijection is enforced (no two a-vars map to the same b-var)."""
    if isinstance(a, Variable) and isinstance(b, Variable):
        if a.name in vmap:
            return vmap if vmap[a.name] == b.name else None
        if b.name in vmap.values():
            return None
        m = dict(vmap)
        m[a.name] = b.name
        return m
    if isinstance(a, Atom) and isinstance(b, Atom):
        return vmap if a.value == b.value else None
    if isinstance(a, Compound) and isinstance(b, Compound):
        if a.functor != b.functor or a.arity != b.arity:
            return None
        for x, y in zip(a.args, b.args):
            vmap = _match(x, y, vmap)
            if vmap is None:
                return None
        return vmap
    return None


def rules_structurally_equivalent(r1: Rule, r2: Rule) -> bool:
    """True iff r1 and r2 are the same clause up to variable renaming and body
    reordering (body as a multiset, single consistent variable bijection)."""
    if len(r1.body) != len(r2.body):
        return False
    base = _match(r1.head, r2.head, {})
    if base is None:
        return False
    n = len(r1.body)
    for perm in permutations(range(n)):
        vmap = dict(base)
        ok = True
        for i in range(n):
            vmap = _match(r1.body[i], r2.body[perm[i]], vmap)
            if vmap is None:
                ok = False
                break
        if ok:
            return True
    return False
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_rule_equivalence.py -v --no-cov`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add dreamlog/rule_equivalence.py tests/test_rule_equivalence.py
git commit -m "feat: structural rule-equivalence checker (variant up to renaming + reorder)"
```

---

## Task 3: Clean 3x2 domains

A `Domain` value carries training facts, the target rule, and a new-entity test split, for one (rule type, vocab) cell. Reuse the EX27 flux/ancestor generators and the family father facts; build the three new ones. Each domain's `target_rule` is the single core rule the proposal probe checks (for within-predicate, the core generalization `can_fly(X):-bird(X)`, since the `not(exception)` refinement is measured by recovery, not the proposal rate).

**Files:**
- Create: `experiments/ex28_domains.py`
- Test: `tests/test_ex28_domains.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ex28_domains.py
import importlib.util, pathlib

def _load(name):
    p = pathlib.Path(__file__).parent.parent / "experiments" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_all_six_cells_present():
    d = _load("ex28_domains")
    cells = d.all_domains(seed=42)
    keys = {(c.rule_type, c.vocab) for c in cells}
    assert keys == {
        ("within_predicate", "canonical"), ("within_predicate", "invented"),
        ("recursive", "canonical"), ("recursive", "invented"),
        ("cross_predicate", "canonical"), ("cross_predicate", "invented"),
    }


def test_each_domain_has_target_and_checks():
    d = _load("ex28_domains")
    for c in d.all_domains(seed=42):
        assert c.base and c.derived and c.new_base and c.new_checks
        assert c.target_rule is not None
        assert any(exp for _, exp, _ in c.new_checks)        # has positives
        assert any(not exp for _, exp, _ in c.new_checks)    # has negatives


def test_invented_vocab_has_no_real_predicates():
    d = _load("ex28_domains")
    invented = [c for c in d.all_domains(seed=42) if c.vocab == "invented"]
    real_words = {"parent", "ancestor", "father", "male", "bird", "can_fly"}
    for c in invented:
        text = " ".join(c.base + c.derived)
        assert not any(w in text for w in real_words), f"{c.name} leaks real vocab"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_ex28_domains.py -v --no-cov`
Expected: FAIL (no `experiments/ex28_domains.py`).

- [ ] **Step 3: Write the domains module**

```python
# experiments/ex28_domains.py
"""Clean 3x2 domains for EX28: {within_predicate, recursive, cross_predicate}
x {canonical, invented}. The invented column invents the PREDICATE names, not
only the entities, so the only recoverable signal is structural/statistical."""
from dataclasses import dataclass
from typing import List, Tuple
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from dreamlog.factories import var, compound, atom
from dreamlog.knowledge import Rule
from dreamlog.recursive_discovery import transitive_closure

X, Y, Z = var("X"), var("Y"), var("Z")


@dataclass
class Domain:
    name: str
    rule_type: str   # within_predicate | recursive | cross_predicate
    vocab: str       # canonical | invented
    base: List[str]
    derived: List[str]
    target_rule: Rule
    new_base: List[str]
    new_checks: List[Tuple[str, bool, str]]


def _within_predicate(vocab, pred, guard, members, exceptions, new_members,
                      new_exceptions):
    """pred(X) holds for all `guard` members except `exceptions`."""
    base = [f"({guard} {m})" for m in members + exceptions]
    derived = [f"({pred} {m})" for m in members]            # exceptions excluded
    target = Rule(compound(pred, X), [compound(guard, X)])  # core generalization
    new_base = [f"({guard} {m})" for m in new_members + new_exceptions]
    checks = ([(f"({pred} {m})", True, f"{m} is {pred}") for m in new_members]
              + [(f"({pred} {m})", False, f"{m} not {pred}") for m in new_exceptions])
    return Domain(f"{pred}_{vocab}", "within_predicate", vocab, base, derived,
                  target, new_base, checks)


def within_canonical():
    return _within_predicate(
        "canonical", "can_fly", "bird",
        members=["robin", "sparrow", "eagle", "hawk", "finch", "wren"],
        exceptions=["penguin", "ostrich"],
        new_members=["raven", "dove"], new_exceptions=["kiwi"])


def within_invented():
    return _within_predicate(
        "invented", "glonk", "zorp",
        members=["mirv", "tup", "wex", "sline", "drof", "blay"],
        exceptions=["quib", "snar"],
        new_members=["yort", "plim"], new_exceptions=["vusk"])


def _recursive(vocab, base_pred, closure_pred, nodes, new_nodes, n_extra=2, seed=42):
    import random
    rng = random.Random(seed)
    order = nodes[:]
    edges = {(order[i], order[i + 1]) for i in range(len(order) - 1)}
    for _ in range(n_extra):
        i = rng.randrange(0, len(order) - 1); j = rng.randrange(i + 1, len(order))
        edges.add((order[i], order[j]))
    closure = transitive_closure(edges)
    base = [f"({base_pred} {a} {b})" for a, b in sorted(edges)]
    derived = [f"({closure_pred} {a} {b})" for a, b in sorted(closure)]
    target = Rule(compound(closure_pred, X, Z),
                  [compound(base_pred, X, Y), compound(closure_pred, Y, Z)])
    full_order = order + new_nodes
    new_edges = {(full_order[i], full_order[i + 1])
                 for i in range(len(order) - 1, len(full_order) - 1)}
    full_closure = transitive_closure(edges | new_edges)
    new_set = set(new_nodes)
    pos = sorted(p for p in full_closure if p[0] in new_set or p[1] in new_set)
    neg = sorted((a, b) for a in full_order for b in full_order
                 if a != b and (a in new_set or b in new_set)
                 and (a, b) not in full_closure)[:max(1, len(pos))]
    new_base = [f"({base_pred} {a} {b})" for a, b in sorted(new_edges)]
    checks = ([(f"({closure_pred} {a} {b})", True, f"{a}->{b}") for a, b in pos]
              + [(f"({closure_pred} {a} {b})", False, f"{a}-/->{b}") for a, b in neg])
    return Domain(f"{closure_pred}_{vocab}", "recursive", vocab, base, derived,
                  target, new_base, checks)


def recursive_canonical(seed=42):
    return _recursive("canonical", "parent", "ancestor",
                      ["al", "bo", "cy", "di", "ed", "fi", "gus", "hy"],
                      ["ike", "jo", "ko", "lu"], seed=seed)


def recursive_invented(seed=42):
    return _recursive("invented", "flux_links", "flux_reaches",
                      ["qux", "vor", "zane", "plix", "drub", "yent", "kosh", "wim"],
                      ["fren", "glor", "snee", "thock"], seed=seed)


def _cross_predicate(vocab, target_pred, rel_pred, prop_pred, pairs, props,
                     new_pairs, new_props, distractor_pairs, new_distractor_pairs):
    """target_pred(X,Y) :- rel_pred(X,Y), prop_pred(X). `props` are the X's that
    have the property; pairs not satisfying both are negatives."""
    base = ([f"({rel_pred} {a} {b})" for a, b in pairs + distractor_pairs]
            + [f"({prop_pred} {x})" for x in props])
    derived = [f"({target_pred} {a} {b})" for a, b in pairs if a in set(props)]
    target = Rule(compound(target_pred, X, Y),
                  [compound(rel_pred, X, Y), compound(prop_pred, X)])
    new_base = ([f"({rel_pred} {a} {b})" for a, b in new_pairs + new_distractor_pairs]
                + [f"({prop_pred} {x})" for x in new_props])
    pos = [(a, b) for a, b in new_pairs if a in set(new_props)]
    neg = [(a, b) for a, b in new_pairs + new_distractor_pairs
           if a not in set(new_props)]
    checks = ([(f"({target_pred} {a} {b})", True, f"{a},{b}") for a, b in pos]
              + [(f"({target_pred} {a} {b})", False, f"{a},{b}") for a, b in neg])
    return Domain(f"{target_pred}_{vocab}", "cross_predicate", vocab, base, derived,
                  target, new_base, checks)


def cross_canonical():
    return _cross_predicate(
        "canonical", "father", "parent", "male",
        pairs=[("tom", "ann"), ("tom", "ben"), ("jim", "cat"), ("jim", "dan")],
        props=["tom", "jim"],
        new_pairs=[("sam", "eve"), ("sam", "fox")], new_props=["sam"],
        distractor_pairs=[("liz", "ann"), ("mae", "ben")],   # mothers (not male)
        new_distractor_pairs=[("ivy", "eve")])


def cross_invented():
    # fully invented predicates: wibble(X,Y) :- frob(X,Y), quax(X)
    return _cross_predicate(
        "invented", "wibble", "frob", "quax",
        pairs=[("ond", "ulp"), ("ond", "esk"), ("arn", "ixt"), ("arn", "obo")],
        props=["ond", "arn"],
        new_pairs=[("zib", "ako"), ("zib", "eln")], new_props=["zib"],
        distractor_pairs=[("uvy", "ulp"), ("ywex", "esk")],
        new_distractor_pairs=[("plon", "ako")])


def all_domains(seed=42):
    return [within_canonical(), within_invented(),
            recursive_canonical(seed), recursive_invented(seed),
            cross_canonical(), cross_invented()]
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_ex28_domains.py -v --no-cov`
Expected: 3 passed. If `test_invented_vocab_has_no_real_predicates` fails, an invented domain is leaking a real word; rename it.

- [ ] **Step 5: Commit**

```bash
git add experiments/ex28_domains.py tests/test_ex28_domains.py
git commit -m "feat: EX28 clean 3x2 domains (invented predicates, not just entities)"
```

---

## Task 4: Op-C disable flag in the dreamer

The within-predicate LLM-only condition needs Operation C off while Op G is on. Add a flag, off by default.

**Files:**
- Modify: `dreamlog/kb_dreamer.py` (`__init__` near line 290; the `_generalize_facts` call in `dream()` at line 348)
- Test: `tests/test_sleep_cycle.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_sleep_cycle.py
from dreamlog.factories import atom, compound
from dreamlog.knowledge import Fact, KnowledgeBase
from dreamlog.kb_dreamer import KnowledgeBaseDreamer


def _artisan_kb():
    kb = KnowledgeBase()
    arts = ["a", "b", "c", "d", "e"]
    for x in arts:
        kb.add_fact(Fact(compound("artisan", atom(x))))
    for x in arts[:4]:   # 4 of 5 are masters -> Op C should generalize
        kb.add_fact(Fact(compound("master", atom(x))))
    return kb


def test_op_c_runs_by_default():
    kb = _artisan_kb()
    KnowledgeBaseDreamer().dream(kb)
    assert any(r.head.functor == "master" and len(r.body) > 0 for r in kb.rules)


def test_op_c_disable_flag_skips_generalization():
    kb = _artisan_kb()
    KnowledgeBaseDreamer(disable_op_c=True).dream(kb)
    assert not any(r.head.functor == "master" and len(r.body) > 0 for r in kb.rules)
```

- [ ] **Step 2: Run to verify the disable test fails**

Run: `python -m pytest tests/test_sleep_cycle.py -k op_c_disable -v --no-cov`
Expected: FAIL with `TypeError: unexpected keyword argument 'disable_op_c'`.

- [ ] **Step 3: Add the flag to `__init__`**

In `KnowledgeBaseDreamer.__init__`, add a parameter and store it (append after the `min_base_facts` param/assignment added by the recursion work):

```python
                 discover_recursion: bool = False,
                 min_base_facts: int = 3,
                 disable_op_c: bool = False):
        ...
        self.discover_recursion = discover_recursion
        self.min_base_facts = min_base_facts
        # Skip Operation C (fact generalization). Off by default; used by the
        # EX28 within-predicate LLM-only ablation condition.
        self.disable_op_c = disable_op_c
```

- [ ] **Step 4: Gate the Op C call in `dream()`**

Change the line `ops.extend(self._generalize_facts(kb, suite))` (line 348) to:

```python
        if not self.disable_op_c:
            ops.extend(self._generalize_facts(kb, suite))
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_sleep_cycle.py -k "op_c" -v --no-cov` then `python -m pytest tests/ -q --no-cov`
Expected: both new tests pass; full suite green (flag off by default = zero drift).

- [ ] **Step 6: Commit**

```bash
git add dreamlog/kb_dreamer.py tests/test_sleep_cycle.py
git commit -m "feat: disable_op_c flag for EX28 within-predicate LLM-only ablation"
```

---

## Task 5: Factor out `_llm_propose` (Op G proposal step)

The proposal probe needs Op G's *proposed and parsed* rules without the accept/verify pipeline. Factor the prompt-build + LLM call + parse + cycle-filter out of `_llm_compress` into `_llm_propose(kb) -> List[Rule]`, and have `_llm_compress` call it.

**Files:**
- Modify: `dreamlog/kb_dreamer.py` (`_llm_compress` at line 1134; uses `_parse_llm_rules` at 1373 and `_filter_cyclic_rules`)
- Test: `tests/test_sleep_cycle.py`

- [ ] **Step 1: Read `_llm_compress`**

Read `dreamlog/kb_dreamer.py:1134-1270` to find exactly where the prompt is built and where `_parse_llm_rules` + `_filter_cyclic_rules` are called (the proposal sub-step), versus where the accept loop begins (the `helper_rules`/`main_rules` split). The factored method returns the cycle-filtered parsed rules; the accept loop stays in `_llm_compress`.

- [ ] **Step 2: Write the failing test**

```python
# add to tests/test_sleep_cycle.py
def test_llm_propose_returns_parsed_rules_without_accepting():
    import json
    from tests.mock_provider import MockLLMProvider
    kb = _ancestor_closure_kb()   # defined earlier in this file
    mock = MockLLMProvider(responses=[json.dumps([
        ["rule", ["ancestor", "X", "Z"],
         [["parent", "X", "Y"], ["ancestor", "Y", "Z"]]],
    ])])
    dreamer = KnowledgeBaseDreamer(llm_client=mock)
    proposed = dreamer._llm_propose(kb)
    assert any(r.head.functor == "ancestor" and len(r.body) == 2 for r in proposed)
    # _llm_propose must NOT mutate the KB (no rules added)
    assert not any(r.head.functor == "ancestor" for r in kb.rules)
```

- [ ] **Step 3: Run to verify it fails**

Run: `python -m pytest tests/test_sleep_cycle.py -k llm_propose -v --no-cov`
Expected: FAIL with `AttributeError: ... has no attribute '_llm_propose'`.

- [ ] **Step 4: Extract `_llm_propose`**

In `kb_dreamer.py`, move the proposal sub-step (build the Op G prompt string, call `_parse_llm_rules`, run `_filter_cyclic_rules`) into a new method that returns the rule list and mutates nothing:

```python
    def _llm_propose(self, kb: KnowledgeBase) -> List[Rule]:
        """Build the Op G prompt, query the LLM, parse and cycle-filter the
        proposed rules. Returns the proposed rules WITHOUT accepting them
        (no KB mutation). Used by _llm_compress and the EX28 proposal probe."""
        if self.llm_client is None:
            return []
        prompt = self._build_op_g_prompt(kb)        # the existing prompt text
        parsed = self._parse_llm_rules(prompt, parse_llm_response=None)
        return _filter_cyclic_rules(parsed)
```

Then in `_llm_compress`, replace the inlined prompt-build + parse + filter with `parsed_rules = self._llm_propose(kb)` and keep the accept loop. Also extract the prompt string into `_build_op_g_prompt(kb)` (it currently lives inline in `_llm_compress`); both methods call it. Keep the existing prompt text verbatim (including the recursion clause added earlier).

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_sleep_cycle.py -k "llm_propose or recursive_proposal or op_g" -v --no-cov` then `python -m pytest tests/ -q --no-cov`
Expected: the new test passes; the existing Op G tests (`test_op_g_accepts_a_recursive_proposal`, the full-pipeline tests) still pass; full suite green. The refactor must be behavior-preserving for `_llm_compress`.

- [ ] **Step 6: Commit**

```bash
git add dreamlog/kb_dreamer.py tests/test_sleep_cycle.py
git commit -m "refactor: extract _llm_propose / _build_op_g_prompt for the EX28 probe"
```

---

## Task 6: Proposal-rate probe

Measure, over N runs, how often Op G proposes a rule structurally equivalent to a domain's target.

**Files:**
- Create: `experiments/ex28_probe.py`
- Test: `tests/test_ex28_probe.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ex28_probe.py
import importlib.util, pathlib, json

def _load(name):
    p = pathlib.Path(__file__).parent.parent / "experiments" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_probe_counts_structurally_equivalent_proposals():
    probe = _load("ex28_probe")
    domains = _load("ex28_domains")
    from tests.mock_provider import MockLLMProvider
    dom = domains.recursive_invented(seed=42)
    # mock proposes the correct recursive rule every time
    correct = json.dumps([
        ["rule", ["flux_reaches", "X", "Z"],
         [["flux_links", "X", "Y"], ["flux_reaches", "Y", "Z"]]],
    ])
    mock = MockLLMProvider(responses=[correct] * 5)
    result = probe.proposal_rate(dom, mock, n_runs=5)
    assert result["rate"] == 1.0
    assert result["hits"] == 5 and result["n"] == 5
    assert len(result["runs"]) == 5            # per-run metadata recorded
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_ex28_probe.py -v --no-cov`
Expected: FAIL (no `experiments/ex28_probe.py`).

- [ ] **Step 3: Write the probe**

```python
# experiments/ex28_probe.py
"""Proposal-rate probe: how often does Op G propose a rule structurally
equivalent to the domain's target rule, over N runs? Records full per-run
metadata (the proposed rules and the verdict)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ex25_generalization import build_kb
from dreamlog.kb_dreamer import KnowledgeBaseDreamer
from dreamlog.rule_equivalence import rules_structurally_equivalent


def proposal_rate(domain, llm_client, n_runs=30):
    hits = 0
    runs = []
    for i in range(n_runs):
        kb = build_kb(domain.base + domain.derived)
        dreamer = KnowledgeBaseDreamer(llm_client=llm_client)
        proposed = dreamer._llm_propose(kb)
        hit = any(rules_structurally_equivalent(r, domain.target_rule)
                  for r in proposed)
        hits += int(hit)
        runs.append({"run": i, "hit": hit,
                     "proposed": [str(r) for r in proposed]})
    return {"hits": hits, "n": n_runs, "rate": hits / n_runs if n_runs else 0.0,
            "runs": runs}
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_ex28_probe.py -v --no-cov`
Expected: PASS (rate 1.0).

- [ ] **Step 5: Commit**

```bash
git add experiments/ex28_probe.py tests/test_ex28_probe.py
git commit -m "feat: EX28 proposal-rate probe (structural-equivalence over N runs)"
```

---

## Task 7: Resumable, metadata-rich harness

A work-unit runner: enumerate `(cell, condition, run_index)`, skip already-completed units (read from an append-only JSONL store), run the rest, append each record with full metadata, and write a run manifest. Supports `--fresh` and `--summarize`.

**Files:**
- Create: `experiments/ex28_harness.py`
- Test: `tests/test_ex28_harness.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ex28_harness.py
import importlib.util, pathlib, json, tempfile, os

def _load(name):
    p = pathlib.Path(__file__).parent.parent / "experiments" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_completed_units_are_skipped_on_resume():
    h = _load("ex28_harness")
    with tempfile.TemporaryDirectory() as d:
        store = os.path.join(d, "results.jsonl")
        calls = {"n": 0}
        def fake_unit(unit):
            calls["n"] += 1
            return {"recovery": 1.0}
        units = [{"cell": "c1", "condition": "symbolic", "run": 0},
                 {"cell": "c1", "condition": "symbolic", "run": 1}]
        h.run_units(units, fake_unit, store, manifest_dir=d, git_sha="abc")
        assert calls["n"] == 2
        # second run: everything is done, fake_unit must not be called again
        h.run_units(units, fake_unit, store, manifest_dir=d, git_sha="abc")
        assert calls["n"] == 2
        # results.jsonl has exactly 2 records, each with metadata
        recs = [json.loads(l) for l in open(store)]
        assert len(recs) == 2
        assert all("git_sha" in r and "ts" in r and "key" in r for r in recs)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_ex28_harness.py -v --no-cov`
Expected: FAIL (no `experiments/ex28_harness.py`).

- [ ] **Step 3: Write the harness**

```python
# experiments/ex28_harness.py
"""Resumable work-unit runner with append-only JSONL metadata."""
import json, os, time, hashlib


def unit_key(unit: dict) -> str:
    raw = json.dumps({k: unit[k] for k in ("cell", "condition", "run")},
                     sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _completed_keys(store: str):
    if not os.path.exists(store):
        return set()
    done = set()
    with open(store) as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["key"])
    return done


def run_units(units, run_one, store: str, manifest_dir: str, git_sha: str,
              fresh: bool = False):
    os.makedirs(os.path.dirname(store) or ".", exist_ok=True)
    if fresh and os.path.exists(store):
        os.remove(store)
    done = _completed_keys(store)
    planned = [u for u in units if unit_key(u) not in done]
    manifest = {"git_sha": git_sha, "total": len(units),
                "already_done": len(done), "to_run": len(planned)}
    with open(os.path.join(manifest_dir, f"manifest-{int(time.time())}.json"), "w") as mf:
        json.dump(manifest, mf, indent=2)
    for u in planned:
        result = run_one(u)                       # the expensive call
        record = {"key": unit_key(u), "ts": time.time(), "git_sha": git_sha,
                  **u, **result}
        with open(store, "a") as f:               # append-only, flush each unit
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())
    return manifest


def summarize(store: str):
    return [json.loads(l) for l in open(store) if l.strip()]
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_ex28_harness.py -v --no-cov`
Expected: PASS (fake_unit called twice across two runs; 2 records).

- [ ] **Step 5: Commit**

```bash
git add experiments/ex28_harness.py tests/test_ex28_harness.py
git commit -m "feat: EX28 resumable work-unit harness (skip-completed, append-only JSONL)"
```

---

## Task 8: EX28 experiment (4 conditions x 6 cells) + registry

Wire the conditions, the probe, recovery, the harness, and a CLI. Reuse `run_domain_test`-style recovery via `dream_kb` and `evaluate_checks` from EX25/EX25b (already pass `discover_recursion`; now also pass `disable_op_c` and `llm_client`).

**Files:**
- Create: `experiments/ex28_llm_role.py`
- Modify: `experiments/experiment_registry.yaml`

- [ ] **Step 1: Read the recovery helpers**

Read `experiments/ex25_generalization.py` (`build_kb`, `dream_kb`, `is_derivable`) and `experiments/ex25b_novel_generalization.py` (`evaluate_checks`, `run_raw_llm_check`). EX28 reuses these for the recovery metric and the raw-LLM baseline. Confirm `dream_kb` forwards `disable_op_c` (added in Task 4); if not, add the one-line passthrough as part of this task.

- [ ] **Step 2: Write the experiment driver**

```python
# experiments/ex28_llm_role.py
"""EX28: isolating the LLM's contribution by rule type, on qwen2.5:3b."""
import argparse, sys, pathlib, subprocess
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ex28_domains import all_domains
from ex28_probe import proposal_rate
from ex28_harness import run_units, summarize, unit_key
from ollama_helper import make_ollama_client
from ex25_generalization import build_kb, dream_kb
from ex25b_novel_generalization import evaluate_checks, run_raw_llm_check

# Which symbolic op each rule type owns (for the symbolic-only / LLM-only split)
SYMBOLIC_OP = {"within_predicate": "op_c", "recursive": "op_i",
               "cross_predicate": None}


def _dream_flags(rule_type, condition):
    """Return (discover_recursion, disable_op_c, use_llm) for a condition."""
    use_llm = condition in ("llm_only", "full")
    discover = (rule_type == "recursive") and condition in ("symbolic_only", "full")
    # LLM-only: turn OFF the rule-type's symbolic op
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
        total = tp + tn + fp + fn
        return {"recovery": tp / (tp + fn) if (tp + fn) else 0.0,
                "precision": tp / (tp + fp) if (tp + fp) else 0.0,
                "tp": tp, "tn": tn, "fp": fp, "fn": fn, "proposal_rate": None}
    discover, disable_c, use_llm = _dream_flags(dom.rule_type, cond)
    kb = build_kb(dom.base + dom.derived)
    dream_kb(kb, llm_client=client if use_llm else None,
             discover_recursion=discover, disable_op_c=disable_c, open_world=True)
    res = evaluate_checks(kb, dom.new_base, dom.new_checks)
    # proposal rate only where the LLM is the route under study
    pr = None
    if use_llm:
        pr = proposal_rate(dom, client, n_runs=n_probe)["rate"]
    return {"recovery": res["recall"], "precision": res["precision"],
            "tp": res["tp"], "tn": res["tn"], "fp": res["fp"], "fn": res["fn"],
            "proposal_rate": pr}


def main():
    ap = argparse.ArgumentParser(description="EX28 LLM-role ablation (qwen2.5:3b)")
    ap.add_argument("--n-probe", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--store", default="experiments/data/ex28/results.jsonl")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()

    if args.summarize:
        for r in summarize(args.store):
            print(r["cell"], r["condition"], r.get("recovery"), r.get("proposal_rate"))
        return

    doms = {d.name: d for d in all_domains(seed=args.seed)}
    client = make_ollama_client()
    conditions = ["symbolic_only", "llm_only", "full", "raw_llm"]
    units = [{"cell": name, "condition": c, "run": 0}
             for name in doms for c in conditions]
    git_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    import os
    os.makedirs(os.path.dirname(args.store), exist_ok=True)
    run_units(units, lambda u: run_one_unit(u, doms, client, args.n_probe),
              args.store, manifest_dir=os.path.dirname(args.store),
              git_sha=git_sha, fresh=args.fresh)
    print(f"EX28 complete. LLM cost (if any): {client.usage}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Register EX28**

Append to `experiments/experiment_registry.yaml` (match the existing entry shape; status `planned` until run):

```yaml
  EX28_llm_role_ablation:
    title: "Isolating the LLM's contribution by rule type (structure vs semantics)"
    date: 2026-06-XX
    script: experiments/ex28_llm_role.py
    status: planned
    motivation: >
      Replace the paper's weak "full == symbolic" observation with a measured
      claim: symbolic compression recovers structure on any vocabulary, while
      the LLM's value is recall of meaning that evaporates on invented
      predicates. Doubles as the rigorous per-mechanism ablation.
    method: >
      Clean 3x2 (within-predicate / recursive / cross-predicate x canonical /
      fully-invented), four conditions (symbolic-only, LLM-only, full, raw-LLM).
      Metrics: recovery and the correct-rule-proposal rate (structural
      equivalence). Local qwen2.5:3b, high N, resumable with full metadata.
    key_result: "TBD after run"
    implications: "TBD after run"
    depends_on: [EX25, EX25b, EX27]
```

- [ ] **Step 4: Smoke test (no GPU): summarize path + dry import**

Run: `python experiments/ex28_llm_role.py --summarize --store /tmp/none.jsonl 2>&1 | head` (should print nothing and exit cleanly) and `python -c "import experiments.ex28_llm_role"` style import via the module loader to confirm no import errors. Fix any helper-name mismatch surfaced against the real EX25 API.

- [ ] **Step 5: Commit**

```bash
git add experiments/ex28_llm_role.py experiments/experiment_registry.yaml experiments/ex25_generalization.py
git commit -m "feat: EX28 experiment driver (4 conditions x 6 cells) + registry"
```

---

## Task 9: Run EX28 on qwen2.5:3b and record results

**Files:**
- Modify: `experiments/experiment_registry.yaml` (fill key_result/implications, set status complete)

- [ ] **Step 1: Run it**

Run: `python experiments/ex28_llm_role.py --n-probe 30` (uses the remote qwen2.5:3b; resumable, so safe to interrupt and re-run). Expect a long run on the shared GPU; if interrupted, re-running resumes.

- [ ] **Step 2: Summarize and read the pattern**

Run: `python experiments/ex28_llm_role.py --summarize`. Build the two tables (recovery, proposal-rate) per cell. Characterize: symbolic-only recovery by rule type and vocab; LLM-only recovery and proposal-rate, especially the **cross-predicate / invented** cell (recall vs inference) and the **recursive / invented** cell (structural-prior transfer).

- [ ] **Step 3: Record in the registry**

Fill `key_result` and `implications` from `experiments/data/ex28/results.jsonl`; set `status: complete` and the run date.

- [ ] **Step 4: Full suite green**

Run: `python -m pytest tests/ -q --no-cov`
Expected: all green (no regression; Op-C flag and `_llm_propose` refactor are behavior-preserving by default).

- [ ] **Step 5: Commit**

```bash
git add experiments/experiment_registry.yaml experiments/data/ex28/results.jsonl
git commit -m "experiments: EX28 complete - LLM-role ablation results recorded"
```

---

## Self-Review

**Spec coverage:**
- 3x2 clean domains with invented predicates: Task 3. Covered.
- Four conditions (symbolic-only / LLM-only / full / raw-LLM): Task 8 (`_dream_flags`, `run_one_unit`). Covered.
- Two metrics (recovery + proposal-rate): Tasks 6, 8. Covered.
- Op-C disable flag for within-predicate LLM-only: Task 4. Covered.
- `_llm_propose` for the probe: Task 5. Covered.
- Ollama `qwen2.5:3b` only, generous timeout: Tasks 1, 8. Covered.
- Resumable + full metadata + manifest + `--fresh`/`--summarize`: Task 7, 8. Covered.
- GPU discipline (single model, sequential, generous timeout): Tasks 1, 8 (units run sequentially in a single loop). Covered.
- Structural-equivalence (variant up to renaming/reorder, distinguishes left/right recursion): Task 2. Covered.

**Placeholder scan:** Tasks 1 to 7 contain complete code. Task 8 has two "confirm against the real EX25 API" steps (the `dream_kb` `disable_op_c` passthrough and the `evaluate_checks`/`run_raw_llm_check` signatures) and Step 1 reads them first; these are scoped read-then-wire steps, not vague placeholders. The within-predicate `target_rule` is deliberately the core `pred(X):-guard(X)` (the proposal probe is primary for the recursive and cross-predicate rows; recovery carries the within-predicate row) and this is stated in Task 3.

**Type consistency:** `Domain` fields (`name`, `rule_type`, `vocab`, `base`, `derived`, `target_rule`, `new_base`, `new_checks`) are used consistently in Tasks 3, 6, 8. `rules_structurally_equivalent(r1, r2)` signature matches across Tasks 2, 6. `proposal_rate(domain, llm_client, n_runs)` matches Tasks 6, 8. `run_units(units, run_one, store, manifest_dir, git_sha, fresh)` and `unit_key`/`summarize` match Tasks 7, 8. `disable_op_c` flag name matches Tasks 4, 8. `_llm_propose(kb)` matches Tasks 5, 6.

**Known follow-ups for the implementer (not blockers):** `dream_kb` may need a one-line `disable_op_c` passthrough (flagged in Task 8 Step 1, committed in Task 8). The `qwen2.5:3b` proposal rates on the structure rows may be near zero (weak model); that is an expected result, not a bug, and the Haiku anchor is deferred per the spec.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-04-llm-role-ablation.md`. Two execution options:

1. **Subagent-Driven (recommended)** - dispatch a fresh subagent per task, review between tasks. Task 1 (the Ollama connectivity probe) is the natural first gate: if the model is unreachable, we stop and fix connectivity before building the rest.
2. **Inline Execution** - execute the tasks in this session with checkpoints.

Which approach?
