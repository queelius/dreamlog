# MDL Unified Gate (P1+P2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the sleep cycle into proposal generators driving ONE description-length gate, with bit-identical behavior (clause-count DL retained), per `docs/superpowers/specs/2026-06-09-mdl-unified-gate-design.md`.

**Architecture:** New `dreamlog/compression/` package (dl, proposal, gate, policies, util, generators/, maintenance). `kb_dreamer.py` keeps its class, `dream()` call skeleton, verification suite, and all existing private method NAMES as thin orchestrators (experiments monkeypatch `_llm_compress` and call `_generalize_facts`; tests call `_llm_propose` etc.). The gate replaces, per op, exactly today's "copy, verify, continue-on-fail, apply, append" block; everything else in each op moves VERBATIM.

**Tech Stack:** Python >= 3.8, pytest (`--no-cov` for targeted runs), DreamLog internals. No LLM calls anywhere in this plan (all regression cells are symbolic).

**Branch:** `mdl-unified-gate` (already checked out, off tag `v0.13.0`).

**Iron rule (G5):** every accept/reject decision today must be reproduced exactly. When in doubt, copy the original lines verbatim and only replace the verify+apply seam. The committed artifacts are the judge (Task 1's test).

**Verified source anchors (read 2026-06-09, file = dreamlog/kb_dreamer.py, 1582 lines):**

| Symbol | Lines | Notes |
|---|---|---|
| `_is_system_predicate` | 33-37 | shared |
| `_next_generated_name` | 38-50 | used by D, E |
| `_strip_llm_noise` | 51-62 | G only |
| `_filter_cyclic_rules` | 63-113 | G only |
| `_collect_user_functors` | 114-126 | G + F |
| `CompressionCandidate` (`mdl_delta`, `is_worth_it`) | 127-139 | stays |
| `VerificationResult/Suite`, `build_verification_suite`, `extend_verification_for_rules` | 142-272 | stays in kb_dreamer |
| `DreamSession` | 274-285 | stays; gains `rejections` |
| `__init__` | 290-315 | unchanged |
| `dream()` | 316-401 | call skeleton unchanged |
| A `_eliminate_subsumed` | 403-437 | A1 rules pass then A2 facts pass, collect-then-apply each |
| B `_prune_redundant_facts(kb, max_calls=0)` | 439-497 | Phase1 probe MUTATES kb (remove/probe/readd); Phase2 batch; Phase3 batch verify; Phase4 sequential fallback on LIVE kb |
| C `_generalize_facts` | 499-626 | interleaved; verify+apply seam = 601-616; rebuild+break = 618-624; groups snapshot at 512-517 excludes new exception functors |
| `_find_guard` | 628-650 | moves with C |
| D `_invent_predicates` | 652-752 | strict delta check `if k + n >= n * k: continue` at 684; name alloc from live kb at 687 (BEFORE verify); seam = 732-750 |
| E `_extract_body_patterns(kb, suite, max_rounds=10)` | 756-828 | round-based on live kb; `failed_keys` memo; seam = 787-826; delta ALWAYS +1 |
| E helpers `_find_best_body_pattern`, `_subseq_structural_key`, `_iter_var_names`, `_compute_interface_vars`, `_map_interface_vars` | 904-1069 | move verbatim with E |
| I `_discover_recursion(kb, suite, max_calls=5000)` | 830-902 | at most ONE candidate per call; seam = 884-900 |
| naming `_name_invented_predicates`, `_rename_predicate` | 1070-1138 | stays in kb_dreamer |
| G `_build_op_g_prompt` / `_llm_propose` / `_llm_compress` / `_parse_llm_rules` / `_build_rule_from_parsed` | 1139-1486 | propose is pure; compress Phase 3 split 1310-1319, Phase 4 per-rule 1321-1383, Phase 5 combined 1385-1398, commit 1400-1407 (approx; re-read before edit) |
| F `_prune_dead_clauses` | 1488-1546 | seed-protected; dream() inlines suite pruning at 340-348 |
| `_frequency_score` | 1548-1552 | H only |
| H `_cache_lemmas` | 1556-1582 | adds facts; no gate |

**External callers that must keep working (AC5):** tests call `_llm_propose` (x6), `_discover_recursion` (x4), `_prune_dead_clauses` (x2), `_prune_redundant_facts` (x1), `_build_op_g_prompt` (x1). Experiments call `_generalize_facts` (x3), `_llm_compress` (x2, one is a MONKEYPATCH in `experiments/ablation_and_scale.py`), `_llm_propose` (x1, `ex28_probe.py`), `_prune_dead_clauses` (x1).

**Repo constraint (hook):** a voice hook rejects any file write containing an em-dash (U+2014) or the banned word spelled l-e-v-e-r-a-g-e. Plain ASCII hyphens only; phrase around the banned word.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `tests/test_mdl_refactor_regression.py` | AC3 baseline vs committed artifacts | 1 |
| `dreamlog/compression/__init__.py` | package exports | 2 |
| `dreamlog/compression/util.py` | shared helpers moved from kb_dreamer | 2 |
| `dreamlog/compression/proposal.py` | `Proposal` dataclass | 2 |
| `dreamlog/compression/dl.py` | description length + delta | 2 |
| `tests/test_compression_dl.py` | dl/proposal unit tests | 2 |
| `dreamlog/compression/gate.py` | apply_proposal, batch variants, results | 3 |
| `dreamlog/compression/policies.py` | per-kind policies | 3 (skeleton) + 5-9 (filled) |
| `tests/test_compression_gate.py` | AC4 property tests | 3 |
| `dreamlog/compression/maintenance.py` | F + H + suite pruning | 4 |
| `dreamlog/compression/generators/__init__.py` | | 5 |
| `dreamlog/compression/generators/reduce.py` | A+B | 5 |
| `dreamlog/compression/generators/generalize.py` | C + `_find_guard` | 6 |
| `dreamlog/compression/generators/factor.py` | D + E + E helpers | 7 |
| `dreamlog/compression/generators/closure.py` | I | 8 |
| `dreamlog/compression/generators/llm.py` | G proposal stage + parse | 9 |
| `dreamlog/kb_dreamer.py` | facade; shrinks to < ~550 lines | 4-10 |
| `CLAUDE.md` | architecture section update | 10 |

Run all commands from the repo root `/home/spinoza/github/beta/dreamlog`.

---

## Task 1: AC3 regression baseline (must PASS before any refactor commit)

This test re-runs the deterministic symbolic cells through the SAME experiment code paths and compares against the committed artifacts. It is the judge for every later task. It must pass against the CURRENT code first.

**Files:**
- Create: `tests/test_mdl_refactor_regression.py`

- [ ] **Step 1: Write the test**

```python
"""AC3 regression: symbolic experiment cells must reproduce committed artifacts.

These cells are deterministic (no LLM). They exercise Operations A, B, C, I and
the full dream() pipeline end to end through the real experiment code paths
(ex25b.run_domain_test and ex28.run_one_unit), so any behavior drift introduced
by the MDL-unified-gate refactor shows up as an exact-value mismatch here.
"""
import importlib.util
import json
import pathlib
import sys

import pytest

EXP = pathlib.Path(__file__).parent.parent / "experiments"
sys.path.insert(0, str(EXP))


def _load(name):
    p = EXP / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.slow
def test_ex25b_crafting_symbolic_reproduces_artifact():
    art = json.loads((EXP / "data" / "ex25b" / "results.json").read_text())
    want = art["domains"]["crafting"]["symbolic"]
    ex25b = _load("ex25b_novel_generalization")
    results = ex25b.run_domain_test(
        "regression-crafting",
        ex25b.crafting_base_facts(),
        ex25b.crafting_derived_facts(),
        ex25b.crafting_negatives(),
        ex25b.NEW_CRAFTING_BASE,
        ex25b.NEW_CRAFTING_CHECKS,
        None,            # no LLM client: symbolic + no_dream conditions only
        n_runs=1,
    )
    got = results["symbolic"]
    assert got["recall"] == pytest.approx(want["recall"], abs=1e-12)
    assert got["precision"] == pytest.approx(want["precision"], abs=1e-12)
    assert got["accuracy"] == pytest.approx(want["accuracy"], abs=1e-12)
    assert got["rules"] == want["runs"][0]["rules"] if "runs" in want else want["rules"]
    assert got["compression"] == pytest.approx(want["compression"], abs=1e-12)


@pytest.mark.slow
def test_ex28_symbolic_column_reproduces_artifact():
    rows = {}
    art_path = EXP / "data" / "ex28_sonnet" / "results.jsonl"
    for line in art_path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["condition"] == "symbolic_only":
                rows[r["cell"]] = r
    assert len(rows) == 6, f"expected 6 symbolic cells, got {sorted(rows)}"

    ex28 = _load("ex28_llm_role")
    doms = {d.name: d for d in ex28.all_domains(seed=42)}
    assert set(doms) == set(rows), "domain names drifted from artifact cells"

    for cell, want in rows.items():
        got = ex28.run_one_unit(
            {"cell": cell, "condition": "symbolic_only", "run": 0},
            doms, client=None, n_probe=0)
        for key in ("recovery", "precision"):
            assert got[key] == pytest.approx(want[key], abs=1e-12), (cell, key)
        for key in ("tp", "tn", "fp", "fn"):
            assert got[key] == want[key], (cell, key)
```

- [ ] **Step 2: Adapt the two flagged uncertainties against the real code**

Before running, read two spots and fix the test if needed (do NOT weaken assertions):
1. `experiments/ex25b_novel_generalization.py:421-470`: confirm `run_domain_test`'s positional signature and which conditions run when `llm_client=None` (expected: `no_dream` and `symbolic`). Confirm the symbolic result dict keys (`recall`, `precision`, `accuracy`, `rules`, `compression`). If symbolic results in the n_runs=1 path have no `runs` key, simplify the `rules` assertion to `got["rules"] == want["rules"]` (artifact stores 5.0; compare with `== pytest.approx(want["rules"])`).
2. `experiments/ex28_llm_role.py` `run_one_unit`: confirm `client=None` is safe for `symbolic_only` (use_llm is False so client is never touched).

- [ ] **Step 3: Run against CURRENT code; it must PASS**

Run: `python -m pytest tests/test_mdl_refactor_regression.py -v --no-cov`
Expected: `2 passed` (roughly 30-90 s). If either fails, STOP: the baseline is wrong, not the code. Diagnose against the artifact before proceeding (likely a signature/keys mismatch from Step 2).

- [ ] **Step 4: Commit**

```bash
git add tests/test_mdl_refactor_regression.py
git commit -m "test: AC3 symbolic regression baseline vs committed EX25b/EX28 artifacts"
```

---

## Task 2: compression package skeleton (util, proposal, dl)

**Files:**
- Create: `dreamlog/compression/__init__.py`, `dreamlog/compression/util.py`, `dreamlog/compression/proposal.py`, `dreamlog/compression/dl.py`
- Modify: `dreamlog/kb_dreamer.py` (helpers become re-imports)
- Test: `tests/test_compression_dl.py`

- [ ] **Step 1: Write the failing unit test**

```python
# tests/test_compression_dl.py
from dreamlog.compression.proposal import Proposal
from dreamlog.compression import dl
from dreamlog.factories import atom, compound
from dreamlog.knowledge import Fact, Rule, KnowledgeBase


def _fact(name, *args):
    return Fact(compound(name, *[atom(a) for a in args]))


def test_description_length_is_clause_count():
    kb = KnowledgeBase()
    kb.add_fact(_fact("p", "a"))
    kb.add_fact(_fact("q", "b"))
    kb.add_rule(Rule(compound("r", atom("x")), [compound("p", atom("x"))]))
    assert dl.description_length(kb) == len(kb) == 3
    assert dl.clause_cost(_fact("p", "a")) == 1


def test_proposal_delta_and_immutability():
    p = Proposal(kind="pruning", remove=(_fact("p", "a"), _fact("p", "b")),
                 add=(), notes={"detector": "derivable"})
    assert dl.proposal_delta(p) == -2
    q = Proposal(kind="llm_compression", remove=(),
                 add=(Rule(compound("r", atom("x")), [compound("p", atom("x"))]),))
    assert dl.proposal_delta(q) == 1
    import dataclasses
    assert dataclasses.is_dataclass(p) and p.__dataclass_params__.frozen
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_compression_dl.py -v --no-cov`
Expected: FAIL with `ModuleNotFoundError: No module named 'dreamlog.compression'`.

- [ ] **Step 3: Create the package**

`dreamlog/compression/util.py`: MOVE these from `kb_dreamer.py` verbatim (cut from kb_dreamer, paste here, keep docstrings): `_is_system_predicate` (33-37), `_next_generated_name` (38-50), `_strip_llm_noise` (51-62), `_filter_cyclic_rules` (63-113), `_collect_user_functors` (114-126). Bring exactly the imports they need (read their bodies for the needed names: `re`/`Set`/`KnowledgeBase`/`Rule`/`Compound` etc.).

In `kb_dreamer.py`, replace the five moved definitions with one import line at the same location:

```python
from .compression.util import (_is_system_predicate, _next_generated_name,
                               _strip_llm_noise, _filter_cyclic_rules,
                               _collect_user_functors)
```

(This keeps every later reference in kb_dreamer working unchanged, including `extend_verification_for_rules`'s use of `_is_system_predicate`.)

`dreamlog/compression/proposal.py`:

```python
"""A Proposal is a pure description of one compression transformation."""
from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple, Union

from ..knowledge import Fact, Rule

Clause = Union[Fact, Rule]


@dataclass(frozen=True)
class Proposal:
    """kind uses today's CompressionCandidate operation labels verbatim:
    subsumption | pruning | generalization | invention | extraction |
    recursion | llm_compression."""
    kind: str
    remove: Tuple[Clause, ...] = ()
    add: Tuple[Clause, ...] = ()
    notes: Mapping[str, Any] = field(default_factory=dict)
```

`dreamlog/compression/dl.py`:

```python
"""Single source of truth for description length. P1: clause count.

P3 will replace these internals with a bits-based symbol encoding (and a
functor-signature charge); the signatures are stable so call sites never
change again.
"""
from ..knowledge import KnowledgeBase
from .proposal import Proposal, Clause


def clause_cost(clause: Clause) -> int:
    return 1


def description_length(kb: KnowledgeBase) -> int:
    return len(kb)


def proposal_delta(p: Proposal) -> int:
    return sum(clause_cost(c) for c in p.add) - sum(clause_cost(c) for c in p.remove)
```

`dreamlog/compression/__init__.py`:

```python
from .proposal import Proposal
from . import dl
```

- [ ] **Step 4: Run the new test, then the FULL suite**

Run: `python -m pytest tests/test_compression_dl.py -v --no-cov`
Expected: 2 passed.
Run: `python -m pytest tests/ -q --no-cov -m "not integration"`
Expected: same green totals as before this task plus 2 (helpers moved by re-import = zero drift). If anything else fails, a helper import was missed; fix before committing.

- [ ] **Step 5: Commit**

```bash
git add dreamlog/compression tests/test_compression_dl.py dreamlog/kb_dreamer.py
git commit -m "feat: compression package skeleton (util/proposal/dl), helpers re-imported"
```

---

## Task 3: the gate

**Files:**
- Create: `dreamlog/compression/gate.py`, `dreamlog/compression/policies.py`
- Modify: `dreamlog/kb_dreamer.py` (DreamSession gains additive `rejections` field)
- Test: `tests/test_compression_gate.py`

- [ ] **Step 1: Read `KnowledgeBase.replace_facts` in `dreamlog/knowledge.py`**

Confirm it is remove-all-then-add-all with no extra side effects (usage counters etc.). The gate's `_apply` below uses typed remove/add loops; the equivalence test in Step 2 proves end-state equality on a mixed proposal. If `replace_facts` does anything beyond remove+add, report BLOCKED with the difference.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_compression_gate.py
from dreamlog.compression.gate import (Accepted, Rejected, apply_proposal,
                                       apply_batch_with_fallback)
from dreamlog.compression.proposal import Proposal
from dreamlog.factories import atom, compound
from dreamlog.knowledge import Fact, KnowledgeBase, Rule


def _fact(name, *args):
    return Fact(compound(name, *[atom(a) for a in args]))


def _kb(*facts):
    kb = KnowledgeBase()
    for f in facts:
        kb.add_fact(f)
    return kb


class AcceptAll:
    operation = "pruning"
    require_negative_delta = False
    def pre_check(self, kb, p): return None
    def verify(self, trial, p): return None


class RejectVerify(AcceptAll):
    def verify(self, trial, p): return "verify_failed"


def _state(kb):
    return (sorted(str(f) for f in kb.facts), sorted(str(r) for r in kb.rules))


def test_accept_commits_and_returns_candidate():
    f1, f2 = _fact("p", "a"), _fact("p", "b")
    kb = _kb(f1, f2)
    p = Proposal(kind="pruning", remove=(f1,))
    res = apply_proposal(kb, p, AcceptAll())
    assert isinstance(res, Accepted)
    assert res.candidate.operation == "pruning"
    assert res.candidate.original_clauses == [f1]
    assert _state(kb)[0] == [str(f2)]


def test_reject_leaves_kb_structurally_identical():
    f1, f2 = _fact("p", "a"), _fact("p", "b")
    kb = _kb(f1, f2)
    before = _state(kb)
    res = apply_proposal(kb, Proposal(kind="pruning", remove=(f1,)), RejectVerify())
    assert isinstance(res, Rejected) and res.reason == "verify_failed"
    assert _state(kb) == before


def test_negative_delta_enforced_when_required():
    class Strict(AcceptAll):
        require_negative_delta = True
    kb = _kb(_fact("p", "a"))
    add_only = Proposal(kind="pruning",
                        add=(Rule(compound("q", atom("x")),
                                  [compound("p", atom("x"))]),))
    res = apply_proposal(kb, add_only, Strict())
    assert isinstance(res, Rejected) and res.reason == "delta"


def test_mixed_apply_matches_replace_facts_end_state():
    f1, f2, f3 = _fact("p", "a"), _fact("p", "b"), _fact("q", "c")
    rule = Rule(compound("p", atom("X")), [compound("q", atom("X"))])
    kb1 = _kb(f1, f2, f3)
    kb2 = kb1.copy()
    res = apply_proposal(kb1, Proposal(kind="generalization",
                                       remove=(f1, f2), add=(rule,)), AcceptAll())
    assert isinstance(res, Accepted)
    kb2.replace_facts([f1, f2], [rule])
    assert _state(kb1) == _state(kb2)


def test_batch_fallback_keeps_independent_items():
    # f_a removable alone, f_b not; batch fails, fallback keeps f_a removed.
    class ItemPolicy(AcceptAll):
        def __init__(self, bad): self.bad = bad
        def verify(self, trial, p):
            return "verify_failed" if p.remove and p.remove[0] == self.bad else None
        def verify_batch(self, trial, props):
            return "verify_failed"
    f_a, f_b = _fact("p", "a"), _fact("p", "b")
    kb = _kb(f_a, f_b)
    props = [Proposal(kind="pruning", remove=(f_a,)),
             Proposal(kind="pruning", remove=(f_b,))]
    accepted, rejected = apply_batch_with_fallback(kb, props, ItemPolicy(bad=f_b))
    assert len(accepted) == 1 and len(rejected) == 1
    assert _state(kb)[0] == [str(f_b)]


def test_recursionerror_maps_to_budget():
    class Boom(AcceptAll):
        def verify(self, trial, p): raise RecursionError
    kb = _kb(_fact("p", "a"))
    res = apply_proposal(kb, Proposal(kind="pruning", remove=(_fact("p", "a"),)), Boom())
    assert isinstance(res, Rejected) and res.reason == "budget"
```

- [ ] **Step 3: Run to verify failure**

Run: `python -m pytest tests/test_compression_gate.py -v --no-cov`
Expected: FAIL with `ModuleNotFoundError` on `dreamlog.compression.gate`.

- [ ] **Step 4: Write the gate**

```python
# dreamlog/compression/gate.py
"""The single accept/verify/rollback mechanism for compression proposals.

Trial-apply on a copy, policy verification, commit to the real KB on success.
RecursionError during verification maps to Rejected("budget"), matching the
catch-and-skip semantics the ops use today. Rejection reasons:
delta | verify_failed | fp_check | budget | policy.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..knowledge import Fact, KnowledgeBase, Rule
from .dl import proposal_delta
from .proposal import Proposal


@dataclass(frozen=True)
class Accepted:
    candidate: "CompressionCandidate"          # noqa: F821 (import cycle; see below)


@dataclass(frozen=True)
class Rejected:
    kind: str
    reason: str
    detail: str = ""


def _apply(kb: KnowledgeBase, p: Proposal) -> None:
    for c in p.remove:
        if isinstance(c, Rule):
            kb.remove_rule_by_value(c)
        else:
            kb.remove_fact_by_value(c)
    for c in p.add:
        if isinstance(c, Rule):
            kb.add_rule(c)
        else:
            kb.add_fact(c)


def _candidate(p: Proposal):
    from ..kb_dreamer import CompressionCandidate
    return CompressionCandidate(operation=p.kind,
                                original_clauses=list(p.remove),
                                new_clauses=list(p.add))


def apply_proposal(kb, p, policy):
    if getattr(policy, "require_negative_delta", False) and proposal_delta(p) >= 0:
        return Rejected(p.kind, "delta")
    reason = policy.pre_check(kb, p)
    if reason:
        return Rejected(p.kind, reason)
    trial = kb.copy()
    _apply(trial, p)
    try:
        reason = policy.verify(trial, p)
    except RecursionError:
        return Rejected(p.kind, "budget")
    if reason:
        return Rejected(p.kind, reason)
    _apply(kb, p)
    return Accepted(_candidate(p))


def apply_batch_with_fallback(kb, proposals, policy):
    """Operation B's semantics: try the whole batch; if the batch-level verify
    fails, fall back to applying items one at a time SEQUENTIALLY against the
    live KB (each commit affects the next item's verification)."""
    accepted: List[Accepted] = []
    rejected: List[Rejected] = []
    if not proposals:
        return accepted, rejected
    trial = kb.copy()
    for p in proposals:
        _apply(trial, p)
    try:
        batch_reason = policy.verify_batch(trial, proposals)
    except RecursionError:
        batch_reason = "budget"
    if batch_reason is None:
        for p in proposals:
            _apply(kb, p)
            accepted.append(Accepted(_candidate(p)))
        return accepted, rejected
    for p in proposals:
        res = apply_proposal(kb, p, policy)
        (accepted if isinstance(res, Accepted) else rejected).append(res)
    return accepted, rejected
```

Note on the local `CompressionCandidate` import: it lives in `kb_dreamer.py`, which will import generators, which import the gate. The function-local import breaks the cycle. (Task 10 may move `CompressionCandidate` into the package if the implementer prefers; not required.)

`dreamlog/compression/policies.py` starts minimal; per-op policies are added in Tasks 5-9 where their checks are lifted:

```python
"""Per-kind acceptance policies, lifted verbatim from the original ops."""
from typing import Optional

from ..knowledge import KnowledgeBase
from .proposal import Proposal


class Policy:
    operation = "generic"
    require_negative_delta = False

    def pre_check(self, kb: KnowledgeBase, p: Proposal) -> Optional[str]:
        return None

    def verify(self, trial_kb: KnowledgeBase, p: Proposal) -> Optional[str]:
        return None

    def verify_batch(self, trial_kb, proposals) -> Optional[str]:
        return None
```

- [ ] **Step 5: Add the additive `rejections` field to `DreamSession`**

In `kb_dreamer.py`, `DreamSession` (line ~274) gains:

```python
    rejections: List[tuple] = field(default_factory=list)   # (kind, reason) pairs
```

(Keep existing field order; append this last so positional constructions elsewhere keep working. Grep for `DreamSession(` call sites and confirm they all use keywords; they do in dream().)

- [ ] **Step 6: Run gate tests + full suite**

Run: `python -m pytest tests/test_compression_gate.py -v --no-cov` -> 6 passed.
Run: `python -m pytest tests/ -q --no-cov -m "not integration"` -> green, totals +6.

- [ ] **Step 7: Commit**

```bash
git add dreamlog/compression/gate.py dreamlog/compression/policies.py tests/test_compression_gate.py dreamlog/kb_dreamer.py
git commit -m "feat: compression gate (trial/verify/commit + batch fallback) and policy base"
```

---

## Task 4: maintenance (F + H out of the compression objective)

**Files:**
- Create: `dreamlog/compression/maintenance.py`
- Modify: `dreamlog/kb_dreamer.py`

- [ ] **Step 1: Move F, `_frequency_score`, H**

Create `maintenance.py` with module docstring "Cache-policy pair, outside the MDL objective: F evicts unused non-seed clauses; H materializes hot lemmas." Move VERBATIM as module functions (self dropped; `min_query_threshold`/`min_derivation_count` keep their defaults):

- `evict_dead_clauses(kb, min_query_threshold=10, seed_terms=None, seed_rules=None)` = body of `_prune_dead_clauses` (1488-1546) unchanged.
- `prune_suite_for_dead(suite, dead_ops)` = the inline block from `dream()` lines 340-348 (collect dead Fact terms from `op.original_clauses`, filter `suite.positive_queries`), guarded by `if suite and dead_ops`.
- `frequency_score(kb, clauses)` = `_frequency_score` (1548-1552).
- `cache_lemmas(kb, min_derivation_count=5)` = `_cache_lemmas` (1556-1582) unchanged.

Imports needed: `math`, `CompressionCandidate` (function-local import from `..kb_dreamer` to avoid the cycle, same pattern as gate.py), `Fact`, `Compound`, `_is_system_predicate` and `_collect_user_functors` from `.util`.

- [ ] **Step 2: Thin facade methods in kb_dreamer**

Replace `_prune_dead_clauses` body with a delegation (signature unchanged: tests call it):

```python
    def _prune_dead_clauses(self, kb, min_query_threshold=10,
                            seed_terms=None, seed_rules=None):
        from .compression import maintenance
        return maintenance.evict_dead_clauses(
            kb, min_query_threshold=min_query_threshold,
            seed_terms=seed_terms, seed_rules=seed_rules)
```

Same pattern for `_cache_lemmas` and `_frequency_score`. In `dream()`, replace the inline suite-pruning block (341-348) with:

```python
        from .compression import maintenance
        maintenance.prune_suite_for_dead(suite, dead_ops)
```

(Keep the `seed_terms`/`seed_rules` computation and the F-first call exactly where they are.)

- [ ] **Step 3: Verify**

Run: `python -m pytest tests/test_sleep_cycle.py -q --no-cov` -> green (F/H tests pass through dream() and the delegators).
Run: `python -m pytest tests/ -q --no-cov -m "not integration"` -> green.
Run: `python -m pytest tests/test_mdl_refactor_regression.py -q --no-cov` -> 2 passed.

- [ ] **Step 4: Commit**

```bash
git add dreamlog/compression/maintenance.py dreamlog/kb_dreamer.py
git commit -m "refactor: relocate F+H to compression.maintenance (cache pair, outside MDL objective)"
```

---

> **Anchor note for Tasks 5-9:** the line anchors in this plan refer to the file BEFORE Task 2 moved the five helpers out (~94 lines). After Task 2, locate code by SYMBOL NAME and the quoted code itself, not by absolute line number. Re-read each op fully before editing it.

## Task 5: reduce (A+B), one generator, two detectors

**Files:**
- Create: `dreamlog/compression/generators/__init__.py` (empty), `dreamlog/compression/generators/reduce.py`
- Modify: `dreamlog/compression/policies.py`, `dreamlog/kb_dreamer.py`

- [ ] **Step 1: Write the detectors (pure detection, no KB mutation of the real KB)**

`reduce.py` lifts the DETECTION halves of A and B verbatim:

```python
"""Reduce: delete clauses implied by the remainder. Detectors lifted from
Operations A (_eliminate_subsumed) and B (_prune_redundant_facts).
Detector order is part of the behavioral contract: A1 (rule-vs-rule
subsumption), then A2 (bodyless-rule-vs-fact), then B (derivability).
A2 must run against the KB AFTER A1's removals are committed."""
from typing import Iterator, List

from ...evaluator import PrologEvaluator
from ...knowledge import KnowledgeBase
from ...unification import clause_subsumes, subsumes
from ..proposal import Proposal


def propose_subsumed_rules(kb: KnowledgeBase) -> List[Proposal]:
    # A1, lifted verbatim from _eliminate_subsumed lines 407-421:
    # same index-set logic, same sorted(reverse=True) emission order.
    rules_to_remove = set()
    rules = kb.rules
    for i, rule_a in enumerate(rules):
        for j, rule_b in enumerate(rules):
            if i == j or j in rules_to_remove:
                continue
            if clause_subsumes(rule_a, rule_b) and rule_a != rule_b:
                rules_to_remove.add(j)
    return [Proposal(kind="subsumption", remove=(rules[idx],))
            for idx in sorted(rules_to_remove, reverse=True)]


def propose_subsumed_facts(kb: KnowledgeBase) -> List[Proposal]:
    # A2, lifted verbatim from lines 423-430. Call ONLY after A1 committed.
    out = []
    bodyless_rules = [r for r in kb.rules if len(r.body) == 0]
    for fact in kb.facts:
        for rule in bodyless_rules:
            if subsumes(rule.head, fact.term):
                out.append(Proposal(kind="pruning", remove=(fact,),
                                    notes={"detector": "bodyless_subsumed"}))
                break
    return out


def propose_redundant_facts(kb: KnowledgeBase, max_calls: int = 0) -> List[Proposal]:
    # B Phase 1, lifted from lines 448-459, but probing on a SCRATCH COPY so
    # detection never mutates the real KB (the original removes/re-adds on the
    # live KB; results are identical because the evaluator sees the same
    # clause sets, and dream() discards op-time usage counters anyway).
    scratch = kb.copy()
    out = []
    for fact in list(scratch.facts):
        scratch.remove_fact_by_value(fact)
        ev = PrologEvaluator(scratch, max_total_calls=max_calls)
        try:
            is_derivable = ev.has_solution(fact.term)
        except RecursionError:
            is_derivable = False
        if is_derivable:
            out.append(Proposal(kind="pruning", remove=(fact,),
                                notes={"detector": "derivable"}))
        scratch.add_fact(fact)
    return out
```

CHECK while implementing: A2's original appends a `Proposal` per fact but today's candidates use operation "subsumption" for A2 removals (line 434-435). Fix the kind above to `"subsumption"` for `propose_subsumed_facts` so `session.operations` labels are identical. (The note text stays.)

- [ ] **Step 2: Policies**

Append to `policies.py`:

```python
class SubsumptionPolicy(Policy):
    operation = "subsumption"
    require_negative_delta = True   # removal-only proposals; witness was detection


class DerivabilityPolicy(Policy):
    """Operation B's acceptance: removed facts must remain derivable."""
    operation = "pruning"
    require_negative_delta = True

    def __init__(self, max_calls: int = 0):
        self.max_calls = max_calls

    def _ev(self, kb):
        from ..evaluator import PrologEvaluator
        return PrologEvaluator(kb, max_total_calls=self.max_calls)

    def verify(self, trial_kb, p):
        ev = self._ev(trial_kb)
        ok = all(ev.has_solution(c.term) for c in p.remove)
        return None if ok else "verify_failed"

    def verify_batch(self, trial_kb, proposals):
        ev = self._ev(trial_kb)
        ok = all(ev.has_solution(c.term)
                 for p in proposals for c in p.remove)
        return None if ok else "verify_failed"
```

(Import path note: policies.py sits in `dreamlog/compression/`, so the evaluator import is `from ..evaluator import PrologEvaluator`. RecursionError propagates to the gate, which maps it to `budget`: per item that means the fact stays, exactly today's `except RecursionError: is_derivable = False`; for the batch it triggers the fallback, exactly today's `still_ok = False`.)

- [ ] **Step 3: Thin orchestrators in kb_dreamer**

Replace the BODIES of `_eliminate_subsumed` and `_prune_redundant_facts` (signatures unchanged):

```python
    def _eliminate_subsumed(self, kb):
        from .compression import gate
        from .compression.generators import reduce as reduce_gen
        from .compression.policies import SubsumptionPolicy
        ops, policy = [], SubsumptionPolicy()
        for p in reduce_gen.propose_subsumed_rules(kb):
            res = gate.apply_proposal(kb, p, policy)
            if isinstance(res, gate.Accepted):
                ops.append(res.candidate)
            else:
                self._rejections.append((p.kind, res.reason))
        for p in reduce_gen.propose_subsumed_facts(kb):   # after A1 commits
            res = gate.apply_proposal(kb, p, policy)
            if isinstance(res, gate.Accepted):
                ops.append(res.candidate)
            else:
                self._rejections.append((p.kind, res.reason))
        return ops

    def _prune_redundant_facts(self, kb, max_calls=0):
        from .compression import gate
        from .compression.generators import reduce as reduce_gen
        from .compression.policies import DerivabilityPolicy
        proposals = reduce_gen.propose_redundant_facts(kb, max_calls=max_calls)
        accepted, rejected = gate.apply_batch_with_fallback(
            kb, proposals, DerivabilityPolicy(max_calls=max_calls))
        self._rejections.extend((r.kind, r.reason) for r in rejected)
        return [a.candidate for a in accepted]
```

Add `self._rejections: list = []` in `__init__`, reset it at the top of `dream()` (`self._rejections = []`), and pass `rejections=list(self._rejections)` when constructing the final `DreamSession` (both return points construct with keywords; the early empty-KB return can keep the default).

Then DELETE the original A and B bodies (the methods above replace them).

- [ ] **Step 4: Verify hard**

Run: `python -m pytest tests/test_sleep_cycle.py -q --no-cov` -> green.
Run: `python -m pytest tests/ -q --no-cov -m "not integration"` -> green.
Run: `python -m pytest tests/test_mdl_refactor_regression.py -v --no-cov` -> 2 passed. This is the first task that can drift; if the regression fails, diff the accepted-candidate sequence (add a temporary print of `[(c.operation, len(c.original_clauses)) for c in session.operations]` on both commits) and find the ordering divergence. Do not weaken anything.

- [ ] **Step 5: Commit**

```bash
git add dreamlog/compression dreamlog/kb_dreamer.py
git commit -m "refactor: A+B become the reduce generator through the gate (bit-identical)"
```

---

## Task 6: generalize (C)

**Files:**
- Create: `dreamlog/compression/generators/generalize.py`
- Modify: `dreamlog/compression/policies.py`, `dreamlog/kb_dreamer.py`

- [ ] **Step 1: Policy**

```python
class SuiteVerifyPolicy(Policy):
    """Verify a trial KB against the dream's suite with an UNBOUNDED evaluator
    (today's C/D/E behavior). suite=None means accept without verification."""
    require_negative_delta = True

    def __init__(self, suite, operation):
        from ..evaluator import PrologEvaluator
        self._PrologEvaluator = PrologEvaluator
        self.suite = suite
        self.operation = operation

    def verify(self, trial_kb, p):
        if self.suite is None:
            return None
        result = self.suite.verify(trial_kb, lambda k: self._PrologEvaluator(k))
        return None if result.passed else "verify_failed"
```

- [ ] **Step 2: Move C as a run-form generator**

`generalize.py` gets `run(kb, suite, gate_apply, policy, min_group_size, rejections)` containing `_generalize_facts`'s ENTIRE body (lines 499-626) moved verbatim, with exactly two changes:
1. `self.min_group_size` -> the `min_group_size` parameter; `self._find_guard(...)` -> module function `find_guard(kb, functor, fact_values)` (move `_find_guard` lines 628-650 verbatim into this file).
2. The verify+apply seam (lines 601-616) is replaced by:

```python
                    proposal = Proposal(kind="generalization",
                                        remove=tuple(facts),
                                        add=tuple(new_clauses))
                    res = gate_apply(kb, proposal, policy)
                    if not isinstance(res, Accepted):
                        rejections.append((proposal.kind, res.reason))
                        continue
                    ops.append(res.candidate)
```

EVERYTHING else stays: the groups snapshot (which deliberately excludes the new `exception_*` functors), the cost check at line 561 (keep it; it is detection, and it guarantees the gate's delta check never fires), the `all_facts` rebuild and `break` after an accept (lines 618-624), and the candidate field shapes (`original_clauses=list(facts)` equivalence holds because the gate builds the candidate from the proposal's remove/add).

NOTE: today's apply uses `kb.replace_facts(facts, new_clauses)`; the gate's `_apply` uses typed remove/add loops. Task 3 Step 1 verified end-state equivalence; the regression test is the empirical check.

- [ ] **Step 3: Thin orchestrator**

```python
    def _generalize_facts(self, kb, suite=None):
        from .compression import gate
        from .compression.generators import generalize
        from .compression.policies import SuiteVerifyPolicy
        return generalize.run(kb, suite, gate.apply_proposal,
                              SuiteVerifyPolicy(suite, "generalization"),
                              self.min_group_size, self._rejections)
```

Delete the original C body and `_find_guard` from kb_dreamer.

- [ ] **Step 4: Verify**

Same three commands as Task 5 Step 4. The EX25b crafting symbolic cell (recall 0.526, 5 rules, compression 0.919) is C-dominated: it is the sharpest drift detector for this task.

- [ ] **Step 5: Commit**

```bash
git add dreamlog/compression dreamlog/kb_dreamer.py
git commit -m "refactor: C becomes the generalize generator through the gate (bit-identical)"
```

---

## Task 7: factor (D+E)

**Files:**
- Create: `dreamlog/compression/generators/factor.py`
- Modify: `dreamlog/compression/policies.py` (nothing new needed if SuiteVerifyPolicy is reused; see delta note), `dreamlog/kb_dreamer.py`

- [ ] **Step 1: Move D**

`factor.py` gets `run_invention(kb, suite, gate_apply, policy, rejections)` = `_invent_predicates` body (652-752) verbatim, with the seam (732-750) replaced by the same proposal+gate pattern (kind="invention", remove=tuple(all_original), add=tuple(all_new); on reject: record + `continue`). KEEP UNTOUCHED: the `if k + n >= n * k: continue` strict-delta detection check (line 684) and `_next_generated_name(kb, "_invented_")` at its EXACT position (line 687, BEFORE verification; name allocation reads the live KB and must stay pre-gate). The `from .skeleton import extract_skeleton` import becomes `from ...skeleton import extract_skeleton`.

D's policy: `SuiteVerifyPolicy(suite, "invention")` with `require_negative_delta = True` (D's own check already guarantees it; the gate check is consistent, never fires).

- [ ] **Step 2: Move E**

`run_extraction(kb, suite, gate_apply, policy, rejections, max_rounds=10)` = `_extract_body_patterns` body (756-828) with one structural change: compute the rewritten rules ONCE (lift the apply-block computation, lines 810-821, up to where the test-KB block sits today) and build:

```python
            proposal = Proposal(kind="extraction",
                                remove=tuple(original_rules),
                                add=(extracted_rule,) + tuple(new_rules))
            res = gate_apply(kb, proposal, policy)
            if not isinstance(res, Accepted):
                rejections.append((proposal.kind, res.reason))
                failed_keys.add(pattern_key)
                continue
            all_ops.append(res.candidate)
```

(The duplicate rewrite computation in today's verify block, lines 791-800, disappears: it was computing the same `new_body` values. The gate's trial applies the same remove/add set, so the verified KB is identical.)

E's policy MUST be `SuiteVerifyPolicy(suite, "extraction")` with `require_negative_delta` OVERRIDDEN to False: E's delta is ALWAYS +1 (spec Section 5.4, settled). Easiest: instantiate and set `policy.require_negative_delta = False`, or add a tiny subclass `ExtractionPolicy(SuiteVerifyPolicy)` with the flag False. Do the subclass (explicit beats mutation).

Move the five E helpers (`_find_best_body_pattern` 904-973, `_subseq_structural_key` 974-1009, `_iter_var_names` 1010-1017, `_compute_interface_vars` 1018-1047, `_map_interface_vars` 1048-1069) verbatim into `factor.py` as module functions; update internal `self._x(...)` calls to `x(...)`.

- [ ] **Step 3: Thin orchestrators**

`_invent_predicates` and `_extract_body_patterns` delegate exactly like Task 6's pattern (max_rounds default preserved on the facade signature). Delete originals + helper methods from kb_dreamer.

- [ ] **Step 4: Verify**

Same three commands. The D/E unit tests in test_sleep_cycle.py (invention + extraction classes) are the focused drift detectors here.

- [ ] **Step 5: Commit**

```bash
git add dreamlog/compression dreamlog/kb_dreamer.py
git commit -m "refactor: D+E become the factor generator through the gate (E delta=+1 preserved)"
```

---

## Task 8: closure (I)

**Files:**
- Create: `dreamlog/compression/generators/closure.py`
- Modify: `dreamlog/compression/policies.py`, `dreamlog/kb_dreamer.py`

- [ ] **Step 1: Policy**

```python
class BoundedSuitePolicy(SuiteVerifyPolicy):
    """Suite verification with a BOUNDED evaluator (Operation I: max_calls=5000)."""
    def __init__(self, suite, operation, max_calls):
        super().__init__(suite, operation)
        self.max_calls = max_calls

    def verify(self, trial_kb, p):
        if self.suite is None:
            return None
        result = self.suite.verify(
            trial_kb, lambda k: self._PrologEvaluator(k, max_total_calls=self.max_calls))
        return None if result.passed else "verify_failed"
```

- [ ] **Step 2: Move I as run-form**

`closure.py` gets `run(kb, suite, gate_apply, policy, min_base_facts, rejections)` = `_discover_recursion` body (830-902) verbatim with the seam (884-900) replaced: proposal (kind="recursion", remove=tuple(r_facts), add=(base_rule, rec_rule)); on Rejected -> record + `continue` (today both `except RecursionError: continue` and `if not result.passed: continue` map to this, since the gate converts RecursionError to `budget`); on Accepted -> `return [res.candidate]`. End of loops -> `return []`. The at-most-one-candidate-per-call contract is preserved by the early return.

- [ ] **Step 3: Thin orchestrator + verify + commit**

`_discover_recursion(self, kb, suite=None, max_calls=5000)` delegates with `BoundedSuitePolicy(suite, "recursion", max_calls)` and `self.min_base_facts`. Tests call this method directly (x4): signature unchanged. Run the three verification commands (the EX27/EX28 recursive cells in the regression are the drift detectors). Commit:

```bash
git add dreamlog/compression dreamlog/kb_dreamer.py
git commit -m "refactor: I becomes the closure generator through the gate (bit-identical)"
```

---

## Task 9: llm (G). The most delicate task: use the strongest available implementer and re-read the whole current method first.

**Files:**
- Create: `dreamlog/compression/generators/llm.py`
- Modify: `dreamlog/compression/gate.py` (add `apply_batch_staged_combined`), `dreamlog/compression/policies.py`, `dreamlog/kb_dreamer.py`

- [ ] **Step 1: Re-read the CURRENT `_build_op_g_prompt`, `_llm_propose`, `_llm_compress`, `_parse_llm_rules`, `_build_rule_from_parsed` in full.** Line anchors have shifted twice by now; locate by name. Map `_llm_compress`'s phases before touching anything: Phase 3 helper/main split (existing_functors = fact functors UNION rule-head functors), MAX_CALLS = max(500, len(kb) * 10), Phase 4 per-rule battery (test_kb = kb.copy() + helpers + rule; >= 2 derivable existing facts with RecursionError -> skip rule; suite verify with bounded ev; false-positive enumeration unless open_world, RecursionError -> treat as FP), Phase 5 combined verify on kb.copy() + ALL accepted (helpers included; failure wipes EVERYTHING including helpers), commit loop adding each accepted rule with one CompressionCandidate("llm_compression", [], [rule]) EACH (helpers get candidates too).

- [ ] **Step 2: Move the pure proposal stage**

`llm.py` gets module functions, moved verbatim: `build_op_g_prompt(kb, max_prompt_facts)`, `propose_rules(kb, llm_client, max_prompt_facts)` (= `_llm_propose` minus self: takes the client and max_prompt_facts explicitly), `parse_llm_rules(llm_client, prompt, parse_llm_response)`, `build_rule_from_parsed(rule_data)`. Helpers `_strip_llm_noise` / `_filter_cyclic_rules` / `_collect_user_functors` come from `..util`.

Facade keeps thin `_build_op_g_prompt(self, kb)` and `_llm_propose(self, kb)` delegators (tests x7 and `ex28_probe.py` call them; signatures unchanged).

- [ ] **Step 3: Add the staged-combined batch to gate.py**

```python
def apply_batch_staged_combined(kb, context_proposals, item_proposals, policy):
    """Operation G Phase 4+5 semantics. Context proposals (helper rules) are
    present in every item trial and committed only if the combined check
    passes; items are verified INDEPENDENTLY against kb + context + item
    (not cumulatively); then the combined set is verified; on combined
    failure NOTHING commits (context included)."""
    staged, rejected = [], []
    for p in item_proposals:
        reason = policy.pre_check(kb, p)
        if reason:
            rejected.append(Rejected(p.kind, reason)); continue
        trial = kb.copy()
        for cp in context_proposals:
            _apply(trial, cp)
        _apply(trial, p)
        try:
            reason = policy.verify(trial, p)
        except RecursionError:
            reason = "budget"
        if reason:
            rejected.append(Rejected(p.kind, reason)); continue
        staged.append(p)
    to_commit = list(context_proposals) + staged
    if to_commit:
        trial = kb.copy()
        for p in to_commit:
            _apply(trial, p)
        try:
            combined_reason = policy.verify_combined(trial)
        except RecursionError:
            combined_reason = "budget"
        if combined_reason:
            rejected.extend(Rejected(p.kind, combined_reason) for p in to_commit)
            return [], rejected
    accepted = []
    for p in to_commit:
        _apply(kb, p)
        accepted.append(Accepted(_candidate(p)))
    return accepted, rejected
```

FIDELITY CHECKS against the re-read in Step 1 (adjust this function, not the semantics, if the original differs): (a) does Phase 5 run when there are helpers but ZERO accepted mains? Today `if accepted_rules and suite is not None` with accepted_rules pre-seeded with helpers -> YES, helpers alone still get combined-verified and committed. The code above matches (`if to_commit`). (b) When `suite is None`, Phase 5 is skipped but the commit still happens: policy.verify_combined must return None when suite is None. (c) Helpers bypass the per-item battery entirely. (d) Candidate ORDER in session.operations today = helpers first, then mains in acceptance order: preserved by `to_commit` ordering.

- [ ] **Step 4: LlmPolicy + thin `_llm_compress`**

`LlmPolicy(suite, max_calls, open_world, kb)` in policies.py carries the Phase 4 battery, lifted verbatim into `verify(trial, p)`: count derivable existing facts with the head functor on the trial (note the trial ALREADY contains kb + helpers + rule, so iterate `kb_snapshot_facts` captured at construction, matching today's `for fact in kb.facts` against test_kb); `< 2 -> "policy"`; suite verify bounded -> `"verify_failed"`; FP enumeration unless open_world -> `"fp_check"`. `verify_combined(trial)` = bounded suite verify or None if suite is None.

`_llm_compress(self, kb, suite=None)` becomes: guard on `self.llm_client`; `parsed = self._llm_propose(kb)`; if none, return []; lift Phase 3 split verbatim (helpers/mains); build context_proposals (one add-only Proposal per helper, kind="llm_compression") and item_proposals (one per main rule); call `apply_batch_staged_combined`; record rejections; return `[a.candidate for a in accepted]`. The monkeypatch in `experiments/ablation_and_scale.py` keeps working because dream() still calls `self._llm_compress(kb, suite)`.

- [ ] **Step 5: Verify HARD, then commit**

Run: `python -m pytest tests/test_sleep_cycle.py -k "op_g or llm_propose or full_pipeline or recursive" -v --no-cov` -> all the AC2 tests pass.
Run: `python -m pytest tests/ -q --no-cov -m "not integration"` -> green.
Run: `python -m pytest tests/test_mdl_refactor_regression.py -q --no-cov` -> 2 passed.

```bash
git add dreamlog/compression dreamlog/kb_dreamer.py
git commit -m "refactor: G routes through the staged-combined gate (Phase 3-5 semantics preserved)"
```

---

## Task 10: facade tidy, rejections, docs

**Files:**
- Modify: `dreamlog/kb_dreamer.py`, `CLAUDE.md`

- [ ] **Step 1:** Remove now-unused imports from kb_dreamer.py (run `python -m pyflakes dreamlog/kb_dreamer.py` or read the import block against remaining uses). Confirm `_name_invented_predicates` / `_rename_predicate` and the verification-suite code still live here and work.
- [ ] **Step 2:** Confirm `wc -l dreamlog/kb_dreamer.py` is under ~600 (spec target ~550; a small overshoot is acceptable, gross overshoot means something did not move).
- [ ] **Step 3:** Confirm `DreamSession.rejections` is populated end to end: add one unit test to `tests/test_compression_gate.py` that dreams a KB where a generalization candidate fails suite verification and asserts `session.rejections` contains a `("generalization", ...)` entry. (Construct: artisan-style KB plus a contradicting negative query is fiddly; simpler: monkeypatch `SuiteVerifyPolicy.verify` to return "verify_failed" for one call and assert the rejection is recorded. Behavior-level enough for an additive observability field.)
- [ ] **Step 4:** Update `CLAUDE.md` Layer 5: describe the compression package (generators + policies + one gate + maintenance), note F/H are cache policies outside the MDL objective, keep the A-I operation descriptions (they still describe behavior), and note `description_length` lives in `dreamlog/compression/dl.py` (clause count in P1).
- [ ] **Step 5:** Full suite + regression green; commit:

```bash
git add dreamlog/kb_dreamer.py CLAUDE.md tests/test_compression_gate.py
git commit -m "refactor: facade tidy, rejections observability, CLAUDE.md architecture update"
```

---

## Task 11: final verification (AC1-AC5 + R4)

- [ ] **Step 1:** `python -m pytest tests/ -q --no-cov -m "not integration"` -> green; record totals (expect prior totals + ~10 new tests).
- [ ] **Step 2:** `python -m pytest tests/test_mdl_refactor_regression.py tests/test_compression_gate.py tests/test_compression_dl.py -v --no-cov` -> all pass.
- [ ] **Step 3 (R4 performance):** `python benchmarks/sleep_cycle_bench.py` and compare against `benchmarks/baseline.json` (CLAUDE.md: these are the canonical performance baselines). Acceptance: no operation slower than ~2x baseline. If reduce/generalize regressed past that, implement the in-place-with-undo fast path inside `gate._apply`'s caller (spec R4) WITHOUT changing accept semantics, and re-run.
- [ ] **Step 4 (AC5):** `grep -rn "kb_dreamer import\|from dreamlog.kb_dreamer\|from .kb_dreamer" experiments/ integrations/ tests/ | head -30` and confirm every imported name still exists. Run the GPU-free smoke: `python -c "import importlib.util,pathlib; p=pathlib.Path('experiments/ex28_llm_role.py'); s=importlib.util.spec_from_file_location('m',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print('import OK')"`.
- [ ] **Step 5:** Commit any final adjustments; report: suite totals, regression status, bench deltas, kb_dreamer.py line count.

```bash
git add -A
git commit -m "refactor: MDL unified gate P1+P2 complete (suite green, regression exact, bench within budget)"
```

---

## Self-Review

**Spec coverage:** G1 gate (Task 3, plus staged-combined in Task 9). G2 generators (Tasks 5-9; run-form per spec amendment). G3 dl.py (Task 2). G4 merges/relocation (Tasks 4, 5, 7). G5 bit-identical (every task gates on the Task 1 regression + full suite; seams quoted from the source read). G6 surface unchanged (thin orchestrators with original names/signatures; AC5 in Task 11). Non-goals respected: no semantic upgrades anywhere; E's delta=+1 explicitly preserved; reduce = union of today's exact checks.

**Placeholder scan:** Tasks 1-8 and 10-11 carry complete code or exact seam replacements quoted against read source. Task 9 deliberately front-loads a full re-read (Step 1) and pins four fidelity checks (a)-(d) instead of inlining 250 moved lines; this mirrors the spec's instruction that policies are lifted by reading, and AC2/AC3 arbitrate. Two flagged uncertainties in Task 1 Step 2 are read-then-adapt steps with explicit expected shapes, not vague TODOs.

**Type consistency:** `Proposal(kind, remove, add, notes)` used identically in Tasks 2, 3, 5-9. `Policy.pre_check/verify/verify_batch` (Task 3) extended by `verify_combined` only for the llm policy (Task 9, where `apply_batch_staged_combined` is also added). `gate.apply_proposal(kb, p, policy)` signature consistent across Tasks 3, 5-8. `self._rejections` introduced in Task 5 and used through Task 9; wired to `DreamSession.rejections` (field added Task 3, populated Task 5, tested Task 10). Operation label strings match today's exactly: subsumption, pruning, generalization, invention, extraction, recursion, llm_compression, dead_clause, lemma_cache.

**Known judgment calls (documented, not hidden):** (1) B's detection probes on a scratch copy instead of the live KB; justified in Task 5 code comments (identical evaluator-visible state; usage counters discarded by dream() anyway); the regression test arbitrates. (2) The gate builds candidates from proposals, so `original_clauses`/`new_clauses` are list copies of the same objects the ops used today. (3) Task 5's A2 `kind` correction is called out explicitly so the label matches today's "subsumption".

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-09-mdl-unified-gate.md`. Two execution options:

1. **Subagent-Driven (recommended):** fresh implementer per task, spec + quality review between tasks. Task 1 (the regression baseline) is the natural first gate: if it does not pass against CURRENT code, nothing else proceeds. Task 9 should use the strongest available implementer model.
2. **Inline Execution:** execute tasks in this session with checkpoints.

Which approach?
