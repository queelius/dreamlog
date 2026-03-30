# Derivation Tree Tracking and Proof-Based Compression

**Date**: 2026-03-29
**Status**: Design
**Scope**: Record proof traces during wake, cache lemmas and extract common subtrees during sleep

## Context

The sleep cycle analyzes the KB statically (what rules exist). It cannot see what
actually happens during resolution: which intermediate terms are derived, which
rule chains fire together, which proof paths are taken repeatedly. Derivation tree
tracking adds this dynamic dimension.

DreamCoder records derivation traces (the program that solved each task) and
compresses them to find common sub-programs. This is the DreamLog equivalent:
record proof trees, find common subtrees, extract as reusable rules or cached
lemmas.

## Part A: Derivation Recording + Lemma Caching

### Recording derived terms

During query resolution, the evaluator produces intermediate derived terms
(sub-goals that are successfully resolved). These are currently invisible: only
the final solution bindings are yielded. We add a `_derivation_log` to the
KnowledgeBase that accumulates derived terms with their derivation depth.

**In the evaluator**, when a goal is successfully resolved (a solution path
exists), record the ground form of the goal term:

```python
# After a fact or rule match yields solutions
if is_ground(resolved_term):
    self.kb.record_derivation(resolved_term)
```

Only record ground terms (fully instantiated). Non-ground intermediate goals
are too general to be useful as lemmas.

**KnowledgeBase additions:**

```python
self._derivation_counts: Dict[int, int] = {}  # hash(term) -> count

def record_derivation(self, term: Term) -> None:
    key = hash(term)
    self._derivation_counts[key] = self._derivation_counts.get(key, 0) + 1

def get_derivation_count(self, term: Term) -> int:
    return self._derivation_counts.get(hash(term), 0)

def get_frequent_derivations(self, min_count: int = 5) -> List[Tuple[Term, int]]:
    """Return terms derived at least min_count times that are NOT already facts."""
    # Implementation scans _derivation_counts and checks against _facts
```

### Operation H: Lemma Caching

During sleep, find terms that are:
1. Frequently derived (count >= threshold, default 5)
2. Not already stored as facts
3. Derivable from the current KB (sanity check)

Add these as facts (cached lemmas). This speeds up future queries: instead of
re-deriving `ancestor(john, alice)` through a chain of parent lookups, it's
retrieved directly as a fact.

**MDL note**: Adding lemma facts increases clause count. The benefit is query
performance, not clause-count compression. This is a different optimization
objective than Operations A-G. We track it separately as "lemma_cache" operations.

**Naming**: Lemma facts are just regular facts (e.g., `ancestor(john, alice).`).
They don't need special naming because they're ground terms identical to what
the rules would derive.

**Verification**: Lemma facts should not change the KB's behavior (they're
already derivable). Verify by checking that no negative query in the verification
suite becomes positive.

### When to record

Recording happens during the evaluator's normal resolution. It adds one hash
lookup per successful goal resolution (negligible overhead, same as usage
tracking).

Recording is enabled/disabled via `kb.enable_derivation_tracking()` and
`kb.disable_derivation_tracking()`. The dreamer disables it during verification
queries (same as usage tracking).

### Interaction with other operations

Operation H runs after all other operations (A-G). Lemma facts are regular facts,
so Op B in the next cycle might prune them if they become derivable through a
shorter path. This is correct: the lemma is no longer needed.

## Part B: Proof Tree Recording + Common Subtree Extraction

### Proof tree data structure

```python
@dataclass
class ProofNode:
    """A node in a proof tree."""
    goal: Term                        # the goal that was resolved
    clause: Optional[Union[Fact, Rule]]  # the clause used to resolve it
    children: List['ProofNode']       # sub-goal resolutions
    depth: int
```

When the evaluator yields a solution, it also yields the proof tree that produced
it. The tree captures: which clause resolved each goal, and what sub-goals were
generated and how they were resolved in turn.

### Recording proof trees

Modify `_solve_goals` to thread a `ProofNode` through the resolution process.
When a fact matches, create a leaf node. When a rule matches and its body goals
are resolved, create an internal node with children. When a solution is yielded,
the accumulated tree is the proof.

**Implementation approach**: Add an optional `proof_tree` parameter to
`_solve_goals`. When enabled, each recursive call builds up the tree. When a
solution is yielded, store the tree in a `_proof_log` on the KB.

**Performance**: Proof tree construction allocates ProofNode objects on each
resolution step. This is more overhead than simple counters. Enable only when
needed (e.g., before a dream cycle, during a "learning" wake phase).

### Common subtree extraction

During sleep, analyze the proof log for common subtrees:

1. Hash each subtree by its structure (clause sequence, ignoring specific bindings)
2. Group identical subtrees
3. For each group with N >= threshold occurrences, extract as a new rule

**Example**: If many proof trees contain the subtree:
```
ancestor(X, Z) via:
  parent(X, Y) [fact]
  ancestor(Y, Z) via:
    parent(Y, Z) [fact]
```

This is a 2-step ancestor derivation. It could be extracted as:
```
_proof_lemma_0(X, Z) :- parent(X, Y), parent(Y, Z).
```

Which is the "grandparent" pattern discovered from proof traces rather than
static rule analysis.

### Difference from Operation E

Operation E finds common body sub-sequences across RULE DEFINITIONS. Proof tree
extraction finds common DERIVATION PATTERNS across QUERY RESULTS. They can
discover different things:

- Op E: "These rules have the same body prefix" (static)
- Proof trees: "These queries always resolve through the same chain" (dynamic)

Example where they differ: If rule `f(X) :- g(X), h(X)` and rule
`p(X) :- g(X), h(X)` share the body `g(X), h(X)`, Op E finds it. But if
`f(X) :- g(X)` and `g(X) :- h(X)` chain through two separate rules to always
resolve as `f->g->h`, only proof tree analysis sees this (the chain crosses
rule boundaries).

## File Plan

| File | Action | Description |
|------|--------|-------------|
| `dreamlog/proof_tree.py` | **New** | ProofNode, proof tree construction utilities |
| `dreamlog/knowledge.py` | **Edit** | Add derivation_counts, proof_log, enable/disable |
| `dreamlog/evaluator.py` | **Edit** | Record derivations + optional proof tree threading |
| `dreamlog/kb_dreamer.py` | **Edit** | Add Operation H (lemma caching), proof subtree extraction |
| `tests/test_proof_tree.py` | **New** | Proof tree construction and comparison tests |
| `tests/test_sleep_cycle.py` | **Edit** | Add Operation H and proof-based compression tests |

## Test Strategy

### Part A tests
- Derivation counts accumulate across queries
- Frequent derivations identified correctly
- Lemma caching adds derived term as fact
- Lemma fact speeds up re-derivation (fact lookup vs rule chain)
- Lemma caching doesn't change KB behavior (verification passes)
- Derivation tracking disabled during dream verification
- Terms already stored as facts not re-cached

### Part B tests
- Proof tree correctly captures fact matches as leaf nodes
- Proof tree captures rule applications with children
- Recursive proofs produce nested trees
- Common subtrees identified across different query proof trees
- Extracted subtree becomes a new rule
- Cross-rule-boundary chains detected (f->g->h across two rules)
- Proof tree disabled when not needed (performance)
