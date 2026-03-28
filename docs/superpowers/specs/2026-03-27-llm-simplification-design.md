# LLM Provider Simplification + Pythonic API Cleanup + LLM-Assisted Sleep Cycle

**Date**: 2026-03-27
**Status**: Design
**Scope**: Simplify LLM providers, clean up pythonic.py, add LLM-assisted naming and compression to sleep cycle

## Part 1: LLM Provider Simplification

### Current state

`llm_providers.py` is 517 lines containing a Protocol, an ABC base class, a
generic HTTP provider, 4 concrete providers (OpenAI, Anthropic, Ollama, URL-based),
an LLMResponse dataclass with a grandparent fallback hack, and a factory function.
Most of this machinery exists to manage HTTP calls via urllib.

### Design

Replace the entire file with a single ~30-line `LLMClient` class that uses the
official OpenAI and Anthropic Python SDKs. Ollama supports the OpenAI HTTP format
at `/v1/chat/completions`, so the OpenAI SDK covers it.

```python
class LLMClient:
    """Thin wrapper around OpenAI/Anthropic SDK."""
    def __init__(self, provider="openai", base_url=None, api_key=None,
                 model="gpt-4o-mini", temperature=0.1, max_tokens=500):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        if provider == "anthropic":
            from anthropic import Anthropic
            self._client = Anthropic(api_key=api_key)
        else:
            from openai import OpenAI
            self._client = OpenAI(base_url=base_url, api_key=api_key or "dummy")

    def complete(self, prompt, **kwargs):
        model = kwargs.get("model", self.model)
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if self.provider == "anthropic":
            resp = self._client.messages.create(
                model=model, max_tokens=max_tokens, temperature=temperature,
                messages=[{"role": "user", "content": prompt}])
            return resp.content[0].text
        else:
            resp = self._client.chat.completions.create(
                model=model, temperature=temperature, max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}])
            return resp.choices[0].message.content
```

No Protocol needed (duck typing). No base class. No factory (just construct
directly). Parameters are public attributes (`client.model = "llama3"`).

For Ollama: `LLMClient(base_url="http://localhost:11434/v1", model="llama3")`

### Location

`LLMClient` lives in `dreamlog/llm_client.py` (new file, ~30 lines). The old
`dreamlog/llm_providers.py` is deleted.

### Dependencies

Add to `pyproject.toml`:
```toml
[project.optional-dependencies]
llm = [
    "openai>=1.0.0",
    "anthropic>=0.20.0",
]
```

The SDKs are imported lazily inside `__init__`, so DreamLog works without them
(the core logic programming + sleep cycle need no LLM). Attempting to create an
LLMClient without the SDK installed raises a clear ImportError.

### What gets deleted

- `BaseLLMProvider` (ABC with parameter management, caching, clone_with_parameters)
- `URLBasedProvider` (generic urllib HTTP client)
- `OllamaProvider` (redundant, OpenAI format works)
- `LLMResponse.from_text` grandparent fallback hack (lines 41-47)
- `create_provider` factory function
- `create_llm_provider` backward compat alias
- All urllib boilerplate (~300 lines)

### What moves

- `LLMResponse` dataclass stays (used by retry wrapper and llm_hook). Moves to
  `llm_response_parser.py` where the parsing logic already lives.
- `generate_knowledge` logic (prompt construction + `complete` + parse response)
  stays in `llm_hook.py` where it's called. It's not a provider responsibility.

### Cascade updates

| File | Change |
|------|--------|
| `dreamlog/llm_client.py` | **New**: LLMClient class |
| `dreamlog/llm_providers.py` | **Delete** |
| `dreamlog/llm_response_parser.py` | Receive LLMResponse dataclass |
| `dreamlog/llm_hook.py` | Import LLMClient from llm_client, inline generate_knowledge |
| `dreamlog/llm_retry_wrapper.py` | Update to use LLMClient interface (complete + public attrs) |
| `dreamlog/llm_judge.py` | Fix `generate()` -> `complete()` bug |
| `dreamlog/correction_retry.py` | Fix `generate()` -> `complete()` bug |
| `dreamlog/pythonic.py` | Use LLMClient directly instead of create_provider |
| `dreamlog/tui.py` | Use `client.model` instead of `provider.set_parameter("model", ...)` |
| `dreamlog/engine.py` | Update import |
| `dreamlog/__init__.py` | Update exports |
| `tests/mock_provider.py` | Simplify to just `complete()` method + public attrs |
| `tests/test_llm_providers.py` | Rewrite for new interface |
| `pyproject.toml` | Add `[llm]` optional dependency group |

### MockLLMProvider

Simplified to match LLMClient interface:

```python
class MockLLMProvider:
    def __init__(self, responses=None, model="mock", temperature=0.1, max_tokens=500):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._responses = responses or []
        self._call_count = 0

    def complete(self, prompt, **kwargs):
        self._call_count += 1
        if self._responses:
            return self._responses[min(self._call_count - 1, len(self._responses) - 1)]
        return self._default_response(prompt)
```

## Part 2: Pythonic API Cleanup

### Delete (~200 lines)

- `visualize()`: Requires networkx + matplotlib, speculative, unused
- `to_dataframe()`: Requires pandas, speculative, unused
- `map_query()` / `filter_query()`: List comprehensions with extra steps
- `demo()`: Example code belongs in examples/, not in the module
- `RuleBuilder.then()`: Alias for `and_()`, unnecessary

### Simplify

- **Shared arg-to-term conversion**: `fact()` and `query()` duplicate arg-to-term
  logic. Extract `_to_term(arg) -> Term` and `_to_terms(*args) -> List[Term]`
  helpers.
- **transaction()**: Use `kb.copy()` / `kb.restore_from()` instead of manually
  manipulating `_facts` / `_rules` / `_rebuild_indices`.
- **LLM initialization**: Use `LLMClient` directly instead of create_provider +
  retry wrapper setup.

### Keep

- `DreamLog` class with `fact()`, `rule()`, `query()`, `ask()`, `find_all()`,
  `find_one()`, `parse()`, `load()`, `save()`, `clear()`, `stats`, `facts()`
- `QueryResult` with dict/attribute access
- `RuleBuilder` with `when()` / `and_()` / `build()`
- `dreamlog()` factory function

### Result

553 lines -> ~250 lines. Same user-facing functionality.

## Part 3: LLM-Assisted Sleep Cycle Features

### Integration point

`KnowledgeBaseDreamer` accepts an optional `LLMClient`:

```python
dreamer = KnowledgeBaseDreamer(llm_client=my_client)
session = dreamer.dream(kb)  # runs symbolic ops + LLM-assisted ops
```

When `llm_client` is None (default), behavior is unchanged (pure symbolic).

### Feature A: LLM-Assisted Naming

After Operations D and E create `_invented_N` and `_extracted_N` predicates,
ask the LLM to suggest descriptive names.

**Prompt**: Show the LLM the invented predicate's rules and ask for a name.

```
This predicate was discovered by compressing a knowledge base:

_invented_0(R, X, Y) :- call(R, X, Y).
_invented_0(R, X, Z) :- call(R, X, Y), _invented_0(R, Y, Z).

It is used as:
  ancestor(X, Y) :- _invented_0(parent, X, Y).
  reachable(X, Y) :- _invented_0(edge, X, Y).

Suggest a short, descriptive name for this predicate (one word or short phrase,
lowercase with underscores). Reply with just the name, nothing else.
```

**Application**: Rename the predicate throughout the KB. Verify the rename
doesn't break anything (same verification suite). If it does, keep the original
name.

**When it runs**: After all symbolic operations, before final verification.
Optional (only when llm_client is provided).

### Feature B: LLM-Assisted Compression

Ask the LLM to propose rules that derive existing facts from other facts. This
catches cross-functor relationships that symbolic methods miss.

**Prompt**: Show the LLM a sample of the KB's facts and ask for rules.

```
Here are facts from a knowledge base:

parent(john, mary). parent(mary, alice).
male(john). male(bob).
female(mary). female(alice).
father(john, mary).

Can you propose rules that derive some of these facts from others?
Use S-expression syntax: (rule_head X Y) :- (body1 X Z), (body2 Z Y)

Reply with rules only, one per line.
```

**Validation**: Parse the response, construct Rule objects, verify each proposed
rule against the KB:
1. The rule must derive at least one existing fact (it's not vacuous)
2. The rule must not derive anything that contradicts the KB (no false positives)
3. Adding the rule must pass the verification suite
4. MDL: the rule + removal of derived facts must decrease clause count

**When it runs**: After symbolic operations, after naming. This is a new
Operation G in the sleep cycle.

**Scope restriction**: Only propose rules where the head predicate already has
facts in the KB. Don't invent entirely new predicates (that's for the LLM
wake-phase hook, not the sleep cycle).

### Operation ordering (updated)

A -> B -> C -> D -> E -> F -> (LLM naming) -> (LLM compression / Op G)

LLM operations only run when `llm_client` is not None.

## File Plan

| File | Action | Description |
|------|--------|-------------|
| `dreamlog/llm_client.py` | **New** | LLMClient class (~30 lines) |
| `dreamlog/llm_providers.py` | **Delete** | Replaced by llm_client.py |
| `dreamlog/llm_response_parser.py` | **Edit** | Receive LLMResponse dataclass |
| `dreamlog/llm_hook.py` | **Edit** | Use LLMClient, inline generate_knowledge |
| `dreamlog/llm_retry_wrapper.py` | **Edit** | Update to LLMClient interface |
| `dreamlog/llm_judge.py` | **Edit** | Fix generate() -> complete() |
| `dreamlog/correction_retry.py` | **Edit** | Fix generate() -> complete() |
| `dreamlog/pythonic.py` | **Edit** | Simplify (553 -> ~250 lines) |
| `dreamlog/tui.py` | **Edit** | Use LLMClient public attrs |
| `dreamlog/engine.py` | **Edit** | Update import |
| `dreamlog/kb_dreamer.py` | **Edit** | Add optional llm_client, naming + compression |
| `dreamlog/__init__.py` | **Edit** | Update exports |
| `tests/mock_provider.py` | **Edit** | Simplify mock to match LLMClient |
| `tests/test_llm_providers.py` | **Edit** | Rewrite for LLMClient |
| `pyproject.toml` | **Edit** | Add llm optional deps |

## Test Strategy

### LLMClient tests
- OpenAI provider creates client with base_url
- Anthropic provider creates client
- complete() returns string
- Public attributes (model, temperature, max_tokens) readable and writable
- ImportError when SDK not installed (mock the import)

### Pythonic cleanup tests
- Existing test_pythonic.py tests still pass after cleanup
- transaction() uses copy/restore_from correctly
- Deleted methods (visualize, to_dataframe, map_query, filter_query) no longer exist

### LLM-assisted naming tests (with MockLLMProvider)
- Invented predicate gets renamed
- Extracted predicate gets renamed
- Rename is consistent throughout KB
- Verification failure -> keeps original name
- No LLM client -> naming skipped

### LLM-assisted compression tests (with MockLLMProvider)
- Proposed rule that derives existing facts is accepted
- Proposed rule that over-generates is rejected
- MDL check: rule + fact removal must decrease clause count
- Verification failure -> rule rejected
- No LLM client -> compression skipped

### Benchmark comparison
- Run benchmarks/sleep_cycle_bench.py before and after
- Compare against baseline.json
- Symbolic compression should be identical (LLM features are additive)
- With mock LLM, verify the pipeline works end-to-end
