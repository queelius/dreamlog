# LLM Simplification + Pythonic Cleanup + LLM-Assisted Sleep Cycle

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace 517-line LLM provider system with 30-line LLMClient using official SDKs, trim pythonic.py from 553 to ~250 lines, add LLM-assisted naming and compression to the sleep cycle.

**Architecture:** LLMClient wraps OpenAI/Anthropic SDKs with public attrs for config. LLMResponse moves to llm_response_parser.py. All 25 import sites get updated. KnowledgeBaseDreamer accepts optional llm_client for naming invented predicates and proposing cross-functor rules.

**Tech Stack:** Python 3.8+, openai SDK, anthropic SDK (optional deps), pytest.

**Spec:** `docs/superpowers/specs/2026-03-27-llm-simplification-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `dreamlog/llm_client.py` | **Create** | LLMClient class (~30 lines) |
| `dreamlog/llm_providers.py` | **Delete** | Replaced by llm_client.py |
| `dreamlog/llm_response_parser.py` | **Edit** | Receive LLMResponse dataclass from old module |
| `dreamlog/llm_hook.py` | **Edit** | Import from llm_client, remove LLMProvider type hint |
| `dreamlog/llm_retry_wrapper.py` | **Edit** | Update to LLMClient interface |
| `dreamlog/llm_judge.py` | **Edit** | Fix generate()->complete(), update import |
| `dreamlog/correction_retry.py` | **Edit** | Fix generate()->complete(), update import |
| `dreamlog/pythonic.py` | **Edit** | Simplify (553->~250 lines), use LLMClient |
| `dreamlog/tui.py` | **Edit** | Use public attrs instead of get/set_parameter |
| `dreamlog/engine.py` | **Edit** | Update import |
| `dreamlog/kb_dreamer.py` | **Edit** | Add optional llm_client, naming + Op G |
| `dreamlog/__init__.py` | **Edit** | Update exports |
| `tests/mock_provider.py` | **Edit** | Simplify to match LLMClient interface |
| `tests/test_llm_providers.py` | **Rewrite** | Test LLMClient instead of old classes |
| `pyproject.toml` | **Edit** | Add llm optional deps |
| `tests/test_sleep_cycle.py` | **Edit** | Add naming + Op G tests |

---

### Task 1: Create LLMClient + update pyproject.toml

**Files:**
- Create: `dreamlog/llm_client.py`
- Modify: `pyproject.toml`
- Create: `tests/test_llm_client.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_llm_client.py
import pytest
from unittest.mock import patch, MagicMock


class TestLLMClient:
    def test_openai_init(self):
        """OpenAI client created with base_url."""
        with patch("openai.OpenAI") as mock_cls:
            from dreamlog.llm_client import LLMClient
            client = LLMClient(provider="openai", base_url="http://localhost:11434/v1",
                               api_key="test", model="llama3")
            assert client.model == "llama3"
            assert client.provider == "openai"
            mock_cls.assert_called_once()

    def test_anthropic_init(self):
        """Anthropic client created."""
        with patch("anthropic.Anthropic") as mock_cls:
            from dreamlog.llm_client import LLMClient
            client = LLMClient(provider="anthropic", api_key="test", model="claude-3-haiku-20240307")
            assert client.provider == "anthropic"
            mock_cls.assert_called_once()

    def test_public_attrs_writable(self):
        """Model, temperature, max_tokens are public and writable."""
        with patch("openai.OpenAI"):
            from dreamlog.llm_client import LLMClient
            client = LLMClient(model="gpt-4")
            assert client.model == "gpt-4"
            client.model = "gpt-3.5-turbo"
            assert client.model == "gpt-3.5-turbo"
            client.temperature = 0.5
            assert client.temperature == 0.5

    def test_complete_openai(self):
        """complete() calls OpenAI chat completions."""
        with patch("openai.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "test response"
            mock_client.chat.completions.create.return_value = mock_response

            from dreamlog.llm_client import LLMClient
            client = LLMClient(model="gpt-4")
            result = client.complete("hello")
            assert result == "test response"

    def test_complete_anthropic(self):
        """complete() calls Anthropic messages."""
        with patch("anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_response = MagicMock()
            mock_response.content = [MagicMock()]
            mock_response.content[0].text = "anthropic response"
            mock_client.messages.create.return_value = mock_response

            from dreamlog.llm_client import LLMClient
            client = LLMClient(provider="anthropic", api_key="test")
            result = client.complete("hello")
            assert result == "anthropic response"

    def test_ollama_via_openai(self):
        """Ollama uses OpenAI SDK with custom base_url."""
        with patch("openai.OpenAI") as mock_cls:
            from dreamlog.llm_client import LLMClient
            client = LLMClient(base_url="http://localhost:11434/v1", model="llama3")
            assert client.provider == "openai"
            mock_cls.assert_called_once_with(
                base_url="http://localhost:11434/v1", api_key="dummy")
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_llm_client.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Create LLMClient**

```python
# dreamlog/llm_client.py
"""
Thin LLM client wrapper around OpenAI/Anthropic SDKs.

Covers OpenAI, Anthropic, Ollama (via OpenAI-compatible endpoint),
and any OpenAI-compatible API by setting base_url.

Usage:
    client = LLMClient(model="gpt-4o-mini")                          # OpenAI
    client = LLMClient(provider="anthropic", model="claude-3-haiku")  # Anthropic
    client = LLMClient(base_url="http://localhost:11434/v1")          # Ollama
"""


class LLMClient:
    def __init__(self, provider="openai", base_url=None, api_key=None,
                 model="gpt-4o-mini", temperature=0.1, max_tokens=500):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        if provider == "anthropic":
            from anthropic import Anthropic
            self._client = Anthropic(api_key=api_key)
        else:
            from openai import OpenAI
            self._client = OpenAI(base_url=base_url, api_key=api_key or "dummy")

    def complete(self, prompt, **kwargs):
        """Send a prompt to the LLM and return the response text."""
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

- [ ] **Step 4: Update pyproject.toml**

Add to `[project.optional-dependencies]`:
```toml
llm = [
    "openai>=1.0.0",
    "anthropic>=0.20.0",
]
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_llm_client.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add dreamlog/llm_client.py tests/test_llm_client.py pyproject.toml
git commit -m "Add LLMClient: thin wrapper around OpenAI/Anthropic SDKs"
```

---

### Task 2: Migrate imports + fix bugs + simplify mock + delete llm_providers.py

This is a large migration task. All changes are mechanical (import updates) except the bug fixes.

**Files:**
- Modify: `dreamlog/llm_response_parser.py` (receive LLMResponse)
- Modify: `dreamlog/llm_hook.py`
- Modify: `dreamlog/llm_retry_wrapper.py`
- Modify: `dreamlog/llm_judge.py` (fix generate()->complete())
- Modify: `dreamlog/correction_retry.py` (fix generate()->complete())
- Modify: `dreamlog/engine.py`
- Modify: `dreamlog/tui.py`
- Modify: `dreamlog/__init__.py`
- Modify: `tests/mock_provider.py`
- Rewrite: `tests/test_llm_providers.py`
- Modify: `tests/test_llm_hook.py`
- Modify: `tests/test_llm_inference.py`
- Modify: `tests/test_llm_judge.py`
- Modify: `tests/test_correction_retry.py`
- Modify: `tests/test_model_commands.py`
- Modify: `tests/test_retry_wrapper.py`
- Modify: `examples/experiment_ollama.py`
- Modify: `examples/ollama_config.py`
- Modify: `examples/wake_sleep_demo.py`
- Modify: `integrations/mcp/dreamlog_mcp_server.py`
- Delete: `dreamlog/llm_providers.py`

- [ ] **Step 1: Move LLMResponse to llm_response_parser.py**

The `LLMResponse` dataclass currently lives in `llm_providers.py`. Move it to
`llm_response_parser.py` (which already imports it and contains the parsing logic).
Remove the grandparent fallback hack from `from_text()`.

- [ ] **Step 2: Simplify mock_provider.py**

Replace the BaseLLMProvider-based mock with a simple class matching LLMClient:

```python
# tests/mock_provider.py
"""Mock LLM provider for deterministic testing."""
import json


class MockLLMProvider:
    """Deterministic mock matching the LLMClient interface."""
    def __init__(self, responses=None, model="mock-model", temperature=0.1, max_tokens=500):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = "mock"
        self._responses = list(responses) if responses else []
        self._default_responses = {}
        self.call_count = 0
        self.last_prompt = None

    def complete(self, prompt, **kwargs):
        self.call_count += 1
        self.last_prompt = prompt
        if self._responses:
            return self._responses[min(self.call_count - 1, len(self._responses) - 1)]
        return self._match_prompt(prompt)

    def add_response(self, response):
        self._responses.append(response)

    def _match_prompt(self, prompt):
        """Return domain-appropriate response based on prompt keywords."""
        prompt_lower = prompt.lower()
        if "compression" in prompt_lower or "redundan" in prompt_lower:
            return json.dumps([])
        if "name" in prompt_lower and "predicate" in prompt_lower:
            return "transitive_closure"
        if "rule" in prompt_lower or "derive" in prompt_lower:
            return json.dumps([["rule", ["father", "X", "Y"],
                               [["parent", "X", "Y"], ["male", "X"]]]])
        return json.dumps([])
```

- [ ] **Step 3: Fix bugs in llm_judge.py and correction_retry.py**

In `dreamlog/llm_judge.py` line 65 and 117: change `self.provider.generate(prompt)` to `self.provider.complete(prompt)`

In `dreamlog/correction_retry.py` line 73: change `self.provider.generate(current_prompt)` to `self.provider.complete(current_prompt)`

- [ ] **Step 4: Update all import sites**

For each file that imports from `llm_providers`:

**Core modules** - replace `from .llm_providers import LLMProvider` with no import (duck typing, no type hint needed) or with a comment. Replace `from .llm_providers import LLMResponse` with `from .llm_response_parser import LLMResponse`. Replace `from .llm_providers import create_provider` with `from .llm_client import LLMClient`.

Specific files and their changes:
- `llm_hook.py:11` - `from .llm_providers import LLMProvider, LLMResponse` -> `from .llm_response_parser import LLMResponse`
- `llm_retry_wrapper.py:19` - `from .llm_providers import LLMProvider, LLMResponse` -> `from .llm_response_parser import LLMResponse`
- `correction_retry.py:11` - `from .llm_providers import LLMProvider` -> remove (duck typing)
- `llm_judge.py:11` - `from .llm_providers import LLMProvider` -> remove (duck typing)
- `engine.py:13` - `from .llm_providers import LLMProvider` -> remove (duck typing)
- `pythonic.py:39` - `from .llm_providers import create_provider` -> `from .llm_client import LLMClient`
- `tui.py:29` - `from .llm_providers import create_provider` -> `from .llm_client import LLMClient`
- `__init__.py` - remove LLMProvider/LLMResponse exports from llm_providers

**Test files** - update similarly. `test_llm_providers.py` gets rewritten to test LLMClient (or deleted if test_llm_client.py covers it). Other test files update their imports.

**Example files** - `OllamaProvider` -> `LLMClient(base_url="http://localhost:11434/v1")`

- [ ] **Step 5: Update tui.py provider parameter access**

Replace all `provider.set_parameter("model", x)` with `provider.model = x`,
`provider.get_parameter("model", default)` with `getattr(provider, "model", default)`,
and `provider.get_metadata()` with a dict of public attrs.

- [ ] **Step 6: Update pythonic.py LLM init**

Replace `create_provider(llm_provider, **llm_config)` with
`LLMClient(provider=llm_provider, **llm_config)`.

- [ ] **Step 7: Delete llm_providers.py**

```bash
rm dreamlog/llm_providers.py
```

- [ ] **Step 8: Run full test suite**

Run: `python -m pytest tests/ -x -q`
Expected: All pass (some LLM-specific tests may need adjustment)

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "Replace llm_providers.py with LLMClient, fix generate() bugs, simplify mock"
```

---

### Task 3: Simplify pythonic.py

**Files:**
- Modify: `dreamlog/pythonic.py`
- Modify: `tests/test_pythonic.py`

- [ ] **Step 1: Delete unused methods and extract shared helper**

Delete from `DreamLog` class:
- `visualize()` (lines 384-422)
- `to_dataframe()` (lines 369-382)
- `map_query()` (lines 426-435)
- `filter_query()` (lines 437-446)

Delete from `RuleBuilder`:
- `then()` (lines 87-89, alias for `and_()`)

Delete from module:
- `demo()` function (lines 495-553)

Extract shared arg-to-term helper:
```python
def _to_term(arg):
    """Convert a Python value to a DreamLog term."""
    if isinstance(arg, Term):
        return arg
    if isinstance(arg, str):
        return var(arg) if arg[0].isupper() else atom(arg)
    return atom(str(arg))

def _to_fact_term(arg):
    """Convert a Python value to a ground term (no variables)."""
    if isinstance(arg, Term):
        return arg
    return atom(str(arg))
```

Use `_to_term` in `query()` and `RuleBuilder._make_term()`.
Use `_to_fact_term` in `fact()`.

Simplify `transaction()` to use `kb.copy()` / `kb.restore_from()`:
```python
@contextmanager
def transaction(self):
    snapshot = self.engine.kb.copy()
    try:
        yield self
    except Exception:
        self.engine.kb.restore_from(snapshot)
        raise
```

- [ ] **Step 2: Update tests**

Remove any tests for deleted methods. Verify existing tests pass.

Run: `python -m pytest tests/test_pythonic.py -v`
Expected: All remaining tests PASS

- [ ] **Step 3: Run full suite**

Run: `python -m pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add dreamlog/pythonic.py tests/test_pythonic.py
git commit -m "Simplify pythonic.py: remove unused methods, extract helpers, fix transaction"
```

---

### Task 4: LLM-assisted naming in sleep cycle

**Files:**
- Modify: `dreamlog/kb_dreamer.py`
- Modify: `tests/test_sleep_cycle.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/test_sleep_cycle.py
from tests.mock_provider import MockLLMProvider


class TestLLMNaming:
    def test_invented_predicate_renamed(self):
        """LLM suggests name for _invented_0, it gets renamed."""
        kb = KnowledgeBase()
        for head, base in [("ancestor","parent"),("reachable","edge"),("connected","link")]:
            kb.add_rule(Rule(compound(head, var("X"), var("Y")),
                             [compound(base, var("X"), var("Y"))]))
            kb.add_rule(Rule(compound(head, var("X"), var("Z")),
                             [compound(base, var("X"), var("Y")),
                              compound(head, var("Y"), var("Z"))]))
        kb.add_fact(compound("parent", atom("a"), atom("b")))
        kb.add_fact(compound("edge", atom("x"), atom("y")))
        kb.add_fact(compound("link", atom("p"), atom("q")))

        mock = MockLLMProvider(responses=["transitive_closure"])
        dreamer = KnowledgeBaseDreamer(llm_client=mock)
        session = dreamer.dream(kb, verify=True)

        # _invented_0 should be renamed to transitive_closure
        rule_functors = {r.head.functor for r in kb.rules if isinstance(r.head, Compound)}
        assert "transitive_closure" in rule_functors
        assert "_invented_0" not in rule_functors

    def test_no_llm_skips_naming(self):
        """Without LLM client, naming is skipped."""
        kb = KnowledgeBase()
        for head, base in [("ancestor","parent"),("reachable","edge"),("connected","link")]:
            kb.add_rule(Rule(compound(head, var("X"), var("Y")),
                             [compound(base, var("X"), var("Y"))]))
            kb.add_rule(Rule(compound(head, var("X"), var("Z")),
                             [compound(base, var("X"), var("Y")),
                              compound(head, var("Y"), var("Z"))]))
        dreamer = KnowledgeBaseDreamer()  # no llm_client
        dreamer.dream(kb, verify=False)
        rule_functors = {r.head.functor for r in kb.rules if isinstance(r.head, Compound)}
        # Should have _invented_0, not a named version
        assert any(f.startswith("_invented_") for f in rule_functors)

    def test_bad_name_keeps_original(self):
        """If LLM suggests invalid name, keep _invented_N."""
        kb = KnowledgeBase()
        for head, base in [("ancestor","parent"),("reachable","edge"),("connected","link")]:
            kb.add_rule(Rule(compound(head, var("X"), var("Y")),
                             [compound(base, var("X"), var("Y"))]))
            kb.add_rule(Rule(compound(head, var("X"), var("Z")),
                             [compound(base, var("X"), var("Y")),
                              compound(head, var("Y"), var("Z"))]))
        # LLM returns something with spaces/special chars
        mock = MockLLMProvider(responses=["this is not a valid name!!!"])
        dreamer = KnowledgeBaseDreamer(llm_client=mock)
        dreamer.dream(kb, verify=False)
        rule_functors = {r.head.functor for r in kb.rules if isinstance(r.head, Compound)}
        assert any(f.startswith("_invented_") for f in rule_functors)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_sleep_cycle.py::TestLLMNaming -v`
Expected: FAIL

- [ ] **Step 3: Implement LLM-assisted naming**

Add `llm_client` parameter to `KnowledgeBaseDreamer.__init__`:
```python
def __init__(self, min_group_size=3, shared_structure_threshold=0.1,
             llm_client=None):
    self.min_group_size = min_group_size
    self.shared_structure_threshold = shared_structure_threshold
    self.llm_client = llm_client
```

Add naming method:
```python
def _name_invented_predicates(self, kb: KnowledgeBase) -> None:
    """Ask LLM to suggest names for _invented_N and _extracted_N predicates."""
    if not self.llm_client:
        return

    import re
    for prefix in ("_invented_", "_extracted_"):
        # Find all generated predicates
        generated = {}
        for rule in kb.rules:
            if isinstance(rule.head, Compound) and rule.head.functor.startswith(prefix):
                generated.setdefault(rule.head.functor, []).append(rule)

        for old_name, rules in generated.items():
            # Build prompt showing the predicate's rules and usage
            rules_str = "\n".join(str(r) for r in rules)
            wrappers = [r for r in kb.rules
                        if isinstance(r.head, Compound)
                        and any(isinstance(g, Compound) and g.functor == old_name
                                for g in r.body)]
            wrappers_str = "\n".join(str(r) for r in wrappers)

            prompt = (
                f"This predicate was discovered by compressing a knowledge base:\n\n"
                f"{rules_str}\n\n"
                f"It is used as:\n{wrappers_str}\n\n"
                f"Suggest a short, descriptive name (lowercase, underscores, "
                f"no spaces). Reply with just the name."
            )

            try:
                suggested = self.llm_client.complete(prompt).strip().lower()
                # Validate: must be a valid identifier
                if not re.match(r'^[a-z][a-z0-9_]*$', suggested):
                    continue
                if len(suggested) > 50:
                    continue
                # Check name doesn't collide with existing predicates
                existing = {r.head.functor for r in kb.rules
                            if isinstance(r.head, Compound)}
                existing.update(f.term.functor for f in kb.facts
                                if isinstance(f.term, Compound))
                if suggested in existing:
                    continue
                # Rename throughout KB
                self._rename_predicate(kb, old_name, suggested)
            except Exception:
                continue  # Keep original name on any error

def _rename_predicate(self, kb: KnowledgeBase, old_name: str,
                      new_name: str) -> None:
    """Rename a predicate throughout the KB."""
    def rename_term(term):
        if isinstance(term, Compound):
            functor = new_name if term.functor == old_name else term.functor
            new_args = [rename_term(a) for a in term.args]
            return Compound(functor, new_args)
        return term

    # Rename in rules
    new_rules = []
    old_rules = list(kb.rules)
    for rule in old_rules:
        new_head = rename_term(rule.head)
        new_body = [rename_term(g) for g in rule.body]
        new_rules.append((rule, Rule(new_head, new_body)))

    for old_rule, new_rule in new_rules:
        if old_rule != new_rule:
            kb.remove_rule_by_value(old_rule)
            kb.add_rule(new_rule)
```

Wire into `dream()` after all symbolic operations and before final verification:
```python
        # LLM-assisted naming (after symbolic ops, before final verify)
        self._name_invented_predicates(kb)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_sleep_cycle.py::TestLLMNaming -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add dreamlog/kb_dreamer.py tests/test_sleep_cycle.py
git commit -m "Add LLM-assisted naming for invented and extracted predicates"
```

---

### Task 5: LLM-assisted compression (Operation G)

**Files:**
- Modify: `dreamlog/kb_dreamer.py`
- Modify: `tests/test_sleep_cycle.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/test_sleep_cycle.py
class TestOperationG:
    def test_cross_functor_rule_proposed(self):
        """LLM proposes father(X,Y) :- parent(X,Y), male(X)."""
        kb = KnowledgeBase()
        for p, c in [("john","mary"),("bob","alice"),("carol","dave")]:
            kb.add_fact(compound("parent", atom(p), atom(c)))
        for n in ["john", "bob"]:
            kb.add_fact(compound("male", atom(n)))
        kb.add_fact(compound("father", atom("john"), atom("mary")))
        kb.add_fact(compound("father", atom("bob"), atom("alice")))

        import json
        rule_json = json.dumps([["rule", ["father", "X", "Y"],
                                [["parent", "X", "Y"], ["male", "X"]]]])
        mock = MockLLMProvider(responses=[rule_json])
        dreamer = KnowledgeBaseDreamer(llm_client=mock)
        session = dreamer.dream(kb, verify=False)

        # father facts should be removed (derivable from new rule)
        father_facts = [f for f in kb.facts
                        if isinstance(f.term, Compound) and f.term.functor == "father"]
        father_rules = [r for r in kb.rules
                        if isinstance(r.head, Compound) and r.head.functor == "father"]
        # Should have the rule and fewer facts
        assert len(father_rules) >= 1 or len(father_facts) < 2

    def test_no_llm_skips_compression(self):
        """Without LLM client, Op G is skipped."""
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        dreamer = KnowledgeBaseDreamer()
        session = dreamer.dream(kb, verify=False)
        # No LLM operations
        assert all(op.operation != "llm_compression" for op in session.operations)

    def test_invalid_rule_rejected(self):
        """LLM proposes rule that over-generates -> rejected."""
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        kb.add_fact(compound("b", atom("y")))
        # LLM proposes a(X) :- b(X) which would make a(y) true (wrong)
        import json
        mock = MockLLMProvider(responses=[
            json.dumps([["rule", ["a", "X"], [["b", "X"]]]])])
        dreamer = KnowledgeBaseDreamer(llm_client=mock)
        dreamer.dream(kb, verify=True)
        # a(y) should NOT be derivable
        from dreamlog.evaluator import PrologEvaluator
        ev = PrologEvaluator(kb)
        assert not ev.has_solution(compound("a", atom("y")))
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_sleep_cycle.py::TestOperationG -v`
Expected: FAIL

- [ ] **Step 3: Implement Operation G**

Add to `KnowledgeBaseDreamer`:

```python
def _llm_compress(self, kb: KnowledgeBase,
                  suite: Optional['VerificationSuite'] = None
                  ) -> List[CompressionCandidate]:
    """Operation G: Ask LLM to propose cross-functor rules."""
    if not self.llm_client:
        return []

    from .evaluator import PrologEvaluator
    from .llm_response_parser import parse_llm_response
    ops = []

    # Sample facts for the prompt
    fact_lines = []
    for fact in kb.facts[:50]:
        fact_lines.append(str(fact.term))
    if not fact_lines:
        return ops

    prompt = (
        "Here are facts from a knowledge base:\n\n"
        + "\n".join(fact_lines)
        + "\n\nCan you propose rules that derive some of these facts from others?\n"
        "Use JSON format: [[\"rule\", [\"head\", \"X\", \"Y\"], "
        "[[\"body1\", \"X\", \"Z\"], [\"body2\", \"Z\", \"Y\"]]]]\n"
        "Reply with a JSON array of rules only."
    )

    try:
        response = self.llm_client.complete(prompt)
        parsed = parse_llm_response(response)
    except Exception:
        return ops

    if not parsed or not parsed.rules:
        return ops

    # Validate each proposed rule
    for rule_data in parsed.rules:
        try:
            rule = self._build_rule_from_parsed(rule_data)
            if rule is None:
                continue

            # Check: rule must derive at least one existing fact
            test_kb = kb.copy()
            test_kb.add_rule(rule)
            ev = PrologEvaluator(test_kb)

            # Find facts the rule could derive
            derivable_facts = []
            for fact in kb.facts:
                if (isinstance(fact.term, Compound)
                        and fact.term.functor == rule.head.functor):
                    if ev.has_solution(fact.term):
                        derivable_facts.append(fact)

            if not derivable_facts:
                continue

            # Verify: adding rule must not break anything
            if suite is not None:
                result = suite.verify(test_kb, lambda k: PrologEvaluator(k))
                if not result.passed:
                    continue

            # MDL: rule + removal of derivable facts must decrease count
            if len(derivable_facts) < 2:  # need to derive 2+ facts to justify the rule
                continue

            # Apply: add rule, then let Op B handle pruning derivable facts
            kb.add_rule(rule)
            ops.append(CompressionCandidate(
                operation="llm_compression",
                original_clauses=[],
                new_clauses=[rule]))

        except Exception:
            continue

    return ops

def _build_rule_from_parsed(self, rule_data):
    """Build a Rule from parsed LLM output."""
    from .factories import atom, var, compound
    try:
        if len(rule_data) < 3:
            return None
        head_data = rule_data[1] if isinstance(rule_data[1], list) else rule_data
        body_data = rule_data[2] if len(rule_data) > 2 else []

        def make_term(data):
            if not isinstance(data, list) or len(data) == 0:
                return None
            functor = data[0]
            args = []
            for a in data[1:]:
                if isinstance(a, str) and a[0].isupper():
                    args.append(Variable(a))
                else:
                    args.append(Atom(str(a)))
            return Compound(functor, args)

        head = make_term(head_data)
        if head is None:
            return None
        body = []
        for b in body_data:
            bt = make_term(b)
            if bt is None:
                return None
            body.append(bt)
        if not body:
            return None
        return Rule(head, body)
    except Exception:
        return None
```

Wire into `dream()` after naming:
```python
        ops.extend(self._llm_compress(kb, suite))
```

Also update the TUI label dict to include `"llm_compression": "LLM-proposed rule"`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_sleep_cycle.py::TestOperationG -v`
Expected: All PASS

- [ ] **Step 5: Run full suite + benchmarks**

Run: `python -m pytest tests/ -x -q`
Run: `python benchmarks/sleep_cycle_bench.py`
Expected: All pass, benchmarks unchanged (no LLM client in benchmarks)

- [ ] **Step 6: Commit**

```bash
git add dreamlog/kb_dreamer.py dreamlog/tui.py tests/test_sleep_cycle.py
git commit -m "Add Operation G (LLM-assisted compression) to sleep cycle"
```
