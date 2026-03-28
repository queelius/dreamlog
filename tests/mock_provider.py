"""Mock LLM provider for deterministic testing."""
import json


class MockLLMProvider:
    """Deterministic mock matching the LLMClient interface."""
    def __init__(self, responses=None, model="mock-model", temperature=0.1, max_tokens=500,
                 knowledge_domain="general", **kwargs):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = "mock"
        self._responses = list(responses) if responses else []
        self._custom_responses = {}  # keyword -> JSON string
        self.call_count = 0
        self.last_prompt = None
        self.knowledge_domain = knowledge_domain

    def complete(self, prompt, **kwargs):
        self.call_count += 1
        self.last_prompt = prompt
        if self._responses:
            return self._responses[min(self.call_count - 1, len(self._responses) - 1)]
        # Check custom keyword-keyed responses
        prompt_lower = prompt.lower()
        for key, response in self._custom_responses.items():
            if key in prompt_lower:
                return response
        return self._match_prompt(prompt)

    def add_response(self, response, facts=None, rules=None):
        """Add a response. Either a raw string, or a keyword with facts/rules."""
        if facts is not None or rules is not None:
            # Keyword-keyed response: response is the keyword
            items = []
            if facts:
                for fact in facts:
                    items.append(["fact", fact])
            if rules:
                for rule in rules:
                    items.append(["rule"] + rule)
            self._custom_responses[response.lower()] = json.dumps(items)
        else:
            self._responses.append(response)

    def generate_knowledge(self, term, context=None, max_items=5):
        """Legacy compatibility: generate structured knowledge."""
        from dreamlog.llm_response_parser import LLMResponse
        raw = self.complete(f"Query: {term}\n{context or ''}")
        return LLMResponse.from_text(raw)

    def _match_prompt(self, prompt):
        """Return domain-appropriate response based on prompt keywords."""
        prompt_lower = prompt.lower()

        # Check for grandparent specifically (common test case)
        if 'query:' in prompt_lower and 'grandparent' in prompt_lower.split('query:')[1].split('\n')[0]:
            return json.dumps([["rule", ["grandparent", "X", "Z"], [["parent", "X", "Y"], ["parent", "Y", "Z"]]]])

        if "compression" in prompt_lower or "redundan" in prompt_lower:
            return json.dumps([])
        if "name" in prompt_lower and "predicate" in prompt_lower:
            return "transitive_closure"
        if "rule" in prompt_lower or "derive" in prompt_lower:
            return json.dumps([["rule", ["father", "X", "Y"],
                               [["parent", "X", "Y"], ["male", "X"]]]])

        # Domain-specific responses for family domain
        if self.knowledge_domain == "family":
            for functor in ["parent", "sibling", "grandparent", "uncle", "healthy"]:
                if functor in prompt_lower:
                    return self._family_response(functor)

        return json.dumps([])

    def _family_response(self, functor):
        """Return family domain responses."""
        responses = {
            "parent": json.dumps([
                ["fact", ["parent", "john", "mary"]],
                ["fact", ["parent", "mary", "alice"]],
                ["rule", ["grandparent", "X", "Z"], [["parent", "X", "Y"], ["parent", "Y", "Z"]]]
            ]),
            "sibling": json.dumps([
                ["fact", ["sibling", "alice", "bob"]],
                ["rule", ["sibling", "X", "Y"], [["parent", "Z", "X"], ["parent", "Z", "Y"], ["different", "X", "Y"]]]
            ]),
            "grandparent": json.dumps([
                ["rule", ["grandparent", "X", "Z"], [["parent", "X", "Y"], ["parent", "Y", "Z"]]]
            ]),
            "healthy": json.dumps([
                ["rule", ["healthy", "X"], [["exercises", "X"], ["eats_well", "X"]]]
            ]),
            "uncle": json.dumps([
                ["rule", ["uncle", "X", "Y"], [["brother", "X", "Z"], ["parent", "Z", "Y"]]]
            ]),
        }
        return responses.get(functor, json.dumps([]))
