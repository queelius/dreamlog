"""
Tests for LLM response parser fixes for common LLM formatting errors
"""

import pytest
from dreamlog.llm_response_parser import DreamLogResponseParser


class TestUnquotedJSONFix:
    """Test fixing unquoted JSON from LLMs"""

    def test_fix_unquoted_variables(self):
        """Test quoting unquoted variable names"""
        parser = DreamLogResponseParser()

        # LLM output with unquoted variables
        raw = '[["rule", ["ancestor", X, Y], [["parent", X, Y]]]]'
        fixed = parser._fix_unquoted_json(raw)

        assert fixed == '[["rule", ["ancestor", "X", "Y"], [["parent", "X", "Y"]]]]'

    def test_fix_unquoted_functors(self):
        """Test quoting unquoted functor names"""
        parser = DreamLogResponseParser()

        raw = '[[rule, [ancestor, X, Y], [[parent, X, Z], [ancestor, Z, Y]]]]'
        fixed = parser._fix_unquoted_json(raw)

        assert 'rule' in fixed  # Should be quoted now
        assert '"rule"' in fixed
        assert '"ancestor"' in fixed
        assert '"parent"' in fixed

    def test_parse_unquoted_llm_response(self):
        """Test parsing actual unquoted LLM response"""
        parser = DreamLogResponseParser()

        # Actual output from phi4-mini
        response = '[["rule", ["ancestor", X, Y], [["parent", X, Z], ["ancestor", Z, Y]]]]'

        parsed = parser.parse(response)

        assert parsed.is_valid
        assert len(parsed.rules) == 1
        assert len(parsed.facts) == 0

        rule = parsed.rules[0]
        assert str(rule.head.functor) == "ancestor"
        assert len(rule.body) == 2

    def test_parse_with_code_block(self):
        """Test parsing JSON inside markdown code blocks"""
        parser = DreamLogResponseParser()

        response = '''```json
[["rule", ["ancestor", X, Y], [["parent", X, Y]]]]
```'''

        parsed = parser.parse(response)

        assert parsed.is_valid
        assert len(parsed.rules) == 1

    def test_dont_quote_json_keywords(self):
        """Test that JSON keywords are not quoted"""
        parser = DreamLogResponseParser()

        raw = '{"valid": true, "count": null, "enabled": false}'
        fixed = parser._fix_unquoted_json(raw)

        # Keywords should remain unquoted
        assert 'true' in fixed
        assert 'false' in fixed
        assert 'null' in fixed
        # But not turned into strings
        assert '"true"' not in fixed
        assert '"false"' not in fixed
        assert '"null"' not in fixed

    def test_mixed_quoted_unquoted(self):
        """Test handling mixed quoted and unquoted strings"""
        parser = DreamLogResponseParser()

        # Some already quoted, some not
        raw = '[["rule", ["ancestor", "X", Y], [["parent", X, "Z"]]]]'
        fixed = parser._fix_unquoted_json(raw)

        # All should end up quoted
        assert '"X"' in fixed
        assert '"Y"' in fixed
        assert '"Z"' in fixed
        assert '"rule"' in fixed


class TestRealWorldLLMResponses:
    """Test with actual LLM responses we've seen"""

    def test_phi4_ancestor_response(self):
        """Test actual phi4-mini response for ancestor"""
        parser = DreamLogResponseParser()

        # First attempt from phi4-mini
        response = '```json\n[["rule", ["ancestor", X, Y], [[parent, X, Z], [ancestor, Z, Y]]]]\n```'

        parsed = parser.parse(response)

        assert parsed.is_valid
        assert len(parsed.rules) == 1
        assert len(parsed.facts) == 0

        rule = parsed.rules[0]
        assert rule.head.functor == "ancestor"
        assert len(rule.head.args) == 2
        assert len(rule.body) == 2

    def test_recursive_ancestor_rule(self):
        """Test parsing a correct recursive ancestor rule"""
        parser = DreamLogResponseParser()

        # Valid JSON array with two rules (correct format)
        response = '''[
["rule", ["ancestor", "X", "Y"], [["parent", "X", "Y"]]],
["rule", ["ancestor", "X", "Y"], [["parent", "X", "Z"], ["ancestor", "Z", "Y"]]]
]'''

        parsed = parser.parse(response)

        # Should get two rules (base case and recursive case)
        assert len(parsed.rules) == 2
