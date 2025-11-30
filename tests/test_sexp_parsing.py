"""
Unit tests for S-expression parsing of facts and rules from LLM responses
"""

import pytest
from dreamlog.llm_response_parser import parse_llm_response
from dreamlog.knowledge import Fact, Rule
from dreamlog.terms import Atom, Variable, Compound


class TestSExpressionParsing:
    """Test parsing of S-expression facts and rules"""
    
    def test_parse_simple_fact(self):
        """Test parsing a simple fact: (parent john mary)"""
        text = "(parent john mary)"
        result = parse_llm_response(text, strict=False, validate=False)
        
        assert result is not None
        parsed_knowledge, validation_report = result
        
        assert len(parsed_knowledge.facts) == 1
        assert len(parsed_knowledge.rules) == 0
        
        fact = parsed_knowledge.facts[0]
        assert fact.term.functor == "parent"
        assert len(fact.term.args) == 2
        assert fact.term.args[0].value == "john"
        assert fact.term.args[1].value == "mary"
    
    def test_parse_multiple_facts(self):
        """Test parsing multiple facts on separate lines"""
        text = """(parent john mary)
(parent mary alice)
(parent john bob)"""
        
        result = parse_llm_response(text, strict=False, validate=False)
        assert result is not None
        parsed_knowledge, validation_report = result
        
        assert len(parsed_knowledge.facts) == 3
        assert len(parsed_knowledge.rules) == 0
        
        # Check first fact
        assert parsed_knowledge.facts[0].term.functor == "parent"
        assert parsed_knowledge.facts[0].term.args[0].value == "john"
        assert parsed_knowledge.facts[0].term.args[1].value == "mary"
        
        # Check second fact
        assert parsed_knowledge.facts[1].term.functor == "parent"
        assert parsed_knowledge.facts[1].term.args[0].value == "mary"
        assert parsed_knowledge.facts[1].term.args[1].value == "alice"
    
    def test_parse_simple_rule(self):
        """Test parsing a simple rule: (rule (grandparent X Z) ((parent X Y) (parent Y Z)))"""
        text = "(rule (grandparent X Z) ((parent X Y) (parent Y Z)))"
        
        result = parse_llm_response(text, strict=False, validate=False)
        assert result is not None
        parsed_knowledge, validation_report = result
        
        assert len(parsed_knowledge.facts) == 0
        assert len(parsed_knowledge.rules) == 1
        
        rule = parsed_knowledge.rules[0]
        
        # Check head
        assert rule.head.functor == "grandparent"
        assert len(rule.head.args) == 2
        assert isinstance(rule.head.args[0], Variable)
        assert rule.head.args[0].name == "X"
        assert isinstance(rule.head.args[1], Variable)
        assert rule.head.args[1].name == "Z"
        
        # Check body
        assert len(rule.body) == 2
        
        # First body term: (parent X Y)
        assert rule.body[0].functor == "parent"
        assert rule.body[0].args[0].name == "X"
        assert rule.body[0].args[1].name == "Y"
        
        # Second body term: (parent Y Z)
        assert rule.body[1].functor == "parent"
        assert rule.body[1].args[0].name == "Y"
        assert rule.body[1].args[1].name == "Z"
    
    def test_parse_sibling_rule(self):
        """Test parsing sibling rule: (rule (sibling X Y) ((parent Z X) (parent Z Y) (different X Y)))"""
        text = "(rule (sibling X Y) ((parent Z X) (parent Z Y) (different X Y)))"
        
        result = parse_llm_response(text, strict=False, validate=False)
        assert result is not None
        parsed_knowledge, validation_report = result
        
        assert len(parsed_knowledge.rules) == 1
        
        rule = parsed_knowledge.rules[0]
        
        # Check head
        assert rule.head.functor == "sibling"
        assert len(rule.head.args) == 2
        
        # Check body has 3 terms
        assert len(rule.body) == 3
        assert rule.body[0].functor == "parent"
        assert rule.body[1].functor == "parent"
        assert rule.body[2].functor == "different"
    
    def test_parse_cousin_rule(self):
        """Test parsing cousin rule: (rule (cousin X Y) ((parent A X) (parent B Y) (sibling A B)))"""
        text = "(rule (cousin X Y) ((parent A X) (parent B Y) (sibling A B)))"
        
        result = parse_llm_response(text, strict=False, validate=False)
        assert result is not None
        parsed_knowledge, validation_report = result
        
        assert len(parsed_knowledge.rules) == 1
        
        rule = parsed_knowledge.rules[0]
        
        # Check head
        assert rule.head.functor == "cousin"
        assert rule.head.args[0].name == "X"
        assert rule.head.args[1].name == "Y"
        
        # Check body
        assert len(rule.body) == 3
        assert rule.body[0].functor == "parent"
        assert rule.body[0].args[0].name == "A"
        assert rule.body[0].args[1].name == "X"
        
        assert rule.body[1].functor == "parent"
        assert rule.body[1].args[0].name == "B"
        assert rule.body[1].args[1].name == "Y"
        
        assert rule.body[2].functor == "sibling"
        assert rule.body[2].args[0].name == "A"
        assert rule.body[2].args[1].name == "B"
    
    def test_parse_mixed_facts_and_rules(self):
        """Test parsing a mix of facts and rules"""
        text = """(parent john mary)
(parent mary alice)
(rule (grandparent X Z) ((parent X Y) (parent Y Z)))
(parent john bob)
(rule (sibling X Y) ((parent Z X) (parent Z Y) (different X Y)))"""
        
        result = parse_llm_response(text, strict=False, validate=False)
        assert result is not None
        parsed_knowledge, validation_report = result
        
        assert len(parsed_knowledge.facts) == 3
        assert len(parsed_knowledge.rules) == 2
        
        # Check facts
        fact_functors = [f.term.functor for f in parsed_knowledge.facts]
        assert all(f == "parent" for f in fact_functors)
        
        # Check rules
        rule_heads = [r.head.functor for r in parsed_knowledge.rules]
        assert "grandparent" in rule_heads
        assert "sibling" in rule_heads
    
    def test_parse_rule_with_single_body_term(self):
        """Test parsing a rule with only one body term"""
        text = "(rule (is_parent X) ((parent X Y)))"
        
        result = parse_llm_response(text, strict=False, validate=False)
        assert result is not None
        parsed_knowledge, validation_report = result
        
        assert len(parsed_knowledge.rules) == 1
        rule = parsed_knowledge.rules[0]
        
        assert rule.head.functor == "is_parent"
        assert len(rule.body) == 1
        assert rule.body[0].functor == "parent"
    
    def test_parse_with_extra_whitespace(self):
        """Test parsing with various whitespace patterns"""
        text = """
        (parent john mary)
        
        (rule (grandparent X Z) ((parent X Y) (parent Y Z)))
        
        (parent   bob   alice)
        """
        
        result = parse_llm_response(text, strict=False, validate=False)
        assert result is not None
        parsed_knowledge, validation_report = result
        
        assert len(parsed_knowledge.facts) == 2
        assert len(parsed_knowledge.rules) == 1
    
    def test_parse_llm_response_with_markdown(self):
        """Test parsing LLM response with markdown code blocks"""
        text = """Here's the rule you need:

```prolog
(rule (cousin X Y) ((parent A X) (parent B Y) (sibling A B)))
```

And some facts:
```
(parent john mary)
(parent bob alice)
```
"""
        
        result = parse_llm_response(text, strict=False, validate=False)
        assert result is not None
        parsed_knowledge, validation_report = result
        
        # Should extract the rule and facts despite markdown
        assert len(parsed_knowledge.rules) >= 1
        assert len(parsed_knowledge.facts) >= 2
    
    @pytest.mark.skip(reason="Error recovery not yet implemented - parser should gracefully handle malformed S-expressions")
    def test_parse_malformed_rule_gracefully(self):
        """Test that malformed rules don't crash the parser

        TODO: Implement error recovery so parser can extract valid parts from malformed input.
        Currently the parser fails completely on malformed input rather than recovering.
        """
        text = """(rule (broken X Y) ((parent X))  # Missing closing paren for body
(parent john mary)  # This should still parse
(rule grandparent X Z) ((parent X Y) (parent Y Z)))  # Missing opening paren for head
"""

        result = parse_llm_response(text, strict=False, validate=False)
        assert result is not None
        parsed_knowledge, validation_report = result

        # Should at least get the valid fact
        assert len(parsed_knowledge.facts) >= 1
        assert parsed_knowledge.facts[0].term.functor == "parent"

        # Should have errors recorded
        assert len(parsed_knowledge.errors) > 0
    
    def test_uppercase_vs_lowercase_detection(self):
        """Test that uppercase are treated as variables, lowercase as atoms"""
        text = "(rule (test X y) ((pred X y Z)))"
        
        result = parse_llm_response(text, strict=False, validate=False)
        assert result is not None
        parsed_knowledge, validation_report = result
        
        assert len(parsed_knowledge.rules) == 1
        rule = parsed_knowledge.rules[0]
        
        # X should be a variable
        assert isinstance(rule.head.args[0], Variable)
        # y should be an atom
        assert isinstance(rule.head.args[1], Atom)
        
        # In body: X and Z should be variables, y should be atom
        body_term = rule.body[0]
        assert isinstance(body_term.args[0], Variable)  # X
        assert isinstance(body_term.args[1], Atom)       # y
        assert isinstance(body_term.args[2], Variable)   # Z


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])