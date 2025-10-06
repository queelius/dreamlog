"""
Comprehensive tests for prefix notation parser.
Supports both JSON arrays and S-expression syntax.
"""

import pytest
import json
from typing import Any, List, Union
from dreamlog.terms import Term, Atom, Variable, Compound
from dreamlog.knowledge import Rule
from dreamlog.prefix_parser import parse_prefix_notation, parse_s_expression


class TestPrefixNotationParsing:
    """Test parsing of prefix notation in various formats"""
    
    def test_json_array_simple_fact(self):
        """Test parsing simple facts in JSON array format"""
        test_cases = [
            (["parent", "john", "mary"], 
             Compound("parent", [Atom("john"), Atom("mary")])),
            
            (["likes", "alice", "bob"],
             Compound("likes", [Atom("alice"), Atom("bob")])),
            
            (["age", "john", 42],
             Compound("age", [Atom("john"), Atom(42)])),
            
            (["active", "server1", True],
             Compound("active", [Atom("server1"), Atom(True)])),
        ]
        
        for input_data, expected in test_cases:
            result = parse_prefix_notation(input_data)
            assert result == expected, f"Failed to parse {input_data}"
    
    def test_json_array_with_variables(self):
        """Test parsing with variables (uppercase identifiers)"""
        test_cases = [
            (["parent", "X", "mary"],
             Compound("parent", [Variable("X"), Atom("mary")])),
            
            (["parent", "john", "Y"],
             Compound("parent", [Atom("john"), Variable("Y")])),
            
            (["ancestor", "X", "Y"],
             Compound("ancestor", [Variable("X"), Variable("Y")])),
            
            (["Person", "Attribute"],  # First is functor, second is variable
             Compound("Person", [Variable("Attribute")])),  # First is functor if compound
        ]
        
        for input_data, expected in test_cases:
            result = parse_prefix_notation(input_data)
            assert result == expected, f"Failed to parse {input_data}"
    
    def test_nested_structures(self):
        """Test parsing nested compound terms"""
        test_cases = [
            (["parent", ["name", "john"], "mary"],
             Compound("parent", [
                 Compound("name", [Atom("john")]),
                 Atom("mary")
             ])),
            
            (["knows", "alice", ["father", "bob"]],
             Compound("knows", [
                 Atom("alice"),
                 Compound("father", [Atom("bob")])
             ])),
            
            (["believes", "X", ["likes", "Y", "Z"]],
             Compound("believes", [
                 Variable("X"),
                 Compound("likes", [Variable("Y"), Variable("Z")])
             ])),
        ]
        
        for input_data, expected in test_cases:
            result = parse_prefix_notation(input_data)
            assert result == expected, f"Failed to parse {input_data}"
    
    def test_quoted_structures(self):
        """Test quoting mechanism to treat structures as data"""
        test_cases = [
            (["quote", ["parent", "john", "mary"]],
             Atom(["parent", "john", "mary"])),  # Entire structure becomes atom
            
            (["assert", ["quote", ["fact", "X"]]],
             Compound("assert", [Atom(["fact", "X"])])),
            
            (["meta", ["quote", "X"]],  # Quote a variable name
             Compound("meta", [Atom("X")])),  # Becomes atom, not variable
        ]
        
        for input_data, expected in test_cases:
            result = parse_prefix_notation(input_data)
            assert result == expected, f"Failed to parse {input_data}"
    
    def test_s_expression_parsing(self):
        """Test parsing S-expression syntax (parent john mary)"""
        test_cases = [
            ("(parent john mary)",
             Compound("parent", [Atom("john"), Atom("mary")])),
            
            ("(likes alice bob)",
             Compound("likes", [Atom("alice"), Atom("bob")])),
            
            ("(parent X mary)",  # With variable
             Compound("parent", [Variable("X"), Atom("mary")])),
            
            ("(age john 42)",  # With number
             Compound("age", [Atom("john"), Atom(42)])),
            
            ("(active server1 true)",  # With boolean
             Compound("active", [Atom("server1"), Atom(True)])),
        ]
        
        for input_str, expected in test_cases:
            result = parse_s_expression(input_str)
            assert result == expected, f"Failed to parse {input_str}"
    
    def test_nested_s_expressions(self):
        """Test parsing nested S-expressions"""
        test_cases = [
            ("(parent (name john) mary)",
             Compound("parent", [
                 Compound("name", [Atom("john")]),
                 Atom("mary")
             ])),
            
            ("(knows alice (father bob))",
             Compound("knows", [
                 Atom("alice"),
                 Compound("father", [Atom("bob")])
             ])),
            
            ("(and (parent X Y) (parent Y Z))",
             Compound("and", [
                 Compound("parent", [Variable("X"), Variable("Y")]),
                 Compound("parent", [Variable("Y"), Variable("Z")])
             ])),
        ]
        
        for input_str, expected in test_cases:
            result = parse_s_expression(input_str)
            assert result == expected, f"Failed to parse {input_str}"
    
    def test_special_characters_and_strings(self):
        """Test handling of special characters and string literals"""
        test_cases = [
            (["name", "john-doe"],  # Hyphenated name
             Compound("name", [Atom("john-doe")])),
            
            (["email", "user@example.com"],  # Email
             Compound("email", [Atom("user@example.com")])),
            
            (["path", "/home/user/file.txt"],  # File path
             Compound("path", [Atom("/home/user/file.txt")])),
            
            (["string", "Hello, World!"],  # String with punctuation
             Compound("string", [Atom("Hello, World!")])),
            
            ('(name "John Doe")',  # Quoted string in S-expr
             Compound("name", [Atom("John Doe")])),
        ]
        
        for input_data, expected in test_cases:
            if isinstance(input_data, str):
                result = parse_s_expression(input_data)
            else:
                result = parse_prefix_notation(input_data)
            assert result == expected, f"Failed to parse {input_data}"
    
    def test_empty_and_nullary_predicates(self):
        """Test empty lists and nullary predicates"""
        test_cases = [
            (["true"],  # Nullary predicate
             Compound("true", [])),
            
            (["false"],
             Compound("false", [])),
            
            (["halt"],
             Compound("halt", [])),
            
            ("(true)",  # S-expr nullary
             Compound("true", [])),
            
            (["list"],  # Empty list predicate
             Compound("list", [])),
        ]
        
        for input_data, expected in test_cases:
            if isinstance(input_data, str):
                result = parse_s_expression(input_data)
            else:
                result = parse_prefix_notation(input_data)
            assert result == expected, f"Failed to parse {input_data}"
    
    def test_rule_representation(self):
        """Test rule representation in prefix notation"""
        test_cases = [
            (["rule", 
              ["grandparent", "X", "Z"],
              [["parent", "X", "Y"], ["parent", "Y", "Z"]]],
             Rule(
                 Compound("grandparent", [Variable("X"), Variable("Z")]),
                 [Compound("parent", [Variable("X"), Variable("Y")]),
                  Compound("parent", [Variable("Y"), Variable("Z")])]
             )),
            
            (["rule",
              ["ancestor", "X", "Y"],
              [["parent", "X", "Y"]]],
             Rule(
                 Compound("ancestor", [Variable("X"), Variable("Y")]),
                 [Compound("parent", [Variable("X"), Variable("Y")])]
             )),
        ]
        
        for input_data, expected in test_cases:
            result = parse_rule(input_data)
            assert result == expected, f"Failed to parse rule {input_data}"
    
    def test_arithmetic_and_operators(self):
        """Test arithmetic and operator expressions"""
        test_cases = [
            (["+", 1, 2],
             Compound("+", [Atom(1), Atom(2)])),
            
            (["*", "X", 3],
             Compound("*", [Variable("X"), Atom(3)])),
            
            (["=", "X", ["+", 2, 3]],
             Compound("=", [Variable("X"), Compound("+", [Atom(2), Atom(3)])])),
            
            (["<", "Age", 18],
             Compound("<", [Variable("Age"), Atom(18)])),
            
            ("(+ 1 2)",  # S-expr arithmetic
             Compound("+", [Atom(1), Atom(2)])),
        ]
        
        for input_data, expected in test_cases:
            if isinstance(input_data, str):
                result = parse_s_expression(input_data)
            else:
                result = parse_prefix_notation(input_data)
            assert result == expected, f"Failed to parse {input_data}"
    
    def test_list_notation(self):
        """Test list notation and operations"""
        test_cases = [
            (["list", 1, 2, 3],
             Compound("list", [Atom(1), Atom(2), Atom(3)])),
            
            (["cons", "H", "T"],
             Compound("cons", [Variable("H"), Variable("T")])),
            
            (["member", "X", ["list", 1, 2, 3]],
             Compound("member", [
                 Variable("X"),
                 Compound("list", [Atom(1), Atom(2), Atom(3)])
             ])),
            
            (["append", ["list", 1, 2], ["list", 3, 4], "Result"],
             Compound("append", [
                 Compound("list", [Atom(1), Atom(2)]),
                 Compound("list", [Atom(3), Atom(4)]),
                 Variable("Result")
             ])),
        ]
        
        for input_data, expected in test_cases:
            result = parse_prefix_notation(input_data)
            assert result == expected, f"Failed to parse {input_data}"
    
    def test_error_cases(self):
        """Test error handling for malformed input"""
        error_cases = [
            [],  # Empty list
            [[]],  # List with empty list
            "",  # Empty string
            "()",  # Empty S-expression
            "(parent",  # Unclosed S-expression
            "parent john mary)",  # Missing opening paren
            ["parent", "john", "mary", ["extra", "nesting", []]],  # Too deep nesting
            [123, "arg"],  # Number as functor
            [True, "arg"],  # Boolean as functor
            '(parent "john mary)',  # Unclosed string
        ]
        
        for input_data in error_cases:
            with pytest.raises((ValueError, SyntaxError, TypeError)):
                if isinstance(input_data, str):
                    parse_s_expression(input_data)
                else:
                    parse_prefix_notation(input_data)
    
    def test_whitespace_handling(self):
        """Test proper whitespace handling in S-expressions"""
        test_cases = [
            ("  (parent john mary)  ",  # Leading/trailing spaces
             Compound("parent", [Atom("john"), Atom("mary")])),
            
            ("(parent  john   mary)",  # Multiple spaces
             Compound("parent", [Atom("john"), Atom("mary")])),
            
            ("(parent\njohn\nmary)",  # Newlines
             Compound("parent", [Atom("john"), Atom("mary")])),
            
            ("(parent\tjohn\tmary)",  # Tabs
             Compound("parent", [Atom("john"), Atom("mary")])),
        ]
        
        for input_str, expected in test_cases:
            result = parse_s_expression(input_str)
            assert result == expected, f"Failed to parse {input_str}"
    
    def test_mixed_case_handling(self):
        """Test handling of mixed case for variables vs atoms"""
        test_cases = [
            (["Parent", "john", "mary"],  # Uppercase functor
             Compound("Parent", [Atom("john"), Atom("mary")])),
            
            (["parent", "John", "mary"],  # Uppercase atom (becomes variable)
             Compound("parent", [Variable("John"), Atom("mary")])),
            
            (["CamelCase", "snake_case", "ALLCAPS"],
             Compound("CamelCase", [Atom("snake_case"), Variable("ALLCAPS")])),
        ]
        
        for input_data, expected in test_cases:
            result = parse_prefix_notation(input_data)
            assert result == expected, f"Failed to parse {input_data}"
    
    def test_escape_sequences(self):
        """Test escape sequences in strings"""
        test_cases = [
            (["print", "Hello\\nWorld"],  # Newline escape
             Compound("print", [Atom("Hello\\nWorld")])),
            
            (["path", "C:\\\\Users\\\\file.txt"],  # Windows path
             Compound("path", [Atom("C:\\\\Users\\\\file.txt")])),
            
            (["quote_test", 'She said "Hello"'],  # Escaped quotes
             Compound("quote_test", [Atom('She said "Hello"')])),
        ]
        
        for input_data, expected in test_cases:
            result = parse_prefix_notation(input_data)
            assert result == expected, f"Failed to parse {input_data}"
    
    def test_unicode_support(self):
        """Test Unicode character support"""
        test_cases = [
            (["greeting", "你好"],  # Chinese
             Compound("greeting", [Atom("你好")])),
            
            (["emoji", "😀"],  # Emoji
             Compound("emoji", [Atom("😀")])),
            
            (["math", "∀x∃y"],  # Math symbols
             Compound("math", [Atom("∀x∃y")])),
        ]
        
        for input_data, expected in test_cases:
            result = parse_prefix_notation(input_data)
            assert result == expected, f"Failed to parse {input_data}"
    
    def test_conversion_between_formats(self):
        """Test conversion between JSON and S-expression formats"""
        test_cases = [
            (["parent", "john", "mary"], "(parent john mary)"),
            (["likes", "X", "Y"], "(likes X Y)"),
            (["age", "john", 42], "(age john 42)"),
            (["parent", ["name", "john"], "mary"], "(parent (name john) mary)"),
        ]
        
        for json_input, sexp_expected in test_cases:
            # Parse JSON to Term
            term = parse_prefix_notation(json_input)
            # Convert Term to S-expression string
            sexp_output = str(term)
            assert sexp_output == sexp_expected, f"Failed to convert {json_input}"
            
            # Parse S-expression back to Term
            term2 = parse_s_expression(sexp_expected)
            assert term == term2, f"Round-trip failed for {json_input}"
    
    def test_performance_large_structures(self):
        """Test parsing performance with large structures"""
        # Create a deeply nested structure
        depth = 100
        nested = "X"
        for i in range(depth):
            nested = ["f" + str(i), nested]
        
        # Should parse without stack overflow
        result = parse_prefix_notation(nested)
        assert isinstance(result, Compound)
        
        # Create a wide structure
        width = 1000
        wide = ["predicate"] + [f"arg{i}" for i in range(width)]
        result = parse_prefix_notation(wide)
        assert isinstance(result, Compound)
        assert len(result.args) == width


class TestParserIntegrationWithExistingCode:
    """Test that new parser integrates correctly with existing Term classes"""
    
    def test_parsed_terms_work_with_unification(self):
        """Ensure parsed terms work with existing unification"""
        from dreamlog.unification import unify
        
        # Parse two terms
        term1 = parse_prefix_notation(["parent", "X", "mary"])
        term2 = parse_prefix_notation(["parent", "john", "Y"])
        
        # Should unify with X=john, Y=mary
        bindings = unify(term1, term2)
        assert bindings is not None
        assert bindings["X"] == Atom("john")
        assert bindings["Y"] == Atom("mary")
    
    def test_parsed_rules_work_with_knowledge_base(self):
        """Ensure parsed rules work with existing knowledge base"""
        from dreamlog.knowledge import KnowledgeBase, Fact
        
        kb = KnowledgeBase()
        
        # Parse and add facts
        fact1 = parse_prefix_notation(["parent", "john", "mary"])
        fact2 = parse_prefix_notation(["parent", "mary", "alice"])
        
        kb.add_fact(Fact(fact1))
        kb.add_fact(Fact(fact2))
        
        # Parse and add rule
        rule_data = ["rule",
                     ["grandparent", "X", "Z"],
                     [["parent", "X", "Y"], ["parent", "Y", "Z"]]]
        rule = parse_rule(rule_data)
        kb.add_rule(rule)
        
        # Query should work
        query = parse_prefix_notation(["grandparent", "john", "alice"])
        # (actual query execution would go here)
    
    def test_json_serialization_roundtrip(self):
        """Test that parsed terms can be serialized and deserialized using prefix notation"""
        # term_to_prefix_json removed - use to_prefix() method instead
        
        test_cases = [
            ["parent", "john", "mary"],
            ["likes", "X", "Y"],
            ["parent", ["name", "john"], "mary"],
        ]
        
        for input_data in test_cases:
            # Parse to term
            term = parse_prefix_notation(input_data)
            
            # Convert back to prefix JSON
            output_data = term.to_prefix()
            
            # Parse again
            reconstructed = parse_prefix_notation(output_data)
            
            assert term == reconstructed, f"Round-trip failed for {input_data}"
            assert output_data == input_data, f"JSON format changed for {input_data}"


# Import the actual parser functions
from dreamlog.prefix_parser import (
    parse_prefix_notation,
    parse_s_expression,
    parse_rule
)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])