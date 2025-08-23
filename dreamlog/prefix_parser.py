"""
Prefix notation parser for DreamLog.
Supports both JSON array notation and S-expression syntax.
"""

import re
import json
from typing import Any, List, Union, Optional
from dreamlog.terms import Term, Atom, Variable, Compound
from dreamlog.knowledge import Rule


def parse_prefix_notation(data: Any) -> Term:
    """
    Parse prefix notation JSON array to Term.
    
    Examples:
        ["parent", "john", "mary"] -> Compound("parent", [Atom("john"), Atom("mary")])
        ["X", "Y"] -> Compound("X", [Atom("Y")])  # X is variable but first position is functor
        "john" -> Atom("john")
        "X" -> Variable("X")
        42 -> Atom(42)
    """
    if not isinstance(data, list):
        # Single value - could be atom or variable
        return _parse_single_value(data)
    
    if len(data) == 0:
        raise ValueError("Empty list cannot be parsed")
    
    # First element is the functor (or special form)
    functor = data[0]
    
    # Handle special forms
    if functor == "quote" and len(data) == 2:
        # Quote turns the next element into an atom (data, not evaluated)
        return Atom(data[1])
    
    if functor == "rule" and len(data) == 3:
        # Rule special form: ["rule", head, body]
        return parse_rule(data)
    
    # Regular compound term
    if not isinstance(functor, str):
        raise TypeError(f"Functor must be a string, got {type(functor)}: {functor}")
    
    # Parse arguments
    args = []
    for arg in data[1:]:
        args.append(parse_prefix_notation(arg))
    
    return Compound(functor, args)


def _parse_single_value(value: Any) -> Term:
    """Parse a single value to appropriate Term type."""
    if isinstance(value, str):
        # Check if it's a variable (starts with uppercase and looks like an identifier)
        if value and value[0].isupper() and _is_valid_identifier(value):
            return Variable(value)
        else:
            return Atom(value)
    elif isinstance(value, (int, float, bool, type(None))):
        return Atom(value)
    else:
        # For complex types, just wrap as atom
        return Atom(value)


def _is_valid_identifier(s: str) -> bool:
    """Check if string is a valid identifier (letters, numbers, underscores only)."""
    if not s:
        return False
    # First character must be letter or underscore
    if not (s[0].isalpha() or s[0] == '_'):
        return False
    # Rest must be alphanumeric or underscore
    return all(c.isalnum() or c == '_' for c in s)


def parse_s_expression(expr: str) -> Term:
    """
    Parse S-expression string to Term.
    
    Examples:
        "(parent john mary)" -> Compound("parent", [Atom("john"), Atom("mary")])
        "(likes X Y)" -> Compound("likes", [Variable("X"), Variable("Y")])
    """
    expr = expr.strip()
    if not expr:
        raise ValueError("Empty expression")
    
    # Check for unmatched closing parenthesis
    if ')' in expr and not expr.startswith('('):
        raise SyntaxError(f"Unmatched closing parenthesis in: {expr}")
    
    if not expr.startswith('('):
        # Single atom or variable
        return _parse_sexp_atom(expr)
    
    # Find matching closing parenthesis
    if not expr.endswith(')'):
        raise SyntaxError(f"Unclosed S-expression: {expr}")
    
    # Remove outer parentheses
    inner = expr[1:-1].strip()
    
    if not inner:
        raise ValueError("Empty S-expression ()")
    
    # Parse the expression
    tokens = _tokenize_sexp(inner)
    if not tokens:
        raise ValueError("No tokens in S-expression")
    
    # First token is the functor
    functor = tokens[0]
    
    # Parse arguments
    args = []
    for token in tokens[1:]:
        args.append(_parse_sexp_token(token))
    
    return Compound(functor, args)


def _tokenize_sexp(expr: str) -> List[str]:
    """
    Tokenize S-expression string, handling nested parentheses and quoted strings.
    """
    tokens = []
    current = []
    depth = 0
    in_string = False
    escape_next = False
    
    for char in expr:
        if escape_next:
            current.append(char)
            escape_next = False
            continue
            
        if char == '\\' and in_string:
            escape_next = True
            current.append(char)
            continue
        
        if char == '"':
            in_string = not in_string
            current.append(char)
        elif in_string:
            current.append(char)
        elif char == '(':
            depth += 1
            current.append(char)
        elif char == ')':
            depth -= 1
            current.append(char)
            if depth < 0:
                raise SyntaxError("Unmatched closing parenthesis")
        elif char in ' \t\n' and depth == 0:
            # Whitespace separates tokens at depth 0
            if current:
                tokens.append(''.join(current))
                current = []
        else:
            current.append(char)
    
    if current:
        tokens.append(''.join(current))
    
    if depth != 0:
        raise SyntaxError("Unmatched parentheses")
    
    if in_string:
        raise SyntaxError("Unclosed string")
    
    return tokens


def _parse_sexp_token(token: str) -> Term:
    """Parse a single S-expression token."""
    token = token.strip()
    
    # Check for nested S-expression
    if token.startswith('('):
        return parse_s_expression(token)
    
    # Check for quoted string
    if token.startswith('"') and token.endswith('"') and len(token) > 1:
        # Remove quotes
        return Atom(token[1:-1])
    
    return _parse_sexp_atom(token)


def _parse_sexp_atom(token: str) -> Term:
    """Parse an atomic S-expression token."""
    # Try to parse as number
    try:
        if '.' in token:
            return Atom(float(token))
        else:
            return Atom(int(token))
    except ValueError:
        pass
    
    # Check for boolean
    if token.lower() == 'true':
        return Atom(True)
    elif token.lower() == 'false':
        return Atom(False)
    
    # Check if it's a variable (starts with uppercase and is valid identifier)
    if token and token[0].isupper() and _is_valid_identifier(token):
        return Variable(token)
    
    # Otherwise it's an atom
    return Atom(token)


def parse_rule(data: List[Any]) -> Rule:
    """
    Parse rule representation to Rule object.
    
    Format: ["rule", head, body]
    Where head is a term and body is a list of terms.
    """
    if len(data) != 3 or data[0] != "rule":
        raise ValueError(f"Invalid rule format: {data}")
    
    head = parse_prefix_notation(data[1])
    
    if not isinstance(data[2], list):
        raise ValueError(f"Rule body must be a list, got: {type(data[2])}")
    
    body = []
    for goal in data[2]:
        body.append(parse_prefix_notation(goal))
    
    return Rule(head, body)


def term_to_sexp(term: Term) -> str:
    """
    Convert Term to S-expression string.
    
    Examples:
        Compound("parent", [Atom("john"), Atom("mary")]) -> "(parent john mary)"
        Variable("X") -> "X"
        Atom("john") -> "john"
    """
    if isinstance(term, Atom):
        value = term.value
        if isinstance(value, str):
            # Check if string needs quotes (contains spaces or special chars)
            if ' ' in value or '(' in value or ')' in value:
                return f'"{value}"'
            return value
        elif isinstance(value, bool):
            return 'true' if value else 'false'
        else:
            return str(value)
    
    elif isinstance(term, Variable):
        return term.name
    
    elif isinstance(term, Compound):
        if len(term.args) == 0:
            return f"({term.functor})"
        
        args_str = ' '.join(term_to_sexp(arg) for arg in term.args)
        return f"({term.functor} {args_str})"
    
    else:
        raise TypeError(f"Unknown term type: {type(term)}")


def term_to_prefix_json(term: Term) -> Any:
    """
    Convert Term to prefix notation JSON format.
    
    Examples:
        Compound("parent", [Atom("john"), Atom("mary")]) -> ["parent", "john", "mary"]
        Variable("X") -> "X"
        Atom("john") -> "john"
    """
    if isinstance(term, Atom):
        return term.value
    
    elif isinstance(term, Variable):
        return term.name
    
    elif isinstance(term, Compound):
        result = [term.functor]
        for arg in term.args:
            result.append(term_to_prefix_json(arg))
        return result
    
    else:
        raise TypeError(f"Unknown term type: {type(term)}")