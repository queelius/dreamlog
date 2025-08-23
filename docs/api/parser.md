# Parser Module

The `dreamlog.prefix_parser` module provides parsing functions for S-expressions and JSON prefix notation.

## Functions

### `parse_s_expression(text: str) -> Term`

Parse an S-expression string into a Term.

```python
from dreamlog.prefix_parser import parse_s_expression

# Parse atoms
atom = parse_s_expression("john")  # Atom("john")

# Parse variables (uppercase)
var = parse_s_expression("X")  # Variable("X")

# Parse compound terms
parent = parse_s_expression("(parent john mary)")
# Compound("parent", [Atom("john"), Atom("mary")])

# Parse nested terms
grandparent = parse_s_expression("(grandparent (father john) mary)")
# Compound("grandparent", [Compound("father", [Atom("john")]), Atom("mary")])
```

#### Syntax Rules

- **Atoms**: Lowercase identifiers or quoted strings
- **Variables**: Uppercase identifiers or underscore-prefixed
- **Compounds**: `(functor arg1 arg2 ...)`
- **Numbers**: Parsed as atoms
- **Lists**: `[a b c]` becomes `(list a b c)`

### `parse_prefix_notation(data: Any) -> Term`

Parse JSON-style prefix notation into a Term.

```python
from dreamlog.prefix_parser import parse_prefix_notation

# Parse from lists
atom = parse_prefix_notation("john")  # Atom("john")
var = parse_prefix_notation("X")      # Variable("X")

parent = parse_prefix_notation(["parent", "john", "mary"])
# Compound("parent", [Atom("john"), Atom("mary")])

# Parse nested structures
rule = parse_prefix_notation([
    "rule",
    ["grandparent", "X", "Z"],
    [["parent", "X", "Y"], ["parent", "Y", "Z"]]
])
```

### `term_to_sexp(term: Term) -> str`

Convert a Term to S-expression string representation.

```python
from dreamlog.prefix_parser import term_to_sexp
from dreamlog.terms import compound, atom, var

term = compound("parent", atom("john"), var("X"))
sexp = term_to_sexp(term)  # "(parent john X)"
```

### `term_to_prefix_json(term: Term) -> Any`

Convert a Term to JSON prefix notation.

```python
from dreamlog.prefix_parser import term_to_prefix_json
from dreamlog.terms import compound, atom, var

term = compound("parent", atom("john"), var("X"))
json_repr = term_to_prefix_json(term)  # ["parent", "john", "X"]
```

## Parsing Rules and Facts

### `parse_fact(data: Any) -> Fact`

Parse a fact from S-expression or prefix notation.

```python
from dreamlog.prefix_parser import parse_fact

# From S-expression
fact1 = parse_fact("(parent john mary)")

# From prefix notation
fact2 = parse_fact(["parent", "john", "mary"])
```

### `parse_rule(data: Any) -> Rule`

Parse a rule from S-expression or prefix notation.

```python
from dreamlog.prefix_parser import parse_rule

# S-expression with :- operator
rule1 = parse_rule("(grandparent X Z) :- (parent X Y), (parent Y Z)")

# Prefix notation
rule2 = parse_rule([
    "rule",
    ["grandparent", "X", "Z"],
    [["parent", "X", "Y"], ["parent", "Y", "Z"]]
])
```

## Batch Parsing

### `parse_knowledge_base(text: str) -> List[Union[Fact, Rule]]`

Parse multiple facts and rules from a text block.

```python
from dreamlog.prefix_parser import parse_knowledge_base

kb_text = """
(parent john mary)
(parent mary alice)

(grandparent X Z) :- (parent X Y), (parent Y Z)
(ancestor X Y) :- (parent X Y)
(ancestor X Z) :- (parent X Y), (ancestor Y Z)
"""

items = parse_knowledge_base(kb_text)
# Returns list of Fact and Rule objects
```

## Error Handling

The parser raises `ParseError` for invalid syntax:

```python
from dreamlog.prefix_parser import parse_s_expression, ParseError

try:
    term = parse_s_expression("(invalid (syntax")
except ParseError as e:
    print(f"Parse error: {e}")
```

## Special Cases

### Numbers

Numbers are parsed as atoms:

```python
term = parse_s_expression("42")      # Atom("42")
term = parse_s_expression("(age john 25)")  
# Compound("age", [Atom("john"), Atom("25")])
```

### Quoted Strings

Quoted strings preserve spaces and special characters:

```python
term = parse_s_expression('"hello world"')  # Atom("hello world")
term = parse_s_expression('(say "Hello, World!")')
# Compound("say", [Atom("Hello, World!")])
```

### Anonymous Variables

Underscore creates anonymous variables:

```python
term = parse_s_expression("(parent _ mary)")
# Compound("parent", [Variable("_"), Atom("mary")])
```

### Lists

Square brackets create list terms:

```python
term = parse_s_expression("[1 2 3]")
# Compound("list", [Atom("1"), Atom("2"), Atom("3")])

term = parse_s_expression("(append [1 2] [3 4] X)")
# Compound("append", [
#     Compound("list", [Atom("1"), Atom("2")]),
#     Compound("list", [Atom("3"), Atom("4")]),
#     Variable("X")
# ])
```

## Performance Considerations

- The parser uses recursive descent for S-expressions
- JSON parsing leverages Python's built-in JSON parser
- Large knowledge bases should be parsed once and cached
- Use batch parsing for multiple items to reduce overhead

## Examples

### Complete Example

```python
from dreamlog.prefix_parser import (
    parse_s_expression,
    parse_prefix_notation,
    term_to_sexp,
    term_to_prefix_json
)

# Parse from S-expression
term = parse_s_expression("(parent john mary)")

# Convert to different formats
sexp_str = term_to_sexp(term)          # "(parent john mary)"
json_repr = term_to_prefix_json(term)  # ["parent", "john", "mary"]

# Parse from JSON
term2 = parse_prefix_notation(["parent", "john", "mary"])

# Terms are equal regardless of parse method
assert term == term2
```