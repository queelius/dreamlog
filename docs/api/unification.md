# Unification Module

The `dreamlog.unification` module provides unification algorithms for pattern matching and variable binding.

## Core Functions

### `unify(term1: Term, term2: Term, subst: Optional[Dict] = None) -> Optional[Dict]`

Perform standard unification between two terms.

```python
from dreamlog.unification import unify
from dreamlog.terms import compound, atom, var

# Unify ground terms
t1 = compound("parent", atom("john"), atom("mary"))
t2 = compound("parent", atom("john"), atom("mary"))
result = unify(t1, t2)  # {} (empty substitution = success)

# Unify with variables
t1 = compound("parent", var("X"), atom("mary"))
t2 = compound("parent", atom("john"), atom("mary"))
result = unify(t1, t2)  # {Variable("X"): Atom("john")}

# Failed unification
t1 = compound("parent", atom("john"), atom("mary"))
t2 = compound("parent", atom("bob"), atom("mary"))
result = unify(t1, t2)  # None
```

### `match(pattern: Term, term: Term, subst: Optional[Dict] = None) -> Optional[Dict]`

One-way pattern matching (only binds variables in pattern).

```python
from dreamlog.unification import match

# Pattern matching
pattern = compound("parent", var("X"), var("Y"))
term = compound("parent", atom("john"), atom("mary"))
result = match(pattern, term)  
# {Variable("X"): Atom("john"), Variable("Y"): Atom("mary")}

# Variables in term are not bound
pattern = compound("parent", atom("john"), atom("mary"))
term = compound("parent", var("X"), var("Y"))
result = match(pattern, term)  # None (can't match atoms to variables)
```

### `subsumes(general: Term, specific: Term) -> bool`

Check if one term subsumes (is more general than) another.

```python
from dreamlog.unification import subsumes

# Variable subsumes atom
general = var("X")
specific = atom("john")
subsumes(general, specific)  # True

# General pattern subsumes specific instance
general = compound("parent", var("X"), var("Y"))
specific = compound("parent", atom("john"), atom("mary"))
subsumes(general, specific)  # True

# Specific doesn't subsume general
subsumes(specific, general)  # False
```

## Unifier Class

The `Unifier` class provides stateful unification operations.

```python
class Unifier:
    def __init__(self, mode: str = "standard", debug: bool = False)
    def unify(self, term1: Term, term2: Term) -> bool
    def get_substitution(self) -> Dict[Variable, Term]
    def apply(self, term: Term) -> Term
    def reset(self) -> None
    def copy(self) -> 'Unifier'
```

### Example Usage

```python
from dreamlog.unification import Unifier
from dreamlog.terms import compound, var, atom

unifier = Unifier(debug=True)

t1 = compound("parent", var("X"), var("Y"))
t2 = compound("parent", atom("john"), atom("mary"))

if unifier.unify(t1, t2):
    subst = unifier.get_substitution()
    # {Variable("X"): Atom("john"), Variable("Y"): Atom("mary")}
    
    # Apply substitution to another term
    t3 = compound("knows", var("X"), var("Z"))
    result = unifier.apply(t3)
    # Compound("knows", Atom("john"), Variable("Z"))
```

## Unification Modes

### Standard Unification

Default mode - bidirectional unification.

```python
unifier = Unifier(mode="standard")
```

### One-Way Matching

Only binds variables in the first term.

```python
unifier = Unifier(mode="match")
```

### Subsumption Checking

Checks if first term is more general.

```python
unifier = Unifier(mode="subsumes")
```

## Advanced Features

### Occurs Check

Prevents infinite structures during unification.

```python
from dreamlog.unification import unify

# Would create infinite structure X = f(X)
t1 = var("X")
t2 = compound("f", var("X"))
result = unify(t1, t2, occurs_check=True)  # None (fails occurs check)
```

### Debugging

Enable debug mode to trace unification steps.

```python
unifier = Unifier(debug=True)
unifier.unify(t1, t2)
# Prints unification trace to stdout
```

### Constraint Handling

Support for constraints during unification.

```python
from dreamlog.unification import UnificationConstraint

class NotEqualConstraint(UnificationConstraint):
    def __init__(self, var1, var2):
        self.var1 = var1
        self.var2 = var2
    
    def check(self, substitution):
        val1 = substitution.get(self.var1)
        val2 = substitution.get(self.var2)
        return val1 != val2 if val1 and val2 else True

# Use with unifier
unifier = Unifier()
unifier.add_constraint(NotEqualConstraint(var("X"), var("Y")))
```

## Substitution Operations

### Applying Substitutions

```python
from dreamlog.unification import apply_substitution

subst = {var("X"): atom("john"), var("Y"): atom("mary")}
term = compound("parent", var("X"), var("Y"))
result = apply_substitution(term, subst)
# Compound("parent", Atom("john"), Atom("mary"))
```

### Composing Substitutions

```python
from dreamlog.unification import compose_substitutions

subst1 = {var("X"): var("Y")}
subst2 = {var("Y"): atom("john")}
composed = compose_substitutions(subst1, subst2)
# {Variable("X"): Atom("john"), Variable("Y"): Atom("john")}
```

### Renaming Variables

```python
from dreamlog.unification import rename_apart

# Rename variables to avoid conflicts
term1 = compound("p", var("X"), var("Y"))
term2 = compound("q", var("X"), var("Z"))

renamed = rename_apart(term2, term1)
# Compound("q", Variable("X_1"), Variable("Z_1"))
```

## Most General Unifier (MGU)

```python
from dreamlog.unification import mgu

# Find the most general unifier
terms = [
    compound("p", var("X"), atom("a")),
    compound("p", atom("b"), var("Y")),
    compound("p", var("Z"), var("W"))
]

result = mgu(terms)
# {Variable("X"): Atom("b"), 
#  Variable("Y"): Atom("a"),
#  Variable("Z"): Atom("b"),
#  Variable("W"): Atom("a")}
```

## Performance Considerations

- **Occurs check**: Disabled by default for performance
- **Caching**: Unifier caches intermediate results
- **Early termination**: Fails fast on incompatible terms
- **Index-based matching**: Uses term structure for quick rejection

## Error Handling

```python
from dreamlog.unification import UnificationError

try:
    result = unify(term1, term2, strict=True)
except UnificationError as e:
    print(f"Unification failed: {e}")
```

## Examples

### Complete Example

```python
from dreamlog.unification import Unifier, unify, match, subsumes
from dreamlog.terms import compound, atom, var

# Create terms
pattern = compound("grandparent", var("X"), var("Z"))
instance = compound("grandparent", atom("john"), atom("alice"))

# Standard unification
subst = unify(pattern, instance)
print(subst)  # {Variable("X"): Atom("john"), Variable("Z"): Atom("alice")}

# Pattern matching
subst = match(pattern, instance)
print(subst)  # Same result

# Subsumption
is_more_general = subsumes(pattern, instance)
print(is_more_general)  # True

# Stateful unifier
unifier = Unifier()
if unifier.unify(pattern, instance):
    # Apply to another term
    query = compound("knows", var("X"), var("Z"))
    result = unifier.apply(query)
    print(result)  # (knows john alice)
```