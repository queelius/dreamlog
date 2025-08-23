# Terms Module

The `dreamlog.terms` module provides the fundamental data structures for representing logical terms.

## Classes

### `Term`

Base class for all terms in DreamLog.

```python
class Term:
    def __eq__(self, other) -> bool
    def __hash__(self) -> int
    def __str__(self) -> str
    def __repr__(self) -> str
    def occurs_in(self, term: 'Term') -> bool
    def get_vars(self) -> Set['Variable']
    def substitute(self, substitution: Dict['Variable', 'Term']) -> 'Term'
    def to_json(self) -> Any
```

#### Methods

- **`occurs_in(term)`** - Check if this term occurs in another term
- **`get_vars()`** - Get all variables in this term
- **`substitute(substitution)`** - Apply a substitution to this term
- **`to_json()`** - Convert to JSON representation

### `Atom`

Represents an atomic constant.

```python
class Atom(Term):
    def __init__(self, name: str)
    
    @property
    def name(self) -> str
```

#### Example

```python
from dreamlog.terms import Atom

john = Atom("john")
mary = Atom("mary")
```

### `Variable`

Represents a logic variable.

```python
class Variable(Term):
    def __init__(self, name: str)
    
    @property
    def name(self) -> str
```

#### Example

```python
from dreamlog.terms import Variable

X = Variable("X")
Y = Variable("Y")
```

### `Compound`

Represents a compound term with a functor and arguments.

```python
class Compound(Term):
    def __init__(self, functor: str, args: List[Term])
    
    @property
    def functor(self) -> str
    
    @property
    def args(self) -> List[Term]
    
    @property
    def arity(self) -> int
```

#### Example

```python
from dreamlog.terms import Compound, Atom, Variable

# (parent john mary)
parent = Compound("parent", [Atom("john"), Atom("mary")])

# (grandparent X Z)
grandparent = Compound("grandparent", [Variable("X"), Variable("Z")])
```

## Factory Functions

### `atom(name: str) -> Atom`

Create an atomic term.

```python
from dreamlog.terms import atom

john = atom("john")
```

### `var(name: str) -> Variable`

Create a variable term.

```python
from dreamlog.terms import var

X = var("X")
```

### `compound(functor: str, *args) -> Compound`

Create a compound term.

```python
from dreamlog.terms import compound, atom, var

# (parent john mary)
parent = compound("parent", atom("john"), atom("mary"))

# (grandparent X Y)
grandparent = compound("grandparent", var("X"), var("Y"))
```

## Utility Functions

### `is_ground(term: Term) -> bool`

Check if a term contains no variables.

```python
from dreamlog.terms import is_ground, atom, var, compound

is_ground(atom("john"))  # True
is_ground(var("X"))      # False
is_ground(compound("parent", atom("john"), var("X")))  # False
```

### `get_variables(term: Term) -> Set[Variable]`

Get all variables in a term.

```python
from dreamlog.terms import get_variables, compound, var

term = compound("parent", var("X"), var("Y"))
vars = get_variables(term)  # {Variable("X"), Variable("Y")}
```

### `rename_variables(term: Term, suffix: str = "_1") -> Term`

Rename all variables in a term by adding a suffix.

```python
from dreamlog.terms import rename_variables, compound, var

term = compound("parent", var("X"), var("Y"))
renamed = rename_variables(term, "_2")
# Results in: (parent X_2 Y_2)
```

## Type Checking

### `is_atom(term: Term) -> bool`
### `is_variable(term: Term) -> bool`
### `is_compound(term: Term) -> bool`

Check the type of a term.

```python
from dreamlog.terms import is_atom, is_variable, is_compound
from dreamlog.terms import atom, var, compound

is_atom(atom("john"))        # True
is_variable(var("X"))        # True
is_compound(compound("f"))   # True
```

## Term Comparison

Terms support equality comparison and hashing, making them suitable for use in sets and as dictionary keys.

```python
from dreamlog.terms import atom, var

a1 = atom("john")
a2 = atom("john")
a3 = atom("mary")

a1 == a2  # True
a1 == a3  # False

terms_set = {a1, a2, a3}  # Contains 2 items
```

## String Representation

Terms have both `str()` and `repr()` representations:

```python
from dreamlog.terms import compound, atom, var

term = compound("parent", atom("john"), var("X"))
str(term)   # "(parent john X)"
repr(term)  # "Compound('parent', [Atom('john'), Variable('X')])"
```