"""
Type aliases and type definitions for DreamLog.

This module provides clear type aliases to improve code readability
and type safety throughout the codebase.
"""

from typing import Dict, List, Tuple, Any, Optional, Iterator, Union, NewType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .terms import Term


# Variable name type - always a string
VarName = NewType('VarName', str)

# Bindings - map variable names to terms
Bindings = Dict[VarName, 'Term']

# Functor/Arity key for indexing
IndexKey = Tuple[str, int]

# Query result - variable name to value mapping
QueryResult = Dict[str, Any]

# Prefix notation types
PrefixData = Union[str, int, float, bool, None, List['PrefixData']]

# Knowledge base size
KBSize = NewType('KBSize', int)

# Recursion depth
Depth = NewType('Depth', int)