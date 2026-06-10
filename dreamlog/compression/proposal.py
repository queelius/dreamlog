"""A Proposal is a pure description of one compression transformation."""
from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple, Union

from ..knowledge import Fact, Rule

Clause = Union[Fact, Rule]


@dataclass(frozen=True)
class Proposal:
    """kind uses today's CompressionCandidate operation labels verbatim:
    subsumption | pruning | generalization | invention | extraction |
    recursion | llm_compression."""
    kind: str
    remove: Tuple[Clause, ...] = ()
    add: Tuple[Clause, ...] = ()
    notes: Mapping[str, Any] = field(default_factory=dict)
