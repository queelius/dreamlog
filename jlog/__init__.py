"""
JLOG - JSON-Log (Prolog-like programming language in JSON) with LLM Integration

A clean, modern implementation of a Prolog-like language that uses JSON for
data representation and integrates with Language Models for automatic
knowledge inference.
"""

from .terms import Term, Atom, Variable, Compound, atom, var, compound, term_from_json
from .knowledge import Fact, Rule, KnowledgeBase
from .unification import Unifier
from .evaluator import PrologEvaluator, Solution
from .llm_hook import LLMHook
from .engine import JLogEngine, create_engine_with_llm, create_family_kb

__version__ = "0.8"
__all__ = [
    "Term", "Atom", "Variable", "Compound", "atom", "var", "compound", "term_from_json",
    "Fact", "Rule", "KnowledgeBase", 
    "Unifier", "PrologEvaluator", "Solution",
    "LLMHook",
    "JLogEngine", "create_engine_with_llm", "create_family_kb"
]
