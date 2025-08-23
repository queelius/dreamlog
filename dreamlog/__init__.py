"""
DreamLog - JSON-Log (Prolog-like programming language in JSON) with LLM Integration

A clean, modern implementation of a Prolog-like language that uses JSON for
data representation and integrates with Language Models for automatic
knowledge inference.
"""

from .terms import Term, Atom, Variable, Compound, atom, var, compound, term_from_prefix
from .knowledge import Fact, Rule, KnowledgeBase
from .unification import Unifier, unify, match, subsumes
from .evaluator import PrologEvaluator, Solution
from .llm_hook import LLMHook
from .engine import DreamLogEngine, create_engine_with_llm, create_family_kb
from .prefix_parser import parse_prefix_notation, parse_s_expression, term_to_sexp, term_to_prefix_json

__version__ = "0.8"
__all__ = [
    "Term", "Atom", "Variable", "Compound", "atom", "var", "compound", "term_from_prefix",
    "Fact", "Rule", "KnowledgeBase", 
    "Unifier", "unify", "match", "subsumes", "PrologEvaluator", "Solution",
    "LLMHook",
    "DreamLogEngine", "create_engine_with_llm", "create_family_kb",
    "parse_prefix_notation", "parse_s_expression", "term_to_sexp", "term_to_prefix_json"
]
