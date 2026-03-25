"""
DreamLog - JSON-Log (Prolog-like programming language in JSON) with LLM Integration

A clean, modern implementation of a Prolog-like language that uses JSON for
data representation and integrates with Language Models for automatic
knowledge inference.
"""

from .terms import Term, Atom, Variable, Compound
from .factories import atom, var, compound, term_from_prefix
from .knowledge import Fact, Rule, KnowledgeBase
from .unification import Unifier, unify, match, subsumes
from .evaluator import PrologEvaluator, Solution, FlounderingError, InstantiationError
from .anti_unification import anti_unify, anti_unify_many, AntiUnificationResult
from .kb_dreamer import KnowledgeBaseDreamer, DreamSession
from .skeleton import extract_skeleton, RuleSetSkeleton
from .llm_hook import LLMHook
from .engine import DreamLogEngine, create_engine_with_llm, create_family_kb
from .prefix_parser import parse_prefix_notation, parse_s_expression

__version__ = "0.8"
__all__ = [
    "Term", "Atom", "Variable", "Compound", "atom", "var", "compound", "term_from_prefix",
    "Fact", "Rule", "KnowledgeBase", 
    "Unifier", "unify", "match", "subsumes", "PrologEvaluator", "Solution",
    "FlounderingError", "InstantiationError",
    "anti_unify", "anti_unify_many", "AntiUnificationResult",
    "KnowledgeBaseDreamer", "DreamSession",
    "extract_skeleton", "RuleSetSkeleton",
    "LLMHook",
    "DreamLogEngine", "create_engine_with_llm", "create_family_kb",
    "parse_prefix_notation", "parse_s_expression"
]
