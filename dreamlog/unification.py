"""
Enhanced unification module for DreamLog

Improvements over the original:
- Functional API alongside class-based
- Pattern matching support
- Debugging/tracing capabilities
- Performance optimizations
- One-way matching mode
"""

from typing import Dict, Optional, Set, Tuple, List, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from .terms import Term, Atom, Variable, Compound


class UnificationMode(Enum):
    """Different unification modes"""
    STANDARD = "standard"  # Standard two-way unification
    MATCH = "match"  # One-way pattern matching (term1 is pattern)
    SUBSUME = "subsume"  # Check if term1 subsumes term2


@dataclass
class UnificationResult:
    """Result of unification with optional metadata"""
    success: bool
    bindings: Optional[Dict[str, Term]] = None
    steps: List[str] = field(default_factory=list)
    
    def __bool__(self) -> bool:
        return self.success


class UnificationTrace:
    """Trace unification steps for debugging"""
    
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.steps: List[str] = []
    
    def add(self, message: str) -> None:
        if self.enabled:
            self.steps.append(message)
    
    def get_trace(self) -> List[str]:
        return self.steps.copy()


# ============================================================================
# Functional API - Simple, clean interface
# ============================================================================

def unify(term1: Term, term2: Term, 
          bindings: Optional[Dict[str, Term]] = None,
          mode: UnificationMode = UnificationMode.STANDARD,
          trace: bool = False) -> Optional[Dict[str, Term]]:
    """
    Simple functional unification interface.
    
    Examples:
        >>> from dreamlog import atom, var, compound
        >>> bindings = unify(compound("p", var("X")), compound("p", atom("a")))
        >>> bindings
        {"X": Atom("a")}
    """
    unifier = Unifier(mode=mode, trace=trace)
    result = unifier.unify(term1, term2, bindings)
    return result.bindings if result.success else None


def match(pattern: Term, term: Term, 
          bindings: Optional[Dict[str, Term]] = None) -> Optional[Dict[str, Term]]:
    """
    One-way pattern matching - only bind variables in pattern.
    
    Examples:
        >>> match(compound("p", var("X")), compound("p", atom("a")))
        {"X": Atom("a")}
        >>> match(compound("p", atom("a")), compound("p", var("X")))
        None  # Won't bind variables in term
    """
    return unify(pattern, term, bindings, mode=UnificationMode.MATCH)


def subsumes(general: Term, specific: Term) -> bool:
    """
    Check if general term subsumes specific term.
    
    A term t1 subsumes t2 if t1 can be made identical to t2 by substitution.
    """
    result = unify(general, specific, mode=UnificationMode.SUBSUME)
    return result is not None


def apply_substitution(term: Term, bindings: Dict[str, Term]) -> Term:
    """Apply substitution to a term"""
    return term.substitute(bindings)


def compose_substitutions(s1: Dict[str, Term], s2: Dict[str, Term]) -> Dict[str, Term]:
    """
    Compose two substitutions: (s1 ∘ s2)(x) = s1(s2(x))
    """
    result = {}
    
    # Apply s1 to all terms in s2
    for var, term in s2.items():
        result[var] = term.substitute(s1)
    
    # Add bindings from s1 that aren't in s2
    for var, term in s1.items():
        if var not in result:
            result[var] = term
    
    return result


# ============================================================================
# Enhanced Unifier Class
# ============================================================================

class Unifier:
    """
    Enhanced unification with multiple modes and tracing.
    """
    
    def __init__(self, mode: UnificationMode = UnificationMode.STANDARD, 
                 trace: bool = False,
                 occurs_check: bool = True,
                 max_depth: int = 1000):
        """
        Initialize unifier with options.
        
        Args:
            mode: Unification mode
            trace: Enable step-by-step tracing
            occurs_check: Enable occurs check (prevent infinite structures)
            max_depth: Maximum recursion depth
        """
        self.mode = mode
        self.trace = UnificationTrace(trace)
        self.occurs_check = occurs_check
        self.max_depth = max_depth
        self._depth = 0
    
    def unify(self, term1: Term, term2: Term, 
              bindings: Optional[Dict[str, Term]] = None) -> UnificationResult:
        """
        Main unification method with all enhancements.
        """
        if bindings is None:
            bindings = {}
        
        # Check recursion depth
        self._depth += 1
        if self._depth > self.max_depth:
            self.trace.add(f"Max depth {self.max_depth} exceeded")
            return UnificationResult(False, None, self.trace.get_trace())
        
        try:
            result_bindings = self._unify_internal(term1, term2, bindings)
            success = result_bindings is not None
            return UnificationResult(success, result_bindings, self.trace.get_trace())
        finally:
            self._depth -= 1
    
    def _unify_internal(self, term1: Term, term2: Term, 
                        bindings: Dict[str, Term]) -> Optional[Dict[str, Term]]:
        """Internal unification logic"""
        
        # Apply existing bindings
        term1 = self._deref(term1, bindings)
        term2 = self._deref(term2, bindings)
        
        self.trace.add(f"Unifying: {term1} with {term2}")
        
        # Same term after dereferencing
        if term1 == term2:
            self.trace.add("Terms are identical")
            return bindings
        
        # Variable unification
        if isinstance(term1, Variable):
            if self.mode == UnificationMode.MATCH:
                # In match mode, only bind variables in pattern (term1)
                return self._bind_variable(term1, term2, bindings)
            else:
                return self._unify_variable(term1, term2, bindings)
        
        if isinstance(term2, Variable):
            if self.mode == UnificationMode.MATCH:
                # In match mode, don't bind variables in term2
                self.trace.add(f"Match mode: Cannot bind variable in term: {term2}")
                return None
            else:
                return self._unify_variable(term2, term1, bindings)
        
        # Compound unification
        if isinstance(term1, Compound) and isinstance(term2, Compound):
            return self._unify_compound(term1, term2, bindings)
        
        # Different types or atoms
        self.trace.add(f"Cannot unify different types: {type(term1)} vs {type(term2)}")
        return None
    
    def _deref(self, term: Term, bindings: Dict[str, Term]) -> Term:
        """
        Dereference a term by following variable bindings.
        More efficient than full substitution.
        """
        if isinstance(term, Variable):
            visited = set()
            while term.name in bindings and term.name not in visited:
                visited.add(term.name)
                bound = bindings[term.name]
                if isinstance(bound, Variable):
                    term = bound
                else:
                    return bound
        return term
    
    def _bind_variable(self, var: Variable, term: Term, 
                       bindings: Dict[str, Term]) -> Optional[Dict[str, Term]]:
        """Bind a variable to a term with occurs check"""
        
        # Occurs check
        if self.occurs_check and self._occurs_in(var.name, term, bindings):
            self.trace.add(f"Occurs check failed: {var.name} occurs in {term}")
            return None
        
        # Create new binding
        new_bindings = bindings.copy()
        new_bindings[var.name] = term
        self.trace.add(f"Bound: {var.name} = {term}")
        return new_bindings
    
    def _unify_variable(self, var: Variable, term: Term, 
                        bindings: Dict[str, Term]) -> Optional[Dict[str, Term]]:
        """Standard variable unification"""
        
        # Check if variable is already bound
        if var.name in bindings:
            return self._unify_internal(bindings[var.name], term, bindings)
        
        # Check if term is a bound variable
        if isinstance(term, Variable) and term.name in bindings:
            return self._unify_internal(var, bindings[term.name], bindings)
        
        return self._bind_variable(var, term, bindings)
    
    def _unify_compound(self, comp1: Compound, comp2: Compound, 
                        bindings: Dict[str, Term]) -> Optional[Dict[str, Term]]:
        """Unify compound terms"""
        
        # Check functor and arity
        if comp1.functor != comp2.functor:
            self.trace.add(f"Different functors: {comp1.functor} vs {comp2.functor}")
            return None
        
        if len(comp1.args) != len(comp2.args):
            self.trace.add(f"Different arity: {len(comp1.args)} vs {len(comp2.args)}")
            return None
        
        # Unify arguments
        current_bindings = bindings
        for i, (arg1, arg2) in enumerate(zip(comp1.args, comp2.args)):
            self.trace.add(f"  Unifying arg {i}: {arg1} with {arg2}")
            result = self._unify_internal(arg1, arg2, current_bindings)
            if result is None:
                return None
            current_bindings = result
        
        return current_bindings
    
    def _occurs_in(self, var_name: str, term: Term, 
                    bindings: Dict[str, Term]) -> bool:
        """Check if variable occurs in term"""
        term = self._deref(term, bindings)
        
        if isinstance(term, Variable):
            return var_name == term.name
        elif isinstance(term, Compound):
            return any(self._occurs_in(var_name, arg, bindings) 
                      for arg in term.args)
        return False


# ============================================================================
# Pattern Building Helpers
# ============================================================================

def build_pattern(template: List[Any]) -> Term:
    """
    Build a pattern term from S-expression list.
    
    Examples:
        >>> build_pattern(["parent", "X", "mary"])
        Compound("parent", [Variable("X"), Atom("mary")])
    """
    from .prefix_parser import parse_prefix_notation
    return parse_prefix_notation(template)


def extract_variables(term: Term) -> Set[str]:
    """Extract all variable names from a term"""
    return term.get_variables()


def is_ground(term: Term) -> bool:
    """Check if term contains no variables"""
    return len(term.get_variables()) == 0


def rename_variables(term: Term, suffix: str = "_1") -> Tuple[Term, Dict[str, str]]:
    """
    Rename all variables in a term with a suffix.
    Returns the new term and the renaming map.
    """
    variables = term.get_variables()
    renaming = {var: f"{var}{suffix}" for var in variables}
    
    bindings = {var: Variable(new_name) 
                for var, new_name in renaming.items()}
    
    new_term = term.substitute(bindings)
    return new_term, renaming


# ============================================================================
# Constraint Support (Experimental)
# ============================================================================

class ConstrainedUnifier(Unifier):
    """
    Unifier with constraint support.
    
    Allows adding constraints like X > 5, X != Y, etc.
    """
    
    def __init__(self, constraints: Optional[List[Callable]] = None, **kwargs):
        super().__init__(**kwargs)
        self.constraints = constraints or []
    
    def add_constraint(self, constraint: Callable[[Dict[str, Term]], bool]) -> None:
        """Add a constraint function"""
        self.constraints.append(constraint)
    
    def _bind_variable(self, var: Variable, term: Term, 
                       bindings: Dict[str, Term]) -> Optional[Dict[str, Term]]:
        """Override to check constraints"""
        
        # First do normal binding
        result = super()._bind_variable(var, term, bindings)
        if result is None:
            return None
        
        # Check all constraints
        for constraint in self.constraints:
            if not constraint(result):
                self.trace.add(f"Constraint failed for binding {var.name} = {term}")
                return None
        
        return result


# Example constraint functions
def numeric_constraint(op: str, var_name: str, value: float) -> Callable:
    """Create a numeric constraint"""
    def check(bindings: Dict[str, Term]) -> bool:
        if var_name in bindings:
            term = bindings[var_name]
            if isinstance(term, Atom) and isinstance(term.value, (int, float)):
                if op == ">":
                    return term.value > value
                elif op == "<":
                    return term.value < value
                elif op == ">=":
                    return term.value >= value
                elif op == "<=":
                    return term.value <= value
                elif op == "==":
                    return term.value == value
                elif op == "!=":
                    return term.value != value
        return True
    return check


def not_equal_constraint(var1: str, var2: str) -> Callable:
    """Constraint that two variables must not be equal"""
    def check(bindings: Dict[str, Term]) -> bool:
        if var1 in bindings and var2 in bindings:
            return bindings[var1] != bindings[var2]
        return True
    return check