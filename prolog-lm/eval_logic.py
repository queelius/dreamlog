from typing import Union, List

class Atom:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name
    
    def __eq__(self, other):
        return self.name == other.name

    
class Symbol:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name
    
    def __eq__(self, other):
        return self.name == other.name

class Term:
    def __init__(self,
                 name : Atom,
                 args : List[Union[Atom, Symbol]]):
        """
        name: the name of the term
        args: the arguments of the term. these can be atoms or symbols

        Example:
        Term(Atom("foo"), [Atom("ben"), Symbol("x")])
        ## to represent the term foo(ben, x)
        """

        self.name = name
        self.args = args

    def __str__(self):
        return f"{self.name}({','.join(self.args)})"
    
    def __eq__(self, other):
        return self.name == other.name and self.args == other.args

class Rule:
    def __init__(self,
                 head : Term,
                 terms : List[Term] = []):
        """
        head: the head of the rule
        terms: the terms of the rule. Default is an empty list of terms, which
               means that the rule is a fact (i.e. it has no conditions)

        Example:
        To represent the rule foo(ben, x) :- bar(ben, x), baz(ben, x)
        Rule(
            Term(Atom("foo"), [Atom("ben"), Symbol("x")]),
                [Term(Atom("bar"), [Atom("ben"), Symbol("x")]),
                 Term(Atom("baz"), [Atom("ben"), Symbol("x")])])
        """

        self.head = head
        self.terms = terms

    def is_fact(self):
        return len(self.terms) == 0
    
    def __str__(self):
        if self.is_fact():
            return f"{self.head}."
        else:
            return f"{self.head} :- {','.join(self.terms)}."
        
    def __eq__(self, other):
        return self.head == other.head and set(self.terms) == set(other.terms)

class KB:
    def __init__(self):
        self.rules : List[Rule] = []

    def __str__(self):
        return "\n".join([str(rule) for rule in self.rules])
    
    def __add__(self, other):
        """
        Merge two KBs

        Example:
        kb1 = KB()
        kb2 = KB()
        kb1.rules = [Rule(Term(Atom("foo"), [Atom("ben"), Symbol("x")]))]
        kb2.rules = [Rule(Term(Atom("bar"), [Atom("ben"), Symbol("x")]))]
        kb3 = kb1 + kb2

        kb3.rules
        [Rule(Term(Atom("foo"), [Atom("ben"), Symbol("x")])),
         Rule(Term(Atom("bar"), [Atom("ben"), Symbol("x")]))]
        """
        new_kb = KB()
        new_kb.rules = self.rules + other.rules
        return new_kb
    
    def __sub__(self, other):
        """
        Remove rules from a KB

        Example:
        kb1 = KB()
        kb1.rules = [Rule(Term(Atom("foo"), [Atom("ben"), Symbol("x")]))]
        kb2 = KB()
        kb2.rules = [Rule(Term(Atom("foo"), [Atom("ben"), Symbol("x")]))]
        kb3 = kb1 - kb2

        kb3.rules
        []
        """
        new_kb = KB()
        new_kb.rules = [r for r in self.rules if r not in other.rules]
        return new_kb
    
    def __eq__(self, other):
        return set(self.rules) == set(other.rules)
    
    # let's define a way to remove rules from a KB
    def remove_rule(self, rule):
        self.rules = [r for r in self.rules if r != rule]

    def remove_rule_by_name(self, name):
        self.rules = [r for r in self.rules if r.name != name]

    def add_rule(self, rule):
        self.rules.append(rule)
    
    def add_rules(self, rules):
        self.rules.extend(rules)

    def get_rule_by_name(self, name):
        return [r for r in self.rules if r.name == name]
    
# let's implement unification now
    
def unify(x : Union[Atom, Symbol],
          y : Union[Atom, Symbol],
          theta : dict = {}):
    
    """
    x: an atom or a symbol
    y: an atom or a symbol
    theta: a dictionary of substitutions

    Example:
    unify(Atom("ben"), Atom("ben"), {})
    ## does not return anything, but the theta dictionary is updated to {'ben': 'ben'}
    ## this means that ben can be substituted by ben
    ## i.e. ben = ben
    ## returns: {}

    unify(Atom("ben"), Symbol("x"), {})
    ## we have a new substitution: x = ben
    ## returns {'x': Atom("ben")}
    """

    if x == y:
        return theta
    
    if isinstance(x, Symbol):
        return unify_var(x, y, theta)

    if isinstance(y, Symbol):
        return unify_var(y, x, theta)
    
    if isinstance(x, Term) and isinstance(y, Term):
        return unify(x.args, y.args, unify(x.name, y.name, theta))
    
    if isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            return None
        else:
            return unify(x[1:], y[1:], unify(x[0], y[0], theta))
        
    return None

def unify_var(var : Symbol,
                x : Union[Atom, Symbol],
                theta : dict):
        """
        var: a symbol
        x: an atom or a symbol
        theta: a dictionary of substitutions
    
        Example:
        unify_var(Symbol("x"), Atom("ben"), {})
        ## we have a new substitution: x = ben
        ## returns {'x': Atom("ben")}
        """
    
        if var in theta:
            return unify(theta[var], x, theta)
        
        if x in theta:
            return unify(var, theta[x], theta)
        
        if occur_check(var, x, theta):
            return None
        
        theta[var] = x
        return theta