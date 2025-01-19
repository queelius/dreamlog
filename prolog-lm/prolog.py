from typing import List, Dict, Optional
import sys, copy, re

class Term:
    def __init__(self, string: str) -> None:
        """
        Initialize a Term object from a string representation.
        A term is expected to be in the format "predicate(arg1,arg2,...)".

        :param string: A string representation of the term.
        """
        if string[-1] != ')':
            fatal("Syntax error in term: %s" % string)
        fields = string.split('(')
        if len(fields) != 2:
            fatal("Syntax error in term: %s" % string)
        self.predicate = fields[0]
        self.arguments = fields[1][:-1].split(',')

    def __repr__(self) -> str:
        return f"{self.predicate}({','.join(self.arguments)})"

class Rule:
    def __init__(self, string: str) -> None:
        """
        Initialize a Rule object from a string representation.
        A rule is expected to be in the format "head :- body1;body2;...".

        :param string: A string representation of the rule.
        """
        fields = string.split(":-")
        self.head = Term(fields[0])
        self.body: List[Term] = []
        if len(fields) == 2:
            body_terms = re.sub("\),", ");", fields[1]).split(";")
            self.body = [Term(term) for term in body_terms]

    def __repr__(self) -> str:
        body_repr = ",".join(str(term) for term in self.body)
        return f"{self.head} :- {body_repr}"

class Goal:
    goal_id = 100

    def __init__(self, rule: Rule, parent: Optional['Goal'] = None, env: Dict[str, str] = {}) -> None:
        """
        Initialize a Goal object, representing a rule at a certain point in its evaluation.

        :param rule: The Rule object associated with this goal.
        :param parent: The parent Goal object, if any, from which this goal was spawned.
        :param env: The current environment mapping variables to their bindings.
        """
        Goal.goal_id += 1
        self.id = Goal.goal_id
        self.rule = rule
        self.parent = parent
        self.env = copy.deepcopy(env)
        self.current_index = 0  # Index to track the current subgoal being processed.

    def __repr__(self) -> str:
        return f"Goal {self.id} rule={self.rule} index={self.current_index} env={self.env}"

# Simplify and add type annotations to other functions as needed...

def fatal(message: str) -> None:
    """
    Prints a fatal error message and exits the program.

    :param message: The error message to be printed.
    """
    sys.stdout.write(f"Fatal: {message}\n")
    sys.exit(1)

def unify(src_term: Term, src_env: Dict[str, str], dest_term: Term, dest_env: Dict[str, str]) -> bool:
    """
    Attempts to unify two terms under their respective environments.

    :param src_term: The source term to unify.
    :param src_env: The environment (variable bindings) of the source term.
    :param dest_term: The destination term to unify.
    :param dest_env: The environment (variable bindings) of the destination term.
    :return: True if unification succeeds, False otherwise.
    """
    if src_term.predicate != dest_term.predicate or len(src_term.arguments) != len(dest_term.arguments):
        return False  # Predicates or arity mismatch

    for src_arg, dest_arg in zip(src_term.arguments, dest_term.arguments):
        src_val: Optional[str] = src_env.get(src_arg, src_arg)
        dest_val: Optional[str] = dest_env.get(dest_arg, dest_arg)

        if src_val.isupper():  # src_arg is a variable
            if dest_val.isupper():  # Both are variables
                if src_val != dest_val:  # Different variables
                    dest_env[dest_arg] = src_val  # Bind variable
            elif src_val != dest_val:
                return False  # Variable and constant mismatch
        elif dest_val.isupper():  # dest_arg is a variable
            dest_env[dest_arg] = src_val  # Bind variable
        elif src_val != dest_val:
            return False  # Constant mismatch

    return True

def search(query: Term) -> None:
    """
    Initiates a search for solutions to a given query based on the rules in the knowledge base.

    :param query: The query term to find solutions for.
    """
    global goalId
    goalId = 0
    if trace:
        print(f"Searching for: {query}")
    initial_goal = Goal(Rule(f"{query}."), None, {})
    stack: List[Goal] = [initial_goal]

    while stack:
        current_goal = stack.pop()
        if trace:
            print(f"Processing goal: {current_goal}")

        if current_goal.current_index >= len(current_goal.rule.body):
            if current_goal.parent is None:
                print(f"Solution found: {current_goal.env}")
                return
            else:
                # Attempt to unify the goal with its parent and continue the search.
                if unify(current_goal.rule.head, current_goal.env, 
                         current_goal.parent.rule.body[current_goal.parent.current_index], current_goal.parent.env):
                    current_goal.parent.current_index += 1
                    stack.append(current_goal.parent)
            continue

        for rule in rules:
            if not unify(current_goal.rule.body[current_goal.current_index], {}, rule.head, {}):
                continue  # Skip to the next rule if unification fails

            new_goal = Goal(rule, current_goal)
            if unify(current_goal.rule.body[current_goal.current_index], current_goal.env, 
                     rule.head, new_goal.env):
                stack.append(new_goal)

    print("No solution found.")
