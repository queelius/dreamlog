from typing import List, Dict, Optional
import sys
import copy
import re

# Initialize global variables
rules: List['Rule'] = []  # Knowledge base to store rules and facts
trace: bool = False       # Trace flag for debugging
goal_id_counter: int = 100  # Counter to assign unique IDs to goals


def fatal(message: str) -> None:
    """
    Prints a fatal error message and exits the program.

    :param message: The error message to be displayed.
    """
    sys.stdout.write(f"Fatal: {message}\n")
    sys.exit(1)


class Term:
    """
    Represents a Prolog term with a predicate and a list of arguments.
    For example, parent(alice, bob) has predicate 'parent' and arguments ['alice', 'bob'].
    """

    def __init__(self, string: str) -> None:
        """
        Initializes a Term object from its string representation.

        :param string: The string representation of the term (e.g., "parent(alice,bob)").
        """
        if not string.endswith(')'):
            fatal(f"Syntax error in term: {string}")
        fields = string.split('(')
        if len(fields) != 2:
            fatal(f"Syntax error in term: {string}")
        self.predicate: str = fields[0]
        args_str = fields[1][:-1]  # Remove the closing parenthesis
        self.arguments: List[str] = [arg.strip() for arg in args_str.split(',')]

    def __repr__(self) -> str:
        return f"{self.predicate}({', '.join(self.arguments)})"


class Rule:
    """
    Represents a Prolog rule with a head and an optional body of goals.
    For example, grandparent(X, Y) :- parent(X, Z), parent(Z, Y).
    """

    def __init__(self, string: str) -> None:
        """
        Initializes a Rule object from its string representation.

        :param string: The string representation of the rule (e.g., "grandparent(X,Y):-parent(X,Z),parent(Z,Y).").
        """
        # Split the rule into head and body
        fields = string.split(":-")
        self.head: Term = Term(fields[0].strip())
        self.body: List[Term] = []

        if len(fields) == 2:
            # Remove trailing period if present and split body into individual goals
            body_str = fields[1].strip().rstrip('.')
            # Split goals by ',' while handling potential nested parentheses
            goals = re.split(r',\s*(?![^()]*\))', body_str)
            self.body = [Term(goal.strip()) for goal in goals]
        elif len(fields) > 2:
            fatal(f"Invalid rule format: {string}")

    def __repr__(self) -> str:
        if self.body:
            body_repr = ", ".join(str(goal) for goal in self.body)
            return f"{self.head} :- {body_repr}."
        else:
            return f"{self.head}."


class Goal:
    """
    Represents a goal in the search process, tracking the rule being evaluated,
    its parent goal, the current environment (variable bindings), and the
    index of the next subgoal to process.
    """

    def __init__(self, rule: Rule, parent: Optional['Goal'] = None, env: Optional[Dict[str, str]] = None) -> None:
        """
        Initializes a Goal object.

        :param rule: The Rule associated with this goal.
        :param parent: The parent Goal that spawned this goal, if any.
        :param env: The current environment (variable bindings) for this goal.
        """
        global goal_id_counter
        goal_id_counter += 1
        self.id: int = goal_id_counter
        self.rule: Rule = rule
        self.parent: Optional['Goal'] = parent
        self.env: Dict[str, str] = copy.deepcopy(env) if env else {}
        self.current_index: int = 0  # Index to track the current subgoal being processed

    def __repr__(self) -> str:
        return (f"Goal {self.id}: {self.rule} "
                f"Index={self.current_index} "
                f"Env={self.env}")


def unify(src_term: Term, src_env: Dict[str, str],
          dest_term: Term, dest_env: Dict[str, str]) -> bool:
    """
    Attempts to unify two terms under their respective environments.

    :param src_term: The source term to unify.
    :param src_env: The environment (variable bindings) of the source term.
    :param dest_term: The destination term to unify.
    :param dest_env: The environment (variable bindings) of the destination term.
    :return: True if unification succeeds, False otherwise.
    """
    # Check if predicates and number of arguments match
    if src_term.predicate != dest_term.predicate or len(src_term.arguments) != len(dest_term.arguments):
        return False

    for src_arg, dest_arg in zip(src_term.arguments, dest_term.arguments):
        # Determine if arguments are variables (start with uppercase letter)
        src_is_var = src_arg[0].isupper()
        dest_is_var = dest_arg[0].isupper()

        # Retrieve bindings if variables are already bound
        src_val = src_env.get(src_arg, src_arg) if src_is_var else src_arg
        dest_val = dest_env.get(dest_arg, dest_arg) if dest_is_var else dest_arg

        if src_is_var:
            if dest_is_var:
                if src_val != dest_val:
                    dest_env[dest_arg] = src_val  # Bind dest variable to src variable's value
            else:
                if src_val != dest_val:
                    dest_env[dest_arg] = src_val  # Bind dest variable to src constant
        else:
            if dest_is_var:
                if src_val != dest_val:
                    dest_env[dest_arg] = src_val  # Bind dest variable to src constant
            else:
                if src_val != dest_val:
                    return False  # Constants do not match

    return True


def search(query: Term) -> None:
    """
    Initiates a search for solutions to a given query based on the rules in the knowledge base.

    :param query: The query term to find solutions for.
    """
    global goal_id_counter
    goal_id_counter = 100  # Reset goal ID counter for each search

    if trace:
        print(f"Searching for: {query}")

    # Create an initial goal with the query
    initial_rule = Rule(f"{query}.")  # Treat the query as a rule with no body
    initial_goal = Goal(initial_rule, None, {})
    initial_goal.body = [query]  # Set the body to contain the query term
    stack: List[Goal] = [initial_goal]

    while stack:
        current_goal = stack.pop()
        if trace:
            print(f"Processing {current_goal}")

        # If all subgoals are processed, check if it's the original query
        if current_goal.current_index >= len(current_goal.body):
            if current_goal.parent is None:
                # Original query has been satisfied
                if current_goal.env:
                    print(f"Solution: {current_goal.env}")
                else:
                    print("Yes.")
                continue
            else:
                # Attempt to unify with the parent goal
                parent = current_goal.parent
                parent_goal = copy.deepcopy(parent)
                success = unify(current_goal.rule.head, current_goal.env,
                                parent.rule.body[parent.current_index], parent.env)
                if success:
                    parent.current_index += 1
                    stack.append(parent)
                continue

        # Get the next subgoal to process
        subgoal = current_goal.body[current_goal.current_index]
        if trace:
            print(f"Next subgoal: {subgoal}")

        # Iterate over all rules to find matches for the subgoal
        for rule in rules:
            if rule.head.predicate != subgoal.predicate or len(rule.head.arguments) != len(subgoal.arguments):
                continue  # Skip rules that don't match the predicate or arity

            # Attempt to unify the subgoal with the rule's head
            child_env = {}
            if unify(subgoal, current_goal.env, rule.head, child_env):
                # Create a new goal for the rule's body
                new_goal = Goal(rule, current_goal, child_env)
                stack.append(new_goal)
                if trace:
                    print(f"Added {new_goal} to the stack")

    print("Search completed.")


def main() -> None:
    """
    Sets up the knowledge base and performs sample queries.
    """
    global rules, trace

    # Sample knowledge base
    rules = [
        Rule("parent(alice, bob)."),            # Alice is Bob's parent
        Rule("parent(bob, charlie)."),          # Bob is Charlie's parent
        Rule("parent(alice, diana)."),          # Alice is Diana's parent
        Rule("parent(diana, elizabeth)."),      # Diana is Elizabeth's parent
        Rule("grandparent(X, Y) :- parent(X, Z), parent(Z, Y).")  # X is Y's grandparent if X is Z's parent and Z is Y's parent
    ]

    # Enable tracing for debugging purposes (set to True to see detailed steps)
    trace = False

    # Sample queries
    queries = [
        "parent(alice, bob)",        # Should return Yes.
        "parent(alice, charlie)",    # Should return No.
        "grandparent(alice, charlie)",    # Should return Yes.
        "grandparent(alice, elizabeth)",  # Should return Yes.
        "grandparent(bob, elizabeth)"     # Should return No.
    ]

    for query_str in queries:
        print(f"\nQuery: {query_str}?")
        query_term = Term(query_str)
        search(query_term)


if __name__ == "__main__":
    main()
