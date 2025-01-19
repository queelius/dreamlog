import pprint

kb = [
    # lower-case relations are facts, upper-case are rules

    # Facts
    (("male", ("bob",)), []),  # 'bob' is a constant representing a specific entity.
    (("female", ("alice",)), []),  # 'alice' is a constant.
    (("parent", ("alice", "bob")), []),  # Another fact, with 'alice' as the parent of 'bob'.
    (("loves", ("alice", "X")), []),  # 'X' is a variable, indicating that 'alice' loves everyone

    # Rule
    (("mother", ("X", "Y")), [("female", ("X",)), ("parent", ("X", "Y"))]),
    # 'X' and 'Y' are variables, indicating this rule can apply to any 'X' and 'Y' 
    # for which 'X' is female and 'X' is a parent of 'Y'.

    (("true", ()), []),    # true is always true, unconditional rule with no arguments

    # we could define "bob" is "male" with:
    #     (("male", "bob,")), [("true", ())],
    # but we don't need to, because we can just make "bob" an unconditional rule.

    # We only say things are true if we can prove them. If we have no proof,
    # we don't say it is false, we leave it unproven. This is a form of "negation as failure".
    # if p(X) <- q(X), then to prove p(X) we must prove q(X). If we can't prove q(X), we can't prove p(X)
    # unless there are other ways in which p(X) can be proven, e.g., if p(X) <- r(X), and we can prove r(X)
    # or there is an unconditional rule p(X) <- true.
]

def show():
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(kb)

def add(rule):
    # Add a rule to the knowledge base
    # a rule has the following form:
    # (("<name>", (<arg1>, <arg2>, ...)), [(<condition1>, (<arg1>, <arg2>, ...)), ...])
    # where <name> is the name of the relation, <arg1>, <arg2>, ... are the arguments
    # and <condition1>, <condition2>, ... are the conditions that must be met for the rule to apply
    #
    # An unconditional rule is a fact, and can be added as a rule with an empty list of conditions.
    # ((<name>, (<arg1>, <arg2>, ...)), [])
    kb.append(rule)

def query_fact(fact):
    # Check if fact is directly in KB
    relation, *args = fact
    if relation in kb["facts"] and args in kb["facts"][relation]:
        return True
    # If not, attempt to derive fact using rules
    for rule in kb["rules"]:
        if rule_can_produce_fact(rule, fact):
            if all(query_fact(cond) for cond in rule[1]):  # Check all conditions
                return True
    return False

def rule_can_produce_fact(rule, fact):
    # Simplified check if a rule can produce a fact
    conclusion, conditions = rule
    return conclusion == fact  # Simplified; actual implementation needs more


def help_command(topic=None):
    """
    Provides detailed help based on the topic.
    """
    if topic == "show":
        print("Show the knowledge base.")
    elif topic == "fact":
        print("Adding a fact:\n"
              "Use the syntax 'fact(relation, (arg1, arg2, ...))' to add a fact to the knowledge base.\n"
              "Example: 'fact(parent, (alice, bob)' declares that alice is a parent of bob.")
    elif topic == "rule":
        print("Adding a rule:\n"
              "Use the syntax 'rule((consequent), [(condition1), (condition2), ...])' to add a rule.\n"
              "Example: 'rule((grandparent(X, Z)), [(parent(X, Y)), (parent(Y, Z))])' declares that if X is a parent of Y and Y is a parent of Z, then X is a grandparent of Z.")
    elif topic == "query":
        print("Making a query:\n"
              "Use the syntax 'query(relation, (arg1, arg2, ...))' to query the knowledge base.\n"
              "Example: 'query(parent, (alice, bob))' asks if alice is a parent of bob, returning 'true' or 'false'.")
    else:
        print("Available commands:\n"
              "- fact: Add a fact to the knowledge base.\n"
              "- rule: Add a rule to the knowledge base.\n"
              "- query: Query the knowledge base.\n"
              "- show: Show the knowledge-base of facts and rules.\n"
              "Type 'help(command)' for more details on a specific command.")

def parse_input(user_input):
    if user_input.startswith("help"):
        topic = user_input[5:].strip("() ")
        return "help", topic
    elif user_input.startswith("fact") or user_input.startswith("query") or user_input.startswith("rule"):
        # Extract the command and the content within parentheses
        command = user_input.split("(")[0]
        content_str = user_input[len(command) + 1:-1]  # Extract string inside parentheses

        # Preparing content for eval by assuming all args are strings without explicit quotes
        content_str = content_str.replace(" ", "")  # Remove spaces to simplify
        args = content_str.split(",")
        args = [f"'{arg}'" for arg in args]  # Enclose each arg in quotes
        content = eval(f"({','.join(args)})", {"__builtins__": None}, {})  # Safely eval to tuple
        return command, content
    else:
        return user_input, None

def repl():
    print("Type 'help' for instructions on how to use this REPL.")
    while True:
        user_input = input("> ").strip()
        action, content = parse_input(user_input)

        print(f"{action=}, {content=}")
        if action == "exit":
            print("Exiting REPL.")
            break
        elif action == "things":
            show_things()
        elif action == "show":
            show_kb()
        elif action == "fact":
            add_fact(content)
            print(f"Added fact: {content}")
        elif action == "rule":
            add_rule(content)
            print(f"Added rule: {content}")
        elif action == "query":
            result = query_fact(content)
            print(f"Query result: {'true' if result else 'false'}")
        elif action == "help":
            help_command(content)
        else:
            print("Unrecognized input. Please try again.")

true_facts = set()  # We'll use a set to store the string representations for simplicity

def is_variable(term):
    return term[0].isupper()

def instantiate_rule(rule, kb_facts):
    """Attempt to instantiate variables in a rule based on known facts."""
    name, conditions = rule
    print(f" {name=} <= {conditions=}")
    for condition in conditions:
        condition_name, condition_args = condition
        print(f"  {condition=} :: {condition_name=}, {condition_args=}")
        print("  Iterating over KB")
        for fact in kb_facts:
            fact_name, fact_args = fact
            print(f"   KB: {fact=} :: {fact_name=}, {fact_args=}")
            if fact_name == condition_name:
                matches = True
                print(f"    {fact_name=} == {condition_name=}")
                for c_arg, f_arg in zip(condition_args, fact_args):
                    print(f"    {c_arg=}, {f_arg=}")
                    if is_variable(c_arg):
                        continue  # Variable matches anything
                    if c_arg != f_arg:
                        matches = False
                        break
                if matches:
                    # Instantiate the rule by replacing variables in the conclusion
                    yield fact


def deduce_new_facts(kb):
    kb_facts = [fact for fact, _ in kb if not _]  # Extract initial facts
    print("Initial facts:")
    print(kb_facts)
    new_facts = True
    i = 0
    while new_facts:
        i = i + 1
        if i == 100:
            break
        #new_facts = False
        for rule in kb:
            print(f"**Trying**: {rule=}")
            for instantiated_fact in instantiate_rule(rule, kb_facts):
                print(f"*Trying*: {instantiated_fact=}")
                if instantiated_fact not in kb_facts:
                    print(f"New fact deduced: {instantiated_fact}")
                    kb_facts.append(instantiated_fact)
                    new_facts = True
                else:
                    print(f"Fact already in KB: {instantiated_fact=}")

    print(f"{kb_facts=}")

deduce_new_facts(kb)
