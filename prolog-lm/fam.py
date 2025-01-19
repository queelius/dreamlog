from prolog2 import Rule, Term, search 

# Initialize the knowledge base with facts and a rule
rules = [
    Rule("parent(alice,bob)."),  # Fact: Alice is Bob's parent
    Rule("parent(bob,charlie)."),  # Fact: Bob is Charlie's parent
    Rule("grandparent(X,Y):-parent(X,Z),parent(Z,Y).")  # Rule for grandparent relationship
]

# Perform query
query = Term("grandparent(alice,charlie)")
search(query)
