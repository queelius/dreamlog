


parent(alice,bob).
parent(bob,charlie).
grandparent(X,Y) :- parent(X,Z), parent(Z,Y).

