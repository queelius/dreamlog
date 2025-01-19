Basic factual relations:
q(jack).
q(oliver).
q(ali).
q(james).
q(simon).
q(harry).
q(beef).
q(potato).
q(fred).
ww(helen).
ww(sophie).
ww(jess).
ww(lily).
ww(sarah).
z(fred,sarah).
z(sarah,beef).
z(beef,jack).
z(jack,jess).
z(jack,lily).
z(helen, jess).
z(helen, lily).
z(oliver,james).
z(sophie, james).
z(jess, simon).
z(ali, simon).
z(lily, harry).
z(james, harry).
 
Composed relations:
g(x,Y) if q(X) and z(X,Z) and z(Z,Y).
k(x,Y) if ww(X) and z(X,Z) and z(Z,Y).
tt(X,Y) if z(X,Y) or (z(X,Z) and tt(Z,Y)).

is sarah related to jess by relation k?