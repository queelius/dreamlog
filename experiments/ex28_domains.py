"""Clean 3x2 domains for EX28: {within_predicate, recursive, cross_predicate}
x {canonical, invented}. The invented column invents the PREDICATE names, not
only the entities, so the only recoverable signal is structural/statistical."""
from dataclasses import dataclass
from typing import List, Tuple
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from dreamlog.factories import var, compound, atom
from dreamlog.knowledge import Rule
from dreamlog.recursive_discovery import transitive_closure

X, Y, Z = var("X"), var("Y"), var("Z")


@dataclass
class Domain:
    name: str
    rule_type: str   # within_predicate | recursive | cross_predicate
    vocab: str       # canonical | invented
    base: List[str]
    derived: List[str]
    target_rule: Rule
    new_base: List[str]
    new_checks: List[Tuple[str, bool, str]]


def _within_predicate(vocab, pred, guard, members, exceptions, new_members,
                      new_exceptions):
    """pred(X) holds for all `guard` members except `exceptions`."""
    base = [f"({guard} {m})" for m in members + exceptions]
    derived = [f"({pred} {m})" for m in members]            # exceptions excluded
    target = Rule(compound(pred, X), [compound(guard, X)])  # core generalization
    new_base = [f"({guard} {m})" for m in new_members + new_exceptions]
    checks = ([(f"({pred} {m})", True, f"{m} is {pred}") for m in new_members]
              + [(f"({pred} {m})", False, f"{m} not {pred}") for m in new_exceptions])
    return Domain(f"{pred}_{vocab}", "within_predicate", vocab, base, derived,
                  target, new_base, checks)


def within_canonical():
    return _within_predicate(
        "canonical", "can_fly", "bird",
        members=["robin", "sparrow", "eagle", "hawk", "finch", "wren"],
        exceptions=["penguin", "ostrich"],
        new_members=["raven", "dove"], new_exceptions=["kiwi"])


def within_invented():
    return _within_predicate(
        "invented", "glonk", "zorp",
        members=["mirv", "tup", "wex", "sline", "drof", "blay"],
        exceptions=["quib", "snar"],
        new_members=["yort", "plim"], new_exceptions=["vusk"])


def _recursive(vocab, base_pred, closure_pred, nodes, new_nodes, n_extra=2, seed=42):
    import random
    rng = random.Random(seed)
    order = nodes[:]
    edges = {(order[i], order[i + 1]) for i in range(len(order) - 1)}
    for _ in range(n_extra):
        i = rng.randrange(0, len(order) - 1); j = rng.randrange(i + 1, len(order))
        edges.add((order[i], order[j]))
    closure = transitive_closure(edges)
    base = [f"({base_pred} {a} {b})" for a, b in sorted(edges)]
    derived = [f"({closure_pred} {a} {b})" for a, b in sorted(closure)]
    target = Rule(compound(closure_pred, X, Z),
                  [compound(base_pred, X, Y), compound(closure_pred, Y, Z)])
    full_order = order + new_nodes
    new_edges = {(full_order[i], full_order[i + 1])
                 for i in range(len(order) - 1, len(full_order) - 1)}
    full_closure = transitive_closure(edges | new_edges)
    new_set = set(new_nodes)
    pos = sorted(p for p in full_closure if p[0] in new_set or p[1] in new_set)
    neg = sorted((a, b) for a in full_order for b in full_order
                 if a != b and (a in new_set or b in new_set)
                 and (a, b) not in full_closure)[:max(1, len(pos))]
    new_base = [f"({base_pred} {a} {b})" for a, b in sorted(new_edges)]
    checks = ([(f"({closure_pred} {a} {b})", True, f"{a}->{b}") for a, b in pos]
              + [(f"({closure_pred} {a} {b})", False, f"{a}-/->{b}") for a, b in neg])
    return Domain(f"{closure_pred}_{vocab}", "recursive", vocab, base, derived,
                  target, new_base, checks)


def recursive_canonical(seed=42):
    return _recursive("canonical", "parent", "ancestor",
                      ["al", "bo", "cy", "di", "ed", "fi", "gus", "hy"],
                      ["ike", "jo", "ko", "lu"], seed=seed)


def recursive_invented(seed=42):
    return _recursive("invented", "flux_links", "flux_reaches",
                      ["qux", "vor", "zane", "plix", "drub", "yent", "kosh", "wim"],
                      ["fren", "glor", "snee", "thock"], seed=seed)


def _cross_predicate(vocab, target_pred, rel_pred, prop_pred, pairs, props,
                     new_pairs, new_props, distractor_pairs, new_distractor_pairs):
    """target_pred(X,Y) :- rel_pred(X,Y), prop_pred(X). `props` are the X's that
    have the property; pairs not satisfying both are negatives."""
    base = ([f"({rel_pred} {a} {b})" for a, b in pairs + distractor_pairs]
            + [f"({prop_pred} {x})" for x in props])
    derived = [f"({target_pred} {a} {b})" for a, b in pairs if a in set(props)]
    target = Rule(compound(target_pred, X, Y),
                  [compound(rel_pred, X, Y), compound(prop_pred, X)])
    new_base = ([f"({rel_pred} {a} {b})" for a, b in new_pairs + new_distractor_pairs]
                + [f"({prop_pred} {x})" for x in new_props])
    pos = [(a, b) for a, b in new_pairs if a in set(new_props)]
    neg = [(a, b) for a, b in new_pairs + new_distractor_pairs
           if a not in set(new_props)]
    checks = ([(f"({target_pred} {a} {b})", True, f"{a},{b}") for a, b in pos]
              + [(f"({target_pred} {a} {b})", False, f"{a},{b}") for a, b in neg])
    return Domain(f"{target_pred}_{vocab}", "cross_predicate", vocab, base, derived,
                  target, new_base, checks)


def cross_canonical():
    return _cross_predicate(
        "canonical", "father", "parent", "male",
        pairs=[("tom", "ann"), ("tom", "ben"), ("jim", "cat"), ("jim", "dan")],
        props=["tom", "jim"],
        new_pairs=[("sam", "eve"), ("sam", "fox")], new_props=["sam"],
        distractor_pairs=[("liz", "ann"), ("mae", "ben")],   # mothers (not male)
        new_distractor_pairs=[("ivy", "eve")])


def cross_invented():
    # fully invented predicates: wibble(X,Y) :- frob(X,Y), quax(X)
    return _cross_predicate(
        "invented", "wibble", "frob", "quax",
        pairs=[("ond", "ulp"), ("ond", "esk"), ("arn", "ixt"), ("arn", "obo")],
        props=["ond", "arn"],
        new_pairs=[("zib", "ako"), ("zib", "eln")], new_props=["zib"],
        distractor_pairs=[("uvy", "ulp"), ("ywex", "esk")],
        new_distractor_pairs=[("plon", "ako")])


def all_domains(seed=42):
    return [within_canonical(), within_invented(),
            recursive_canonical(seed), recursive_invented(seed),
            cross_canonical(), cross_invented()]
