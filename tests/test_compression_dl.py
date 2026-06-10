from dreamlog.compression.proposal import Proposal
from dreamlog.compression import dl
from dreamlog.factories import atom, compound
from dreamlog.knowledge import Fact, Rule, KnowledgeBase


def _fact(name, *args):
    return Fact(compound(name, *[atom(a) for a in args]))


def test_description_length_is_clause_count():
    kb = KnowledgeBase()
    kb.add_fact(_fact("p", "a"))
    kb.add_fact(_fact("q", "b"))
    kb.add_rule(Rule(compound("r", atom("x")), [compound("p", atom("x"))]))
    assert dl.description_length(kb) == len(kb) == 3
    assert dl.clause_cost(_fact("p", "a")) == 1


def test_proposal_delta_and_immutability():
    p = Proposal(kind="pruning", remove=(_fact("p", "a"), _fact("p", "b")),
                 add=(), notes={"detector": "derivable"})
    assert dl.proposal_delta(p) == -2
    q = Proposal(kind="llm_compression", remove=(),
                 add=(Rule(compound("r", atom("x")), [compound("p", atom("x"))]),))
    assert dl.proposal_delta(q) == 1
    import dataclasses
    assert dataclasses.is_dataclass(p) and p.__dataclass_params__.frozen
