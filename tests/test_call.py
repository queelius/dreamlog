# tests/test_call.py
import pytest
from dreamlog.factories import atom, var, compound
from dreamlog.knowledge import KnowledgeBase, Fact, Rule
from dreamlog.evaluator import PrologEvaluator, InstantiationError


class TestCallN:
    def test_basic_call(self):
        kb = KnowledgeBase()
        kb.add_fact(compound("parent", atom("john"), atom("mary")))
        ev = PrologEvaluator(kb)
        sols = list(ev.query([compound("call", atom("parent"), atom("john"), atom("mary"))]))
        assert len(sols) == 1

    def test_call_with_variable(self):
        kb = KnowledgeBase()
        kb.add_fact(compound("parent", atom("john"), atom("mary")))
        ev = PrologEvaluator(kb)
        sols = list(ev.query([compound("call", atom("parent"), var("X"), atom("mary"))]))
        assert len(sols) == 1
        assert sols[0].bindings.get("X") == atom("john")

    def test_call_functor_bound_via_unification(self):
        kb = KnowledgeBase()
        kb.add_fact(compound("parent", atom("john"), atom("mary")))
        kb.add_fact(compound("rel", atom("parent")))
        kb.add_rule(Rule(
            compound("test_rel", var("X"), var("Y")),
            [compound("rel", var("F")),
             compound("call", var("F"), var("X"), var("Y"))]))
        ev = PrologEvaluator(kb)
        sols = list(ev.query([compound("test_rel", var("X"), var("Y"))]))
        assert len(sols) == 1

    def test_call_unbound_functor_raises(self):
        kb = KnowledgeBase()
        ev = PrologEvaluator(kb)
        with pytest.raises(InstantiationError):
            list(ev.query([compound("call", var("F"), atom("x"))]))

    def test_call_compound_functor_raises(self):
        kb = KnowledgeBase()
        ev = PrologEvaluator(kb)
        with pytest.raises(InstantiationError):
            list(ev.query([compound("call", compound("f", atom("a")), atom("x"))]))

    def test_call_arity_1_raises(self):
        kb = KnowledgeBase()
        ev = PrologEvaluator(kb)
        with pytest.raises(InstantiationError):
            list(ev.query([compound("call", atom("foo"))]))

    def test_call_in_rule_body(self):
        kb = KnowledgeBase()
        kb.add_fact(compound("parent", atom("john"), atom("mary")))
        kb.add_fact(compound("parent", atom("mary"), atom("alice")))
        kb.add_rule(Rule(
            compound("closure", var("R"), var("X"), var("Y")),
            [compound("call", var("R"), var("X"), var("Y"))]))
        kb.add_rule(Rule(
            compound("closure", var("R"), var("X"), var("Z")),
            [compound("call", var("R"), var("X"), var("Y")),
             compound("closure", var("R"), var("Y"), var("Z"))]))
        ev = PrologEvaluator(kb)
        sols = list(ev.query([compound("closure", atom("parent"), atom("john"), atom("mary"))]))
        assert len(sols) >= 1
        sols = list(ev.query([compound("closure", atom("parent"), atom("john"), atom("alice"))]))
        assert len(sols) >= 1

    def test_call_triggers_unknown_hook(self):
        hook_called = []
        def hook(term, evaluator):
            hook_called.append(str(term))
        kb = KnowledgeBase()
        ev = PrologEvaluator(kb, unknown_hook=hook)
        list(ev.query([compound("call", atom("undefined"), atom("x"))]))
        assert len(hook_called) > 0

    def test_call_inside_not(self):
        kb = KnowledgeBase()
        kb.add_fact(compound("bird", atom("tweety")))
        ev = PrologEvaluator(kb)
        sols = list(ev.query([compound("not", compound("call", atom("bird"), atom("fido")))]))
        assert len(sols) == 1
        sols = list(ev.query([compound("not", compound("call", atom("bird"), atom("tweety")))]))
        assert len(sols) == 0
