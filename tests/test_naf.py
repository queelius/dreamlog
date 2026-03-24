# tests/test_naf.py
import pytest
from dreamlog.factories import atom, var, compound
from dreamlog.knowledge import KnowledgeBase, Fact, Rule
from dreamlog.evaluator import PrologEvaluator, FlounderingError


class TestNAF:
    def _make_kb(self):
        kb = KnowledgeBase()
        kb.add_fact(compound("bird", atom("tweety")))
        kb.add_fact(compound("bird", atom("opus")))
        kb.add_fact(compound("penguin", atom("opus")))
        kb.add_rule(Rule(
            compound("flies", var("X")),
            [compound("bird", var("X")),
             compound("not", compound("penguin", var("X")))]))
        return kb

    def test_not_known_fact_fails(self):
        kb = self._make_kb()
        ev = PrologEvaluator(kb)
        sols = list(ev.query([compound("not", compound("bird", atom("tweety")))]))
        assert len(sols) == 0

    def test_not_unknown_fact_succeeds(self):
        kb = self._make_kb()
        ev = PrologEvaluator(kb)
        sols = list(ev.query([compound("not", compound("bird", atom("fido")))]))
        assert len(sols) == 1

    def test_not_derivable_from_rule_fails(self):
        kb = self._make_kb()
        ev = PrologEvaluator(kb)
        sols = list(ev.query([compound("not", compound("flies", atom("tweety")))]))
        assert len(sols) == 0

    def test_exception_clause_pattern(self):
        kb = self._make_kb()
        ev = PrologEvaluator(kb)
        assert len(list(ev.query([compound("flies", atom("tweety"))]))) == 1
        assert len(list(ev.query([compound("flies", atom("opus"))]))) == 0

    def test_double_negation(self):
        kb = self._make_kb()
        ev = PrologEvaluator(kb)
        sols = list(ev.query([
            compound("not", compound("not", compound("bird", atom("tweety"))))]))
        assert len(sols) == 1
        sols = list(ev.query([
            compound("not", compound("not", compound("bird", atom("fido"))))]))
        assert len(sols) == 0

    def test_floundering_error(self):
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        ev = PrologEvaluator(kb)
        with pytest.raises(FlounderingError):
            list(ev.query([compound("not", compound("a", var("X")))]))

    def test_naf_suppresses_unknown_hook(self):
        hook_called = []
        def hook(term, evaluator):
            hook_called.append(term)
        kb = KnowledgeBase()
        ev = PrologEvaluator(kb, unknown_hook=hook)
        sols = list(ev.query([compound("not", compound("undefined", atom("x")))]))
        assert len(sols) == 1
        assert len(hook_called) == 0


class TestHasSolution:
    def test_true(self):
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        ev = PrologEvaluator(kb)
        assert ev.has_solution(compound("a", atom("x"))) is True

    def test_false(self):
        kb = KnowledgeBase()
        ev = PrologEvaluator(kb)
        assert ev.has_solution(compound("a", atom("x"))) is False

    def test_via_rule(self):
        kb = KnowledgeBase()
        kb.add_fact(compound("b", atom("x")))
        kb.add_rule(Rule(compound("a", var("X")), [compound("b", var("X"))]))
        ev = PrologEvaluator(kb)
        assert ev.has_solution(compound("a", atom("x"))) is True
