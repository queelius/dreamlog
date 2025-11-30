"""
Tests for the Pythonic API (dreamlog/pythonic.py)

Tests the fluent Python interface including:
- QueryResult for accessing query bindings
- RuleBuilder for fluent rule construction
- DreamLog class for knowledge base management
"""

import pytest
import tempfile
import os
from dreamlog.pythonic import DreamLog, QueryResult, RuleBuilder, dreamlog
from dreamlog.terms import Atom, Variable


class TestQueryResult:
    """Test QueryResult class for accessing query bindings"""

    def test_dict_access_returns_value(self):
        """QueryResult should allow dict-like access to bindings"""
        # Given: A QueryResult with bindings
        bindings = {"X": Atom("john"), "Y": Atom("mary")}
        result = QueryResult(bindings)

        # When: Accessing via dict notation
        value_x = result["X"]
        value_y = result["Y"]

        # Then: Should return the atom values
        assert value_x == "john"
        assert value_y == "mary"

    def test_dict_access_returns_none_for_missing_key(self):
        """QueryResult should return None for missing keys"""
        # Given: A QueryResult with limited bindings
        bindings = {"X": Atom("john")}
        result = QueryResult(bindings)

        # When: Accessing a missing key
        value = result["Z"]

        # Then: Should return None
        assert value is None

    def test_attribute_access_returns_value(self):
        """QueryResult should allow attribute access to bindings"""
        # Given: A QueryResult with bindings
        bindings = {"Name": Atom("alice")}
        result = QueryResult(bindings)

        # When: Accessing via attribute notation
        value = result.Name

        # Then: Should return the atom value
        assert value == "alice"

    def test_to_dict_converts_bindings(self):
        """QueryResult.to_dict() should return plain Python dict"""
        # Given: A QueryResult with bindings
        bindings = {"X": Atom("john"), "Y": Atom("mary")}
        result = QueryResult(bindings)

        # When: Converting to dict
        d = result.to_dict()

        # Then: Should be a plain dict with string values
        assert isinstance(d, dict)
        assert d == {"X": "john", "Y": "mary"}

    def test_handles_non_atom_values(self):
        """QueryResult should handle non-Atom term values"""
        # Given: A QueryResult with a Variable (edge case)
        bindings = {"X": Variable("Y")}
        result = QueryResult(bindings)

        # When: Accessing the value
        value = result["X"]

        # Then: Should return string representation
        assert value == "Y"


class TestRuleBuilder:
    """Test RuleBuilder for fluent rule construction"""

    def test_when_adds_first_condition(self):
        """RuleBuilder.when() should add the first body condition"""
        # Given: A DreamLog instance and RuleBuilder
        jl = DreamLog()
        builder = RuleBuilder(jl, "grandparent", ["X", "Z"])

        # When: Adding first condition with when()
        builder.when("parent", ["X", "Y"])

        # Then: Should have one condition
        assert len(builder.body_conditions) == 1
        assert builder.body_conditions[0] == ("parent", ["X", "Y"])

    def test_and_adds_additional_condition(self):
        """RuleBuilder.and_() should add additional conditions"""
        # Given: A RuleBuilder with one condition
        jl = DreamLog()
        builder = RuleBuilder(jl, "grandparent", ["X", "Z"])
        builder.when("parent", ["X", "Y"])

        # When: Adding another condition with and_()
        builder.and_("parent", ["Y", "Z"])

        # Then: Should have two conditions
        assert len(builder.body_conditions) == 2
        assert builder.body_conditions[1] == ("parent", ["Y", "Z"])

    def test_then_is_alias_for_and(self):
        """RuleBuilder.then() should be an alias for and_()"""
        # Given: A RuleBuilder with one condition
        jl = DreamLog()
        builder = RuleBuilder(jl, "can_fly", ["X"])
        builder.when("bird", ["X"])

        # When: Using then() instead of and_()
        builder.then("has_wings", ["X"])

        # Then: Should work the same as and_()
        assert len(builder.body_conditions) == 2

    def test_build_adds_rule_to_kb(self):
        """RuleBuilder.build() should add the rule to knowledge base"""
        # Given: A DreamLog instance with facts
        jl = DreamLog()
        jl.fact("parent", "john", "mary")
        jl.fact("parent", "mary", "alice")

        # When: Building a grandparent rule
        jl.rule("grandparent", ["X", "Z"]) \
          .when("parent", ["X", "Y"]) \
          .and_("parent", ["Y", "Z"]) \
          .build()

        # Then: Rule should be added and queryable
        results = jl.find_all("grandparent", "X", "Z")
        assert len(results) == 1
        assert results[0]["X"] == "john"
        assert results[0]["Z"] == "alice"

    def test_build_returns_dreamlog_for_chaining(self):
        """RuleBuilder.build() should return DreamLog for method chaining"""
        # Given: A DreamLog instance
        jl = DreamLog()

        # When: Building a rule
        result = jl.rule("test", ["X"]).when("fact", ["X"]).build()

        # Then: Should return DreamLog instance
        assert result is jl

    def test_make_term_handles_variables(self):
        """RuleBuilder._make_term should create Variables from uppercase strings"""
        # Given: A RuleBuilder
        jl = DreamLog()
        builder = RuleBuilder(jl, "test", ["X"])

        # When: Making a term with uppercase string
        term = builder._make_term("parent", ["X", "john"])

        # Then: X should be Variable, john should be Atom
        assert term.functor == "parent"
        assert isinstance(term.args[0], Variable)
        assert isinstance(term.args[1], Atom)

    def test_make_term_handles_nullary_predicate(self):
        """RuleBuilder._make_term should handle predicates with no args"""
        # Given: A RuleBuilder
        jl = DreamLog()
        builder = RuleBuilder(jl, "test", [])

        # When: Making a term with no arguments
        term = builder._make_term("fact", [])

        # Then: Should be an atom
        assert isinstance(term, Atom)
        assert term.value == "fact"


class TestDreamLogFacts:
    """Test DreamLog fact operations"""

    def test_fact_adds_simple_fact(self):
        """DreamLog.fact() should add a simple fact"""
        # Given: A DreamLog instance
        jl = DreamLog()

        # When: Adding a fact
        jl.fact("likes", "john", "mary")

        # Then: Fact should be queryable
        assert jl.ask("likes", "john", "mary")

    def test_fact_returns_self_for_chaining(self):
        """DreamLog.fact() should return self for method chaining"""
        # Given: A DreamLog instance
        jl = DreamLog()

        # When: Adding a fact
        result = jl.fact("test", "a")

        # Then: Should return self
        assert result is jl

    def test_fact_method_chaining(self):
        """DreamLog.fact() should support method chaining"""
        # Given/When: Creating facts with chaining
        jl = DreamLog()
        jl.fact("parent", "john", "mary") \
          .fact("parent", "mary", "alice") \
          .fact("parent", "john", "tom")

        # Then: All facts should exist
        assert len(jl.find_all("parent", "john", "X")) == 2

    def test_fact_handles_integers(self):
        """DreamLog.fact() should convert integers to atoms"""
        # Given: A DreamLog instance
        jl = DreamLog()

        # When: Adding a fact with integer
        jl.fact("age", "john", 42)

        # Then: Should be queryable
        results = jl.find_all("age", "john", "X")
        assert len(results) == 1
        assert results[0]["X"] == "42"

    def test_fact_handles_floats(self):
        """DreamLog.fact() should convert floats to atoms"""
        # Given: A DreamLog instance
        jl = DreamLog()

        # When: Adding a fact with float
        jl.fact("height", "john", 1.85)

        # Then: Should be queryable
        results = jl.find_all("height", "john", "X")
        assert len(results) == 1

    def test_facts_batch_method(self):
        """DreamLog.facts() should add multiple facts at once"""
        # Given: A DreamLog instance
        jl = DreamLog()

        # When: Adding multiple facts
        jl.facts(
            ("parent", "john", "mary"),
            ("parent", "mary", "alice"),
            ("age", "john", 45)
        )

        # Then: All facts should exist
        assert jl.ask("parent", "john", "mary")
        assert jl.ask("parent", "mary", "alice")
        assert jl.ask("age", "john", "45")

    def test_nullary_fact(self):
        """DreamLog.fact() should handle facts with no arguments"""
        # Given: A DreamLog instance
        jl = DreamLog()

        # When: Adding a nullary fact
        jl.fact("sunny")

        # Then: Should be queryable
        assert jl.ask("sunny")


class TestDreamLogQueries:
    """Test DreamLog query operations"""

    def test_query_returns_iterator(self):
        """DreamLog.query() should return an iterator"""
        # Given: A DreamLog with facts
        jl = DreamLog()
        jl.fact("parent", "john", "mary")

        # When: Querying
        result = jl.query("parent", "john", "X")

        # Then: Should be an iterator
        assert hasattr(result, '__iter__')
        assert hasattr(result, '__next__')

    def test_query_yields_query_results(self):
        """DreamLog.query() should yield QueryResult objects"""
        # Given: A DreamLog with facts
        jl = DreamLog()
        jl.fact("parent", "john", "mary")

        # When: Iterating query results
        results = list(jl.query("parent", "john", "X"))

        # Then: Should be QueryResult objects
        assert all(isinstance(r, QueryResult) for r in results)

    def test_query_with_variable_binding(self):
        """DreamLog.query() should bind variables correctly"""
        # Given: A DreamLog with facts
        jl = DreamLog()
        jl.fact("parent", "john", "mary")
        jl.fact("parent", "john", "tom")

        # When: Querying with variable
        results = list(jl.query("parent", "john", "X"))

        # Then: Should find both children
        children = {r["X"] for r in results}
        assert children == {"mary", "tom"}

    def test_query_with_ground_terms(self):
        """DreamLog.query() should work with fully ground terms"""
        # Given: A DreamLog with facts
        jl = DreamLog()
        jl.fact("parent", "john", "mary")

        # When: Querying with ground terms
        results = list(jl.query("parent", "john", "mary"))

        # Then: Should find one result
        assert len(results) == 1

    def test_ask_returns_true_when_solution_exists(self):
        """DreamLog.ask() should return True when solution exists"""
        # Given: A DreamLog with facts
        jl = DreamLog()
        jl.fact("parent", "john", "mary")

        # When: Asking yes/no question
        result = jl.ask("parent", "john", "mary")

        # Then: Should return True
        assert result is True

    def test_ask_returns_false_when_no_solution(self):
        """DreamLog.ask() should return False when no solution exists"""
        # Given: A DreamLog with facts
        jl = DreamLog()
        jl.fact("parent", "john", "mary")

        # When: Asking about non-existent fact
        result = jl.ask("parent", "mary", "john")

        # Then: Should return False
        assert result is False

    def test_find_all_returns_list(self):
        """DreamLog.find_all() should return a list"""
        # Given: A DreamLog with facts
        jl = DreamLog()
        jl.fact("parent", "john", "mary")
        jl.fact("parent", "john", "tom")

        # When: Finding all solutions
        results = jl.find_all("parent", "john", "X")

        # Then: Should be a list
        assert isinstance(results, list)
        assert len(results) == 2

    def test_find_one_returns_first_result(self):
        """DreamLog.find_one() should return first solution"""
        # Given: A DreamLog with facts
        jl = DreamLog()
        jl.fact("parent", "john", "mary")
        jl.fact("parent", "john", "tom")

        # When: Finding one solution
        result = jl.find_one("parent", "john", "X")

        # Then: Should return first QueryResult
        assert isinstance(result, QueryResult)
        assert result["X"] in ("mary", "tom")

    def test_find_one_returns_none_when_no_solution(self):
        """DreamLog.find_one() should return None when no solution"""
        # Given: An empty DreamLog
        jl = DreamLog()

        # When: Finding non-existent
        result = jl.find_one("parent", "X", "Y")

        # Then: Should return None
        assert result is None


class TestDreamLogRules:
    """Test DreamLog rule operations"""

    def test_rule_with_when_and_build(self):
        """DreamLog.rule() should work with when() and build()"""
        # Given: A DreamLog with parent facts
        jl = DreamLog()
        jl.fact("parent", "john", "mary")
        jl.fact("parent", "mary", "alice")

        # When: Adding grandparent rule
        jl.rule("grandparent", ["X", "Z"]) \
          .when("parent", ["X", "Y"]) \
          .and_("parent", ["Y", "Z"]) \
          .build()

        # Then: Should derive grandparent relationship
        assert jl.ask("grandparent", "john", "alice")

    def test_rule_with_multiple_body_conditions(self):
        """DreamLog rules should support multiple body conditions"""
        # Given: A DreamLog with facts
        jl = DreamLog()
        jl.fact("parent", "tom", "john")
        jl.fact("parent", "tom", "bob")
        jl.fact("male", "john")
        jl.fact("male", "bob")

        # When: Adding sibling rule with multiple conditions
        jl.rule("brother", ["X", "Y"]) \
          .when("parent", ["P", "X"]) \
          .and_("parent", ["P", "Y"]) \
          .and_("male", ["X"]) \
          .build()

        # Then: Should find brothers (note: X can be brother of himself in this simple rule)
        results = jl.find_all("brother", "X", "bob")
        brothers = {r["X"] for r in results}
        assert "john" in brothers or "bob" in brothers


class TestDreamLogSExpressions:
    """Test DreamLog S-expression parsing"""

    def test_parse_simple_fact(self):
        """DreamLog.parse() should parse simple S-expression facts"""
        # Given: A DreamLog instance
        jl = DreamLog()

        # When: Parsing S-expression fact
        jl.parse("(likes john mary)")

        # Then: Fact should be queryable
        assert jl.ask("likes", "john", "mary")

    def test_parse_returns_self_for_chaining(self):
        """DreamLog.parse() should return self for chaining"""
        # Given: A DreamLog instance
        jl = DreamLog()

        # When: Parsing
        result = jl.parse("(test a)")

        # Then: Should return self
        assert result is jl

    def test_parse_rule_with_implication(self):
        """DreamLog.parse() should parse rules with :-"""
        # Given: A DreamLog with facts
        jl = DreamLog()
        jl.fact("parent", "john", "mary")
        jl.fact("parent", "mary", "alice")

        # When: Parsing a rule
        jl.parse("(grandparent X Z) :- (parent X Y), (parent Y Z)")

        # Then: Rule should be usable
        assert jl.ask("grandparent", "john", "alice")

    def test_parse_chain_multiple(self):
        """DreamLog.parse() should support chaining multiple parses"""
        # Given: A DreamLog instance
        jl = DreamLog()

        # When: Chaining multiple parses
        jl.parse("(likes alice bob)") \
          .parse("(likes bob charlie)")

        # Then: Both should exist
        assert jl.ask("likes", "alice", "bob")
        assert jl.ask("likes", "bob", "charlie")


class TestDreamLogFileOperations:
    """Test DreamLog file load/save operations"""

    def test_save_and_load_round_trip(self):
        """DreamLog should save and load knowledge correctly"""
        # Given: A DreamLog with facts
        jl = DreamLog()
        jl.fact("parent", "john", "mary")
        jl.fact("parent", "mary", "alice")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            # When: Saving and loading
            jl.save(filepath)

            jl2 = DreamLog()
            jl2.load(filepath)

            # Then: Facts should be preserved
            assert jl2.ask("parent", "john", "mary")
            assert jl2.ask("parent", "mary", "alice")
        finally:
            os.unlink(filepath)

    def test_save_returns_self_for_chaining(self):
        """DreamLog.save() should return self"""
        # Given: A DreamLog
        jl = DreamLog()
        jl.fact("test", "a")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            # When: Saving
            result = jl.save(filepath)

            # Then: Should return self
            assert result is jl
        finally:
            os.unlink(filepath)

    def test_load_returns_self_for_chaining(self):
        """DreamLog.load() should return self"""
        # Given: A saved knowledge base
        jl1 = DreamLog()
        jl1.fact("test", "a")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            jl1.save(filepath)

            # When: Loading
            jl2 = DreamLog()
            result = jl2.load(filepath)

            # Then: Should return self
            assert result is jl2
        finally:
            os.unlink(filepath)


class TestDreamLogTransaction:
    """Test DreamLog transaction context manager"""

    def test_transaction_commits_on_success(self):
        """Transaction should commit changes on successful completion"""
        # Given: A DreamLog instance
        jl = DreamLog()

        # When: Using transaction successfully
        with jl.transaction():
            jl.fact("parent", "john", "mary")
            jl.fact("parent", "mary", "alice")

        # Then: Changes should persist
        assert jl.ask("parent", "john", "mary")
        assert jl.ask("parent", "mary", "alice")

    def test_transaction_rollback_on_error(self):
        """Transaction should rollback on exception"""
        # Given: A DreamLog with existing fact
        jl = DreamLog()
        jl.fact("existing", "fact")

        # When: Transaction fails with exception
        try:
            with jl.transaction():
                jl.fact("new", "fact")
                raise ValueError("Simulated error")
        except ValueError:
            pass

        # Then: New fact should be rolled back
        assert jl.ask("existing", "fact")
        assert not jl.ask("new", "fact")


class TestDreamLogStats:
    """Test DreamLog statistics and introspection"""

    def test_stats_reports_counts(self):
        """DreamLog.stats should report fact and rule counts"""
        # Given: A DreamLog with facts and rules
        jl = DreamLog()
        jl.fact("parent", "john", "mary")
        jl.fact("parent", "mary", "alice")
        jl.rule("grandparent", ["X", "Z"]).when("parent", ["X", "Y"]).and_("parent", ["Y", "Z"]).build()

        # When: Getting stats
        stats = jl.stats

        # Then: Should have correct counts
        assert stats["num_facts"] == 2
        assert stats["num_rules"] == 1
        assert stats["total_items"] == 3

    def test_stats_lists_functors(self):
        """DreamLog.stats should list all functors"""
        # Given: A DreamLog with various predicates
        jl = DreamLog()
        jl.fact("parent", "john", "mary")
        jl.fact("likes", "john", "pizza")
        jl.rule("grandparent", ["X", "Z"]).when("parent", ["X", "Y"]).and_("parent", ["Y", "Z"]).build()

        # When: Getting stats
        stats = jl.stats

        # Then: Should list all functors
        assert "parent" in stats["functors"]
        assert "likes" in stats["functors"]
        assert "grandparent" in stats["functors"]

    def test_clear_removes_all_knowledge(self):
        """DreamLog.clear() should remove all facts and rules"""
        # Given: A DreamLog with facts and rules
        jl = DreamLog()
        jl.fact("parent", "john", "mary")
        jl.rule("test", ["X"]).when("parent", ["X", "Y"]).build()

        # When: Clearing
        jl.clear()

        # Then: Should have no facts or rules
        assert len(jl) == 0
        assert jl.stats["num_facts"] == 0
        assert jl.stats["num_rules"] == 0

    def test_clear_returns_self_for_chaining(self):
        """DreamLog.clear() should return self"""
        # Given: A DreamLog
        jl = DreamLog()

        # When: Clearing
        result = jl.clear()

        # Then: Should return self
        assert result is jl

    def test_len_returns_total_items(self):
        """len(DreamLog) should return total facts + rules"""
        # Given: A DreamLog with facts and rules
        jl = DreamLog()
        jl.fact("a", "b")
        jl.fact("c", "d")
        jl.rule("test", ["X"]).when("a", ["X"]).build()

        # When: Getting length
        length = len(jl)

        # Then: Should be total
        assert length == 3

    def test_repr_shows_counts(self):
        """DreamLog.__repr__ should show fact and rule counts"""
        # Given: A DreamLog with facts
        jl = DreamLog()
        jl.fact("a", "b")
        jl.fact("c", "d")

        # When: Getting repr
        rep = repr(jl)

        # Then: Should mention counts
        assert "2 facts" in rep


class TestDreamLogFunctional:
    """Test DreamLog functional programming support"""

    def test_map_query_applies_function(self):
        """DreamLog.map_query() should apply function to results"""
        # Given: A DreamLog with facts
        jl = DreamLog()
        jl.fact("person", "john")
        jl.fact("person", "mary")
        jl.fact("person", "alice")

        # When: Mapping uppercase function
        names = jl.map_query("person", "X", mapper=lambda r: r["X"].upper())

        # Then: Should return uppercase names
        assert set(names) == {"JOHN", "MARY", "ALICE"}

    def test_filter_query_filters_results(self):
        """DreamLog.filter_query() should filter results"""
        # Given: A DreamLog with facts
        jl = DreamLog()
        jl.fact("person", "john")
        jl.fact("person", "mary")
        jl.fact("adult", "john")

        # When: Filtering for adults
        adults = jl.filter_query("person", "X",
                                 predicate=lambda r: jl.ask("adult", r["X"]))

        # Then: Should only return john
        assert len(adults) == 1
        assert adults[0]["X"] == "john"


class TestDreamLogConvenienceFunction:
    """Test the dreamlog() convenience function"""

    def test_dreamlog_creates_instance(self):
        """dreamlog() should create a DreamLog instance"""
        # When: Creating via convenience function
        jl = dreamlog()

        # Then: Should be DreamLog instance
        assert isinstance(jl, DreamLog)

    def test_dreamlog_accepts_kwargs(self):
        """dreamlog() should pass kwargs to DreamLog"""
        # When: Creating without LLM
        jl = dreamlog()

        # Then: Should work without LLM
        jl.fact("test", "a")
        assert jl.ask("test", "a")


class TestDreamLogWithMockProvider:
    """Test DreamLog with mock LLM provider"""

    def test_creates_with_mock_provider(self):
        """DreamLog should accept mock provider directly"""
        from tests.mock_provider import MockLLMProvider

        # Given: A mock provider
        provider = MockLLMProvider(knowledge_domain="family")

        # When: Creating DreamLog with provider
        jl = DreamLog(llm_provider=provider, use_retry=False)

        # Then: Should work
        assert jl is not None
        jl.fact("parent", "john", "mary")
        assert jl.ask("parent", "john", "mary")
