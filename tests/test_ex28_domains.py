import importlib.util, pathlib

def _load(name):
    p = pathlib.Path(__file__).parent.parent / "experiments" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_all_six_cells_present():
    d = _load("ex28_domains")
    cells = d.all_domains(seed=42)
    keys = {(c.rule_type, c.vocab) for c in cells}
    assert keys == {
        ("within_predicate", "canonical"), ("within_predicate", "invented"),
        ("recursive", "canonical"), ("recursive", "invented"),
        ("cross_predicate", "canonical"), ("cross_predicate", "invented"),
    }


def test_each_domain_has_target_and_checks():
    d = _load("ex28_domains")
    for c in d.all_domains(seed=42):
        assert c.base and c.derived and c.new_base and c.new_checks
        assert c.target_rule is not None
        assert any(exp for _, exp, _ in c.new_checks)        # has positives
        assert any(not exp for _, exp, _ in c.new_checks)    # has negatives


def test_invented_vocab_has_no_real_predicates():
    d = _load("ex28_domains")
    invented = [c for c in d.all_domains(seed=42) if c.vocab == "invented"]
    # Real words AND removed leaky predicates / reachability-graph terms: the
    # invented cells must give a capable LLM no lexical or semantic foothold.
    real_words = {"parent", "ancestor", "father", "male", "bird", "can_fly",
                  "flux", "links", "reaches", "reach", "wibble", "frob", "quax",
                  "link", "edge", "path", "graph", "node", "child"}
    for c in invented:
        text = " ".join(c.base + c.derived)
        assert not any(w in text for w in real_words), f"{c.name} leaks real vocab"
