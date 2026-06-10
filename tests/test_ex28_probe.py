import importlib.util, pathlib, json

def _load(name):
    p = pathlib.Path(__file__).parent.parent / "experiments" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_probe_counts_structurally_equivalent_proposals():
    probe = _load("ex28_probe")
    domains = _load("ex28_domains")
    from tests.mock_provider import MockLLMProvider
    dom = domains.recursive_invented(seed=42)
    # mock proposes the correct recursive rule every time. Build it FROM the
    # domain's own target so a future vocabulary rename cannot desynchronize
    # this mock again (the tessik/vorlan rename broke the old hardcoded names).
    head = dom.target_rule.head
    base_pred = dom.target_rule.body[0].functor
    correct = json.dumps([
        ["rule", [head.functor, "X", "Z"],
         [[base_pred, "X", "Y"], [head.functor, "Y", "Z"]]],
    ])
    mock = MockLLMProvider(responses=[correct] * 5)
    result = probe.proposal_rate(dom, mock, n_runs=5)
    assert result["rate"] == 1.0
    assert result["hits"] == 5 and result["n"] == 5
    assert len(result["runs"]) == 5            # per-run metadata recorded
