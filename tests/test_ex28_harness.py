import importlib.util, pathlib, json, tempfile, os

def _load(name):
    p = pathlib.Path(__file__).parent.parent / "experiments" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_completed_units_are_skipped_on_resume():
    h = _load("ex28_harness")
    with tempfile.TemporaryDirectory() as d:
        store = os.path.join(d, "results.jsonl")
        calls = {"n": 0}
        def fake_unit(unit):
            calls["n"] += 1
            return {"recovery": 1.0}
        units = [{"cell": "c1", "condition": "symbolic", "run": 0},
                 {"cell": "c1", "condition": "symbolic", "run": 1}]
        h.run_units(units, fake_unit, store, manifest_dir=d, git_sha="abc")
        assert calls["n"] == 2
        # second run: everything is done, fake_unit must not be called again
        h.run_units(units, fake_unit, store, manifest_dir=d, git_sha="abc")
        assert calls["n"] == 2
        # results.jsonl has exactly 2 records, each with metadata
        recs = [json.loads(l) for l in open(store)]
        assert len(recs) == 2
        assert all("git_sha" in r and "ts" in r and "key" in r for r in recs)
