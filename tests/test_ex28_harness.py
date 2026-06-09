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


def test_resume_tolerates_corrupt_trailing_line():
    """A store whose last line is truncated (mid-write kill) must not crash on resume.

    The corrupt line was never fsynced so its unit should simply be re-run.
    The harness must:
    - not raise when reading the store,
    - skip the corrupt line and still recognise the previously-completed key,
    - run (only) the new unit,
    - return parseable records from summarize() without raising.
    """
    h = _load("ex28_harness")
    with tempfile.TemporaryDirectory() as d:
        store = os.path.join(d, "results.jsonl")

        # Build a valid first record whose key we control.
        valid_unit = {"cell": "c1", "condition": "symbolic", "run": 0}
        valid_key = h.unit_key(valid_unit)
        valid_record = {"key": valid_key, "ts": 1000000.0, "git_sha": "deadbeef",
                        "cell": "c1", "condition": "symbolic", "run": 0,
                        "recovery": 1.0}

        # Write: one complete JSON line then a truncated/garbage line with no
        # closing brace, simulating a kill mid-write.
        with open(store, "w") as f:
            f.write(json.dumps(valid_record) + "\n")
            f.write('{"key": "abc12", "cell":')   # truncated - no closing brace

        # A new unit whose key differs from the valid record above.
        new_unit = {"cell": "c2", "condition": "llm", "run": 0}
        assert h.unit_key(new_unit) != valid_key

        calls = {"n": 0}
        def fake_unit(unit):
            calls["n"] += 1
            return {"recovery": 0.5}

        # Must not raise despite the corrupt trailing line.
        h.run_units([valid_unit, new_unit], fake_unit, store,
                    manifest_dir=d, git_sha="deadbeef")

        # Only the new unit was run; the valid first record was recognised as done.
        assert calls["n"] == 1

        # summarize must return the parseable records without raising.
        recs = h.summarize(store)
        keys_found = {r["key"] for r in recs}
        # The valid record from the pre-seeded store and the newly appended record
        # are both present; the corrupt line is absent.
        assert valid_key in keys_found
        assert h.unit_key(new_unit) in keys_found
