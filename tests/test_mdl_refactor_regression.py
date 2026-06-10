"""AC3 regression: symbolic experiment cells must reproduce committed artifacts.

These cells are deterministic (no LLM). They exercise Operations A, B, C, I and
the full dream() pipeline end to end through the real experiment code paths
(ex25b.run_domain_test and ex28.run_one_unit), so any behavior drift introduced
by the MDL-unified-gate refactor shows up as an exact-value mismatch here.
"""
import importlib.util
import json
import pathlib
import sys

import pytest

EXP = pathlib.Path(__file__).parent.parent / "experiments"
sys.path.insert(0, str(EXP))


def _load(name):
    p = EXP / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.slow
def test_ex25b_crafting_symbolic_reproduces_artifact():
    art = json.loads((EXP / "data" / "ex25b" / "results.json").read_text())
    # n_runs=1 returns per-run flat dicts; the artifact was stored with n_runs=5
    # so want lives under domains.crafting.symbolic at the mean level.
    want = art["domains"]["crafting"]["symbolic"]
    ex25b = _load("ex25b_novel_generalization")
    # llm_client=None -> only no_dream and symbolic conditions run (full/raw_llm
    # are skipped entirely). n_runs=1 returns flat result dicts (no "runs" list).
    results = ex25b.run_domain_test(
        "regression-crafting",
        ex25b.crafting_base_facts(),
        ex25b.crafting_derived_facts(),
        ex25b.crafting_negatives(),
        ex25b.NEW_CRAFTING_BASE,
        ex25b.NEW_CRAFTING_CHECKS,
        None,            # no LLM client: symbolic + no_dream conditions only
        n_runs=1,
    )
    got = results["symbolic"]
    assert got["recall"] == pytest.approx(want["recall"], abs=1e-12)
    assert got["precision"] == pytest.approx(want["precision"], abs=1e-12)
    assert got["accuracy"] == pytest.approx(want["accuracy"], abs=1e-12)
    assert got["rules"] == pytest.approx(want["rules"])
    assert got["compression"] == pytest.approx(want["compression"], abs=1e-12)


@pytest.mark.slow
def test_ex28_symbolic_column_reproduces_artifact():
    rows = {}
    art_path = EXP / "data" / "ex28_sonnet" / "results.jsonl"
    for line in art_path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["condition"] == "symbolic_only":
                rows[r["cell"]] = r
    assert len(rows) == 6, f"expected 6 symbolic cells, got {sorted(rows)}"

    ex28 = _load("ex28_llm_role")
    # ex28_llm_role re-exports all_domains ("from ex28_domains import all_domains")
    doms = {d.name: d for d in ex28.all_domains(seed=42)}
    assert set(doms) == set(rows), "domain names drifted from artifact cells"

    for cell, want in rows.items():
        # run_one_unit signature: (unit, domains_by_name, client, n_probe)
        # client=None is safe for symbolic_only: use_llm=False, client never touched
        # return key is "recovery" (alias for recall), not "recall"
        got = ex28.run_one_unit(
            {"cell": cell, "condition": "symbolic_only", "run": 0},
            doms, client=None, n_probe=0)
        for key in ("recovery", "precision"):
            assert got[key] == pytest.approx(want[key], abs=1e-12), (cell, key)
        for key in ("tp", "tn", "fp", "fn"):
            assert got[key] == want[key], (cell, key)
