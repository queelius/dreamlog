"""P3 decision diff: score every gate decision under BOTH DL encodings.

Replays the EX25b crafting symbolic cell, the six EX28 symbolic domains, and
the eight benchmark scenarios, running each dream TWICE (dl_mode="clauses" =
live behavior; dl_mode="bits") with the decision recorder attached, and
writes a flip report for review before any default change.

SCOPE LIMITATION: this tool passes NO llm_client, so Operation G (LLM-assisted
compression) NEVER fires. Only symbolic gate decisions are recorded: kinds that
appear here are from Operations A-F and I only. The bits vs clauses comparison
covers exactly the symbolic path; G's priced criterion (Task 4) is exercised
only by the unit tests in tests/test_dl_bits.py.

Usage:  python experiments/dl_decision_diff.py
Writes: experiments/data/p3_decision_diff/{report.md, decisions.jsonl}
"""
import importlib.util
import json
import pathlib
import subprocess
import sys
import time
import datetime

HERE = pathlib.Path(__file__).parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from dreamlog.kb_dreamer import KnowledgeBaseDreamer  # noqa: E402


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------

def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------------
# Bench scenario thunk adapter
# ---------------------------------------------------------------------------

def _bench_kb_thunk(builder):
    """Return a zero-arg callable that yields a fresh KnowledgeBase per call.

    A bench scenario builder is a zero-arg function that returns a 3-tuple
    (name: str, kb: KnowledgeBase, checks: list).  We extract the KB (index 1)
    each time the thunk is called so both DL-mode runs get independent copies.
    """
    def thunk():
        _name, kb, _checks = builder()
        return kb
    return thunk


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

def _scenarios():
    """Yield (name, kb_builder) pairs; kb_builder() -> fresh KnowledgeBase.

    Covers:
    - EX25b crafting symbolic cell (1 scenario)
    - EX28 six symbolic domains (6 scenarios)
    - Bench scenarios (8 scenarios)
    Total: 15 scenarios x 2 DL modes = 30 dream runs.
    """
    # -- EX25b crafting --
    ex25b = _load(HERE / "ex25b_novel_generalization.py", "ex25b")
    ex25 = _load(HERE / "ex25_generalization.py", "ex25")

    # crafting_derived_facts() -> Dict[str, List[str]]
    # Flatten all derived fact strings across predicate groups.
    derived_all = [f for fs in ex25b.crafting_derived_facts().values() for f in fs]
    base_all = ex25b.crafting_base_facts()
    combined = base_all + derived_all

    yield ("ex25b_crafting",
           lambda: ex25.build_kb(combined))

    # -- EX28 six symbolic domains --
    # Load ex28_domains directly (ground truth per regression test pattern).
    ex28d = _load(HERE / "ex28_domains.py", "ex28_domains")
    for dom in ex28d.all_domains(seed=42):
        # discover_recursion is needed for recursive rule types.
        # Capture dom in default arg to avoid late-binding closure bug.
        yield (
            f"ex28_{dom.name}",
            lambda d=dom: ex25.build_kb(d.base + d.derived),
        )

    # -- Bench scenarios (8) --
    bench = _load(REPO_ROOT / "benchmarks" / "sleep_cycle_bench.py", "bench")
    for sname, builder in bench.SCENARIOS.items():
        yield (f"bench_{sname}", _bench_kb_thunk(builder))


# ---------------------------------------------------------------------------
# Single-scenario runner
# ---------------------------------------------------------------------------

def run_scenario(name, kb_builder, dl_mode, discover_recursion):
    """Dream one fresh KB under the given mode; return (records, session)."""
    records = []
    kb = kb_builder()
    dreamer = KnowledgeBaseDreamer(
        dl_mode=dl_mode,
        decision_recorder=records.append,
        discover_recursion=discover_recursion,
    )
    session = dreamer.dream(kb)
    for r in records:
        r.update({"scenario": name, "dl_mode": dl_mode})
    return records, session


# ---------------------------------------------------------------------------
# Flip table
# ---------------------------------------------------------------------------

def _flip_table(rec_clauses, rec_bits):
    """Compare two parallel decision lists (same scenario, different DL modes).

    Match records by key = (kind, tuple(removed), tuple(added)).
    A FLIP is a key present in both runs but with different 'decision' values.
    CLAUSES_ONLY and BITS_ONLY rows are cascade effects (one mode fired a
    pass that changed the KB in a way the other did not).

    Returns a list of dicts with fields:
      key, kind, delta_clauses, delta_bits,
      decision_clauses, decision_bits, flipped (bool),
      note ('both'|'clauses_only'|'bits_only')
    """
    def _key(r):
        return (r["kind"], tuple(r["removed"]), tuple(r["added"]))

    clauses_map = {}
    for r in rec_clauses:
        k = _key(r)
        clauses_map.setdefault(k, []).append(r)

    bits_map = {}
    for r in rec_bits:
        k = _key(r)
        bits_map.setdefault(k, []).append(r)

    all_keys = list(dict.fromkeys(
        [_key(r) for r in rec_clauses] + [_key(r) for r in rec_bits]
    ))

    rows = []
    for key in all_keys:
        c_list = clauses_map.get(key, [])
        b_list = bits_map.get(key, [])

        if c_list and b_list:
            c = c_list[0]
            b = b_list[0]
            flipped = c["decision"] != b["decision"]
            rows.append({
                "key": key,
                "kind": c["kind"],
                "delta_clauses": c["delta_clauses"],
                "delta_bits": b["delta_bits"],
                "decision_clauses": c["decision"],
                "decision_bits": b["decision"],
                "flipped": flipped,
                "note": "both",
            })
        elif c_list:
            c = c_list[0]
            rows.append({
                "key": key,
                "kind": c["kind"],
                "delta_clauses": c["delta_clauses"],
                "delta_bits": c["delta_bits"],
                "decision_clauses": c["decision"],
                "decision_bits": "-",
                "flipped": False,
                "note": "clauses_only",
            })
        else:
            b = b_list[0]
            rows.append({
                "key": key,
                "kind": b["kind"],
                "delta_clauses": b["delta_clauses"],
                "delta_bits": b["delta_bits"],
                "decision_clauses": "-",
                "decision_bits": b["decision"],
                "flipped": False,
                "note": "bits_only",
            })

    # Sort: flipped first, then by kind
    rows.sort(key=lambda r: (0 if r["flipped"] else 1, r["kind"]))
    return rows


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _write_report(path, summary, all_records, git_sha):
    """Write a deterministic Markdown report (apart from ts and git_sha).

    `summary` is a list of tuples:
      (name, n_rec_clauses, n_rec_bits, flip_rows, ratio_clauses, ratio_bits)
    where flip_rows is the output of _flip_table.
    """
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    # Aggregate totals
    total_flips_by_kind: dict = {}
    total_clauses_only = 0
    total_bits_only = 0
    total_flips = 0

    lines = [
        "# P3 Decision Diff: clauses vs bits DL modes",
        "",
        f"Git SHA: `{git_sha}` | Generated: {ts}",
        "",
        "## Scope",
        "",
        "This report covers symbolic Operations A-F and I only. Operation G",
        "(LLM-assisted compression) is EXCLUDED: no llm_client is passed to any",
        "dreamer in this tool, so G never fires. The bits vs clauses comparison",
        "is therefore complete only for the symbolic pipeline. G's priced",
        "criterion (Task 4) is covered by tests/test_dl_bits.py unit tests.",
        "",
        "---",
        "",
        "## Per-Scenario Decision Tables",
        "",
    ]

    for (name, n_c, n_b, flip_rows, ratio_c, ratio_b) in summary:
        n_flips = sum(1 for r in flip_rows if r["flipped"])
        n_c_only = sum(1 for r in flip_rows if r["note"] == "clauses_only")
        n_b_only = sum(1 for r in flip_rows if r["note"] == "bits_only")

        total_flips += n_flips
        total_clauses_only += n_c_only
        total_bits_only += n_b_only

        for r in flip_rows:
            if r["flipped"]:
                total_flips_by_kind[r["kind"]] = (
                    total_flips_by_kind.get(r["kind"], 0) + 1
                )

        ratio_c_str = f"{ratio_c:.3f}" if ratio_c is not None else "n/a"
        ratio_b_str = f"{ratio_b:.3f}" if ratio_b is not None else "n/a"
        lines += [
            f"### {name}",
            "",
            f"Decisions: {n_c} (clauses) / {n_b} (bits) | "
            f"Compression ratio: {ratio_c_str} (clauses) / {ratio_b_str} (bits) | "
            f"Flips: {n_flips} | Cascade: {n_c_only} clauses-only, "
            f"{n_b_only} bits-only",
            "",
        ]

        if not flip_rows:
            lines.append("*(no decisions recorded)*")
            lines.append("")
        else:
            lines.append(
                "| kind | delta_clauses | delta_bits | "
                "decision_clauses | decision_bits | FLIPPED | note |"
            )
            lines.append(
                "|------|--------------|------------|"
                "-----------------|---------------|---------|------|"
            )
            for r in flip_rows:
                flipped_str = "YES" if r["flipped"] else ""
                dc = r["delta_clauses"]
                db = r["delta_bits"]
                dc_str = f"{dc:+d}" if isinstance(dc, int) else str(dc)
                db_str = (
                    f"{db:+.2f}" if isinstance(db, float) else str(db)
                )
                lines.append(
                    f"| {r['kind']} | {dc_str} | {db_str} | "
                    f"{r['decision_clauses']} | {r['decision_bits']} | "
                    f"{flipped_str} | {r['note']} |"
                )
            lines.append("")

    lines += [
        "---",
        "",
        "## Totals",
        "",
        f"Total scenarios: {len(summary)}",
        f"Total flips (same key, different decision): {total_flips}",
        f"Total clauses-only cascade rows: {total_clauses_only}",
        f"Total bits-only cascade rows: {total_bits_only}",
        "",
        "### Flips by operation kind",
        "",
    ]

    if total_flips_by_kind:
        lines.append("| kind | flip count |")
        lines.append("|------|-----------|")
        for kind, cnt in sorted(total_flips_by_kind.items(),
                                key=lambda x: -x[1]):
            lines.append(f"| {kind} | {cnt} |")
    else:
        lines.append("*(no flips across any scenario)*")
    lines.append("")

    lines += [
        "### Compression ratios per mode per scenario",
        "",
        "| scenario | ratio_clauses | ratio_bits |",
        "|----------|--------------|-----------|",
    ]
    for (name, _nc, _nb, _fr, ratio_c, ratio_b) in summary:
        rc = f"{ratio_c:.3f}" if ratio_c is not None else "n/a"
        rb = f"{ratio_b:.3f}" if ratio_b is not None else "n/a"
        lines.append(f"| {name} | {rc} | {rb} |")
    lines.append("")

    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _discover_recursion_for(name, dom=None):
    """Heuristic: enable Op I for scenarios that involve recursive predicates.

    For EX28 domains we can inspect the rule_type attribute directly.
    For bench scenarios we use the scenario name.
    The bench transitive_closures scenario already has the recursive rules
    PRESENT as KB rules (built in), so Op I would be redundant; do NOT enable
    it there (Op I would try to re-discover and get rejected by subsumption).
    """
    if dom is not None:
        # EX28 domain object: use the authoritative rule_type field.
        return dom.rule_type == "recursive"
    # Name-based heuristic for bench and ex25b.
    # Recursive bench scenarios: none of the 8 currently require Op I.
    # EX25b crafting: no recursive predicates.
    lower = name.lower()
    return "tessik" in lower or "ancestor_canonical" in lower or "ancestor_invented" in lower


def main():
    out_dir = HERE / "data" / "p3_decision_diff"
    out_dir.mkdir(parents=True, exist_ok=True)

    gp = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
        cwd=str(REPO_ROOT),
    )
    git_sha = (gp.stdout.strip() if gp.returncode == 0 else "") or "unknown"

    all_records = []
    summary = []

    # Load the ex28 domains map once so we can pass dom objects to the heuristic.
    ex28d = _load(HERE / "ex28_domains.py", "ex28_domains")
    dom_by_name = {d.name: d for d in ex28d.all_domains(seed=42)}

    scenarios_list = list(_scenarios())
    print(f"Running {len(scenarios_list)} scenarios x 2 DL modes "
          f"= {len(scenarios_list) * 2} dream runs ...")

    for idx, (name, builder) in enumerate(scenarios_list, 1):
        # Determine whether Op I (recursive discovery) should be enabled.
        # For ex28_* scenarios, use the domain object; else use name heuristic.
        dom = None
        if name.startswith("ex28_"):
            dom_key = name[len("ex28_"):]
            dom = dom_by_name.get(dom_key)
        discover = _discover_recursion_for(name, dom=dom)

        print(f"  [{idx}/{len(scenarios_list)}] {name}  "
              f"(discover_recursion={discover})", end="", flush=True)
        t0 = time.perf_counter()

        try:
            rec_c, sess_c = run_scenario(name, builder, "clauses", discover)
        except Exception as exc:
            print(f" ERROR in clauses mode: {exc}")
            summary.append((name, 0, 0, [], None, None))
            all_records.append({
                "scenario": name, "dl_mode": "clauses", "kind": "ERROR",
                "removed": [], "added": [], "delta_clauses": 0, "delta_bits": 0,
                "decision": "error", "reason": str(exc),
            })
            continue

        try:
            rec_b, sess_b = run_scenario(name, builder, "bits", discover)
        except Exception as exc:
            print(f" ERROR in bits mode: {exc}")
            summary.append((name, len(rec_c), 0,
                            _flip_table(rec_c, []),
                            getattr(sess_c, "compression_ratio", None), None))
            all_records.extend(rec_c)
            all_records.append({
                "scenario": name, "dl_mode": "bits", "kind": "ERROR",
                "removed": [], "added": [], "delta_clauses": 0, "delta_bits": 0,
                "decision": "error", "reason": str(exc),
            })
            continue

        elapsed = (time.perf_counter() - t0) * 1000
        print(f" {elapsed:.0f}ms  clauses:{len(rec_c)} bits:{len(rec_b)}")

        all_records.extend(rec_c + rec_b)
        flip_rows = _flip_table(rec_c, rec_b)
        n_flips = sum(1 for r in flip_rows if r["flipped"])
        summary.append((
            name,
            len(rec_c),
            len(rec_b),
            flip_rows,
            getattr(sess_c, "compression_ratio", None),
            getattr(sess_b, "compression_ratio", None),
        ))
        if n_flips:
            print(f"    *** {n_flips} FLIP(s) ***")

    # Write JSONL
    jsonl_path = out_dir / "decisions.jsonl"
    with open(jsonl_path, "w") as f:
        for r in all_records:
            row = dict(r)
            row["git_sha"] = git_sha
            row["ts"] = time.time()
            f.write(json.dumps(row) + "\n")

    # Write Markdown report
    report_path = out_dir / "report.md"
    _write_report(report_path, summary, all_records, git_sha)

    total_records = len(all_records)
    total_flips = sum(
        sum(1 for r in flip_rows if r["flipped"])
        for (_name, _nc, _nb, flip_rows, _rc, _rb) in summary
    )
    print(
        f"\nWrote {report_path} and {jsonl_path} "
        f"({total_records} records, {total_flips} flips total)"
    )


if __name__ == "__main__":
    main()
