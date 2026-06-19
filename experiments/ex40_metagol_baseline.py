#!/usr/bin/env python3
"""
EX40: Metagol ILP Baseline on Family-Tree Domain (EX25/EX27 protocol).

Meta-interpretive learning (Cropper & Muggleton 2016) comparison against
DreamLog (EX25/EX27) and Popper (EX26) on two learning targets:
  - father/2  (cross-predicate: parent + male => father)
  - ancestor/2 (recursion: transitive closure of parent)

Metagol is a meta-interpretive learner that searches for a logic program that
proves positive examples and refutes negative ones, guided by user-supplied
METARULES (rule templates).  This is the key methodological distinction:
Metagol requires the user to supply metarule templates (e.g., chain, tailrec)
specifying the SHAPE of allowed clauses; DreamLog (Ops C/I) and Popper derive
rule structure without such templates.  The metarules supplied here are the
standard Metagol metarules from the literature (ident and tailrec for ancestor,
conjunction for father); they constitute additional inductive bias not available
to the other two systems.

Setup: SWI-Prolog 10.0.2 + Metagol commit 7b61799 (cloned to
experiments/vendor/metagol/).

No LLM used (--no-llm by design).  Zero DreamLog library changes.

Run:
    python experiments/ex40_metagol_baseline.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EXPERIMENTS_DIR))

from _harness import experiment_run  # type: ignore

# Path to vendored metagol.pl
METAGOL_PL = EXPERIMENTS_DIR / "vendor" / "metagol" / "metagol.pl"

# -------------------------------------------------------------------------
# Family tree facts (reused from EX25 build_family_tree / family_base_facts)
# We reproduce them here as Prolog strings so this file is self-contained
# (the Python import path issues from running standalone would otherwise
# require inserting the experiments dir, which is already on sys.path, but
# having them inline also makes the Prolog task generation clearer).
# -------------------------------------------------------------------------

FAMILY_PARENT_FACTS: List[str] = [
    "parent(albert, henry).", "parent(albert, margaret).", "parent(albert, william).",
    "parent(martha, henry).", "parent(martha, margaret).", "parent(martha, william).",
    "parent(charles, patricia).", "parent(charles, robert).",
    "parent(dorothy, patricia).", "parent(dorothy, robert).",
    "parent(henry, james).", "parent(henry, sarah).", "parent(henry, thomas).",
    "parent(elizabeth, james).", "parent(elizabeth, sarah).", "parent(elizabeth, thomas).",
    "parent(robert, michael).", "parent(robert, jennifer).",
    "parent(margaret, michael).", "parent(margaret, jennifer).",
    "parent(william, david).", "parent(william, emily).", "parent(william, daniel).",
    "parent(catherine, david).", "parent(catherine, emily).", "parent(catherine, daniel).",
    "parent(james, emma).", "parent(james, jack).",
    "parent(lisa, emma).", "parent(lisa, jack).",
    "parent(thomas, oliver).",
    "parent(rachel, oliver).",
    "parent(michael, sophia).", "parent(michael, noah).",
    "parent(amanda, sophia).", "parent(amanda, noah).",
    "parent(david, lucas).",
    "parent(christine, lucas).",
]

FAMILY_MALE_FACTS: List[str] = [
    "male(albert).", "male(charles).", "male(henry).", "male(william).", "male(robert).",
    "male(james).", "male(thomas).", "male(michael).", "male(david).", "male(daniel).",
    "male(jack).", "male(oliver).", "male(noah).", "male(lucas).",
]

FAMILY_FEMALE_FACTS: List[str] = [
    "female(martha).", "female(dorothy).", "female(margaret).", "female(patricia).",
    "female(elizabeth).", "female(catherine).", "female(sarah).", "female(jennifer).",
    "female(emily).", "female(lisa).", "female(rachel).", "female(amanda).",
    "female(christine).", "female(emma).", "female(sophia).",
]

# Training positive/negative examples for father/2
FATHER_POS: List[str] = [
    "father(albert, henry)", "father(albert, margaret)", "father(albert, william)",
    "father(charles, patricia)", "father(charles, robert)",
    "father(henry, james)", "father(henry, sarah)", "father(henry, thomas)",
    "father(robert, michael)", "father(robert, jennifer)",
    "father(william, david)", "father(william, emily)", "father(william, daniel)",
    "father(james, emma)", "father(james, jack)",
    "father(thomas, oliver)",
    "father(michael, sophia)", "father(michael, noah)",
    "father(david, lucas)",
]

# Negatives: female parents that should NOT be fathers
FATHER_NEG: List[str] = [
    "father(martha, henry)", "father(dorothy, patricia)",
    "father(elizabeth, james)",
    "father(margaret, michael)",
    "father(catherine, david)",
    "father(lisa, emma)",
    "father(rachel, oliver)",
    "father(amanda, sophia)",
    "father(christine, lucas)",
]

# Training positives for ancestor/2 (transitive closure of parent)
# Using a representative subset (not all O(n^2) pairs) to keep the task manageable
ANCESTOR_POS: List[str] = [
    # depth-1 (direct parent)
    "ancestor(albert, henry)", "ancestor(albert, margaret)", "ancestor(albert, william)",
    "ancestor(martha, henry)",
    "ancestor(henry, james)", "ancestor(henry, sarah)", "ancestor(henry, thomas)",
    "ancestor(robert, michael)", "ancestor(robert, jennifer)",
    "ancestor(william, david)", "ancestor(william, emily)", "ancestor(william, daniel)",
    "ancestor(james, emma)", "ancestor(james, jack)",
    "ancestor(david, lucas)",
    # depth-2 (grandparent)
    "ancestor(albert, james)", "ancestor(albert, sarah)", "ancestor(albert, thomas)",
    "ancestor(albert, michael)", "ancestor(albert, jennifer)",
    "ancestor(albert, david)", "ancestor(albert, emily)", "ancestor(albert, daniel)",
    "ancestor(henry, emma)", "ancestor(henry, jack)", "ancestor(henry, oliver)",
    "ancestor(robert, sophia)", "ancestor(robert, noah)",
    "ancestor(william, lucas)",
    # depth-3 (great-grandparent)
    "ancestor(albert, emma)", "ancestor(albert, jack)", "ancestor(albert, oliver)",
    "ancestor(albert, sophia)", "ancestor(albert, noah)",
    "ancestor(albert, lucas)",
]

ANCESTOR_NEG: List[str] = [
    "ancestor(emma, albert)",
    "ancestor(jack, henry)",
    "ancestor(lucas, david)",
    "ancestor(sophia, robert)",
    "ancestor(oliver, thomas)",
    "ancestor(henry, albert)",
]

# -------------------------------------------------------------------------
# New-entity held-out evaluation (same protocol as EX25/EX26/EX27)
# -------------------------------------------------------------------------

# These new individuals have base facts only (parent/male/female).
# They are NOT in the training set.
NEW_FAMILY_PARENT_FACTS: List[str] = [
    "parent(new_gfather, new_dad).", "parent(new_gmother, new_dad).",
    "parent(new_dad, new_son).", "parent(new_mom, new_son).",
    "parent(new_dad, new_daughter).", "parent(new_mom, new_daughter).",
]

NEW_FAMILY_GENDER_FACTS: List[str] = [
    "male(new_gfather).", "female(new_gmother).",
    "male(new_dad).", "female(new_mom).",
    "male(new_son).", "female(new_daughter).",
]

# (target_pred, args_str, expected_true, description)
NEW_ENTITY_CHECKS: List[Tuple[str, str, bool, str]] = [
    ("father", "new_dad, new_son",      True,  "father from parent+male"),
    ("father", "new_dad, new_daughter", True,  "father from parent+male"),
    ("father", "new_mom, new_son",      False, "female not father"),
    ("father", "new_gmother, new_dad",  False, "female not father (grandparent level)"),
    ("ancestor", "new_dad, new_son",         True,  "ancestor depth-1"),
    ("ancestor", "new_gfather, new_son",     True,  "ancestor depth-2"),
    ("ancestor", "new_gmother, new_daughter",True,  "ancestor depth-2 (female line)"),
    ("ancestor", "new_son, new_dad",         False, "reverse direction not ancestor"),
    ("ancestor", "new_son, new_gfather",     False, "deep reverse not ancestor"),
]


# -------------------------------------------------------------------------
# Metagol Prolog program builder
# -------------------------------------------------------------------------

def _prolog_term_list(items: List[str]) -> str:
    """Render a Python list as a Prolog list literal."""
    return "[\n    " + ",\n    ".join(items) + "\n  ]"


def build_father_program(metagol_path: Path) -> str:
    """Build the Prolog program for learning father/2."""
    bk = "\n".join(FAMILY_PARENT_FACTS + FAMILY_MALE_FACTS + FAMILY_FEMALE_FACTS)
    pos_list = _prolog_term_list(FATHER_POS)
    neg_list = _prolog_term_list(FATHER_NEG)
    return f"""\
:- use_module('{metagol_path}').

body_pred(parent/2).
body_pred(male/1).
body_pred(female/1).

%% conjunction metarule: P(A,B) :- Q(A,B), R(A)
%% This is the standard Metagol metarule for binary + unary conjunction.
%% It covers "father(X,Y) :- parent(X,Y), male(X)" structurally.
metarule([P,Q,R], [P,A,B], [[Q,A,B],[R,A]]).

%% Background knowledge
{bk}

:-
  Pos = {pos_list},
  Neg = {neg_list},
  (learn(Pos, Neg) ->
      true
  ;
      writeln('METAGOL_FAIL: unable to learn father/2')
  ).
"""


def build_ancestor_program(metagol_path: Path) -> str:
    """Build the Prolog program for learning ancestor/2 (recursive)."""
    bk = "\n".join(FAMILY_PARENT_FACTS)
    pos_list = _prolog_term_list(ANCESTOR_POS)
    neg_list = _prolog_term_list(ANCESTOR_NEG)
    return f"""\
:- use_module('{metagol_path}').

metagol:max_clauses(2).

body_pred(parent/2).

%% ident metarule: P(A,B) :- Q(A,B)
%% (base case: ancestor(X,Y) :- parent(X,Y))
metarule([P,Q], [P,A,B], [[Q,A,B]]).

%% tailrec metarule: P(A,B) :- Q(A,C), P(C,B)
%% (recursive case: ancestor(X,Z) :- parent(X,Y), ancestor(Y,Z))
metarule([P,Q], [P,A,B], [[Q,A,C],[P,C,B]]).

%% Background knowledge
{bk}

:-
  Pos = {pos_list},
  Neg = {neg_list},
  (learn(Pos, Neg) ->
      true
  ;
      writeln('METAGOL_FAIL: unable to learn ancestor/2')
  ).
"""


def build_eval_program(
    metagol_path: Path,
    target_pred: str,
    learned_clauses: List[str],
    checks: List[Tuple[str, str, bool, str]],
) -> str:
    """Build a Prolog program that loads the learned rules + new-entity facts
    and queries each check, printing a result line per check.

    Each check is its own top-level directive so variable scoping does not
    bleed across checks.  Format: CHECK <pred> <args_no_space> <expected> <got>
    """
    bk_new = "\n".join(
        NEW_FAMILY_PARENT_FACTS + NEW_FAMILY_GENDER_FACTS
        + FAMILY_PARENT_FACTS + FAMILY_MALE_FACTS + FAMILY_FEMALE_FACTS
    )
    clauses_block = "\n".join(learned_clauses)
    # Each check is an independent top-level directive
    directives = []
    for pred, args, expected, desc in checks:
        if pred != target_pred:
            continue
        atom = f"{pred}({args})"
        exp_atom = "true" if expected else "false"
        # use a compact key without spaces for easy parsing
        args_key = args.replace(" ", "")
        directives.append(
            f":- (({atom}) -> Got = true ; Got = false),\n"
            f"   format('CHECK {pred} {args_key} {exp_atom} ~w~n', [Got])."
        )
    if not directives:
        return ""
    directives_block = "\n".join(directives)
    return f"""\
{bk_new}

{clauses_block}

{directives_block}
"""


# -------------------------------------------------------------------------
# Subprocess runner
# -------------------------------------------------------------------------

def run_swipl(prolog_source: str, timeout_s: int = 60) -> Tuple[str, str, int, float]:
    """Write prolog_source to a temp file, invoke swipl, return
    (stdout, stderr, returncode, elapsed_s)."""
    with tempfile.NamedTemporaryFile(suffix=".pl", mode="w",
                                    delete=False) as tmp:
        tmp.write(prolog_source)
        tmp_path = tmp.name
    try:
        start = time.perf_counter()
        proc = subprocess.run(
            ["swipl", "-g", "halt", tmp_path],
            capture_output=True, text=True, timeout=timeout_s,
        )
        elapsed = time.perf_counter() - start
        return proc.stdout, proc.stderr, proc.returncode, elapsed
    except subprocess.TimeoutExpired:
        elapsed = timeout_s
        return "", "TIMEOUT", -1, float(elapsed)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# -------------------------------------------------------------------------
# Output parser
# -------------------------------------------------------------------------

def parse_learned_rules(stdout: str, target_pred: str) -> Tuple[bool, List[str]]:
    """Extract learned rule lines from Metagol stdout.

    Metagol prints learned clauses as standard Prolog clauses, e.g.:
        father(A,B):-parent(A,B),male(A).

    Returns (success, list_of_clause_strings).
    success=False if 'METAGOL_FAIL' or 'unable to learn' appears in output.
    """
    if "METAGOL_FAIL" in stdout or "unable to learn" in stdout:
        return False, []
    lines = stdout.strip().splitlines()
    rules = []
    for line in lines:
        line = line.strip()
        # Skip progress lines like "% learning ...", "% clauses: N"
        if line.startswith("%"):
            continue
        # Keep non-empty non-comment lines that look like Prolog clauses
        if line and not line.startswith("//"):
            rules.append(line)
    return bool(rules), rules


def parse_check_results(stdout: str, target_pred: str,
                        checks: List[Tuple[str, str, bool, str]],
                        ) -> Dict:
    """Parse CHECK lines printed by the eval program.

    Format: CHECK <pred> <args_no_spaces> <expected> <got>
    All four tokens are single words (args compacted by removing spaces).
    Returns dict with per_check list and aggregate recall/precision/accuracy.
    """
    check_lines = [ln for ln in stdout.splitlines()
                   if ln.startswith("CHECK ")]
    per_check = []
    tp = fp = tn = fn = 0
    for ln in check_lines:
        parts = ln.split()
        if len(parts) != 5:
            continue
        _, pred, args_key, expected_val, got_val = parts
        expected = (expected_val == "true")
        got = (got_val == "true")
        if expected and got:
            outcome = "TP"; tp += 1
        elif expected and not got:
            outcome = "FN"; fn += 1
        elif not expected and got:
            outcome = "FP"; fp += 1
        else:
            outcome = "TN"; tn += 1
        # find description from original checks table
        desc = ""
        for p, a, e, d in checks:
            if p == pred and a.replace(" ", "") == args_key and e == expected:
                desc = d
                break
        per_check.append({"pred": pred, "args": args_key, "expected": expected,
                          "got": got, "outcome": outcome, "desc": desc})

    total = tp + fp + tn + fn
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    accuracy = (tp + tn) / total if total > 0 else None
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "recall": recall, "precision": precision, "accuracy": accuracy,
        "per_check": per_check,
    }


# -------------------------------------------------------------------------
# Per-target runner
# -------------------------------------------------------------------------

METARULES_FATHER = [
    "conjunction: P(A,B) :- Q(A,B), R(A)  [Metagol standard]",
]
METARULES_ANCESTOR = [
    "ident:   P(A,B) :- Q(A,B)             [Metagol standard]",
    "tailrec: P(A,B) :- Q(A,C), P(C,B)    [Metagol standard]",
]


def run_target(
    target: str,
    program_src: str,
    metagol_path: Path,
    timeout_s: int = 60,
) -> Dict:
    """Run a single Metagol learning task and evaluate on new entities."""
    print(f"\n  -- {target}/2 --")
    stdout, stderr, rc, elapsed = run_swipl(program_src, timeout_s=timeout_s)
    if rc == -1:
        print(f"     TIMEOUT after {elapsed:.1f}s")
        return {
            "target": target, "status": "timeout",
            "learned_rules": [], "elapsed_s": elapsed,
            "eval": None,
        }

    success, rules = parse_learned_rules(stdout, target)
    if not success:
        print(f"     FAIL: Metagol could not learn {target}/2")
        print(f"     stdout: {stdout[:400]}")
        return {
            "target": target, "status": "no_rule",
            "learned_rules": [], "elapsed_s": elapsed,
            "stdout": stdout, "stderr": stderr[:400],
            "eval": None,
        }

    print(f"     Learned ({len(rules)} clause(s)):")
    for r in rules:
        print(f"       {r}")

    # Evaluate on new entities by building a Prolog program that
    # loads the learned clauses + new-entity BK and queries each check
    eval_src = build_eval_program(metagol_path, target, rules, NEW_ENTITY_CHECKS)
    eval_stdout, eval_stderr, eval_rc, eval_elapsed = run_swipl(
        eval_src, timeout_s=30,
    )
    eval_result = parse_check_results(eval_stdout, target, NEW_ENTITY_CHECKS)
    r_val = eval_result["recall"]
    p_val = eval_result["precision"]
    r_str = f"{r_val:.0%}" if r_val is not None else "N/A"
    p_str = f"{p_val:.0%}" if p_val is not None else "N/A"
    print(f"     recall={r_str} precision={p_str}"
          f" (tp={eval_result['tp']} fp={eval_result['fp']}"
          f" fn={eval_result['fn']} tn={eval_result['tn']})")

    return {
        "target": target,
        "status": "ok",
        "learned_rules": rules,
        "elapsed_s": elapsed,
        "eval": eval_result,
        "eval_stdout": eval_stdout,
    }


# -------------------------------------------------------------------------
# Driver
# -------------------------------------------------------------------------

def main() -> None:
    print("EX40: Metagol ILP baseline (father + ancestor, family-tree domain)")
    print(f"  metagol.pl : {METAGOL_PL}")
    print(f"  swipl      : ", end="", flush=True)
    ver_proc = subprocess.run(["swipl", "--version"], capture_output=True, text=True)
    print(ver_proc.stdout.strip() or ver_proc.stderr.strip())

    metagol_commit = subprocess.run(
        ["git", "-C", str(METAGOL_PL.parent), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()
    print(f"  metagol git: {metagol_commit}")

    with experiment_run(
        exp_id="ex40",
        name="Metagol ILP baseline (recursion + cross-predicate)",
        description=(
            "Metagol meta-interpretive learning baseline on the EX25/EX27 "
            "family-tree domain. Two targets: father/2 (cross-predicate, "
            "conjunction metarule) and ancestor/2 (recursion, ident+tailrec "
            "metarules). New-entity held-out evaluation matches EX25/EX26/EX27 "
            "protocol. No LLM. Key methodological point: Metagol requires "
            "user-supplied metarule templates (extra inductive bias absent from "
            "DreamLog and Popper)."
        ),
        script=__file__,
        params={
            "targets": ["father", "ancestor"],
            "metarules_father": METARULES_FATHER,
            "metarules_ancestor": METARULES_ANCESTOR,
            "metagol_commit": metagol_commit,
            "timeout_s_per_target": 60,
        },
        seeds={},
    ) as run:
        father_src = build_father_program(METAGOL_PL)
        ancestor_src = build_ancestor_program(METAGOL_PL)

        father_result = run_target("father", father_src, METAGOL_PL)
        ancestor_result = run_target("ancestor", ancestor_src, METAGOL_PL)

        run.results["father"] = {
            "status": father_result["status"],
            "learned_rules": father_result["learned_rules"],
            "elapsed_s": father_result["elapsed_s"],
            "eval": father_result.get("eval"),
        }
        run.results["ancestor"] = {
            "status": ancestor_result["status"],
            "learned_rules": ancestor_result["learned_rules"],
            "elapsed_s": ancestor_result["elapsed_s"],
            "eval": ancestor_result.get("eval"),
        }

        run.results["metarules_supplied"] = {
            "father": METARULES_FATHER,
            "ancestor": METARULES_ANCESTOR,
            "total_count": len(METARULES_FATHER) + len(METARULES_ANCESTOR),
        }

        run.results["methodological_note"] = (
            "Metagol requires the user to supply METARULES (rule templates) that "
            "define the allowed clause structure before learning begins. For "
            "father/2 we supplied 1 metarule (conjunction: P(A,B):-Q(A,B),R(A)); "
            "for ancestor/2 we supplied 2 metarules (ident and tailrec). "
            "These metarules constitute additional inductive bias that is NOT "
            "available to DreamLog (Ops C/I discover structure from data) or "
            "Popper (Popper searches over clause shapes without pre-specified "
            "templates). An honest comparison must account for this asymmetry: "
            "Metagol's success on the recursive case is guaranteed-by-design "
            "once the tailrec metarule is supplied, whereas DreamLog's Operation I "
            "and Popper must discover the recursive structure from examples alone."
        )

        # Build human-readable summary
        def _fmt_eval(ev: Optional[Dict]) -> str:
            if ev is None:
                return "N/A"
            r = ev["recall"]
            p = ev["precision"]
            rs = f"{r:.0%}" if r is not None else "N/A"
            ps = f"{p:.0%}" if p is not None else "N/A"
            return f"recall={rs} precision={ps}"

        run.summary_lines.extend([
            "EX40: Metagol ILP baseline -- family-tree domain",
            "",
            "TARGET: father/2",
            f"  status        : {father_result['status']}",
            f"  learned rules : {father_result['learned_rules']}",
            f"  new-entity    : {_fmt_eval(father_result.get('eval'))}",
            f"  elapsed       : {father_result['elapsed_s']:.2f}s",
            f"  metarules     : {METARULES_FATHER}",
            "",
            "TARGET: ancestor/2",
            f"  status        : {ancestor_result['status']}",
            f"  learned rules : {ancestor_result['learned_rules']}",
            f"  new-entity    : {_fmt_eval(ancestor_result.get('eval'))}",
            f"  elapsed       : {ancestor_result['elapsed_s']:.2f}s",
            f"  metarules     : {METARULES_ANCESTOR}",
            "",
            "METHODOLOGICAL NOTE:",
            "  Metagol requires hand-supplied metarule templates (extra inductive",
            "  bias). DreamLog (Ops C/I) and Popper discover rule structure from",
            "  data without templates. Count: 1 metarule for father, 2 for ancestor.",
            "  Metagol's recursion success is guaranteed-by-design once tailrec",
            "  is supplied; DreamLog's Operation I discovers recursion with no",
            "  template -- the stronger result.",
        ])

    print(f"\n  father   : {_fmt_eval(father_result.get('eval'))}")
    print(f"  ancestor : {_fmt_eval(ancestor_result.get('eval'))}")
    print(f"\n  Run dir: {run.run_dir}")
    print(f"Done. latest.json: {run.latest_path}")


if __name__ == "__main__":
    main()
