"""Resumable work-unit runner with append-only JSONL metadata.

Each unit is identified by a deterministic 16-hex key derived from its
(cell, condition, run) fields. Completed keys are stored in a JSONL file so
a run interrupted mid-flight can resume without re-executing finished units.

Durability contract: each record is written, flushed, and fsynced before the
next unit starts. If the process is killed mid-write the last line may be
truncated. On resume, ``_iter_records`` silently skips that corrupt trailing
line (it was never fsynced, so re-running the unit is correct) and emits a
``logging.warning``.
"""
import json
import logging
import os
import time
import hashlib
from typing import Generator, Any

# Fields written by the harness itself; no run_one result may use these names.
_RESERVED_FIELDS = {"key", "ts", "git_sha"}


def unit_key(unit: dict) -> str:
    """Return a 16-hex deterministic key for *unit* based on (cell, condition, run).

    The key is a SHA-1 prefix over the canonically JSON-serialised subset so
    that the same logical unit always maps to the same key regardless of dict
    insertion order or extra fields the caller may have attached.
    """
    raw = json.dumps({k: unit[k] for k in ("cell", "condition", "run")},
                     sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _iter_records(store: str) -> Generator[dict, None, None]:
    """Yield parsed JSON records from *store*, skipping blank and unparseable lines.

    Returns immediately (yields nothing) if *store* does not exist. A corrupt
    or truncated trailing line -- the normal casualty of a mid-write kill --
    is logged at WARNING level and skipped; the caller should re-run any unit
    whose key is absent.
    """
    if not os.path.exists(store):
        return
    with open(store) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logging.warning(
                    "ex28_harness: skipping unparseable line %d in %s "
                    "(truncated mid-write?)", lineno, store
                )


def _completed_keys(store: str) -> set:
    """Return the set of unit keys that are already recorded in *store*.

    Reads via ``_iter_records`` so corrupt/truncated trailing lines are
    tolerated silently. Returns an empty set if *store* does not exist.
    """
    return {rec["key"] for rec in _iter_records(store) if "key" in rec}


def run_units(units, run_one, store: str, manifest_dir: str, git_sha: str,
              fresh: bool = False):
    """Run each unit in *units* that has not already been recorded in *store*.

    Parameters
    ----------
    units:
        Sequence of unit dicts, each containing at least ``cell``,
        ``condition``, and ``run`` keys used to build the deduplication key.
    run_one:
        Callable ``(unit) -> result_dict``. Must not return a dict whose keys
        overlap with the reserved harness fields ``{"key", "ts", "git_sha"}``.
    store:
        Path to the append-only JSONL results file. Completed keys are read
        from this file at startup; new records are appended one-per-unit.
    manifest_dir:
        Directory for the run-manifest JSON file. The filename encodes the
        Unix timestamp and PID to avoid same-second collisions and to preserve
        a resume history across interrupted runs.
    git_sha:
        Commit SHA written into every record for provenance.
    fresh:
        If True, delete *store* before starting so all units are re-run. Use
        only for deliberate full reruns; the normal path leaves *store* intact
        so interrupted runs resume from where they stopped.

    Durability guarantee: each record is written, flushed, and fsynced before
    the next unit begins. A corrupt trailing line left by a mid-write kill is
    skipped by ``_completed_keys`` on the next resume; the affected unit will
    simply be re-run.
    """
    os.makedirs(os.path.dirname(store) or ".", exist_ok=True)
    if fresh and os.path.exists(store):
        os.remove(store)
    done = _completed_keys(store)
    planned = [u for u in units if unit_key(u) not in done]
    manifest = {"git_sha": git_sha, "total": len(units),
                "already_done": len(done), "to_run": len(planned)}
    manifest_filename = f"manifest-{int(time.time())}-{os.getpid()}.json"
    with open(os.path.join(manifest_dir, manifest_filename), "w") as mf:
        json.dump(manifest, mf, indent=2)
    for u in planned:
        result = run_one(u)                       # the expensive call
        overlap = _RESERVED_FIELDS & result.keys()
        if overlap:
            raise ValueError(
                f"run_one result contains reserved harness field(s) {overlap!r}; "
                "rename those keys in run_one to avoid clobbering provenance metadata."
            )
        record = {"key": unit_key(u), "ts": time.time(), "git_sha": git_sha,
                  **u, **result}
        # If the file ended with a truncated line (no newline - e.g. a
        # previous mid-write kill), prepend a newline so the new record
        # starts on its own line rather than merging with the corrupt tail.
        needs_leading_newline = False
        if os.path.exists(store) and os.path.getsize(store) > 0:
            with open(store, "rb") as peek:
                peek.seek(-1, os.SEEK_END)
                needs_leading_newline = peek.read(1) != b"\n"
        line = json.dumps(record) + "\n"
        with open(store, "a") as f:               # append-only, flush each unit
            if needs_leading_newline:
                f.write("\n")
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
    return manifest


def summarize(store: str) -> list:
    """Return all parseable records from *store* as a list of dicts.

    Uses ``_iter_records`` so corrupt or truncated trailing lines are skipped
    rather than raising. Returns an empty list if *store* does not exist.
    """
    return list(_iter_records(store))
