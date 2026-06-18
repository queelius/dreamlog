"""Shared experiment-records harness: comprehensive, reproducible run records.

Every experiment writes through `experiment_run(...)` so provenance is uniform
and rich. On-disk layout (one directory per experiment, fully separated):

    experiments/data/<exp_id>/
        runs/
            <run_id>/                 run_id = UTC timestamp + short git sha
                meta.json             the full metadata envelope (below)
                results.json          the experiment's structured output
                summary.txt           human-readable summary (if provided)
        latest.json                   pointer to the most recent run + its key fields

The metadata envelope (meta.json) captures, for reproducibility:
  experiment: id, name, description, script path + sha256 of the script source
  run:        run_id, started/finished UTC, wall-clock duration
  git:        commit, branch, dirty flag, describe
  env:        python version, platform, machine, processor, hostname
  packages:   installed versions of the relevant packages
  params:     the experiment's parameter grid / config (caller-supplied)
  seeds:      any RNG seeds (caller-supplied)
  llm:        call/token/cost accounting (0 for symbolic experiments)
  outputs:    sha256 + byte size of each written result file

`experiments/data/` is tracked by git (only popper/ is ignored), so run records
are committable as provenance. Deterministic symbolic experiments reproduce
byte-identically, so committing the canonical run is cheap and auditable.

Usage:

    from _harness import experiment_run   # experiments insert their dir on sys.path

    with experiment_run(
            exp_id="ex30",
            name="...", description="...",
            script=__file__,
            params={"k_range": [2, 12], "l_values": [2, 3, 4, 5]},
            seeds={"numpy": 42}) as run:
        run.results["extraction"] = extraction_sweep()
        run.note("crossover K* falls with body depth")
        run.summary_lines.append("EXTRACTION crossover: ...")
    # on exit: writes runs/<run_id>/{meta,results,summary} and latest.json
    # run.run_dir and run.latest_path are available after the block
"""
import datetime
import hashlib
import json
import pathlib
import platform
import socket
import subprocess
import sys
import time


_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_DATA_ROOT = pathlib.Path(__file__).resolve().parent / "data"
_RELEVANT_PACKAGES = ("dreamlog", "pytest", "numpy", "sympy", "pyyaml",
                      "scipy", "anthropic", "openai")


def _git(*args, default=""):
    try:
        out = subprocess.run(["git", *args], cwd=str(_REPO_ROOT),
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else default
    except Exception:
        return default


def _git_metadata():
    return {
        "commit": _git("rev-parse", "HEAD", default="unknown"),
        "commit_short": _git("rev-parse", "--short", "HEAD", default="unknown"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD", default="unknown"),
        "describe": _git("describe", "--always", "--dirty", "--tags",
                         default="unknown"),
        "dirty": bool(_git("status", "--porcelain")),
    }


def _env_metadata():
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
    }


def _package_versions():
    try:
        from importlib.metadata import version, PackageNotFoundError
    except Exception:
        return {}
    out = {}
    for pkg in _RELEVANT_PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
        except Exception:
            continue
    return out


def _sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc)


class ExperimentRun:
    """Accumulates structured results + metadata; writes the run record on exit."""

    def __init__(self, exp_id, name, description, script=None,
                 params=None, seeds=None):
        self.exp_id = exp_id
        self.name = name
        self.description = description
        self.script = script
        self.params = params or {}
        self.seeds = seeds or {}
        self.results = {}
        self.summary_lines = []
        self.notes = []
        self.llm = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                    "est_cost_usd": 0.0}
        self._start_perf = None
        self._started = None
        self.run_dir = None
        self.latest_path = None

    # -- caller helpers --

    def note(self, text):
        self.notes.append(text)

    def log_llm(self, calls=0, input_tokens=0, output_tokens=0, cost_usd=0.0):
        self.llm["calls"] += calls
        self.llm["input_tokens"] += input_tokens
        self.llm["output_tokens"] += output_tokens
        self.llm["est_cost_usd"] += cost_usd

    # -- lifecycle --

    def _start(self):
        self._start_perf = time.perf_counter()
        self._started = _utc_now()

    def _build_meta(self, finished, duration_s, output_records):
        script_info = {}
        if self.script:
            sp = pathlib.Path(self.script).resolve()
            script_info = {
                "path": str(sp.relative_to(_REPO_ROOT))
                        if str(sp).startswith(str(_REPO_ROOT)) else str(sp),
                "sha256": _sha256_file(sp) if sp.exists() else None,
            }
        return {
            "experiment": {
                "id": self.exp_id,
                "name": self.name,
                "description": self.description,
                "script": script_info,
            },
            "run": {
                "run_id": self._run_id,
                "started_utc": self._started.isoformat(),
                "finished_utc": finished.isoformat(),
                "duration_s": round(duration_s, 4),
            },
            "git": _git_metadata(),
            "env": _env_metadata(),
            "packages": _package_versions(),
            "params": self.params,
            "seeds": self.seeds,
            "llm": self.llm,
            "outputs": output_records,
        }

    def _finish_and_save(self):
        finished = _utc_now()
        duration_s = time.perf_counter() - self._start_perf
        sha = _git("rev-parse", "--short", "HEAD", default="nogit")
        self._run_id = "%s-%s" % (
            self._started.strftime("%Y%m%dT%H%M%SZ"), sha)

        exp_dir = _DATA_ROOT / self.exp_id
        self.run_dir = exp_dir / "runs" / self._run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # results.json
        results_path = self.run_dir / "results.json"
        results_path.write_text(json.dumps(self.results, indent=2,
                                           sort_keys=True))
        # summary.txt (optional)
        output_records = {
            "results.json": {
                "sha256": _sha256_file(results_path),
                "bytes": results_path.stat().st_size,
            }
        }
        if self.summary_lines:
            summary_path = self.run_dir / "summary.txt"
            summary_path.write_text("\n".join(self.summary_lines) + "\n")
            output_records["summary.txt"] = {
                "sha256": _sha256_file(summary_path),
                "bytes": summary_path.stat().st_size,
            }

        # meta.json
        meta = self._build_meta(finished, duration_s, output_records)
        if self.notes:
            meta["notes"] = self.notes
        (self.run_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True))

        # latest.json pointer (small, stable path for citation)
        self.latest_path = exp_dir / "latest.json"
        self.latest_path.write_text(json.dumps({
            "run_id": self._run_id,
            "run_dir": str(self.run_dir.relative_to(_REPO_ROOT)),
            "git_commit_short": meta["git"]["commit_short"],
            "finished_utc": finished.isoformat(),
            "results_sha256": output_records["results.json"]["sha256"],
        }, indent=2, sort_keys=True))


class _RunContext:
    def __init__(self, run):
        self.run = run

    def __enter__(self):
        self.run._start()
        return self.run

    def __exit__(self, exc_type, exc, tb):
        # Always write a record, even on failure, so partial runs are auditable.
        self.run._finish_and_save()
        return False


def experiment_run(exp_id, name, description, script=None,
                   params=None, seeds=None):
    """Context manager that records a fully-provenanced experiment run.

    Fill `run.results` (a dict) inside the block; append human-readable lines to
    `run.summary_lines`. On exit, writes runs/<run_id>/{meta,results,summary} and
    updates latest.json under experiments/data/<exp_id>/.
    """
    return _RunContext(ExperimentRun(exp_id, name, description, script,
                                     params, seeds))
