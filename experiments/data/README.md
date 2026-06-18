# Experiment records

One directory per experiment, fully separated. Records are written through
`experiments/_harness.py` (`experiment_run(...)`), which captures comprehensive,
reproducible provenance.

## Layout (harness convention, EX29 onward)

```
experiments/data/<exp_id>/
    runs/
        <run_id>/                 run_id = <UTC timestamp>-<short git sha>
            meta.json             full metadata envelope (see below)
            results.json          the experiment's structured output
            summary.txt           human-readable summary (if the script emits one)
    latest.json                   pointer to the most recent run + key fields
```

`runs/` keeps the full history; `latest.json` is the stable path to cite. Because
these experiments are deterministic and symbolic, re-running reproduces results
byte-for-byte (verify via the `outputs[*].sha256` in `meta.json`).

## meta.json envelope

| key | contents |
|-----|----------|
| `experiment` | id, name, description, script path + **sha256 of the script source** |
| `run` | run_id, started/finished UTC, wall-clock duration |
| `git` | commit, branch, describe, dirty flag |
| `env` | python version + implementation, platform, machine, processor, hostname |
| `packages` | installed versions of the relevant packages (dreamlog, pytest, numpy, ...) |
| `params` | the experiment's parameter grid / config |
| `seeds` | RNG seeds (or a note that the run is deterministic) |
| `llm` | call / token / cost accounting (0 for symbolic experiments) |
| `outputs` | sha256 + byte size of every written result file |

## Legacy experiments (predate the harness)

These use their own ad-hoc conventions and are kept as-is because tests and the
paper read their exact paths:

- `ex25b/results.json` -- crafting generalization (read by the refactor regression test)
- `ex27/run_*.log` -- recursion discovery run log
- `ex28/`, `ex28_sonnet/` -- LLM-role ablation (`results.jsonl` + thin `manifest-*.json`)
- `popper/` -- Popper baseline (gitignored: `results.jsonl`, `cells/`)

New experiments should use the harness so provenance stays uniform and rich.
