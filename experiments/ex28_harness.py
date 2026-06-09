"""Resumable work-unit runner with append-only JSONL metadata."""
import json, os, time, hashlib


def unit_key(unit: dict) -> str:
    raw = json.dumps({k: unit[k] for k in ("cell", "condition", "run")},
                     sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _completed_keys(store: str):
    if not os.path.exists(store):
        return set()
    done = set()
    with open(store) as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["key"])
    return done


def run_units(units, run_one, store: str, manifest_dir: str, git_sha: str,
              fresh: bool = False):
    os.makedirs(os.path.dirname(store) or ".", exist_ok=True)
    if fresh and os.path.exists(store):
        os.remove(store)
    done = _completed_keys(store)
    planned = [u for u in units if unit_key(u) not in done]
    manifest = {"git_sha": git_sha, "total": len(units),
                "already_done": len(done), "to_run": len(planned)}
    with open(os.path.join(manifest_dir, f"manifest-{int(time.time())}.json"), "w") as mf:
        json.dump(manifest, mf, indent=2)
    for u in planned:
        result = run_one(u)                       # the expensive call
        record = {"key": unit_key(u), "ts": time.time(), "git_sha": git_sha,
                  **u, **result}
        with open(store, "a") as f:               # append-only, flush each unit
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())
    return manifest


def summarize(store: str):
    return [json.loads(l) for l in open(store) if l.strip()]
