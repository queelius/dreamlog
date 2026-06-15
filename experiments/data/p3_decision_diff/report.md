# P3 Decision Diff: clauses vs bits DL modes

Git SHA: `f246ce3` | Generated: 2026-06-15T04:00:26Z

## Scope

This report covers symbolic Operations A-F and I only. Operation G
(LLM-assisted compression) is EXCLUDED: no llm_client is passed to any
dreamer in this tool, so G never fires. The bits vs clauses comparison
is therefore complete only for the symbolic pipeline. G's priced
criterion (Task 4) is covered by tests/test_dl_bits.py unit tests.

---

## Per-Scenario Decision Tables

### ex25b_crafting

Decisions: 5 (clauses) / 5 (bits) | Compression ratio: 0.919 (clauses) / 1.000 (bits) | Flips: 5 | Cascade: 0 clauses-only, 0 bits-only

| kind | delta_clauses | delta_bits | decision_clauses | decision_bits | FLIPPED | note |
|------|--------------|------------|-----------------|---------------|---------|------|
| generalization | -2 | +228.09 | accepted | rejected | YES | both |
| generalization | -5 | +229.48 | accepted | rejected | YES | both |
| generalization | -1 | +349.55 | accepted | rejected | YES | both |
| generalization | -2 | +347.67 | accepted | rejected | YES | both |
| generalization | -4 | +271.50 | accepted | rejected | YES | both |

### ex28_can_fly_canonical

Decisions: 1 (clauses) / 1 (bits) | Compression ratio: 0.786 (clauses) / 1.000 (bits) | Flips: 1 | Cascade: 0 clauses-only, 0 bits-only

| kind | delta_clauses | delta_bits | decision_clauses | decision_bits | FLIPPED | note |
|------|--------------|------------|-----------------|---------------|---------|------|
| generalization | -3 | +204.00 | accepted | rejected | YES | both |

### ex28_glonk_invented

Decisions: 1 (clauses) / 1 (bits) | Compression ratio: 0.786 (clauses) / 1.000 (bits) | Flips: 1 | Cascade: 0 clauses-only, 0 bits-only

| kind | delta_clauses | delta_bits | decision_clauses | decision_bits | FLIPPED | note |
|------|--------------|------------|-----------------|---------------|---------|------|
| generalization | -3 | +188.00 | accepted | rejected | YES | both |

### ex28_ancestor_canonical

Decisions: 1 (clauses) / 1 (bits) | Compression ratio: 0.278 (clauses) / 0.278 (bits) | Flips: 0 | Cascade: 0 clauses-only, 0 bits-only

| kind | delta_clauses | delta_bits | decision_clauses | decision_bits | FLIPPED | note |
|------|--------------|------------|-----------------|---------------|---------|------|
| recursion | -26 | -419.49 | accepted | accepted |  | both |

### ex28_tessik_invented

Decisions: 1 (clauses) / 1 (bits) | Compression ratio: 0.278 (clauses) / 0.278 (bits) | Flips: 0 | Cascade: 0 clauses-only, 0 bits-only

| kind | delta_clauses | delta_bits | decision_clauses | decision_bits | FLIPPED | note |
|------|--------------|------------|-----------------|---------------|---------|------|
| recursion | -26 | -419.49 | accepted | accepted |  | both |

### ex28_father_canonical

Decisions: 0 (clauses) / 0 (bits) | Compression ratio: 1.000 (clauses) / 1.000 (bits) | Flips: 0 | Cascade: 0 clauses-only, 0 bits-only

*(no decisions recorded)*

### ex28_brulit_invented

Decisions: 0 (clauses) / 0 (bits) | Compression ratio: 1.000 (clauses) / 1.000 (bits) | Flips: 0 | Cascade: 0 clauses-only, 0 bits-only

*(no decisions recorded)*

### bench_family_small

Decisions: 0 (clauses) / 0 (bits) | Compression ratio: 1.000 (clauses) / 1.000 (bits) | Flips: 0 | Cascade: 0 clauses-only, 0 bits-only

*(no decisions recorded)*

### bench_family_with_guards

Decisions: 4 (clauses) / 4 (bits) | Compression ratio: 0.907 (clauses) / 0.977 (bits) | Flips: 3 | Cascade: 0 clauses-only, 0 bits-only

| kind | delta_clauses | delta_bits | decision_clauses | decision_bits | FLIPPED | note |
|------|--------------|------------|-----------------|---------------|---------|------|
| extraction | +1 | +105.02 | accepted | rejected | YES | both |
| generalization | -3 | +255.86 | accepted | rejected | YES | both |
| invention | -1 | +335.25 | accepted | rejected | YES | both |
| pruning | -1 | -20.04 | accepted | accepted |  | both |

### bench_transitive_closures

Decisions: 1 (clauses) / 1 (bits) | Compression ratio: 0.909 (clauses) / 1.000 (bits) | Flips: 1 | Cascade: 0 clauses-only, 0 bits-only

| kind | delta_clauses | delta_bits | decision_clauses | decision_bits | FLIPPED | note |
|------|--------------|------------|-----------------|---------------|---------|------|
| invention | -2 | +283.11 | accepted | rejected | YES | both |

### bench_body_patterns

Decisions: 1 (clauses) / 1 (bits) | Compression ratio: 1.091 (clauses) / 1.000 (bits) | Flips: 1 | Cascade: 0 clauses-only, 0 bits-only

| kind | delta_clauses | delta_bits | decision_clauses | decision_bits | FLIPPED | note |
|------|--------------|------------|-----------------|---------------|---------|------|
| extraction | +1 | +87.77 | accepted | rejected | YES | both |

### bench_dead_clauses

Decisions: 1 (clauses) / 1 (bits) | Compression ratio: 0.938 (clauses) / 1.000 (bits) | Flips: 1 | Cascade: 0 clauses-only, 0 bits-only

| kind | delta_clauses | delta_bits | decision_clauses | decision_bits | FLIPPED | note |
|------|--------------|------------|-----------------|---------------|---------|------|
| generalization | -1 | +295.59 | accepted | rejected | YES | both |

### bench_cascading

Decisions: 4 (clauses) / 4 (bits) | Compression ratio: 0.885 (clauses) / 0.962 (bits) | Flips: 3 | Cascade: 0 clauses-only, 0 bits-only

| kind | delta_clauses | delta_bits | decision_clauses | decision_bits | FLIPPED | note |
|------|--------------|------------|-----------------|---------------|---------|------|
| extraction | +1 | +104.29 | accepted | rejected | YES | both |
| generalization | -2 | +234.31 | accepted | rejected | YES | both |
| invention | -1 | +263.60 | accepted | rejected | YES | both |
| pruning | -1 | -18.63 | accepted | accepted |  | both |

### bench_stress

Decisions: 3 (clauses) / 3 (bits) | Compression ratio: 0.838 (clauses) / 0.843 (bits) | Flips: 2 | Cascade: 0 clauses-only, 0 bits-only

| kind | delta_clauses | delta_bits | decision_clauses | decision_bits | FLIPPED | note |
|------|--------------|------------|-----------------|---------------|---------|------|
| extraction | +1 | +118.16 | accepted | rejected | YES | both |
| invention | -2 | +218.57 | accepted | rejected | YES | both |
| generalization | -29 | -444.14 | accepted | accepted |  | both |

### bench_subsumption_edge

Decisions: 1 (clauses) / 1 (bits) | Compression ratio: 0.889 (clauses) / 0.889 (bits) | Flips: 0 | Cascade: 0 clauses-only, 0 bits-only

| kind | delta_clauses | delta_bits | decision_clauses | decision_bits | FLIPPED | note |
|------|--------------|------------|-----------------|---------------|---------|------|
| subsumption | -1 | -45.77 | accepted | accepted |  | both |

---

## Totals

Total scenarios: 15
Total flips (same key, different decision): 18
Total clauses-only cascade rows: 0
Total bits-only cascade rows: 0

### Flips by operation kind

| kind | flip count |
|------|-----------|
| generalization | 10 |
| extraction | 4 |
| invention | 4 |

### Compression ratios per mode per scenario

| scenario | ratio_clauses | ratio_bits |
|----------|--------------|-----------|
| ex25b_crafting | 0.919 | 1.000 |
| ex28_can_fly_canonical | 0.786 | 1.000 |
| ex28_glonk_invented | 0.786 | 1.000 |
| ex28_ancestor_canonical | 0.278 | 0.278 |
| ex28_tessik_invented | 0.278 | 0.278 |
| ex28_father_canonical | 1.000 | 1.000 |
| ex28_brulit_invented | 1.000 | 1.000 |
| bench_family_small | 1.000 | 1.000 |
| bench_family_with_guards | 0.907 | 0.977 |
| bench_transitive_closures | 0.909 | 1.000 |
| bench_body_patterns | 1.091 | 1.000 |
| bench_dead_clauses | 0.938 | 1.000 |
| bench_cascading | 0.885 | 0.962 |
| bench_stress | 0.838 | 0.843 |
| bench_subsumption_edge | 0.889 | 0.889 |

