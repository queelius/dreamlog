# P3 Decision Diff: clauses vs bits DL modes

Git SHA: `6a64d8d` | Generated: 2026-06-18T11:26:43Z

## Scope

This report covers symbolic Operations A-F and I only. Operation G
(LLM-assisted compression) is EXCLUDED: no llm_client is passed to any
dreamer in this tool, so G never fires. The bits vs clauses comparison
is therefore complete only for the symbolic pipeline. G's priced
criterion (Task 4) is covered by tests/test_dl_bits.py unit tests.

---

## Per-Scenario Decision Tables

### ex25b_crafting

Decisions: 5 (clauses) / 5 (bits) | Compression ratio: 0.919 (clauses) / 0.925 (bits) | Flips: 1 | Cascade: 0 clauses-only, 0 bits-only

| kind | delta_clauses | delta_bits | decision_clauses | decision_bits | FLIPPED | note |
|------|--------------|------------|-----------------|---------------|---------|------|
| generalization | -1 | +15.05 | accepted | rejected | YES | both |
| generalization | -2 | -45.91 | accepted | accepted |  | both |
| generalization | -5 | -56.58 | accepted | accepted |  | both |
| generalization | -2 | -18.99 | accepted | accepted |  | both |
| generalization | -4 | -40.17 | accepted | accepted |  | both |

### ex28_can_fly_canonical

Decisions: 1 (clauses) / 1 (bits) | Compression ratio: 0.786 (clauses) / 0.786 (bits) | Flips: 0 | Cascade: 0 clauses-only, 0 bits-only

| kind | delta_clauses | delta_bits | decision_clauses | decision_bits | FLIPPED | note |
|------|--------------|------------|-----------------|---------------|---------|------|
| generalization | -3 | -6.00 | accepted | accepted |  | both |

### ex28_glonk_invented

Decisions: 1 (clauses) / 1 (bits) | Compression ratio: 0.786 (clauses) / 0.786 (bits) | Flips: 0 | Cascade: 0 clauses-only, 0 bits-only

| kind | delta_clauses | delta_bits | decision_clauses | decision_bits | FLIPPED | note |
|------|--------------|------------|-----------------|---------------|---------|------|
| generalization | -3 | -6.00 | accepted | accepted |  | both |

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

Decisions: 4 (clauses) / 4 (bits) | Compression ratio: 0.907 (clauses) / 0.907 (bits) | Flips: 2 | Cascade: 0 clauses-only, 0 bits-only

| kind | delta_clauses | delta_bits | decision_clauses | decision_bits | FLIPPED | note |
|------|--------------|------------|-----------------|---------------|---------|------|
| extraction | +1 | +3.19 | accepted | rejected | YES | both |
| invention | -1 | +22.24 | accepted | rejected | YES | both |
| generalization | -3 | -34.14 | accepted | accepted |  | both |
| pruning | -1 | -20.04 | accepted | accepted |  | both |

### bench_transitive_closures

Decisions: 1 (clauses) / 1 (bits) | Compression ratio: 0.909 (clauses) / 0.909 (bits) | Flips: 0 | Cascade: 0 clauses-only, 0 bits-only

| kind | delta_clauses | delta_bits | decision_clauses | decision_bits | FLIPPED | note |
|------|--------------|------------|-----------------|---------------|---------|------|
| invention | -2 | -14.89 | accepted | accepted |  | both |

### bench_body_patterns

Decisions: 1 (clauses) / 1 (bits) | Compression ratio: 1.091 (clauses) / 1.091 (bits) | Flips: 0 | Cascade: 0 clauses-only, 0 bits-only

| kind | delta_clauses | delta_bits | decision_clauses | decision_bits | FLIPPED | note |
|------|--------------|------------|-----------------|---------------|---------|------|
| extraction | +1 | -13.23 | accepted | accepted |  | both |

### bench_dead_clauses

Decisions: 1 (clauses) / 1 (bits) | Compression ratio: 0.938 (clauses) / 1.000 (bits) | Flips: 1 | Cascade: 0 clauses-only, 0 bits-only

| kind | delta_clauses | delta_bits | decision_clauses | decision_bits | FLIPPED | note |
|------|--------------|------------|-----------------|---------------|---------|------|
| generalization | -1 | +5.59 | accepted | rejected | YES | both |

### bench_cascading

Decisions: 4 (clauses) / 4 (bits) | Compression ratio: 0.885 (clauses) / 0.885 (bits) | Flips: 2 | Cascade: 0 clauses-only, 0 bits-only

| kind | delta_clauses | delta_bits | decision_clauses | decision_bits | FLIPPED | note |
|------|--------------|------------|-----------------|---------------|---------|------|
| extraction | +1 | +2.62 | accepted | rejected | YES | both |
| invention | -1 | +15.02 | accepted | rejected | YES | both |
| generalization | -2 | -7.69 | accepted | accepted |  | both |
| pruning | -1 | -18.63 | accepted | accepted |  | both |

### bench_stress

Decisions: 3 (clauses) / 3 (bits) | Compression ratio: 0.838 (clauses) / 0.843 (bits) | Flips: 2 | Cascade: 0 clauses-only, 0 bits-only

| kind | delta_clauses | delta_bits | decision_clauses | decision_bits | FLIPPED | note |
|------|--------------|------------|-----------------|---------------|---------|------|
| extraction | +1 | +17.16 | accepted | rejected | YES | both |
| invention | -2 | +14.57 | accepted | rejected | YES | both |
| generalization | -29 | -734.14 | accepted | accepted |  | both |

### bench_subsumption_edge

Decisions: 1 (clauses) / 1 (bits) | Compression ratio: 0.889 (clauses) / 0.889 (bits) | Flips: 0 | Cascade: 0 clauses-only, 0 bits-only

| kind | delta_clauses | delta_bits | decision_clauses | decision_bits | FLIPPED | note |
|------|--------------|------------|-----------------|---------------|---------|------|
| subsumption | -1 | -45.77 | accepted | accepted |  | both |

---

## Totals

Total scenarios: 15
Total flips (same key, different decision): 8
Total clauses-only cascade rows: 0
Total bits-only cascade rows: 0

### Flips by operation kind

| kind | flip count |
|------|-----------|
| extraction | 3 |
| invention | 3 |
| generalization | 2 |

### Compression ratios per mode per scenario

| scenario | ratio_clauses | ratio_bits |
|----------|--------------|-----------|
| ex25b_crafting | 0.919 | 0.925 |
| ex28_can_fly_canonical | 0.786 | 0.786 |
| ex28_glonk_invented | 0.786 | 0.786 |
| ex28_ancestor_canonical | 0.278 | 0.278 |
| ex28_tessik_invented | 0.278 | 0.278 |
| ex28_father_canonical | 1.000 | 1.000 |
| ex28_brulit_invented | 1.000 | 1.000 |
| bench_family_small | 1.000 | 1.000 |
| bench_family_with_guards | 0.907 | 0.907 |
| bench_transitive_closures | 0.909 | 0.909 |
| bench_body_patterns | 1.091 | 1.091 |
| bench_dead_clauses | 0.938 | 1.000 |
| bench_cascading | 0.885 | 0.885 |
| bench_stress | 0.838 | 0.843 |
| bench_subsumption_edge | 0.889 | 0.889 |

