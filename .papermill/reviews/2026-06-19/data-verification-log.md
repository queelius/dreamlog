# Data Verification Log (2026-06-19)

Every Critical and Major finding, and every headline numeric claim, was cross-checked against the
committed experiment artifacts under `experiments/data/`, the registry, and the implementation.
This log records what was checked and the result.

## Headline numeric claims: manuscript vs committed data

| Claim (manuscript) | Source artifact | Committed value | Verdict |
|---|---|---|---|
| EX38 correlation r=0.52, p=0.014, n=22 | `data/ex38/runs/.../results.json` `aggregate.clause_compression_ratio_vs_recovery` | pearson_r=0.5177, pearson_p=0.0136, n_domains=22 | MATCH (rounds to 0.52/0.014) |
| EX38 grand-mean recovery 72.8% +/- 23.2% | same | grand_mean_recovery=0.7277, all 22 nonzero, prec 1.0 | MATCH |
| EX38 bits ratio (NOT used as headline) | same | pearson_r=0.4094, p=0.0585 (non-significant) | Paper correctly does NOT lean on this |
| EX25b crafting symbolic 53% rec / 59% prec / 5 rules | `data/ex25b/results.json` (5-run, sha 719ad2a) | recall 0.5263, prec 0.5882, rules 5.0 | MATCH (53%/59%/5) |
| EX25b crafting full 63% rec / 63% prec / 19 rules | same | recall 0.6316, prec 0.6316, rules 18.6 | MATCH (63%/63%/~19) |
| EX27 flux symbolic 100%/100%, raw-LLM 0% | registry `EX27_recursion.key_result` | symbolic 100% prec 100%, raw_llm 0% (TP=0/38) | MATCH |
| EX27 family ancestor raw-LLM 2.6% (1/38) | same | raw_llm 2.6% (TP=1/38) | MATCH |
| EX28 Sonnet grid (Table II) all cells | `data/ex28_sonnet/results.jsonl` | see cell map below | MATCH (every cell) |
| EX30 extraction crossover K*=5,3,3,2,2 (L=2,3,4,5,8) | `data/ex30/runs/.../summary.txt` | K*=5,3,3,2,2 exactly | MATCH |
| EX30 bits/rule -7.6,-15.8,-24.1,-32.6,-58.7 | same | slopes -7.6,-15.8,-24.1,-32.6,-58.7 | MATCH |
| EX39 cross-pred recall 54.2% +/- 22.4%, prec 100%, n=6 | registry `EX39_llm_multidomain.key_result` | full 54.2 +/- 22.4, prec 100, n=6 | MATCH; correctly flagged as stricter slice |
| EX36 k=1 amortization +5.66 -> -2.67 bits | registry `EX36_context_effect.key_result` | +5.66 (k=0 reject) -> -2.67 (k=1 accept), 8.33-bit discount | MATCH |

## EX28 Sonnet per-cell map (paper Table II)

```
cell                  symbolic  llm_only  full   raw_llm   (paper match)
within  canon (can_fly)  1.00     0.00     1.00    0.00     YES (world-knowledge double edge)
within  synth (glonk)    1.00     1.00     1.00    0.00     YES
recurse canon (ancestor) 1.00     1.00     1.00    1.00     YES (memorized cell)
recurse synth (tessik)   1.00     1.00     1.00    0.00     YES (not-memorization survives)
cross   canon (father)   0.00     1.00     1.00    1.00     YES
cross   synth (brulit)   0.00     1.00     1.00    0.00     YES (structural induction)
```
Within-synthetic proposal rate in data = 0.5333; paper says "proposal 0.53". MATCH.

## Implementation checks

- **Elias gamma decl cost** (`dl.py` `_symbol_decl_bits`): arity 0 -> 1 bit, arity 1/2 -> 3 bits,
  arity 3/4 -> 5 bits. Recomputed independently: identical. Matches Section sec:dl prose.
- **Bits-DL formula** (`dl.py` module docstring + code): 4-bit clause header, 2-bit type tag,
  log2(|F|)/log2(|C|)/log2(V_c) payloads, delta computed exactly with table-size repricing. Matches
  Section sec:dl exactly.
- **Operation I open-world subset gate** (`generators/closure.py` lines 72-78): requires
  `r_ext.issubset(C)`, coverage >= min_closure_coverage (tau default 0.5), base-size guard; carries
  `notes={"predicted_closure": frozenset(C)}`. Synthesized clauses identical to exact path. Confirms
  the paper's safety argument: the rule computes exactly TC(B), so it derives no pair outside the
  closure. MATCH.

## Build / format checks

- pdfinfo: 20 pages.
- Citations: 42 cited unique == 42 bib entries == 42 bbl bibitems. Zero unused, zero undefined.
- `\ref` resolution: 35 labels defined, 22 refs used, 0 dangling.
- Figures: all 3 PDFs valid, 1 page each, all referenced (Figs fig:extraction-crossover,
  fig:compression-recall, fig:opi-coverage).
- Log: 49 underfull + 2 overfull hboxes, all cosmetic. No undefined-reference warnings.

## Provenance discrepancy found (not a paper defect)

The registry `EX25b_novel_generalization.key_result` prose (50%/75%; 67%/80%/16) is STALE: it
disagrees with both the committed `data/ex25b/results.json` and the manuscript, which agree with
each other (53%/59%/5; 63%/63%/19). The paper matches its data artifact. Action: update the
registry prose (finding m6). The same stale-prose risk applies to EX08 (Op E sign, MJ1).

## Prior-review findings confirmed resolved

- Crafting-number cluster (2026-06-02 MJ2/MJ3): the manuscript now uses one consistent crafting
  figure (53%/63%) everywhere (abstract, intro, EX25b, EX26, discussion, conclusion), matching the
  committed 5-run artifact. Resolved in the paper.
- ellis2021dreamcoder author transposition (2026-06-02 MJ8): bib now reads "Morales, Lucas" and
  "Cary, Luc" (both correct). Resolved.
