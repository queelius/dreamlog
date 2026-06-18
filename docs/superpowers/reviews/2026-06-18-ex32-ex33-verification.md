# EX32 / EX33 Independent Rigor Audit

Date: 2026-06-18
Auditor: Claude Code (claude-sonnet-4-6), commissioned by Alex Towell
Experiments: EX32 (noise filter) and EX33 (compression predicts generalization)
Git commit under test: 1c9c8ba (master, dirty -- uncommitted work in tree)
Reproducing runs: EX32 new run sha256 e2951509... (byte-identical to committed);
                  EX33 new run sha256 787dc90e... (byte-identical to committed)


## Executive summary

Both experiments reproduce byte-identically. The headline numbers are real. However,
both share the same underlying confound: the bits delta for any proposal is priced
against the GLOBAL symbol table at the moment the gate scores it, not against an
isolated symbol table for that group alone. In EX32 this creates an ordering
dependency that scopes the claim; in EX33 it causes cross-group acceptance in
subfamily C that makes "bits is more selective than clauses for CL_ONLY groups"
hold only for the subfamily-B subfamily, not across the full 30-domain family.

EX32 verdict:  CONFIRMED-WITH-SCOPE
EX33 verdict:  CONFIRMED-WITH-SCOPE


---

## EX32 audit: bits-mode noise filter

### Claim

Bits mode eliminates all spurious generalizations (precision 1.00 at all noise
levels L=0..5) while clause count degrades to precision 0.17 at L=5.

### Reproduced numbers

Sweep table, reproduced exactly from the committed run:

  L   f      noise | CLAUSES true/spur/prec | BITS true/spur/prec
  0   0.000  0     | Y / 0 / 1.00          | Y / 0 / 1.00
  1   0.250  8     | Y / 1 / 0.50          | Y / 0 / 1.00
  2   0.400  16    | Y / 2 / 0.33          | Y / 0 / 1.00
  3   0.500  24    | Y / 3 / 0.25          | Y / 0 / 1.00
  4   0.571  32    | Y / 4 / 0.20          | Y / 0 / 1.00
  5   0.625  40    | Y / 5 / 0.17          | Y / 0 / 1.00

Bits deltas for noise proposals (rejected by bits):
  L=1: +10.90   L=2: +8.03 (x2)   L=3: +6.28 (x3)
  L=4: +5.06 (x4)   L=5: +4.14 (x5)

These numbers match the committed results.json byte-for-byte (sha256 e2951509).


### The ordering confound

EX32 inserts noise facts BEFORE the true-pattern facts (student/grade). The script
acknowledges this explicitly in its docstring: "noise facts are inserted BEFORE the
student/grade facts so Op C evaluates them while the symbol table is still small."

To bound this, I constructed a throwaway script that reverses insertion order
(true pattern FIRST, noise groups AFTER) and ran it. Results:

  L=1: bits mode accepted spurious=1, noise accepted delta_bits = -5.04
  L=2: bits mode accepted spurious=2, noise accepted delta_bits ~ -5.80
  L=3: bits mode accepted spurious=3, noise accepted delta_bits ~ -6.40
  L=4: bits mode accepted spurious=4, noise accepted delta_bits ~ -6.89
  L=5: bits mode accepted spurious=5, noise accepted delta_bits ~ -7.32

Under true-first ordering the bits filter completely erodes: at every L >= 1,
the noise groups are ACCEPTED by bits mode (delta_bits < 0), not rejected.

The mechanism is the global-DL amortization effect. Op C processes groups in
insertion order. When the true pattern (grade/student) is processed first and
the dream accepts it, it adds 'not' and the exception functor to the symbol
table. This raises log2(|F|) for ALL subsequent groups, making each new
functor+exception declaration cheaper relative to the per-occurrence payload
savings. The crossover analysis shows:

  n_good student-fact pairs needed before noise delta_bits flips sign:
    n_good=0  (no grade rule): delta_bits = +5.66  (REJECTED)
    n_good=2  (no grade rule): delta_bits = +3.51  (REJECTED)
    n_good=4  (grade rule): delta_bits = -3.66  (ACCEPTED)
    n_good=6  (grade rule): delta_bits = -4.18  (ACCEPTED)
    n_good=12 (grade rule): delta_bits = -5.04  (ACCEPTED)

The crossover occurs precisely when n_good >= 4 -- the minimum needed for
Op C to accept the true pattern (building the grade generalization rule).
Once the grade rule is in the KB, 'not' and exception_grade are amortized,
and the CL_ONLY-sized noise groups become cheaper.


### Precise scope of the EX32 claim

The bits-filter result is VALID under the condition that noise groups are
evaluated by Op C before the true-pattern generalization has been accepted.
This is the noise-first insertion order used in EX32.

The claim does NOT hold when any "good" generalization (n_pass >= 4, n_exc=0)
fires before the noise groups are evaluated, because the resulting 'not' and
exception functor declarations amortize the cost of subsequent CL_ONLY proposals.

For realistic KBs, the ordering is typically determined by insertion order of
facts, which is under the user's or program's control. In the EX32 scenario
(a polluted learning stream), the ordering assumption is plausible: noise arrives
interleaved with data, and the dream runs on the current KB regardless. But in a
scenario where a large, compressible true pattern precedes noise, the filter
dissolves.

EX32's noise design -- fresh, disjoint predicates per noise group -- is also
a clean-room construction that avoids the second confound (cross-group exception
inflation that would cause clause-count mode to reject too). Real noise is
unlikely to be this neatly isolated.


### EX32 verdict: CONFIRMED-WITH-SCOPE

The claim holds as stated for the noise-first, isolated-predicates design in EX32.
The scope caveat is: the filter requires that no prior generalization has amortized
the 'not' and exception-functor declarations in the global symbol table. Quantified
bound: a GOOD pattern with n_pass >= 4 firing first is sufficient to flip the noise
group's delta_bits from +5 to -5 (a 10-bit swing), dissolving the filter entirely.

Paper recommendation: state the noise-filter claim with an explicit scope note --
"under the condition that no prior generalization in the dream has amortized the
exception infrastructure in the shared symbol table." Do not present this as a
general structural property of bits mode without that qualifier.


---

## EX33 audit: compression predicts generalization

### Claim

Across a family of 30 synthetic domains, Spearman correlation between compression
and recovery is 0.59-0.92; bits is the stronger predictor; mean recovery 53.8%;
range 0-100%.

### Reproduced numbers

All numbers match the committed results.json byte-for-byte (sha256 787dc90e).

Recovery statistics (N=60 domain x mode pairs):
  Min=0.0%  Max=100.0%  Mean=53.8%  Nonzero=42/60  Partial(0<r<1)=21

Correlations (bits mode, N=30):
  bits_saved vs recovery:      Spearman 0.908  Pearson 0.840
  clauses_saved vs recovery:   Spearman 0.918  Pearson 0.896

Correlations (clauses mode, N=30):
  bits_saved vs recovery:      Spearman 0.630  Pearson 0.511
  clauses_saved vs recovery:   Spearman 0.694  Pearson 0.645


### Recovery is genuine (not an artifact)

Verification confirms that in D02 (Subfamily A, n_good=1), 25% recovery comes
from a RULE that derives held-out new-entity facts:
  (prop0 X v0) :- (cat0 X), (not (exception_prop0_v0_cat0 X)).
  -> DERIVED: (prop0 ng0_0 v0)  (from rule, not leftover ground fact)

In D12 (Subfamily B, n_cl=1, n_inc=3, bits mode), 0% recovery is confirmed:
bits mode fires no rules, and new-entity facts are not derivable.

The new-entity protocol is sound: new entities are added only by their guard facts
(not their derived facts), and recovery counts only facts derived via rules. The
held-out facts are genuinely absent from the KB during dreaming.


### The subfamily-C contamination

The script design for subfamily C states: "bits recovery = n_good / (n_good + n_cl_only)."
This prediction is INCORRECT for any domain with n_good > 0 and n_cl_only > 0.

Actual vs expected bits-mode recovery for subfamily C:

  D20  ng=0 nc=4  expected=0%    actual=0%    OK
  D21  ng=0 nc=3  expected=0%    actual=0%    OK
  D22  ng=1 nc=3  expected=25%   actual=100%  CONTAMINATED
  D23  ng=1 nc=2  expected=33%   actual=100%  CONTAMINATED
  D24  ng=2 nc=2  expected=50%   actual=100%  CONTAMINATED
  D25  ng=2 nc=1  expected=67%   actual=100%  CONTAMINATED
  D26  ng=3 nc=1  expected=75%   actual=100%  CONTAMINATED
  D27  ng=3 nc=3  expected=50%   actual=100%  CONTAMINATED
  D28  ng=4 nc=2  expected=67%   actual=100%  CONTAMINATED
  D29  ng=4 nc=0  expected=100%  actual=100%  OK

8 out of 10 subfamily-C domains are contaminated. The mechanism (confirmed by
throwaway script) is identical to the EX32 ordering confound: when the GOOD
group's dream fires first (Op C processes groups in insertion order; GOOD
groups are inserted before CL_ONLY groups in build_domain_c), 'not' and the
exception functor are added to the symbol table. The CL_ONLY group's
generalization proposal is then scored against the enriched table.

Measured: CL_ONLY group in isolation -> delta_bits = +5.71 (REJECTED).
Same CL_ONLY group after GOOD group compressed first -> delta_bits = -3.66 (ACCEPTED).

The comment in EX33's build_domain_c docstring ("Groups use unique predicates so no
symbol-table pollution") refers to predicate-NAME pollution (no shared functors between
groups), not symbol-TABLE amortization. The amortization happens through shared
symbol-table ENTRIES for the 'not' functor and the arity-1 class of exception
functors -- these are added once and reduce the per-occurrence pricing for ALL
subsequent proposals.

Note also: CL_ONLY groups do NOT self-amortize within a pure-CL_ONLY KB at the
small group counts used in EX33 (n_cl <= 4). At n_cl=5 bits still rejects all
(min delta_bits = +1.80). However, self-amortization does occur at large scale
(n_cl=10: all accepted, min delta_bits = -7.91). This is an additional bound not
tested in EX33 (the family stays within n_cl <= 4).


### Impact of contamination on the reported correlations

The 8 contaminated subfamily-C domains all have bits_saved > 0 and recovery = 100%
(instead of the expected 25-75%). This inflates the correlation between bits_saved
and recovery in bits mode.

Correlations computed with and without subfamily C:

  All 30 domains (bits mode):   sp(bits_saved, rec) = 0.908
  Subfamily A+B only (N=20):    sp(bits_saved, rec) = 0.984
  Subfamily A+B only (N=20):    sp(clauses_saved, rec) = 0.987

The contamination DOES NOT REDUCE the correlation; it changes the interpretation.
In A+B-only (uncontaminated), both metrics are nearly perfect predictors (0.98+)
because subfamily A gives a graded range (GOOD groups predict linearly) and
subfamily B gives a bimodal split (bits: all-0; clauses: graded).

The reported 0.908 for bits_saved in bits mode across all 30 domains is driven by:
  (1) Subfamily A: genuine graded prediction (bits_saved rises with n_good -> recovery rises)
  (2) Subfamily B: all-zero anchor (bits mode gives 0 compression and 0 recovery)
  (3) Subfamily C: contaminated 0/100% split (bits_saved>0 iff n_good>0, recovery always 100% when n_good>0)

This means the correlation for bits mode is high for two different reasons in
two different subfamilies, which the single reported Spearman across all 30 conflates.


### "Bits is the better predictor" claim

The EX33 summary states "bits and clause-count predict similarly" (the script verdict).
This is defensible because the Spearman gap is small (0.908 vs 0.918). However:

- In bits mode, BOTH bits_saved and clauses_saved are high predictors (0.908 / 0.918).
  This is because subfamily C now has recovery=100% for all n_good>0 domains in
  bits mode, making it look like "more compression = more recovery" regardless of metric.
- The intended demonstration -- that bits_saved uniquely predicts recovery when
  clauses would disagree (the CL_ONLY groups) -- is only cleanly shown in
  subfamily B, where bits mode correctly shows 0 recovery (no compression) and
  clauses mode shows non-zero recovery (spurious compression accepted).


### Subfamily B is the clean family

Subfamily B (CL_ONLY + INCOMP, n_good=0) shows clean mode divergence:
  Bits mode:    recovery=0% for all 10 domains (bits never accepts CL_ONLY)
  Clauses mode: recovery varies 0-100%, positively correlated with n_cl_only/(n_cl_only+n_incomp)

This is the honest demonstration of the claim: clauses mode accepts CL_ONLY
generalizations that lead to real recovery; bits mode correctly rejects them,
producing zero recovery. The "bits is more selective" claim is unambiguously
demonstrated in subfamily B alone (N=10 domains per mode).

Subfamily A shows both modes agree (as designed).
Subfamily C shows both modes agree (accidentally, due to contamination).


### EX33 verdict: CONFIRMED-WITH-SCOPE

The headline numbers reproduce exactly. Recovery is genuine (from rules, not
leftover facts). The Spearman correlations are real. However:

1. Subfamily C is contaminated: bits mode incorrectly accepts CL_ONLY groups
   in the presence of GOOD groups (the GOOD group amortizes 'not' and exception
   functors), so 8/10 subfamily-C domains show bits recovery = 100% instead of
   the predicted n_good/(n_good+n_cl_only). The EX33 note in the script about
   "symbol-table pollution only occurs when OTHER groups have been compressed first"
   is exactly what happens here -- the script's own design triggers the effect.

2. The clean demonstration of "bits more selective than clauses" lives in subfamily
   B, not across all 30 domains. Subfamily A is a noise-free baseline; subfamily C
   accidentally confirms the claim but for the wrong reason.

3. The reported correlations (0.59-0.92) are real and significant, but the
   interpretation -- that bits_saved is a better predictor than clauses_saved --
   is not supported by the between-modes comparison: both metrics are equally
   strong predictors when evaluated within bits mode across the full 30-domain
   family (0.908 vs 0.918). The apparent advantage of bits_saved over clauses_saved
   appears only in the clauses-mode correlations (0.630 vs 0.694), where clauses_saved
   has a modest edge.

Paper recommendation: restrict the "bits more selective" claim to subfamily B
(which is the clean, uncontaminated family). Note that subfamily C demonstrates
a different phenomenon: GOOD-group amortization makes bits mode accept CL_ONLY
groups it would reject in isolation. Frame this as a positive finding (the code
compresses more when structurally similar groups share amortized infrastructure) rather than a confound, but do not use it to
argue for mode divergence. Separately note the subfamily-B result as the clean
evidence that bits-mode precision outperforms clauses-mode on low-quality
(N=3+exception) generalizations.


---

## Cross-check: the shared global-DL amortization effect

Both confounds are instances of the same phenomenon: the bits DL for a proposal
is priced against the GLOBAL symbol table at the moment the gate scores it, not
against a local or isolated table for that proposal's predicates.

When any previously accepted proposal adds 'not' and an exception functor to the
symbol table, log2(|F|) for the post-change table is higher, which means:
  - Each existing functor occurrence is slightly more expensive (larger denominator).
  - But a NEW exception functor (arity 1) is amortized across a richer dictionary
    -- the per-occurrence cost of log2(|F|+1) is only marginally higher than
    log2(|F|), while the declaration cost (_symbol_decl_bits(1) = 3 bits, fixed)
    is shared.

This is correct behavior for a proper two-part MDL code (the dictionary cost is
amortized globally), but it means single-group experiments like EX32's L=1
analysis and EX33's "CL_ONLY in isolation" analysis do not predict behavior when
a global symbol table is pre-populated by prior compressions.

The EX32 framing (noise-first, lean symbol table) is a valid scenario; it should
be presented as "the filter holds when evaluated before any other generalization
has populated the symbol table." The EX33 design flaw is that the described
expected behavior of subfamily C depended on per-group isolation that the global
DL code does not provide.


---

## Reproduced numbers vs committed results

EX32:
  Committed sha256:  e2951509c155790732c3246f3d9e78a5bc151a3ccefeed1dad0a6f188dd8b036
  Re-run sha256:     e2951509c155790732c3246f3d9e78a5bc151a3ccefeed1dad0a6f188dd8b036
  Match: YES (byte-identical)

EX33:
  Committed sha256:  787dc90e48b7d38e2bf0ed1f8458268b8213785e098dcc78949867e126debf11
  Re-run sha256:     787dc90e48b7d38e2bf0ed1f8458268b8213785e098dcc78949867e126debf11
  Match: YES (byte-identical)

Both experiments are fully deterministic (no RNG) and reproduce under the same
Python environment (CPython 3.12.3, dreamlog 0.9.0, scipy 1.16.1).


---

## One-line paper recommendations

EX32: Add a scope qualifier: "the bits filter holds when noise groups are scored
against a lean symbol table, before any prior generalization has amortized the
exception infrastructure; once a mature pattern fires, the filter dissolves."

EX33: Restrict the "bits more selective" finding to subfamily B (10 domains,
clean mode divergence); note that subfamily C shows inadvertent amortization
cross-talk (the GOOD group inflates the symbol table so CL_ONLY groups are
accepted), and frame it separately as a co-compression effect rather than as
evidence for mode divergence.
