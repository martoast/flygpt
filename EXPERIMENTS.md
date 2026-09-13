# FlyGPT experiment registry

## G1 — Finite-Grammar Functional Encoding

G1 asks whether a finite learned mapping can be approximately encoded in a
fixed MaleCNS recurrent graph. The 512-update milestone is frozen at code
commit `da3713a`, with checkpoint and provenance hashes in
`results/malecns_v1/target/milestone_0512_archive.json`.
That preservation commit differs from the recorded training commit; the
historical dirty-worktree flag and source hashes remain part of provenance.

Validation sentences occur in training. The measured 0.851 nats/byte and
3.107 zero-edge loss support approximate functional encoding and recurrent
necessity on this task. They do not establish sequence generalization, exact
reproduction of the teacher, a benefit specific to distillation versus CE-only
training, or a benefit specific to biological wiring. G1 must retain this
label even if later experiments succeed. The target is now frozen at 1,024
updates: CE 0.395344 crossed the existing 0.425 near-teacher threshold, so no
2,048-update continuation was launched. Final G1 test CE is 0.403671 versus
teacher 0.341370 on 2,272 bytes; zero-edge CE is 3.467236. This test has the
same finite-grammar overlap limitation. The control replays the 512 and 1,024
stage budgets and then receives identical final-test evaluation.

## G2 — Finite-Grammar Compositional Generalization

G2 separates known lexical pieces from previously unseen combinations.
`python -m scripts.prepare_grammar_v2` constructs and audits the versioned
dataset without training a model or touching G1. Training has 2,048 unique
sentences. Validation and test each have three distinct 64-sentence subsets:

- Attribute: unseen subject size–color pairs, familiar subject–verb pairs.
- Relation: unseen subject animal–verb pairs, familiar size–color pairs.
- Combined: both pairs unseen simultaneously.

Pair membership is fixed by a cyclic partition before any G2 result exists.
Every lexical item occurs in training. Validation and test hold out different
pairs; neither their sentences nor their held-out pairs occur in training.
Object attributes stay in training support to isolate the subject-side test.
The JSONL records retain each sentence's generating factors. Leakage audits
check all split intersections, pair membership, uniqueness and lexical coverage.

This is a distribution-shift test. Training explicitly excludes some pairings;
the next word may be intrinsically ambiguous. A lower G1 loss is not a target
for G2. Low average byte loss alone could reflect predictable spelling and
function words while held-out composition fails.

The separate training/evaluation configuration is now frozen in
`configs/g2_v1.json`, before G2 training. It specifies a fresh 1,000-update
TinyGPT teacher (batch 8), followed by 512-update students (batch 1), all using
whole training sentences. Five paired seeds run real KD, rewired KD, real CE,
rewired CE, and GRU CE. No early stopping or hyperparameter search is scheduled.
Student AdamW learning rate is 0.001; teacher rate is 0.0003. Validation
trajectories use the first eight audited sentences per axis; final evaluations
use all 64 per axis. Runs are serial, with resumable student checkpoints and
disk-space checks. `python -m scripts.run_g2` waits for the G1 matched control
and its final test before launching this schedule.

Evaluation rules:

The n-gram reference is fitted to the complete training corpus and is not
matched to the student's sampled-byte budget. The unrestricted symbolic
grammar oracle is also an external reference, not a learned model. Initial
student validation scores provide normalized teacher-gap closure on the same
fixed validation sentences; this quantity is not evidence of a causal KD
benefit without comparison to the CE-only student.

- Train a fresh teacher on G2 training data only. Never distill logits from
  validation/test sentences; use the same prohibition for every student.
- Compare real and degree-preserving rewired graphs across at least five
  paired seeds, with the same data order, budgets and hyperparameter searches.
  Include CE-only students and conventional sequence baselines.
- Evaluate complete sentences with state reset per sentence, preserving the
  whole held-out composition. Do not reuse G1's 32-byte window evaluation,
  which can truncate the dependency being tested.
- Report CE/bits per byte, teacher KL/agreement, edge ablation, and separate
  conditional log loss on the held-out attribute and verb spans, alongside
  familiar-composition performance. Aggregate uncertainty over seeds and
  held-out pair groups, not individual bytes as independent observations.
- Report an explicit grammar-oracle reference and simple n-gram baseline to
  distinguish irreducible ambiguity from failed transfer. The teacher must
  itself show composition transfer before student imitation is interpreted
  as transfer of a generalizing function.
- Validation may guide a prespecified search. Test stays unopened for model
  evaluation until choices are frozen. Dataset leakage auditing is allowed.

G2 is running under the frozen protocol. A post hoc span-KL diagnostic was
added in a separate read-only script after observing real seed-zero step 256;
it changes neither the objective nor model evaluation's test lock. Its matched
256/512 results and teacher qualifications are documented in
`results/g2_v1/INTERPRETATION.md`. Lexical, length and structural splits are subsequent
separate tests, not claims already covered by this dataset. Truly unseen words
need an explicit learning/compositional rule; a byte vocabulary alone does not
provide their meaning. Length tests must preserve the target dependency across
longer contexts. Structural tests must hold out productions while documenting
which prerequisite productions remain observed.

## G2b — Teacher-qualified compositional transduction

The prospective G2b successor is documented in [G2B_PROTOCOL.md](G2B_PROTOCOL.md).
Its deterministic transduction dataset and teacher qualification gate are
specified before G2b training. No teacher has qualified and no G2b student
training has been enabled by a passing gate yet. G2's relation focus is a post hoc
exploratory analysis; attribute and combined outcomes remain reported.

## G3 — Natural-Language Modeling

Future work on a corpus with documented train/validation/test decontamination.
No G3 result exists. G1 success cannot be relabeled as G2 or G3 success.
