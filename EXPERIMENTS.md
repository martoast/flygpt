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
label even if later experiments succeed. Its ongoing continuation and primary
matched rewired control remain unchanged. The earlier adaptive controller can
extend through 2,048 updates; the control replays the completed stage budgets.

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

Before launching G2, freeze a separate training/evaluation configuration:

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

G2 is prepared, not run. Lexical, length and structural splits are subsequent
separate tests, not claims already covered by this dataset. Truly unseen words
need an explicit learning/compositional rule; a byte vocabulary alone does not
provide their meaning. Length tests must preserve the target dependency across
longer contexts. Structural tests must hold out productions while documenting
which prerequisite productions remain observed.

## G3 — Natural-Language Modeling

Future work on a corpus with documented train/validation/test decontamination.
No G3 result exists. G1 success cannot be relabeled as G2 or G3 success.
