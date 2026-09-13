# G2b — Teacher-qualified compositional transduction

This is a prospective successor to G2, not a repair to its active run. No G2b
model has been trained. The candidate dataset is generated and audited by
`python -m scripts.prepare_g2b`: 8,192 training prompt/answer pairs and 128
pairs per held-out axis in each of validation, qualification, and test.
Numerical gates are specified in
`configs/g2b_qualification_v1.json` before seeing G2b outcomes.

## Why change the task?

G2 assigns probability to unseen combinations in a generative grammar that
deliberately excludes those pairings during training. Multiple continuations
are valid; training does not uniquely determine that missing pairings should
be admitted. Its teacher does not outperform the n-gram on all axes.

G2b will instead provide a sentence as input and request a deterministic
transformation. The answer is fully specified by the input and operation.
For example:

```
swap|the small red dog follows the large blue cat.
>the large blue cat follows the small red dog.
```

`swap` exchanges the explicit subject and object noun phrases while retaining
the verb. It is a syntactic operation, not a claim that the reversed event is
true. `roles` emits `subject | verb | object` from the same input. Both
operations occur in training; held-out lexical compositions must never occur
in a training prompt or answer. Evaluation scores response bytes only and
also requires exact greedy autoregressive answers, so syntax and copied
ground-truth response prefixes cannot dominate the success criterion.

## Separate four roles for data

The dataset uses a larger cyclic pair partition than G2: eight lexical values per
factor, leaving disjoint pair sets for training, validation, teacher
qualification, and final student test. Every atomic lexical item and every
operation must occur in training. Audit both prompts and answers for verbatim
and pair leakage, and retain generating factors and response masks.

Validation screens the teacher. A separate teacher-qualification split checks
the gate once, only after validation passes. Neither may be used to report
student test generalization. Final student-test composition pairs remain
separate and unseen by teacher qualification. If either gate fails, no
MaleCNS training is launched under this version. Dataset implementation and
audits precede any model run; the G2 dataset is not reused as a G2b test set.

## Qualify the teacher before spending on connectome training

Train five teacher seeds and five approximately parameter-matched GRU seeds
with the same fixed teacher-stage budget, optimizer settings and data order.
Include training-only n-grams and nearest-training-prompt answer retrieval.
Report all six axis-by-operation categories separately.

For every category, require mean teacher exact-response accuracy at least
80%, at least 10 percentage points above the strongest baseline, and at least
20% lower response CE than the strongest probabilistic baseline. Require a
positive lower bound on the paired seed/composition bootstrap accuracy
margin. Report all seed outcomes and group-level uncertainty; do not select
the best teacher seed. Retrieval is evaluated by accuracy, since its exact
stored answer does not define a calibrated probabilistic language model.

These are stringent prospective gates, not evidence that this candidate task
will pass. A simple RNN may match the transformer, in which case the teacher
does not qualify under this protocol. No claim of a uniquely transformer
function follows even if it passes. The purpose is to establish a clear
generalization target before attempting transfer.

After qualification, freeze a separate matched student protocol before
training students. Use the full MaleCNS topology, rewiring and CE-only controls;
evaluate response CE, exact response accuracy and teacher KL on identical
target positions. G1 and G2 remain separately reported experiments.
