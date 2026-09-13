# Frozen substitution compiler benchmark — C1 study

The original G2c substitution data and teacher are permanent reference artifacts.
The primary evaluation remains `data/raw/g2c_v1/substitute_6/test_extension.json`:
128 unseen complete instances, with the same inputs, labels and generation rule.
It has already been inspected. This adaptive tournament is explicitly exploratory.
We will not call repeated evaluation of this benchmark a fresh blinded test.

First complete five paired full-MaleCNS CE / degree-preserving rewired CE seeds.
Original real CE seed zero is reused transparently; seeds 1–4 are independent
replications. All new batch-one runs use the same 1,024 updates, example order,
learning rate, optimizer, clipping, I/O population seed and recurrent architecture.
At six checkpoints, preserve optimizer and sampling state and evaluate validation.
Final exact answers are generated autoregressively from prompts, with no true
answer prefix supplied. Record CE, KL, teacher agreement and zero-edge ablation.

C1 generates all training answers autoregressively using only prompts and the
frozen teacher. An independent audit compares those generated answers to original
labels, but never repairs them. If the labels coincide, hard-label CE and ground
truth CE are mathematically the same objective. Train a separate seed-zero C1
checkpoint and compare model/optimizer/RNG tensors. Exact equivalence avoids
redundant deterministic reruns; it is reported as equivalence, not five measured
hard-label replications. Otherwise complete hard-label seeds 1–4 as well.
This is a teacher-only training path, not evidence of improved sample efficiency
or direct translation of transformer parameters into synapses.

The batch-one screen fixes T=0.5,1,2,4,8 with alpha=0.5, a linear hard-to-KL
curriculum, and hidden alignment. All compiler objectives use generated training
answers; original labels are used only for baseline training and evaluation.
C0's historical ground-truth/KL mixture is retained; T=2 is run again with teacher
hard labels to establish a teacher-only version. C4 adds a training-only linear
projection from output-neuron states to final normalized teacher representations
at response-prediction positions. The auxiliary penalty is the mean squared
Euclidean distance between L2-normalized projected and teacher features, weight 1.
The projection is discarded for inference; no recurrent edge or query bypass is
introduced. Extra auxiliary parameters and teacher compute are disclosed rather
than claimed to have identical FLOPs to plain CE.

C5 matches pairwise cosine similarities across examples at response-prediction
positions. Its batch-two cohort has CE, hard-teacher and T=2 KD controls, all
512 updates / 1,024 examples. This separates a relational objective from a batch
size change. Recurrent state never carries across independent examples.

Select the batch-one screen's highest full-validation exact accuracy (CE, then
method name break ties), before evaluating any candidate on the reused final
test. Every candidate gets the same seed-zero budget and is reported, including
negative results. Replicate the selected method on seeds 1–4 regardless of sign.
Replicate C5's entire matched cohort on seeds 1–4 only if its validation exact
accuracy beats hard-teacher CE. Both selection rules are exploratory.

Report paired per-seed differences, mean and seed standard deviation, separating
historical seed zero from fresh seeds. The descriptive parity target is compiled
accuracy within two percentage points of CE; it is not a formal noninferiority
test. No harder task, new test, larger budget, different architecture or rescue
hyperparameter search is authorized by this protocol. All scientific choices and
source/data/teacher hashes are frozen before new training in `frozen_plan.json`.

The previous overnight run is preserved. Reversal is a secondary negative result;
rotation teacher qualification happened before the scheduling handoff, but no
rotation student or further task experiment is scheduled in this study.
