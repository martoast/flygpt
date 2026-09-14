# Compiler acquisition before saturation — design draft

Status: prospective design only, created while the five-seed compiler confirmation is running. Not a preregistration, executable configuration, or authorization to start another training queue. The current 30-model experiment, its frozen endpoints, test lock and stopping rule remain unchanged. Do not spend competing M4 training compute on this draft.

## Question

At a limited acquisition budget, does correctly matched teacher information improve unseen-input exact accuracy or response CE relative to teacher-hard supervision, and do internal representations outperform their matched shuffled controls?

Lower CE with equal exact accuracy means improved probabilistic predictive quality. It does not alone prove faster acquisition, calibration, new algorithmic capability, or a uniquely necessary hidden representation. Soft-logit transfer remains a full candidate method. Representation methods must beat both answers-only and shuffled correspondence on the claimed endpoint.

## Preferred next experiment: limit acquisition before changing the task

Keep the six v3 objectives, qualified teacher, substitution function, alphabet, sequence length, training examples, fixed MaleCNS topology, optimizer, batch eight and initialization rules unchanged. All methods run on one machine. Train each seed/method once to **256 updates**, saving the **128-update prefix** as well. Do not train separate replicas from scratch for each budget: the saved prefix permits a paired trajectory comparison without tripling training compute.

| Budget | Example presentations | Planned role |
|---|---:|---|
| 128 updates | 1,024 | Prespecified early acquisition diagnostic |
| 256 updates | 2,048 | Primary fixed-budget endpoint and stopping point |
| 512 updates | 4,096 | Current study's ceiling reference, not a newly matched arm |

The pilot teacher-hard validation accuracy at 256 was 69.92%; it reached 93.75% at 512. This motivates 256 as a development-informed choice. The desired 50–80% hard-baseline regime is **not** an eligibility criterion, a promise, or a reason to replace a seed. If the next baseline is unexpectedly near 0% or 100%, preserve that result rather than adjusting its budget. Do not pick a favorable budget from the new final test.

The current replication's 128/256 validation curves are already recorded. They can inform development and remain secondary under that study's existing protocol. Do not retrofit new primary intermediate-checkpoint final-test comparisons into the current confirmation. Its 512 results cannot be treated as a matched next-study budget arm, since seeds and test allocations differ. If a within-study 512 arm is later deemed essential, amend this draft and freeze that choice before any next-study training; then extend every trajectory identically to 512.

## Candidate endpoint and inference plan to freeze before launch

- Primary at 256: exact accuracy and response CE for B−A, C−D, E−F, C−A and E−A, with the same explicitly defined benefit directions and Holm family of ten comparisons as the current protocol. No selecting the winning endpoint after evaluation.
- Secondary: the 128 prefix, validation learning-curve area over 32–256, observed threshold crossings at 50% and 75% exact accuracy, full loss/gradient/state traces, teacher agreement and KL. Thresholds are design choices now, not fitted to the next cohort.
- Candidate validation schedule: every 32 updates from 32 through 256. Report crossing intervals and right censoring, not an interpolated exact crossing step. Count all validation overhead in end-to-end timing.
- Report examples presented separately from unique training examples. The fixed-group sampler revisits examples; fewer updates here tests acquisition/optimization efficiency, not learning from fewer distinct labeled examples.
- Report teacher inference, target/representation extraction, projection work and student training in standalone cost. Equal updates do not imply equal compute. A raw wall-clock advantage requires all compared methods on the same hardware with all preparation costs counted.
- Recommended next cohort: **10 new paired seeds**, six conditions each, separate from pilot and current replication. Fix the seed list before launch. This improves the small-N limitations of five seeds; it is not a guarantee of power. No pooling prior seeds or adaptive additions based on significance.
- Preserve zero-edge ablation at the primary endpoint. Maintain fixed groups, fixed deterministic derangements and the numerical non-invariance audit; failure stops the cohort without resampling.

Any eventual claim must name the endpoint: higher accuracy at 256 is a fixed-budget acquisition advantage; lower CE at 256 is a probabilistic-quality advantage. Neither by itself estimates how many updates would achieve the same quality. Threshold results can support that stronger efficiency claim only when they separate with their observation intervals and costs accounted for.

## Test budget and leakage protection

After all existing allocations, including the current locked confirmation, only **384** of the 4,096 substitution inputs remain unused. Proposed allocation: 256 next-study final-test inputs, leaving 128 untouched. This document does **not** select, inspect, or evaluate that test set. Before launch, deterministically select and hash the new split, audit it against every earlier allocation, and publish the full protocol and inputs. Reuse the original training/validation corpus; do not consume the unused pool for tuning. The final test remains closed until the complete next cohort and endpoint choices are frozen and all required checkpoints exist.

Ten seeds share a teacher, task and corpus; this remains conditional evidence about that setting, not ten independent tasks. The limited finite test domain constrains further fresh-test replications. A larger-domain benchmark will be needed for broader claims.

## Harder-function fallback: a separate development track

If the completed current study and development evidence show that restricted-budget substitution still gives little separation, move to a new task **before** freezing the next confirmatory cohort, not by modifying one in progress.

Prefer a deterministic composition of two noncommuting transformations, such as position permutation followed by a position-dependent symbol substitution. A fixed uniform substitution commutes with reversal, so merely reversing the order of those two operations does not create a meaningful transformation-order test. Define the functions and generate an exhaustive oracle before training.

Use a larger finite input domain and disjoint train/development/teacher-qualification/final-test instances by construction. Treat longer sequences, larger alphabets and held-out transformation compositions as separate named changes; do not vary all of them at once and attribute an effect to one feature. Training-budget sweeps and task selection belong only to development.

For any changed function/domain, train and qualify a fresh transformer on separate locked qualification inputs (target >=95% exact, ideally near 100%, in each primary category), compare n-gram/memorization and GRU baselines, and audit leakage before expensive MaleCNS training. A teacher that fails qualification cannot support a clean compilation test; retain the failure and return to development without consulting the final test.

## Immediate action

Finish the current frozen replication. Preserve and report CE, real-versus-shuffled contrasts and the already registered learning-curve metrics even if exact accuracy saturates. Then freeze one next protocol based on development evidence, publish it before training, and keep its conclusions separate from the current study. No additional models are launched by this document.
