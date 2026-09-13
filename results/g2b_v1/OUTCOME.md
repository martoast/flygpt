# G2b v1 outcome: teacher not qualified

All five TinyGPT seeds and all five approximately parameter-matched GRU seeds
completed their fixed 1,000-update training budgets. G2b failed its **validation
qualification gate in all six categories**. No connectome student was launched.
The separate qualification and final student-test splits were not evaluated.

Teacher greedy exact-response accuracy was **0/1,920** across the six validation
categories and five seeds. Saved predictions contain substantive word/content
errors, not just mismatched whitespace or stopping characters. GRU and the
other baselines also had zero exact-response accuracy. The teacher's response
CE was worse than the GRU's in every category:

| Category | Teacher CE | GRU CE |
|---|---:|---:|
| Attribute, swap | 0.6389 | 0.3963 |
| Attribute, roles | 0.5258 | 0.3692 |
| Relation, swap | 0.6120 | 0.3290 |
| Relation, roles | 0.5355 | 0.3286 |
| Combined, swap | 0.6293 | 0.3934 |
| Combined, roles | 0.5546 | 0.3879 |

These are response-only nats/byte, averaged over five training seeds using the
same validation cases. Every gate check failed: minimum exact accuracy,
accuracy margin, CE margin, and positive bootstrap lower bound. The detailed
gate evidence is preserved in `gate_validation.json` and `GATE_VALIDATION.md`.

This version did not establish a generalizing teacher function suitable for
the planned transfer experiment. It therefore gives no evidence that the
connectome compiler fails to transfer such a function, and does not establish
a representational limit of the MaleCNS topology. No rescue training, revised
threshold, or additional dataset selection was performed after the failed gate.

Separately, G2 exploratory seed zero closed without a distillation benefit on
its post hoc relation metric: CE-only MaleCNS reached span CE 2.1752 and teacher
KL 1.9614, compared with distilled MaleCNS CE 2.3972 and KL 2.1399. Rewired KD
reached CE 2.6177 and KL 2.3519. That one-seed comparison remains exploratory.
