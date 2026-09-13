# G2 interpretation checkpoint: real seed 0 at 256 updates

These are the fixed eight validation sentences per axis, not final test data.
No architecture, objective, optimizer, sampling, or 512-update budget changed.

| Model/reference | Attribute span CE | Relation span CE | Combined span CE |
|---|---:|---:|---:|
| Frozen TinyGPT teacher | 2.100339 | 1.176763 | 2.016162 |
| MaleCNS KD, step 256 | 2.557614 | 2.991643 | 2.768647 |
| Training-corpus n-gram (order 4) | 1.533844 | 1.586332 | 1.610255 |
| Unrestricted symbolic grammar oracle | 0.221807 | 0.213276 | 0.224048 |

All values are nats per selected target byte; lower is better. Selected spans
are the subject's color word and/or verb, including the trailing space. The
first byte of each word is included, using the full preceding sentence prefix.
The n-gram sees the complete training corpus, not the student's sampled-byte
budget. The oracle encodes the unrestricted grammar externally; it is not a
learned or matched baseline.

The teacher beats the n-gram on relation holdouts but not on attribute or
combined holdouts. Therefore a strong teacher that generalizes across all
three axes is not established. Failure to reproduce generalization absent in
the teacher would not isolate a failure of the compiler. Neither would it
establish that the substrate lacks capacity.

The student improves whole-sentence prediction from 128 to 256 updates while
attribute-span loss is nearly flat and relation/combined-span losses increase.
This is negative evidence for the desired compositional behavior so far,
not a new binary pass/fail rule imposed after seeing the results. G2's frozen
protocol specifies metrics and budgets rather than a numerical generalization
success cutoff. Final student, rewired, and CE-only comparisons remain pending.

These grammars permit multiple next words. The task tests probability assigned
to valid but unseen combinations; a single next word is not uniquely entailed
by the preceding phrase. Good spelling/syntax or ordinary byte accuracy can
mask poor probability on held-out lexical choices.

## Added diagnostic: span-restricted teacher KL

At the user's request, after inspecting the 256-update outcome, a separate
read-only process evaluates forward KL(teacher || student) at exactly the
same held-out target positions, summing over all 256 possible next-byte values.
It reports T=1 KL per selected byte, T=2 training-scaled KL, teacher/student
span CE and argmax agreement. It uses validation only. This is a **post hoc
diagnostic**, not a change to training, stopping or test access.

The process watches matched 256/512 checkpoints for every planned student
condition and seed. Earlier real-seed-zero weight snapshots were not retained,
so their span KL cannot be recovered from saved aggregate CE/KL alone. No
unmeasured historical values will be reconstructed or relabeled. Results and
checkpoint hashes are recorded in `span_diagnostics/`; available diagnostic
checkpoints are copied and hash-verified on the Seagate when mounted.

Lower span KL measures transfer of the teacher's predictive distribution.
It is useful even when CE changes little, but copying a teacher's errors is
also capable of lowering KL. A claim of transferred generalization requires
the teacher's ground-truth performance, the student's ground-truth performance,
and the matched CE-only/control results together.
