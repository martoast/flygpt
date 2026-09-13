# FlyGPT decision tree

Current priority: establish whether the paired CE topology gap replicates,
then transfer the fixed substitution function using teacher-derived training
signals. The current five-seed jobs, data order, optimizer, architecture and
1,024-update budget remain unchanged.

1. **Complete paired CE replication.** For every seed report
   `Delta_s = accuracy(real CE, s) - accuracy(rewired CE, s)` on the same
   128 substitution test instances. Show all five pairs and the fresh seeds
   1–4 separately from historical seed zero. Report mean, seed variability,
   range and direction consistency. A consistent positive difference supports
   a task- and budget-specific topology effect; mixed differences or large
   variability leave it unresolved. Five exploratory seeds do not establish
   universal biological superiority. This outcome does not gate the compiler
   question: function transfer may work even without biological advantage.

2. **Retain the frozen substitution benchmark.** Keep the existing teacher,
   dataset and final test, including negative results and zero-edge ablations.
   The final test has already been inspected; subsequent method comparisons
   remain exploratory. Do not use the 83.6% seed-zero result as a universal
   target: each compiler is compared with its paired CE reference, and the
   replicated CE distribution defines the attainable benchmark under this budget.

3. **Run the frozen compiler tournament after all CE pairs finish.** Test hard
   teacher labels, the prescribed temperature sweep, gradual KD, hidden-state
   alignment and relational alignment. Compiler training receives teacher-derived
   targets only. Original labels are reserved for the explicit CE controls and
   evaluation. C5 has its own batch-two controls. Selection uses validation,
   and the selected batch-one objective receives independent seed replication.
   All candidates and failures are reported. The existing within-two-percentage-
   points parity target is descriptive, not a formal noninferiority test.

   - If hard labels equal original labels and the independently trained model,
     optimizer and RNG match CE exactly, report teacher-only training equivalence.
     Do not call it faster compilation, compressed parameter translation or an
     independent measurement for seeds that were not rerun.
   - If a non-equivalent objective reaches comparable CE performance across
     paired seeds, investigate teacher-function transfer under that procedure.
   - If objectives remain worse or unstable, preserve the negative result and
     diagnose optimization/representation issues on this same function. Do not
     compensate by changing the test or advancing to language.

4. **Architecture independence comes next, conditional on compiler parity.**
   Freeze a separate matched protocol before new training: same teacher,
   substitution task and compiler into full MaleCNS, degree-rewired MaleCNS,
   random sparse, smaller and modular graphs. Keep ground-truth CE controls
   for every substrate to distinguish insufficient learnability/capacity from
   compiler failure. Match examples, optimizer/search opportunities, seeds and
   evaluation. Record neuron/edge/parameter counts and actual runtime; a smaller
   graph comparison is a capacity intervention, not an isolated topology test.
   Compare both absolute accuracy and compiled-minus-CE gaps. New graph families
   are not launched by the current compiler controller.

5. **Harder functions follow the architecture comparison.** Reversal, multiple
   deterministic tasks, then language. Each new function requires its own
   qualified teacher and frozen evaluation protocol before expensive students.

Execution is tracked in `COMPILER_REPORT.md`. The current controller already
finishes every CE pair before running any compiler objective; no training
change is needed for this priority order. Architecture-independence experiments
require their own frozen specifications once the compiler evidence is available.
