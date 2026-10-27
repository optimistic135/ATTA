# Static review: oracle-target isolation

No Python, CLI help, compile, tests or experiments were executed. Source reads,
call-path/schema review and Git diff checks only.

- Additive package; previous current-target and feedback studies remain unchanged.
- D100, PACS P→A→C→S, bs32, three seeds, five trigger windows only.
- Original SP formal path inherited; no oracle predictions or scores flow into it.
- Base three candidates fork before fresh labels enter formal truth memory.
- Oracle packets retain exactly the same CPU snapshot, old-truth draw, image draw,
  source/repair probability vectors and original masks.
- Current evaluation labels become available only in the post-batch scoring hook.
  They are read exclusively for target scoring and explicit oracle copies; never
  for ordinary candidates, repair-head training, query selection or SP gates.
- Oracle candidates train from saved pre-update snapshot, not the post-formal model.
  All five use empty SGD state, identical learning rate, momentum and RNG reset.
- Swap argmax/true-class probabilities from immutable q into a clone; already
  correct vectors unchanged; masks never recomputed using evaluation labels.
- Assert sorted probability values unchanged and selected argmax equals truth.
- Main model and repair head fingerprints unchanged after oracle computation.
  Isolated randomness restores the post-formal RNG after diagnostic construction.
- Oracle models inserted into the same active-probe dictionary before future
  scoring. Frozen scoring begins next batch, never trigger batch.
- All correctness/pair fields generated consistently from ARMS/PAIRS. Oracle
  changed-selected fields initialized on every event before later assignment.
  No heterogeneous late CSV keys; empty files retain fixed header lists.
- Zero masks imply zero auxiliary loss; zero corrected selected count logged.
  Do not treat these events as evidence against useful corrected targets.
- Complete windows only; future domain and source/repair acceptance flags preserved.
  No oracle acceptance decision exists.
- Sample-weighted within seed; paired correction gains and oracle-vs-TRUE gains
  aggregated across seeds with sample SD and valid n, not across events as seeds.
- Resume requires its own matching manifest/defaults/method. Required oracle audit
  output included; incomplete seeds archived and restarted, complete ones skipped.
- Existing untracked outputs/ untouched. Framework execution, trajectory equality
  and oracle runtime behavior remain unverified until server execution.

Interpretation limits: correct top-1 is not a perfect conditional distribution;
permutation changes gradients and does not enforce equal gradient norm. Oracle is
nondeployable and not a mathematical upper bound. Negative one-step effects do
not exclude different coefficients, parameters, sample plans or longer training.
