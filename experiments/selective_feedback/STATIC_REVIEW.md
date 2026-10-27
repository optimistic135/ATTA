# Static review

Only source reading and Git diff checks. No Python, compile, CLI invocation,
unit test, training or experiment executed locally.

- Additive package; existing methods/results and user's untracked evidence document untouched.
- PACS P→A→C→S, batch32, D100, three seeds, three groups only.
- SP baseline delegates its formal path unchanged. Risk initialization is RNG-isolated.
- Source encoder/classifier remain fixed. Copied head only trains on legally returned truth.
- Query features, branch predictions and scores stored before return, not reconstructed
  after training. Supervised correctness labels created only from released events.
- Current masks/targets formed before fresh feedback is appended or modules trained.
- Original SP gate and _loss unmodified; current loss exists in a separate formal helper.
- Fresh-feedback validation images excluded from auxiliary masks. Local candidates use
  only previous returned truth. No oracle/prospective target labels enter learning.
- Source correction requires class disagreement, a positive reliability margin and
  confidence/consistency tests. Agreement does not independently imply reliability.
- Warmup/zero-selected cases fall back to historical SP student loss; module quota
  still spent. No claim of identical trajectory under fallback.
- Teacher/head and risk features detached from backbone; module steps cannot update
  online model. Current targets stop gradient; only student sees its auxiliary loss.
- True coefficient1 on all student steps. Historical/current nominal auxiliary sum
  stays at original SP coefficient. Empty sampled mask triggers history fallback.
- Last two positive-return slots allocated to head/risk; quota1 risk only. All
  original truth/pseudo RNG draws consumed, even for module slots; total checked.
- Current minibatch generators private. Probe hashing restored after diagnostic draw.
- Two new controls share feedback descriptors, risk training, head training, current
  draws and module budgets. Summary verifies hashes and counts; accepting flags may differ.
- Fixed schemas for all update kinds and target rows; None present for unused loss
  fields. No subset-only late CSV keys. Empty files have explicit headers.
- Evaluation truth read only in post-batch scoring hook, separate from risk memory.
  Target precision weighted by actual draws, domains taken from actual image IDs.
- Local TRUE/HISTORY/SELECTED copies use one snapshot and draws; state fingerprint
  unchanged. Future scores cannot affect learning. Complete windows only.
- Active local metric excludes zero-selection probes; HISTORY comparator separates
  old auxiliary benefit from current replacement. No accepted-subset superiority claim.
- Resume uses JSON-normalized defaults (including tuple windows), required output
  files and total student+head+risk budget. Incomplete runs archived, not deleted.
- Three-seed sample SD/valid n; paired stream differences, no event-as-seed inflation.
- Runtime compatibility, numerical trajectory equality and effectiveness require server validation.
