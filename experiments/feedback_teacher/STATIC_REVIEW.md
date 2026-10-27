# Static review

Only source reading and Git diff checks performed. No Python invocation, compile,
CLI help, unit test, training, or experiment was run locally.

- Additive package: original sp_atta and delayed_feedback are unchanged.
- D100 / Photo → Art → Cartoon → Sketch / bs32 / three seeds / nine jobs only.
- Original trace checks source checkpoint hashes, stream shape, IDs, clocks and quotas.
- Predictions precede oracle release; pending labels never enter any loss.
- The repaired head is a copy; original source encoder/head remain unchanged.
- Query-time teacher predictions are stored before truth access, scored only on return.
- Return CE compares both teachers before any fresh-label training.
- Fresh evidence changes target selection next batch; current fresh images are excluded
  from shadow auxiliary draws. Candidate truth uses only previously returned labels.
- Cached targets predate the current head update, which runs last and affects next batch.
- Both Current groups train their repair head at identical reserved steps and draws;
  Current-Source never uses that head for student targets.
- Teacher loss only differentiates through the copied linear head; encoder features
  and initial head anchoring tensors are detached.
- Each formal quota is consumed exactly once: student + head. All original random
  truth/pseudo index draws are consumed, including reserved head iterations.
- Window sampling uses a separate CPU generator and no labels/domain IDs; fresh IDs
  only exclude held-out images. No domain-boundary reset.
- Masks, zero reliable samples, no returns, no previous truth and zero quotas fall
  back safely. Source teacher is always available; no evidence means no repaired targets.
- Gates use isolated randomness and discarded model copies; model fingerprints checked.
- Current-source vs current-feedback permits different shadow eligibility/costs, which
  are logged; no matched FLOPs or matched candidate-count claim.
- Parent checks feedback audits and total quotas. Summary checks nine identities,
  per-seed draw/pool hashes, formal counts, reference hashes and feedback totals.
- Complete local windows only; future labels score already-frozen candidates and
  never feed back into learning. These compare AUX against TRUE, not SKIP.
- Per-seed aggregation uses sample SD and valid metric n. Paired differences computed
  within seed. Query teacher accuracy is conditional, not whole-stream accuracy.
- New output directories only; logs and partial rows preserved on failure.

Limitations: no runtime validation; dataset identity is checked by stream shape and
reference IDs, not a full image-content hash. A repaired linear head may lack capacity;
feedback evidence may fail under domain changes. Neither is assumed effective in advance.
