# Static review

No local Python, CLI help, compile, tests, offline analysis or training executed.
Source reads, schema/call-path review and Git diff checks only.

- Independent package; old studies and user evidence/outputs/.DS_Store untouched.
- Waiting analyzer reads existing selection logs. Optional selected_steps argument
  leaves prior batch_rankings default warmup semantics unchanged.
- Waiting includes first return batch because prediction precedes release; names
  changed from warm_batches to evaluated_batches in waiting report only.
- Core D100 experiment: three factual SP trajectories, four shadow arms, five
  fixed trigger intervals. No learned reliability or copied trainable source head.
- Source selector receives data/IDs only. Confidence/flip masks contain no truth.
  Private random tie order does not advance formal RNG; tie rule logged/documented.
- Four arms use same pre-update snapshot and old-truth/history draws. Fresh release
  excluded from truth/history probe training. No evaluation truth trains candidates.
- Current q produced under no_grad; KL updates only copied online model. K=0 never
  forwards an empty image batch; CURRENT then falls back to full HISTORY.
- REDUCED uses exactly CURRENT's history coefficient. Active subset excludes empty
  selection when interpreting CURRENT−REDUCED; no false attribution to history scaling.
- Current loss mean over selected K; no masked/random resampling dilution. All
  arms use one SGD step, original lr/momentum, empty state and RNG reset.
- Original SP model fingerprint unchanged by probes, main quota/feedback verified.
- Selected targets' online predictions saved before update; source correction
  opportunities/risks scored only post-batch, not used to filter candidates.
- Distinct active containers for original SP and elite probes. Future evaluation
  begins next batch; frozen models never affect formal decisions; complete only.
- All CSV records initialize fixed keys/None before later values. Empty files have
  explicit headers; zero-selection targets yield N/A, not zero accuracy.
- Domain-aware future scoring; sample weighting within seed, SD/n across three seeds.
  No event count inflation or source precision interpreted as deployed gains.
- Resume validates method/config/reference/formal budget and complete output marker.
  Archives partial attempts and old reports; preserves completed seeds; directory lock.
- Runtime compatibility, faithful SP trajectory and intervention effectiveness await
  server validation. Equal steps not equal FLOPs; diagnostics are extra overhead.
