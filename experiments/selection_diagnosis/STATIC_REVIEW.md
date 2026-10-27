# Static review: screening usability

Source reads, call-path/schema inspection and Git diff checks only. No local
Python, CLI help, compile, tests, diagnosis or training was run.

- Independent package. Existing methods, user evidence document, outputs and .DS_Store untouched.
- D100, PACS P→A→C→S, bs32, three seeds. One SP trajectory per seed, not nine methods.
- Inherit original source-copy/risk initialization and query descriptors. Formal
  cluster_train explicitly delegates to SPATTA, never the selective student loss.
- Original SP gate unchanged. No new current-target candidates or formal updates.
- Copied head/risk training are shadow overhead, not formal slots. Replay same
  virtual last-two draw identities, private generators, original learning settings.
- Current observations made before fresh feedback training; no prediction is
  reconstructed after learning to evaluate its earlier correctness.
- Query phase/warmup/update counts captured at query. Returned labels alone create
  risk-memory targets; pending labels never consumed.
- Online base-rate comparator uses only already returned correctness records.
- Current evaluation labels attached post-batch to observation rows only. They
  never enter risk memory, thresholds, source-copy training or SP losses.
- RNG isolation and main-model fingerprint check surround shadow training.
- Window truth cache and source-feature cache both keep four batches; stream ID
  bounds from the fixed trace align current/post-batch observations.
- Score transfer to CPU batched; all observations, including rejected targets,
  logged. Source/repair branches have their own consistency/score conditions.
- Funnel cumulative source/repair chains monotone; SP gate recorded after original
  gate evaluates, and priority action counts logged separately.
- All raw CSV rows initialize identical keys, including pre-learning/null-label
  placeholders; no late extra keys on only some rows. Empty files have headers.
- Fixed top/bottom10% ranking excludes the risk threshold, uses same base pool and
  count as confidence comparator. Fractional boundary ties avoid order bias.
- Phase after_warmup is explicitly a subset of after_learning. Pending and active
  query bias disclosed; repeated window images not treated as seed replicates.
- Domain report uses image domain for observations and query domain for Brier.
- Fixed score bins and rank fraction are diagnostic, not a test-label search.
- Each section summarized per seed then sample SD/valid n across seeds. No fake
  precision on empty pools. Baseline and query counts avoid double-counting branches.
- Resume checks method/defaults/source/reference/seed/formal quota and required
  outputs; incomplete attempts archived; complete seeds reused; directory lock.
- Numerical module replay equivalence, SP trajectory equality and runtime behavior
  require server verification. No accuracy or calibration claim made in advance.
