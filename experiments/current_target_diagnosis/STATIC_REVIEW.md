# Static protocol review

No Python, CLI help, compile, tests, diagnosis or training was executed locally.
Only source reading, schema/call-path review and Git diff checks were performed.

- Additive package; no existing experiment or result modified.
- D100 only; three seeds, one factual SP trajectory per seed, five small probes.
- Fixed-trace constructor verifies source checkpoint, seed, clocks, stream shape,
  query IDs, source configuration and formal quotas using existing SP checks.
- Original SP selection, pseudo pool, release hook, optimizer and gate retained.
- Diagnostic current images/features have no labels. Separate evaluation truth
  buffer is only read by the scoring hook; no training/gating access exists.
- Repair head updates only use self.received after release; original encoder/head
  have no gradient from repair. It runs after formal updates, next-batch use only.
- New probe truth uses self.received before current release is merged. Fresh image
  IDs are excluded from both candidate auxiliary masks.
- One CPU snapshot, common truth/current indices and reset RNG for each arm;
  private random generator; formal model fingerprint unchanged after candidates.
- SOURCE and REPAIR own masks may differ. Same-image target quality also includes
  common-mask and disagreement subsets so mask quality is visible.
- Three candidate objectives use true coefficient1, auxiliary0.25 and denominator B.
  Empty masks produce zero auxiliary loss without redrawing images.
- Candidate forward hooks counted separately; source computation counted via
  existing teacher counters. Diagnostic head steps kept outside formal quota.
- Original SP active probes and new diagnostic probes use distinct containers.
- Future evaluation starts next batch, freezes models, never feeds results back.
  Completeness recorded consistently on all future rows before CSV writing.
- Future domain grouping uses scored domain; target grouping uses image domain.
  Acceptance flags for source, repair and original SP are not conflated.
- All event/target/future CSV records initialize fixed keys, including null or
  incomplete cases; empty outputs use explicit header lists. No late key insertion
  into a subset of rows as in the previous CSV bug.
- Per-seed tables retain zero-denominator N/A. Aggregate across seeds, not events;
  shared accepted subset available to avoid comparing different event populations.
- Resume validates root protocol, final marker, diagnostics, reference and formal
  budget; complete results skipped. Incomplete attempts/old summaries archived.
  Directory lock blocks simultaneous updated launchers; stop legacy tasks first.
- Runtime/framework compatibility and baseline trajectory equality remain to be
  verified on the server. Different data with identical shape is not excluded by
  a full content hash. No deployment/superiority claim made.
