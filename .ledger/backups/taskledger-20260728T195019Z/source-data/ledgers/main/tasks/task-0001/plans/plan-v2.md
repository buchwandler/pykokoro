---
schema_version: 1
object_type: plan
file_version: v2
task_id: task-0001
plan_id: plan-v2
version: 2
plan_version: 2
status: accepted
created_at: "2026-07-26T10:26:18Z"
created_by:
  actor_type: agent
  actor_name: u0_a992
  tool: null
  session_id: null
  host: localhost
  pid: 29734
  actor_id: null
  role: null
  harness_id: null
  command_pid: null
  pid_scope: null
supersedes: 1
question_refs: []
criteria:
  - id: ac-0001
    text:
      Per-call pipeline generation overrides accept GenerationConfig and mapping forms,
      combine correctly with lang, reject invalid fields clearly, and never mutate the
      base configuration.
    mandatory: true
  - id: ac-0002
    text:
      G2P and backend caches are keyed by complete effective configuration snapshots,
      including nested mutable values, and backend replacement is transactional and
      ownership-safe on construction failure and close.
    mandatory: true
  - id: ac-0003
    text:
      Download and cache paths validate existing artifacts, enforce
      pinned/integrity-checked artifacts, retry validator failures classified as
      transient or recoverable, recover stale locks conservatively, and support explicit
      offline/cache-only behavior.
    mandatory: true
  - id: ac-0004
    text:
      G2P disk-cache entries use a validated versioned schema, malformed or obsolete
      entries recompute safely, and warnings are deterministic on cache hits and misses.
    mandatory: true
  - id: ac-0005
    text:
      Lightweight pykokoro utilities import without ONNX Runtime, ONNX-backed operations
      fail with an actionable capability error, and kokorog2p absence is distinguished
      from initialization/import failures.
    mandatory: true
  - id: ac-0006
    text:
      GenerationConfig and ShortSentenceConfig reject non-real, boolean, non-finite,
      out-of-range, and malformed values with field-specific errors.
    mandatory: true
  - id: ac-0007
    text:
      Source and built artifacts contain valid license/version/package metadata, and
      ONNX Runtime provider extras install according to a documented non-conflicting
      one-provider policy.
    mandatory: true
  - id: ac-0008
    text:
      Ruff, mypy, formatting, compile, test, artifact-install, and supported-environment
      release checks are represented by runnable repository commands and pass when their
      declared dependencies are available.
    mandatory: true
  - id: ac-0009
    text:
      README, AGENTS, examples, provider priority/fallback behavior, supported Python
      versions, and CI documentation match the repository and tested implementation.
    mandatory: true
  - id: ac-0010
    text:
      The large-function maintenance hotspots identified as PK-014 are refactored only
      after characterization tests and correctness fixes are green, without intentional
      behavior changes.
    mandatory: true
todos:
  - id: plan-todo-0001
    text:
      "Phase 1: Add deterministic failing reproductions for PK-001 through PK-004
      covering the override matrix, backend lifecycle, mutable configuration snapshots,
      and G2P cache matrix before implementation edits."
    done: false
    created_at: "2026-07-26T10:26:18Z"
    updated_at: "2026-07-26T10:26:18Z"
    source: plan
    mandatory: true
    status: open
    active_at: null
    blocked_reason: null
    done_at: null
    skipped_at: null
    completed_by: null
    completed_in_harness: null
    skipped_by: null
    evidence: []
    artifact_refs: []
    change_refs: []
    command_refs: []
    source_plan_id: null
    source_question_ids: []
    validation_hint:
      Run the new focused regression tests and confirm each intended defect is
      reproduced before the corresponding fix.
  - id: plan-todo-0002
    text:
      "Phase 2: Fix generation override normalization, complete G2P cache keys,
      transactional backend replacement, and immutable value-based backend configuration
      snapshots; add and update focused tests."
    done: false
    created_at: "2026-07-26T10:26:18Z"
    updated_at: "2026-07-26T10:26:18Z"
    source: plan
    mandatory: true
    status: open
    active_at: null
    blocked_reason: null
    done_at: null
    skipped_at: null
    completed_by: null
    completed_in_harness: null
    skipped_by: null
    evidence: []
    artifact_refs: []
    change_refs: []
    command_refs: []
    source_plan_id: null
    source_question_ids: []
    validation_hint:
      Run pipeline, lifecycle, G2P, and session tests; inspect that base configs remain
      unchanged and owned backends close exactly once.
  - id: plan-todo-0003
    text:
      "Phase 3: Harden ONNX/model downloads and G2P disk caches with pinned revisions,
      digest/format validation, corrupt-cache replacement, retryable validation
      failures, stale-lock metadata/recovery, offline mode, schema validation, and
      warning preservation using local deterministic fixtures."
    done: false
    created_at: "2026-07-26T10:26:18Z"
    updated_at: "2026-07-26T10:26:18Z"
    source: plan
    mandatory: true
    status: open
    active_at: null
    blocked_reason: null
    done_at: null
    skipped_at: null
    completed_by: null
    completed_in_harness: null
    skipped_by: null
    evidence: []
    artifact_refs: []
    change_refs: []
    command_refs: []
    source_plan_id: null
    source_question_ids: []
    validation_hint:
      Run download/cache tests for valid, undersized, invalid, checksum, retry,
      stale-lock, concurrent, forced, offline, malformed-schema, and cache-warning cases
      without public network access.
  - id: plan-todo-0004
    text:
      "Phase 4: Repair release packaging by adding the actual license artifact, making
      source-only builds produce deliberate release-safe versions, removing stale
      setup/package discovery configuration, inspecting sdist/wheel contents, and
      redesigning ONNX Runtime extras so exactly one provider distribution is selected
      per install."
    done: false
    created_at: "2026-07-26T10:26:18Z"
    updated_at: "2026-07-26T10:26:18Z"
    source: plan
    mandatory: true
    status: open
    active_at: null
    blocked_reason: null
    done_at: null
    skipped_at: null
    completed_by: null
    completed_in_harness: null
    skipped_by: null
    evidence: []
    artifact_refs: []
    change_refs: []
    command_refs: []
    source_plan_id: null
    source_question_ids: []
    validation_hint:
      Build sdist and wheel, inspect metadata and contents programmatically, install the
      wheel in a clean virtual environment, and exercise each documented provider-extra
      policy.
  - id: plan-todo-0005
    text:
      "Phase 5: Move dependency-light public types/defaults out of ONNX-facing imports,
      use TYPE_CHECKING for annotations, provide actionable lazy capability errors, and
      narrow kokorog2p import exception handling with subprocess tests."
    done: false
    created_at: "2026-07-26T10:26:18Z"
    updated_at: "2026-07-26T10:26:18Z"
    source: plan
    mandatory: true
    status: open
    active_at: null
    blocked_reason: null
    done_at: null
    skipped_at: null
    completed_by: null
    completed_in_harness: null
    skipped_by: null
    evidence: []
    artifact_refs: []
    change_refs: []
    command_refs: []
    source_plan_id: null
    source_question_ids: []
    validation_hint:
      Run subprocess tests with ONNX Runtime import failure and kokorog2p absent/broken;
      verify lightweight imports succeed and ONNX requests explain remediation.
  - id: plan-todo-0006
    text:
      "Phase 6: Add strict numeric and short-sentence configuration validation,
      consolidate Ruff configuration and non-mutating check behavior, repair
      contributor/docs paths, align provider priority/fallback documentation with exact
      tested lists, and add visible CI/release-gate documentation."
    done: false
    created_at: "2026-07-26T10:26:18Z"
    updated_at: "2026-07-26T10:26:18Z"
    source: plan
    mandatory: true
    status: open
    active_at: null
    blocked_reason: null
    done_at: null
    skipped_at: null
    completed_by: null
    completed_in_harness: null
    skipped_by: null
    evidence: []
    artifact_refs: []
    change_refs: []
    command_refs: []
    source_plan_id: null
    source_question_ids: []
    validation_hint:
      Run configuration edge-case tests, ruff check, ruff format --check, mypy,
      compileall, and documentation/path consistency checks.
  - id: plan-todo-0007
    text:
      "Phase 7: After all correctness tests are green, extract the PK-014 maintenance
      hotspots into focused helpers with characterization coverage and no intentional
      behavior changes."
    done: false
    created_at: "2026-07-26T10:26:18Z"
    updated_at: "2026-07-26T10:26:18Z"
    source: plan
    mandatory: true
    status: open
    active_at: null
    blocked_reason: null
    done_at: null
    skipped_at: null
    completed_by: null
    completed_in_harness: null
    skipped_by: null
    evidence: []
    artifact_refs: []
    change_refs: []
    command_refs: []
    source_plan_id: null
    source_question_ids: []
    validation_hint:
      Run the complete non-slow suite plus targeted audio, splitter, pipeline, cache,
      packaging, and provider tests before and after refactoring.
  - id: plan-todo-0008
    text:
      Run the complete supported-environment release gate, including Python 3.10–3.13
      coverage where available, clean artifact installation, pinned real CPU ONNX smoke
      inference, and platform/provider smoke checks; record any unavailable native/model
      checks explicitly.
    done: false
    created_at: "2026-07-26T10:26:18Z"
    updated_at: "2026-07-26T10:26:18Z"
    source: plan
    mandatory: true
    status: open
    active_at: null
    blocked_reason: null
    done_at: null
    skipped_at: null
    completed_by: null
    completed_in_harness: null
    skipped_by: null
    evidence: []
    artifact_refs: []
    change_refs: []
    command_refs: []
    source_plan_id: null
    source_question_ids: []
    validation_hint:
      Run pytest, ruff, format check, mypy, compileall, build/artifact checks, and real
      CPU/provider smoke jobs in the supported environment.
generation_reason: initial
based_on_question_ids: []
based_on_answer_hash: null
supersedes_plan_id: plan-v1
approved_at: "2026-07-26T10:26:48Z"
approved_by:
  actor_type: user
  actor_name: u0_a992
  tool: manual
  session_id: null
  host: null
  pid: null
  actor_id: null
  role: null
  harness_id: null
  command_pid: null
  pid_scope: null
approval_note: "User approved in harness: approve"
approval_source: explicit_chat
approved_plan_hash: 3574cb768e6b5fb49f86f0ed75e7c27836f1ed27ef472f9d8e9f5c4089a6c7e3
goal:
  Implement all seven phases of PYKOKORO_MAINTAINER_REVIEW-4 with regression coverage
  and release-readiness validation.
files:
  - "@PYKOKORO_MAINTAINER_REVIEW-4.md"
  - "@pykokoro/pipeline.py"
  - "@pykokoro/pipeline_config.py"
  - "@pykokoro/stages/g2p/kokorog2p.py"
  - "@pykokoro/onnx_backend.py"
  - "@pykokoro/onnx_session.py"
  - "@pykokoro/audio_generator.py"
  - "@pykokoro/generation_config.py"
  - "@pykokoro/short_sentence_handler.py"
  - "@pyproject.toml"
  - "@setup.py"
  - "@README.md"
  - "@AGENTS.md"
  - "@tests"
test_commands:
  - python -m pytest -q -m 'not slow'
  - ruff check .
  - ruff format --check .
  - mypy pykokoro
  - python -m compileall -q pykokoro tests
  - python -m build
expected_outputs:
  - All mandatory regression and supported-environment checks pass.
  - Built sdist and wheel contain the intended non-0.0.0 version and license metadata.
todos_waived_reason: null
---

# Maintainer Review 4 — Phases 1–7

## Summary

Implement the complete remediation sequence in `PYKOKORO_MAINTAINER_REVIEW-4.md`. The
work is deliberately staged: first reproduce the deterministic state/configuration
defects, then fix correctness and cache/download integrity, repair packaging and import
boundaries, tighten validation and documentation, and only then perform the recommended
structural refactors. The release gate must distinguish checks that pass locally from
native/model/platform checks that require the supported CI environment.

## Implementation Changes

- Phase 1: add failing tests for per-call generation overrides, G2P cache identity,
  backend lifecycle failure handling, and mutable nested backend configuration.
- Phase 2: normalize overrides, make G2P/backend cache identity reflect effective
  values, and make backend replacement transactional with explicit ownership behavior.
- Phase 3: pin and integrity-check downloaded artifacts, validate existing files,
  improve retry/lock/offline behavior, and version/validate G2P disk-cache payloads
  while preserving warnings.
- Phase 4: restore license and release-safe version metadata, remove stale packaging
  configuration, verify built artifacts, and make provider extras non-conflicting.
- Phase 5: separate lightweight types from ONNX imports, use focused capability
  diagnostics, and preserve the distinction between missing and broken `kokorog2p`
  installations.
- Phase 6: validate finite real configuration values and short-sentence templates,
  consolidate tooling configuration, repair repository documentation, align provider
  policy, and add CI/release-gate coverage.
- Phase 7: refactor the large hotspots only after characterization and correctness tests
  pass, keeping behavior stable.
- Register each meaningful source/config/test change and each verification command in
  taskledger implementation evidence.

## Tests

- Focused regression tests for every PK-001 through PK-013 behavior, including local
  download/cache fixtures and subprocess import isolation.
- Existing audio, short-sentence, session, pipeline, splitter, and G2P unit suites.
- `python -m pytest -q -m "not slow"`, Ruff lint/check, formatting check, mypy, and
  compileall.
- Sdist/wheel build, metadata/content inspection, clean-wheel installation, and
  lightweight installed-package smoke tests.
- Supported-environment CI matrix for Python 3.10–3.13 and relevant Linux/Windows/macOS
  provider smoke coverage.
- Pinned real CPU ONNX inference and validated-cache reuse; unavailable native/model
  checks must be recorded as blocked rather than silently treated as passing.

## Assumptions

- The maintainer review is the source of truth for scope, including all findings PK-001
  through PK-015 and its Phase 1–7 sequence.
- No public network access or unpinned model download is required for unit tests;
  deterministic local fixtures will cover download/cache behavior.
- Provider extras are alternatives, and the final policy will document one intended ONNX
  Runtime distribution per installation target.
- A native/model-supported CI environment is required to claim the real inference and
  platform acceptance criteria.

## Out of Scope

- Unrelated API redesigns, broad refactors not identified by PK-014, or changing
  existing behavior solely to make a test pass.
- Adding large binary model/audio artifacts to the repository.
- Claiming native ONNX, Android, or accelerator support based only on mocks when the
  required environment/assets are unavailable.
