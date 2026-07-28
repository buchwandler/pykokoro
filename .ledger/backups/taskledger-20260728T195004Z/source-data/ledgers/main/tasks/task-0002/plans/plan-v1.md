---
schema_version: 1
object_type: plan
file_version: v2
task_id: task-0002
plan_id: plan-v1
version: 1
plan_version: 1
status: accepted
created_at: "2026-07-26T14:04:56Z"
created_by:
  actor_type: agent
  actor_name: u0_a992
  tool: null
  session_id: null
  host: localhost
  pid: 29593
  actor_id: null
  role: null
  harness_id: null
  command_pid: null
  pid_scope: null
supersedes: null
question_refs: []
criteria:
  - id: ac-0001
    text:
      "All live documentation and examples use native SSMD annotations or directives; no
      Markdown-link SSMD syntax or @voice: directive remains outside intentional
      negative regression coverage."
    mandatory: true
  - id: ac-0002
    text:
      Native inline and block voice syntax works, and block language, voice fields, and
      prosody are inherited into PyKokoro metadata with field-by-field inline
      precedence.
    mandatory: true
  - id: ac-0003
    text:
      Adjacent SSMD breaks accumulate duration, X-SAMPA annotations use the SSMD
      converter API, and annotation detection matches the supported SSMD grammar.
    mandatory: true
  - id: ac-0004
    text:
      Phoneme-dictionary generated annotations safely escape special attribute
      characters and round-trip through SSMD parsing.
    mandatory: true
  - id: ac-0005
    text:
      The targeted and full test suites, compile check, Ruff, and mypy pass, or any
      environment limitation is recorded explicitly with criterion evidence.
    mandatory: true
  - id: ac-0006
    text:
      A changelog entry records the legacy-to-native SSMD migration, and the upstream
      SSMD X-SAMPA data-file packaging limitation is explicitly documented if it cannot
      be fixed in this repository.
    mandatory: true
todos:
  - id: plan-todo-0001
    text:
      Inspect the applied SSMD correction patch against the handoff and repository
      contracts; identify any missing implementation, documentation, changelog, or test
      work.
    done: false
    created_at: "2026-07-26T14:04:56Z"
    updated_at: "2026-07-26T14:04:56Z"
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
      Review git diff, the handoff acceptance checklist, and the relevant PyKokoro/SSMD
      interfaces.
  - id: plan-todo-0002
    text:
      Complete the PyKokoro SSMD parser, metadata inheritance, cumulative break, X-SAMPA
      dispatch, and phoneme-dictionary escaping behavior described by the handoff.
    done: false
    created_at: "2026-07-26T14:04:56Z"
    updated_at: "2026-07-26T14:04:56Z"
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
    validation_hint: Run targeted SSMD and phoneme-dictionary tests.
  - id: plan-todo-0003
    text:
      Complete native SSMD syntax migrations in the README, docs, examples, and
      configuration documentation, including the multi-voice podcast and end-to-end
      prosody examples.
    done: false
    created_at: "2026-07-26T14:04:56Z"
    updated_at: "2026-07-26T14:04:56Z"
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
    validation_hint: Audit live files for legacy syntax and compile all Python files.
  - id: plan-todo-0004
    text:
      Add or update the changelog migration note and document the SSMD X-SAMPA
      package-data limitation and upstream coordination status.
    done: false
    created_at: "2026-07-26T14:04:56Z"
    updated_at: "2026-07-26T14:04:56Z"
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
      Inspect the changelog entry and documentation for both required release notes.
  - id: plan-todo-0005
    text:
      Run the complete validation matrix and record targeted, full-suite, compile, Ruff,
      mypy, and example checks with any dependency limitations.
    done: false
    created_at: "2026-07-26T14:04:56Z"
    updated_at: "2026-07-26T14:04:56Z"
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
      Run the listed test and static-check commands and preserve their results.
generation_reason: initial
based_on_question_ids: []
based_on_answer_hash: null
supersedes_plan_id: null
approved_at: "2026-07-26T14:08:14Z"
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
approval_note: "User approved in harness: approve."
approval_source: explicit_chat
approved_plan_hash: 5fb35ded9224cd210924a07094e718f678211690e1c36b58c67c07c92a8419ee
goal:
  Complete and verify the SSMD integration correction handoff in PyKokoro, including
  native syntax, metadata inheritance, break and phoneme handling, documentation,
  examples, regression tests, and release-risk documentation.
files:
  - "@pykokoro/ssmd_parser.py"
  - "@pykokoro/phoneme_dictionary.py"
  - "@tests/test_ssmd.py"
  - "@tests/test_phoneme_dictionary.py"
  - "@README.md"
  - "@docs/basic_usage.rst"
  - "@docs/pipeline_stages.rst"
  - "@examples/english.py"
  - "@examples/podcast.py"
  - "@examples/prosody_demo.py"
  - "@examples/say_as_demo.py"
  - "@examples/ssmd_demo.py"
  - "@pykokoro/generation_config.py"
  - "@CHANGELOG.md"
test_commands:
  - python -m compileall -q pykokoro examples tests
  - python -m pytest -q tests/test_ssmd.py tests/test_phoneme_dictionary.py
  - python -m pytest -q
  - ruff check .
  - mypy pykokoro
expected_outputs:
  - Compilation, targeted tests, full tests, Ruff, and mypy pass in the supported
    environment.
todos_waived_reason: null
---

<!-- Required: keep this body. It is the implementation handoff context.
     Run `taskledger plan check --file ./plan.md` before upsert. -->

# Complete PyKokoro SSMD maintainer handoff

## Summary

The applied correction patch is the starting point for completing the maintainer
handoff. This task will verify that the implementation matches the installed SSMD
contract, fill any gaps, and leave PyKokoro consistently using native SSMD annotations
and directives. It includes the requested code, tests, documentation, examples, release
note, and explicit handling of the upstream X-SAMPA package-data limitation.

## Implementation Changes

- Verify and, where necessary, finish `ssmd_parser.py` and `phoneme_dictionary.py`
  behavior for grammar-aligned detection, directive inheritance, voice merging,
  cumulative breaks, X-SAMPA conversion, and escaped generated annotations.
- Verify the regression tests for native syntax, legacy literal behavior, inheritance,
  phoneme conversion, breaks, and escaping; add narrowly scoped tests for any uncovered
  handoff requirement.
- Verify all live README, documentation, examples, and configuration snippets use native
  SSMD syntax; keep legacy syntax only in negative tests or explanatory migration text.
- Finish the podcast and prosody examples against `KokoroPipeline`, and ensure examples
  remain compilable and use current public APIs.
- Add a concise changelog migration note and explicitly document the upstream X-SAMPA
  data-file packaging limitation if the dependency cannot be changed here.

## Tests

- `python -m compileall -q pykokoro examples tests`
- `python -m pytest -q tests/test_ssmd.py tests/test_phoneme_dictionary.py`
- `python -m pytest -q`
- `ruff check .` and `mypy pykokoro`
- Audit live syntax with repository search and inspect the changed examples; run
  lightweight examples only when model/dependency assets are available.

## Assumptions

- The user has already applied `pykokoro_ssmd_corrections.patch`; existing working-tree
  changes are in scope and will be preserved.
- The upstream SSMD repository/package is outside this workspace, so its missing X-SAMPA
  data file can be documented and tested for explicitly here but cannot be repaired
  upstream from this task.
- Full audio examples may require model assets and optional dependencies; unavailable
  environment prerequisites will be recorded rather than masked.

## Out of Scope

- No automatic compatibility rewrite for legacy Markdown links will be reintroduced.
- No embedded duplicate X-SAMPA conversion table will be added to PyKokoro.
- No unrelated refactoring or generated artifact changes.

## Plan input checklist before upsert

- [x] I ran `taskledger plan check --file plan.md`.
- [x] Every acceptance criterion uses `text`, not `description`.
- [x] Todo mappings use supported keys only: `id`, `id_hint`, `text`, `mandatory`,
      `validation_hint`, `worker_step`.
- [x] File references are plan-level `files:` entries or are mentioned in todo
      text/body; todo-level `files:` is not captured.
- [x] The Markdown body explains enough context for implementation handoff.
