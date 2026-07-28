---
schema_version: 1
object_type: plan
file_version: v2
task_id: task-0003
plan_id: plan-v1
version: 1
plan_version: 1
status: accepted
created_at: "2026-07-26T22:44:51Z"
created_by:
  actor_type: agent
  actor_name: u0_a992
  tool: null
  session_id: null
  host: localhost
  pid: 9713
  actor_id: null
  role: null
  harness_id: null
  command_pid: null
  pid_scope: null
supersedes: null
question_refs: []
criteria:
  - id: ac-0001
    text: All nine .rst files were renamed and converted to .md
    mandatory: true
  - id: ac-0002
    text: No .rst files remain anywhere in the repository
    mandatory: true
  - id: ac-0003
    text: myst-parser>=2.0.0 is present in docs/requirements.txt
    mandatory: true
  - id: ac-0004
    text: myst_parser is enabled in docs/conf.py
    mandatory: true
  - id: ac-0005
    text: Sphinx source suffixes are Markdown-only
    mandatory: true
  - id: ac-0006
    text: docs/README.md is excluded from Sphinx discovery
    mandatory: true
  - id: ac-0007
    text: colon_fence is enabled
    mandatory: true
  - id: ac-0008
    text: heading anchors cover levels 1 through 4
    mandatory: true
  - id: ac-0009
    text: the toctree order is unchanged
    mandatory: true
  - id: ac-0010
    text: every code block retains its language
    mandatory: true
  - id: ac-0011
    text: all four notes render as notes
    mandatory: true
  - id: ac-0012
    text: all 14 cross-document links work
    mandatory: true
  - id: ac-0013
    text: all three generated-index references work
    mandatory: true
  - id: ac-0014
    text: all 24 autodoc objects render
    mandatory: true
  - id: ac-0015
    text: sphinx-build -W --keep-going -b html succeeds
    mandatory: true
  - id: ac-0016
    text:
      dirhtml, linkcheck, and doctest builds succeed or have only explicitly documented
      pre-existing failures
    mandatory: true
  - id: ac-0017
    text: pre-commit run --all-files succeeds
    mandatory: true
  - id: ac-0018
    text: the normal test suite succeeds
    mandatory: true
  - id: ac-0019
    text: Read the Docs still uses docs/conf.py and the RTD theme
    mandatory: true
  - id: ac-0020
    text: no unrelated prose or API changes are mixed into the migration
    mandatory: true
todos:
  - id: plan-todo-0001
    text:
      "Phase 1: Build RST baseline - install dependencies and run Sphinx html build to
      record warnings, page list, toctree order, API objects, and generated pages"
    done: false
    created_at: "2026-07-26T22:44:51Z"
    updated_at: "2026-07-26T22:44:51Z"
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
    validation_hint: Check docs/_build/rst-baseline/ for generated HTML
  - id: plan-todo-0002
    text: "Phase 2: Install rst-to-myst and run initial conversion on all 9 .rst files"
    done: false
    created_at: "2026-07-26T22:44:51Z"
    updated_at: "2026-07-26T22:44:51Z"
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
    validation_hint: Check that .md files are created from .rst files
  - id: plan-todo-0003
    text: "Phase 3a: Update docs/requirements.txt to add myst-parser>=2.0.0"
    done: false
    created_at: "2026-07-26T22:44:51Z"
    updated_at: "2026-07-26T22:44:51Z"
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
    validation_hint: Verify myst-parser is in requirements.txt
  - id: plan-todo-0004
    text:
      "Phase 3b: Update docs/conf.py - add myst_parser extension, source_suffix,
      myst_enable_extensions, myst_heading_anchors, exclude README.md"
    done: false
    created_at: "2026-07-26T22:44:51Z"
    updated_at: "2026-07-26T22:44:51Z"
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
    validation_hint: Verify conf.py has all required MyST settings
  - id: plan-todo-0005
    text:
      "Phase 3c: Update docs/README.md - replace .rst references with .md, describe MyST
      syntax"
    done: false
    created_at: "2026-07-26T22:44:51Z"
    updated_at: "2026-07-26T22:44:51Z"
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
    validation_hint: Verify README.md describes MyST workflow
  - id: plan-todo-0006
    text:
      "Phase 3d: Update .pre-commit-config.yaml - remove .rst exclusion from
      check-merge-conflict"
    done: false
    created_at: "2026-07-26T22:44:51Z"
    updated_at: "2026-07-26T22:44:51Z"
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
    validation_hint: Verify .pre-commit-config.yaml has no .rst exclusions
  - id: plan-todo-0007
    text: "Phase 3e: Update .codecrate.toml - remove .rst include pattern"
    done: false
    created_at: "2026-07-26T22:44:51Z"
    updated_at: "2026-07-26T22:44:51Z"
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
    validation_hint: Verify .codecrate.toml has no .rst patterns
  - id: plan-todo-0008
    text:
      "Phase 4a: Manually review and fix docs/index.md - preserve headings, convert
      toctree, code blocks, refs, bare URLs"
    done: false
    created_at: "2026-07-26T22:44:51Z"
    updated_at: "2026-07-26T22:44:51Z"
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
    validation_hint: Run sphinx build and check index.html
  - id: plan-todo-0009
    text:
      "Phase 4b: Manually review and fix docs/quickstart.md - preserve headings, convert
      code blocks, doc roles, nested lists"
    done: false
    created_at: "2026-07-26T22:44:51Z"
    updated_at: "2026-07-26T22:44:51Z"
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
    validation_hint: Run sphinx build and check quickstart.html
  - id: plan-todo-0010
    text:
      "Phase 4c: Manually review and fix docs/installation.md - convert code blocks,
      note, bare URL"
    done: false
    created_at: "2026-07-26T22:44:51Z"
    updated_at: "2026-07-26T22:44:51Z"
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
    validation_hint: Run sphinx build and check installation.html
  - id: plan-todo-0011
    text:
      "Phase 4d: Manually review and fix docs/basic_usage.md - convert code blocks,
      note, doc roles"
    done: false
    created_at: "2026-07-26T22:44:51Z"
    updated_at: "2026-07-26T22:44:51Z"
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
    validation_hint: Run sphinx build and check basic_usage.html
  - id: plan-todo-0012
    text:
      "Phase 4e: Manually review and fix docs/advanced_features.md - convert code
      blocks, note, doc roles"
    done: false
    created_at: "2026-07-26T22:44:51Z"
    updated_at: "2026-07-26T22:44:51Z"
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
    validation_hint: Run sphinx build and check advanced_features.html
  - id: plan-todo-0013
    text:
      "Phase 4f: Manually review and fix docs/pipeline_stages.md - convert code blocks,
      preserve H1-H4 hierarchy"
    done: false
    created_at: "2026-07-26T22:44:51Z"
    updated_at: "2026-07-26T22:44:51Z"
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
    validation_hint: Run sphinx build and check pipeline_stages.html
  - id: plan-todo-0014
    text:
      "Phase 4g: Manually review and fix docs/api_reference.md - wrap all 24 autodoc
      directives in eval-rst, convert code blocks and doc roles"
    done: false
    created_at: "2026-07-26T22:44:51Z"
    updated_at: "2026-07-26T22:44:51Z"
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
    validation_hint: Run sphinx build and verify all 24 API objects render
  - id: plan-todo-0015
    text:
      "Phase 4h: Manually review and fix docs/examples.md - convert code blocks, note,
      doc roles"
    done: false
    created_at: "2026-07-26T22:44:51Z"
    updated_at: "2026-07-26T22:44:51Z"
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
    validation_hint: Run sphinx build and check examples.html
  - id: plan-todo-0016
    text:
      "Phase 4i: Manually review and fix docs/changelog.md - convert headings, code
      block, inline literals"
    done: false
    created_at: "2026-07-26T22:44:51Z"
    updated_at: "2026-07-26T22:44:51Z"
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
    validation_hint: Run sphinx build and check changelog.html
  - id: plan-todo-0017
    text:
      "Phase 5: Run strict validation - sphinx html, dirhtml, linkcheck, doctest builds;
      pre-commit; pytest; verify no .rst files remain"
    done: false
    created_at: "2026-07-26T22:44:51Z"
    updated_at: "2026-07-26T22:44:51Z"
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
    validation_hint: All builds pass, no .rst files found
  - id: plan-todo-0018
    text:
      "Phase 6: Semantic parity checks - compare baseline vs migrated HTML, verify
      navigation order, API smoke checks, README.md exclusion"
    done: false
    created_at: "2026-07-26T22:44:51Z"
    updated_at: "2026-07-26T22:44:51Z"
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
    validation_hint: All parity checks pass
generation_reason: initial
based_on_question_ids: []
based_on_answer_hash: null
supersedes_plan_id: null
approved_at: "2026-07-26T22:46:12Z"
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
approval_note: User approved in harness.
approval_source: explicit_chat
approved_plan_hash: e4147b64858fd102e9c8506e99297c05e37a423b8cd3eabc3fd6f69802222ded
goal:
  Convert all Sphinx documentation from reStructuredText to MyST Markdown while
  preserving rendered content, navigation, API reference, page URLs, theme, and Read the
  Docs setup.
files:
  - "@docs/requirements.txt"
  - "@docs/conf.py"
  - "@docs/README.md"
  - "@.pre-commit-config.yaml"
  - "@.codecrate.toml"
  - "@docs/index.md"
  - "@docs/quickstart.md"
  - "@docs/installation.md"
  - "@docs/basic_usage.md"
  - "@docs/advanced_features.md"
  - "@docs/pipeline_stages.md"
  - "@docs/api_reference.md"
  - "@docs/examples.md"
  - "@docs/changelog.md"
test_commands:
  - python -m sphinx -W --keep-going -b html docs docs/_build/html
  - python -m sphinx -W --keep-going -b dirhtml docs docs/_build/dirhtml
  - python -m sphinx -W --keep-going -b linkcheck docs docs/_build/linkcheck
  - python -m sphinx -W --keep-going -b doctest docs docs/_build/doctest
  - pre-commit run --all-files
  - pytest
  - find . -type f -name '*.rst' -print
  - git grep -nE '\.rst\b|reStructuredText|index\.rst|quickstart\.rst'
expected_outputs:
  - Sphinx html build exits 0 with no warnings
  - Sphinx dirhtml build exits 0
  - Sphinx linkcheck build exits 0 or only pre-existing failures
  - Sphinx doctest build exits 0 or only pre-existing failures
  - pre-commit exits 0
  - pytest exits 0
  - No .rst files found
  - No stale .rst references in tracked files
todos_waived_reason: null
---

# Migrate Sphinx Docs from RST to MyST Markdown

## Summary

Convert all nine Sphinx documentation source files from reStructuredText (.rst) to MyST
Markdown (.md) while preserving rendered content, navigation, API reference, page URLs,
theme, and Read the Docs setup. This is a focused format migration, not a content
rewrite.

## Implementation Changes

1. **Dependencies**: Add `myst-parser>=2.0.0` to `docs/requirements.txt`
2. **Sphinx Config**: Update `docs/conf.py` with MyST parser settings
3. **File Conversions**: Convert all 9 .rst files to .md using rst-to-myst tool + manual
   review
4. **Maintainer Docs**: Update `docs/README.md` to describe MyST workflow
5. **Pre-commit**: Remove .rst exclusion from `.pre-commit-config.yaml`
6. **Codecrate**: Remove .rst include pattern from `.codecrate.toml`

## Tests

- Sphinx html build with `-W --keep-going` (warnings as errors)
- Sphinx dirhtml, linkcheck, doctest builds
- `pre-commit run --all-files`
- `pytest` test suite
- Verify no .rst files remain
- Verify no stale .rst references in tracked files
- Verify all 24 autodoc objects render
- Verify toctree order unchanged
- Verify README.md excluded from Sphinx

## Assumptions

- Network-enabled environment for pip installs and Sphinx builds
- Current docs have no pre-existing blocking warnings
- RTD theme and builder behavior remain unchanged
- No content corrections during format migration

## Out of Scope

- Content corrections (URLs, license text, Python versions, legacy API examples)
- Theme changes (staying with sphinx_rtd_theme)
- Builder changes (staying with html, not dirhtml)
- Switching from sphinx.ext.autodoc to sphinx-autodoc2
- Enabling linkify, commonmark_only, or gfm_only
- Adding Furo, Poetry, or sphinx-copybutton
- Rewriting NumPy-style docstrings as Markdown
