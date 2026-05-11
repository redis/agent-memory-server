---
description: Internal AI-agent guide to the Agent Memory Server source tree.
---

# For AI Agents Modifying Agent Memory

This section is the internal counterpart to the user-facing
[AGENTS.md](https://github.com/redis/agent-memory-server/blob/main/AGENTS.md).
It exists for an agent that has been asked to *change* the library: add a
new feature, fix a bug, extend a subsystem.

Start with the [**Repository map**](REPOSITORY_MAP.md) to find the
subsystem you need. Run [**Build and test**](BUILD_AND_TEST.md) before
and after any change. If something looks broken, check
[**Failure modes**](FAILURE_MODES.md) before assuming it's a bug —
several behaviors are intentional and documented there.

<div class="grid cards" markdown>

-   :material-map:{ .lg .middle } **[Repository map](REPOSITORY_MAP.md)**

    ---

    Top-down map of every package and module with its responsibility.

-   :material-hammer-wrench:{ .lg .middle } **[Build and test](BUILD_AND_TEST.md)**

    ---

    The exact commands CI runs, plus the local equivalents.

-   :material-alert-circle:{ .lg .middle } **[Failure modes](FAILURE_MODES.md)**

    ---

    Intentional behaviors that look like bugs and where they live.

-   :material-book-edit:{ .lg .middle } **[Authoring standard](AUTHORING_STANDARD.md)**

    ---

    System prompt for generating or revising docs in this repo.

</div>

## Decision tree

| Task | Read first |
|---|---|
| Run tests | [Build and test](BUILD_AND_TEST.md) |
| Diagnose "this looks broken" | [Failure modes](FAILURE_MODES.md) before assuming a bug |
| Find a subsystem | [Repository map](REPOSITORY_MAP.md) |
| Write or revise docs | [Authoring standard](AUTHORING_STANDARD.md) |
