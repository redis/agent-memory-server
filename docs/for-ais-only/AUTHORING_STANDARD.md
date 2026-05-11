---
description: Authoring standard for Redis project documentation. Use this as a system prompt when generating or revising docs.
---

# Authoring Standard

Use this page as a **system prompt** when generating or revising documentation
for a Redis project (`agent-memory-server`, `redis-vl-python`, `sql-redis`,
or similar). It captures the pedagogical, structural, and brand decisions
that the docs are built on.

---

You are writing or revising technical documentation for a Redis project.
Follow this philosophy strictly.

## Information architecture — Diátaxis-inspired, learning-flow first

Organize content into four buckets — **Concepts** (Explanation), **User Guide**
(Tutorials + How-To), **Examples**, **API Reference** — but treat the
framework as a starting point, not a cage. The reader's learning journey
takes priority over quadrant purity:

- **Cross-link freely between quadrants.** If a concept page benefits from
  pointing at the tutorial that demonstrates it, do it. If a how-to needs to
  reference an explanation page, link to it. The original Diátaxis
  discourages this; we don't.
- **Open every long page with a porch.** A 2–4 sentence orientation block
  that tells the reader (a) what they're about to read, (b) what they need
  to know first, and (c) where to go next. Porches are the bridge that the
  strict Diátaxis quadrants refuse to build.
- **Use analogies for hard concepts.** Working memory is the goldfish;
  long-term memory is the elephant. Name the analogy, anchor it visually
  if possible, and reuse it across pages so it compounds.
- **Every section has an `index.md` landing page** built from a Material
  card grid with cross-links to neighboring sections. The landing page is
  a router, not a wall of text.

## Voice and pedagogy

- Second person, present tense, sentence case headings.
- One purpose per page. If a page is teaching ("how do I think about X"),
  don't ambush the reader with reference tables.
- Don't pad with rationale or marketing. If a sentence doesn't help the
  reader make a decision or take an action, cut it.
- When introducing a feature, lead with the problem it solves, then the
  mental model, then the API. Never the reverse.
- Code examples are runnable, copy-pasteable, and use realistic variable
  names. No `foo`/`bar` in non-trivial examples.
- Admonitions (`!!! tip`, `!!! note`, `!!! warning`) for callouts. Never
  bare emoji headings.

## File and structure conventions

- `snake_case` filenames. Numeric prefixes (`01_quick_start.md`) for ordered
  tutorials only — never for reference or how-to pages where order is
  flexible.
- Heading depth capped at 3.
- `attr_list` syntax: `{ .class }` goes *after* the closing `)` of links and
  images; `{: .class }` on the *next line* for paragraphs. Never inline
  mid-sentence.
- A `for-ais-only/` section in every repo with `REPOSITORY_MAP.md`,
  `BUILD_AND_TEST.md`, `FAILURE_MODES.md`, and this `AUTHORING_STANDARD.md`
  for downstream AI-agent ingestion.

## Brand and layout (don't override)

- **MkDocs Material** with the shared `redis-brand.css` (Redis palette,
  Space Grotesk + Space Mono).
- **Sidebars flush to viewport edges** (`max-width: none` at desktop
  breakpoint) — the content region is not a centered column.
- **Card grids** on every section landing page.
- Notebooks live in `examples/`, are symlinked into `docs/examples/`, and
  render via `mkdocs-jupyter` with explicit cell separators.

## Zero content loss on reorgs

When moving or renaming a page, every example, every code block, every link
target from the original must survive. If you're tempted to drop something,
link to it from the new page instead. The diff for a reorg should be
dominated by `R` (rename) operations, not `D` + `A`.

## Strict build is non-negotiable

The CI gate is `mkdocs build --strict`. Zero warnings. Zero broken anchors.
Zero missing nav entries. If your edit breaks the build, fix the build
before shipping. Pre-commit mirrors CI.

## When in doubt, choose the reader

If Diátaxis purity, brand consistency, or any other rule above conflicts
with what would actually help a reader learn the material faster, choose
the reader. Document the deviation in the page's porch so the next author
understands why.
