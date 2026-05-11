---
name: agent-memory
version: 0.5.0
description: |
  (Redis Agent Memory Server.)
  Use this skill whenever the user asks to (list 3-5 concrete triggers).
runtime:
  language: python
  package: agent_memory_server
  install: "pip install agent_memory_server"
links:
  docs: https://ai.redis.io/agent-memory/
  llms_txt: https://ai.redis.io/agent-memory/llms.txt
  llms_full_txt: https://ai.redis.io/agent-memory/llms-full.txt
  repository: https://github.com/redis/agent-memory-server
---

# Agent Memory Agent Skill

## When to use

- (List 3-5 trigger phrases / situations. Be specific about what this
  library does that no other portfolio repo does.)

Do **not** use this skill for (...). Fall back to {{ALTERNATIVE_LIBRARY}}
for those cases.

## Minimal install

```bash
pip install "agent_memory_server>=...,<..."
```

State any optional extras with one-line context:

```bash
pip install "agent_memory_server[extra]"   # adds X
```

## Core patterns

### 1. (Most common usage)

```python
# 10-15 lines maximum. Show the library doing the thing it is best at.
```

### 2. (Second most common)

```python
# ...
```

### 3. (Third most common)

```python
# ...
```

(Add a 4th and 5th if there is room. Stop at 5; SKILL.md is a prompt, not a
manual.)

## Common gotchas

- **Async vs sync**: (if both are present, mention not to mix them)
- **Type X must be Y**: (e.g., vector dtype must be float32)
- **Reserved characters**: (escape rules, if relevant)
- **Storage type**: (hash vs JSON, schema-time-only choice)

## Agent execution policy

When this skill is loaded:

1. Always confirm the user's Redis version.
2. (Other policy items, e.g., "Prefer `IndexSchema.from_yaml` over inline
   dicts", "Always use the async client when called from async code".)
3. Never invent classes; only those documented at `links.docs`.

## Reference

- API reference: https://ai.redis.io/agent-memory/api/
- User guide: https://ai.redis.io/agent-memory/user_guide/
- Examples: (link to recipes if applicable)
