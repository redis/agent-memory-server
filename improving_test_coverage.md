# ✅ Test Coverage Improvement Plan

NOTE: Keep track of your work at the end of the file after "PROGRESS ON WORK."
NOTE: The command to run coverage is: `uv run pytest --cov agent_memory_server --cov-report=term-missing --run-api-tests`

## 🎯 Goal
Increase total coverage from **51% to 75%** with minimal disruption and maximum impact.

---

## 🔝 Priority Targets (Uncovered Critical Logic)

### 1. `cli.py` (0%)
- **Tests to write:**
  - `version`, `api`, `mcp`, `schedule_task`, `task_worker`, `migrate_memories`, `rebuild_index`
- **Approach:** Use `click.testing.CliRunner`. Mock Redis and async functions.

### 2. `long_term_memory.py` (46%)
- **Tests to write:**
  - `promote_working_memory_to_long_term`, `index_long_term_memories`, `search_memories`
- **Approach:** Parametrize edge cases (no matches, Redis down, invalid filter). Mock Redis and vector indexing.

### 3. `summarization.py` (14%)
- **Tests to write:**
  - `_incremental_summary`, edge cases for token limits and empty input
- **Approach:** Mock LLM client. Use short token windows to force edge behavior.

### 4. `docket_tasks.py` (0%)
- **Tests to write:**
  - Task registration, task collection membership, verify Redis URL use
- **Approach:** Mock Docket client, test task registration logic

---

## ⚙️ Medium-Priority

### 5. `filters.py` (51%)
- **Tests to write:**
  - Each filter type (`eq`, `gt`, `between`, etc.)
- **Approach:** Use parametric testing, validate Redis query expressions

### 6. `llms.py` (66%)
- **Tests to write:**
  - Model selection logic, token limit behaviors, prompt formatting
- **Approach:** Mock OpenAI and Anthropic clients

### 7. `mcp.py` (66%)
- **Tests to write:**
  - Tool interface dispatch, SSE vs stdio routing
- **Approach:** Patch `run_stdio_async`, `run_sse_async`, assert logs and Redis effects

---

## 🧹 Low-Hanging Fruit (Fast Wins)

### 8. `utils/api_keys.py` (0%)
- **Add tests for:** key parsing, fallback logic

### 9. `logging.py` (50%)
- **Add tests for:** `configure_logging`, custom level configs

### 10. `dependencies.py` (92%)
- **Add tests for:** `add_task` fallback path (no Docket)

---

## 🧪 Integration Tests

- Test full flow:
  - POST to `working_memory` → promotion task triggers → data lands in `long_term_memory`
  - GET from memory search endpoint returns expected result

---

## 🗂️ Progress Tracking Template

| File | Coverage Before | Target | Status |
|------|------------------|--------|--------|
| `cli.py` | 0% | 80%+ | ⬜️ |
| `long_term_memory.py` | 46% | 70%+ | ⬜️ |
| `summarization.py` | 14% | 60%+ | ⬜️ |
| `filters.py` | 51% | 75% | ⬜️ |
| `llms.py` | 66% | 80% | ⬜️ |
| `docket_tasks.py` | 0% | 80% | ⬜️ |
| `mcp.py` | 66% | 85% | ⬜️ |
| `api_keys.py` | 0% | 100% | ⬜️ |
| `logging.py` | 50% | 100% | ⬜️ |
| `dependencies.py` | 92% | 100% | ⬜️ |

---

## 🧰 Tools & Tips

- Use `pytest-cov` with `--cov-report=term-missing`
- Use `pytest --durations=10` to find slow tests
- Group new tests by file in `tests/unit` or `tests/integration`

---

## 📈 Exit Criteria

- At least **75%** overall test coverage
- 80%+ coverage on top priority files
- All major logic paths exercised with mocks or real integration

---

## PROGRESS ON WORK
