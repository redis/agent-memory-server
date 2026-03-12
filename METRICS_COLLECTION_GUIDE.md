# Metrics Collection Guide for system_test.md Validation

## Overview

This guide explains how to collect metrics from the travel agent test data using `replay_session_script.py` to validate the requirements in `system_test.md`.

## Prerequisites

1. **Start the Agent Memory Server**:
   ```bash
   source .venv/bin/activate
   uv run agent-memory api --port 8001
   ```

2. **Ensure Redis is running**:
   ```bash
   docker-compose up redis
   ```

3. **Set API keys**:
   ```bash
   export OPENAI_API_KEY=your-key-here
   ```

## Test Scenarios

### Scenario 1: Short Conversation (10 messages)

**Purpose**: Validate O(1) latency and basic message storage

**Data**: `tests/system/test_data_travel_agent.json` → `short_conversation`

**Command**:
```bash
python3 replay_session_script.py \
  temp_fixtures/short_weekend_trip.json \
  --base-url http://localhost:8001 \
  --model-name gpt-4o-mini \
  --reset-session \
  --snapshot-file metrics/short_conversation.jsonl
```

**Expected Metrics** (from system_test.md):
- ✅ PUT latency: < 100ms per message
- ✅ GET latency: < 50ms per message
- ✅ All 10 messages preserved
- ✅ Messages in chronological order
- ✅ No summarization (conversation too short)

---

### Scenario 2: Greece Trip with Summarization

**Purpose**: Validate summarization behavior when context window fills

**Data**: `tests/system/test_data_travel_agent.json` → `greece_trip`

**Command**:
```bash
python3 replay_session_script.py \
  temp_fixtures/greece_trip.json \
  --base-url http://localhost:8001 \
  --model-name gpt-4o-mini \
  --context-window-max 4000 \
  --reset-session \
  --snapshot-file metrics/greece_trip.jsonl
```

**Expected Metrics** (from system_test.md):
- ✅ Summary created when context window fills
- ✅ Recent messages (last 8-10) still present as full messages
- ✅ Summary contains key information (destinations, budget, preferences)
- ✅ PUT/GET latency remains O(1) even after summarization
- ✅ Message order preserved

---

### Scenario 3: Returning Client - Multiple Trips

**Purpose**: Validate long-term memory across multiple sessions

**Data**: `tests/system/test_data_travel_agent.json` → `returning_client_scenario`

**Commands** (run each trip separately):
```bash
# Trip 1: Paris (June 2023)
python3 replay_session_script.py \
  temp_fixtures/trip_1_paris.json \
  --base-url http://localhost:8001 \
  --session-id trip-1-paris-2023 \
  --user-id sarah-johnson-001 \
  --namespace travel-agent \
  --reset-session \
  --snapshot-file metrics/trip_1_paris.jsonl

# Trip 2: Italy (March 2024)
python3 replay_session_script.py \
  temp_fixtures/trip_2_italy.json \
  --base-url http://localhost:8001 \
  --session-id trip-2-italy-2024 \
  --user-id sarah-johnson-001 \
  --namespace travel-agent \
  --reset-session \
  --snapshot-file metrics/trip_2_italy.jsonl

# Trip 3: Japan (October 2024)
python3 replay_session_script.py \
  temp_fixtures/trip_3_japan.json \
  --base-url http://localhost:8001 \
  --session-id trip-3-japan-2024 \
  --user-id sarah-johnson-001 \
  --namespace travel-agent \
  --reset-session \
  --snapshot-file metrics/trip_3_japan.jsonl
```

**Expected Metrics**:
- ✅ Each session stored independently
- ✅ All sessions retrievable by session_id
- ✅ Sessions linked by user_id (sarah-johnson-001)
- ✅ Consistent latency across all trips

---

## Interpreting Metrics

### Latency Metrics (from snapshot files)

Each JSONL snapshot contains per-turn metrics:
```json
{
  "turn_index": 5,
  "put_latency_ms": 45.23,
  "get_latency_ms": 23.45,
  "visible_message_count": 5,
  "context_present": false,
  "context_length": 0
}
```

**What to check**:
- `put_latency_ms` should be < 100ms (O(1) requirement)
- `get_latency_ms` should be < 50ms
- Latency should NOT increase with `turn_index` (validates O(1))

### Summarization Metrics

When summarization occurs:
```json
{
  "turn_index": 15,
  "context_present": true,
  "context_length": 1247,
  "visible_message_count": 8,
  "context_percentage_total_used": 68.5
}
```

**What to check**:
- `context_present` becomes `true` when summarization triggers
- `visible_message_count` drops (older messages summarized)
- `context_length` > 0 (summary text exists)
- Recent messages still in `visible_message_ids`

---

## Mapping to system_test.md Requirements

| Requirement | Metric | Pass Criteria |
|-------------|--------|---------------|
| O(1) latency | `put_latency_ms` | < 100ms, no growth with conversation length |
| Summarization triggers | `context_present` | `true` when context window fills |
| Recent messages preserved | `visible_message_count` | Last 8-10 messages still visible |
| Message order | `visible_message_ids` | IDs in chronological order |
| Session readable after summarization | Final GET succeeds | 200 status, valid response |

---

## Automated Metrics Collection

Use the provided `run_travel_agent_replay.py` script:

```bash
python3 run_travel_agent_replay.py
```

This will:
1. Create conversation fixtures from `test_data_travel_agent.json`
2. Run replay script for each scenario
3. Save metrics to `metrics/*.jsonl`
4. Print summary report

---

## Next Steps

1. Run the replay scripts for each scenario
2. Analyze the JSONL snapshot files
3. Validate metrics against system_test.md requirements
4. Document any failures or performance issues
5. Include metrics in team review

