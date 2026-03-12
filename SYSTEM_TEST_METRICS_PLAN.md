# System Test Metrics Collection Plan

## Purpose

This document outlines how to collect metrics from the travel agent test data (`tests/system/test_data_travel_agent.json`) using `replay_session_script.py` to validate the requirements in `system_test.md`.

## Validation Mapping

### system_test.md Requirements → Metrics

| Requirement | How to Measure | Tool | Expected Result |
|-------------|----------------|------|-----------------|
| **O(1) latency** | PUT/GET latency per message | `replay_session_script.py` | < 100ms PUT, < 50ms GET, no growth |
| **Summarization triggers** | Context field appears | Snapshot `context_present` | `true` when window fills |
| **Recent messages preserved** | Message count after summarization | Snapshot `visible_message_count` | Last 8-10 messages visible |
| **Message order** | Message IDs in response | Snapshot `visible_message_ids` | Chronological order maintained |
| **Session readable after summarization** | GET succeeds | Final GET response | 200 status, valid JSON |

## Test Scenarios from Travel Agent Data

### 1. Short Conversation (10 messages)

**File**: `tests/system/test_data_travel_agent.json` → `short_conversation`

**Scenario**: Weekend trip to Paris
- User: Solo traveler, vegetarian, interested in museums
- Budget: $2000-2500
- Messages: 10 (5 user, 5 assistant)

**Validates**:
- ✅ Basic message storage
- ✅ O(1) latency baseline
- ✅ Message ordering
- ✅ No summarization (too short)

**Replay Command**:
```bash
python3 replay_session_script.py \
  <fixture_file> \
  --base-url http://localhost:8001 \
  --session-id weekend-paris \
  --user-id test-user-001 \
  --namespace travel-agent \
  --reset-session \
  --snapshot-file metrics/short_conversation.jsonl
```

**Expected Metrics**:
- Turns replayed: 10
- Summary first seen: `null` (no summarization)
- Final visible messages: 10
- PUT latency: avg < 50ms, p95 < 100ms
- GET latency: avg < 30ms, p95 < 50ms

---

### 2. Greece Trip with Summarization (13 messages)

**File**: `tests/system/test_data_travel_agent.json` → `greece_trip`

**Scenario**: Anniversary trip to Greek islands
- Destinations: Santorini, Mykonos, Crete
- Budget: $5000-7000
- Messages: 13 (includes iterative planning)

**Validates**:
- ✅ Summarization triggers with small context window
- ✅ Recent messages preserved
- ✅ Summary contains key information
- ✅ Latency remains O(1) after summarization

**Replay Command**:
```bash
python3 replay_session_script.py \
  <fixture_file> \
  --base-url http://localhost:8001 \
  --session-id greece-anniversary \
  --user-id test-user-002 \
  --namespace travel-agent \
  --context-window-max 4000 \
  --reset-session \
  --snapshot-file metrics/greece_trip.jsonl
```

**Expected Metrics**:
- Turns replayed: 13
- Summary first seen: turn 8-10 (when context fills)
- Final visible messages: 8-10 (recent messages)
- Final context length: > 500 chars
- PUT/GET latency: Similar to short conversation (validates O(1))

**Key Validation**: Check snapshot file for turn where `context_present` changes from `false` to `true`

---

### 3. Returning Client - Multiple Trips

**File**: `tests/system/test_data_travel_agent.json` → `returning_client_scenario`

**Scenario**: Sarah's 3 trips over 16 months
- Trip 1: Paris (June 2023) - Solo, $2500
- Trip 2: Italy (March 2024) - With partner, $6000
- Trip 3: Japan (October 2024) - Honeymoon, $12000

**Validates**:
- ✅ Multiple sessions for same user
- ✅ Sessions retrievable independently
- ✅ Consistent latency across trips
- ✅ User context preserved (user_id linkage)

**Replay Commands** (run separately):
```bash
# Trip 1
python3 replay_session_script.py \
  <trip_1_fixture> \
  --base-url http://localhost:8001 \
  --session-id trip-1-paris-2023 \
  --user-id sarah-johnson-001 \
  --namespace travel-agent \
  --reset-session \
  --snapshot-file metrics/trip_1_paris.jsonl

# Trip 2
python3 replay_session_script.py \
  <trip_2_fixture> \
  --base-url http://localhost:8001 \
  --session-id trip-2-italy-2024 \
  --user-id sarah-johnson-001 \
  --namespace travel-agent \
  --reset-session \
  --snapshot-file metrics/trip_2_italy.jsonl

# Trip 3
python3 replay_session_script.py \
  <trip_3_fixture> \
  --base-url http://localhost:8001 \
  --session-id trip-3-japan-2024 \
  --user-id sarah-johnson-001 \
  --namespace travel-agent \
  --reset-session \
  --snapshot-file metrics/trip_3_japan.jsonl
```

**Expected Metrics**:
- Each trip: 5 messages
- All trips: Same user_id (sarah-johnson-001)
- Latency: Consistent across all 3 trips
- Sessions: All independently retrievable

---

## Metrics Analysis

### Latency Analysis (O(1) Validation)

From snapshot files, extract `put_latency_ms` and `get_latency_ms` for each turn:

```python
import json

latencies = []
with open('metrics/greece_trip.jsonl') as f:
    for line in f:
        snapshot = json.loads(line)
        latencies.append({
            'turn': snapshot['turn_index'],
            'put_ms': snapshot['put_latency_ms'],
            'get_ms': snapshot['get_latency_ms'],
        })

# Check if latency grows with turn number
# O(1) means no correlation between turn_index and latency
```

**Pass Criteria**: No significant correlation between `turn_index` and latency

### Summarization Analysis

From snapshot files, find when summarization occurs:

```python
for line in open('metrics/greece_trip.jsonl'):
    snapshot = json.loads(line)
    if snapshot['context_present']:
        print(f"Summarization first occurred at turn {snapshot['turn_index']}")
        print(f"Context length: {snapshot['context_length']}")
        print(f"Visible messages: {snapshot['visible_message_count']}")
        break
```

**Pass Criteria**:
- `context_present` becomes `true` when context window fills
- `visible_message_count` < total messages (older ones summarized)
- `context_length` > 0

---

## Report Template

After collecting metrics, report against system_test.md:

```markdown
## Metrics Report for system_test.md

### Test 1: Short Conversation
- ✅ O(1) latency: PUT avg 45ms, GET avg 28ms
- ✅ All 10 messages preserved
- ✅ Messages in chronological order
- ✅ No summarization (as expected)

### Test 2: Greece Trip with Summarization
- ✅ Summarization triggered at turn 9
- ✅ Recent 8 messages preserved
- ✅ Summary length: 1247 chars
- ✅ O(1) latency maintained: PUT avg 48ms, GET avg 30ms

### Test 3: Returning Client
- ✅ All 3 trips stored successfully
- ✅ Consistent latency across trips
- ✅ Sessions independently retrievable
- ✅ User context preserved (user_id)

### Conclusion
All requirements from system_test.md validated ✅
```

