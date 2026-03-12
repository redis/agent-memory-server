# System Testing Guide

> **Latest Test Results**: See [Test Results](#test-results-2026-03-12) section below for current status (19/25 passing, 6 failures documented)

## Overview

This directory contains comprehensive system tests for the Agent Memory Server, focusing on production readiness validation for the **Long Conversation Memory** feature.

**Purpose**: Find breaking points before customers do through realistic, production-scale testing.

## Test Suites

### 1. Scale Tests (`test_long_conversation_scale.py`)

Tests system behavior under load with varying conversation sizes and concurrency.

**Test Classes**:
- `TestLongConversationPrepare`: Create conversations of various sizes
- `TestLongConversationRun`: Operational scenarios (parallel, concurrent)
- `TestLongConversationCheck`: Correctness validation (summarization, ordering)
- `TestScaleMetrics`: Comprehensive performance reporting

**Configuration** (via environment variables):
- `SCALE_SHORT_MESSAGES`: Default 10
- `SCALE_MEDIUM_MESSAGES`: Default 50
- `SCALE_LONG_MESSAGES`: Default 200
- `SCALE_PARALLEL_SESSIONS`: Default 5
- `SCALE_CONCURRENT_UPDATES`: Default 10

### 2. Travel Agent Scenarios (`test_travel_agent_scenarios.py`)

Tests with realistic travel planning conversations to validate domain-specific behavior.

**Test Classes**:
- `TestTravelAgentShortConversations`: Weekend trip inquiries (10 messages)
- `TestTravelAgentMediumConversations`: Family vacation planning (50 messages)
- `TestTravelAgentLongConversations`: Honeymoon planning (200 messages)
- `TestTravelAgentConcurrentScenarios`: Multi-agent updates, parallel clients
- `TestTravelAgentSummarization`: Summarization quality with domain keywords
- `TestReturningClientScenarios`: **Multiple trips over time, long-term memory**

**Key Scenarios**:
- **Short**: Paris weekend trip - vegetarian, museums, cooking class
- **Medium**: Family Italy trip - allergies, kid-friendly, multiple cities
- **Long**: European honeymoon - multi-phase planning, 200 messages
- **Returning Client**: Sarah's journey - 3 trips over 16 months (Paris → Italy → Japan)

### 3. Returning Client Scenario (Critical)

**The Problem**: Testing "multiple different trips with the same user who would then have older memories"

**Sarah's Journey**:
```
Trip 1: Paris (June 2023)     - Solo, $2,500, establishes preferences
Trip 2: Italy (March 2024)    - With partner, $6,000, references Trip 1
Trip 3: Japan (October 2024)  - Honeymoon, $12,000, references Trips 1 & 2
```

**What Gets Tested**:
1. **Multiple Sessions, Same User**: 3 separate sessions linked by user_id
2. **Long-term Memory Creation**: Preferences, history, relationships, patterns
3. **Context Switching**: Planning Greece, asks "What hotel in Florence?"
4. **Preference Recognition**: System knows she's vegetarian without asking

**Expected Long-term Memories**:
- Preferences: Vegetarian, cultural experiences, boutique hotels
- History: Trip records with budgets and destinations
- Relationships: Solo → Couple → Engaged → Married
- Patterns: Budget trend ($2.5K → $6K → $12K)

## Running Tests

### Quick Start
```bash
# Prerequisites
source .venv/bin/activate
docker-compose up redis
uv run agent-memory api --port 8001
export OPENAI_API_KEY=your-key

# Run all system tests
make test-system

# Run specific suites
make test-system-quick          # Fast smoke test
make test-travel-agent          # Travel agent scenarios only
make test-system-production     # Full production scale
```

### Individual Test Classes
```bash
# Scale tests
uv run pytest tests/system/test_long_conversation_scale.py --run-api-tests -v -s

# Travel agent scenarios
uv run pytest tests/system/test_travel_agent_scenarios.py --run-api-tests -v -s

# Returning client only
uv run pytest tests/system/test_travel_agent_scenarios.py::TestReturningClientScenarios --run-api-tests -v -s
```

## What We're Looking For

### ✅ Success Criteria

**Performance**:
- Message storage: < 50ms per message
- Update operations: < 200ms average
- No degradation with conversation length (O(1) latency)
- Parallel sessions complete without timeouts

**Correctness**:
- All messages in chronological order
- Recent messages always preserved
- Summarization triggers when context window fills
- No data loss during concurrent updates

**Long-term Memory**:
- Preferences extracted across multiple sessions
- Cross-session search returns relevant results
- Context switching works (retrieve old session during new planning)
- Consistent preferences recognized

### ❌ What We Expect to Break

From `long_conversation_memory.md`:
- Recent messages getting lost
- Messages coming back in wrong order
- Summaries not appearing or being empty
- Session reads becoming inconsistent after many updates
- Long-term memories not being created
- Context switching failing or corrupting sessions
- Preference recognition not working across trips

## Test Data

### Scale Test Data
- **Short**: 10 generic messages
- **Medium**: 50 generic messages
- **Long**: 200 generic messages
- **Very Large**: Individual messages with 5000+ characters

### Travel Agent Data (`test_data_travel_agent.json`)
- **Short Conversation**: Paris weekend (10 messages)
- **Medium Conversation**: Italy family trip (50 messages)
- **Long Conversation**: European honeymoon (200 messages)
- **Returning Client**: Sarah's 3 trips with cross-references
- **Concurrent Updates**: Multiple agents updating same booking

## Critical Test Cases

### 1. Summarization Quality
```python
# Test: test_summarization_with_greece_trip
# Expected: Summary contains keywords (Greece, Santorini, islands, budget)
# Expected: Recent messages preserved (mentions of Crete)
# Expected: Context percentage > 60% when summarization occurs
```

### 2. Context Switching
```python
# Test: test_context_switching_in_conversation
# Scenario: Planning Greece, user asks about Florence hotel from Italy trip
# Expected: Retrieve Italy session, answer question, return to Greece
# Expected: Both sessions remain intact
```

### 3. Preference Consistency
```python
# Test: test_preference_consistency_across_trips
# Expected: Vegetarian in all 3 trips
# Expected: Cultural experiences in all 3 trips
# Expected: Long-term memories reflect these patterns
```

### 4. Concurrent Updates
```python
# Test: test_multiple_agents_updating_booking
# Scenario: 4 agents update same session simultaneously
# Expected: All updates present in final state
# Expected: No data loss or corruption
```

## Files

- `test_long_conversation_scale.py`: Scale and performance tests
- `test_travel_agent_scenarios.py`: Domain-specific realistic scenarios
- `test_data_travel_agent.json`: Realistic conversation data
- `travel_agent_data.py`: Data loader and generator
- `run_scale_tests.sh`: Convenience script with profiles
- `README_CONSOLIDATED.md`: This file
- `SYSTEM_TESTING.md`: High-level overview (can be removed)
- `GETTING_STARTED.md`: Quick start guide (can be removed)
- `TRAVEL_AGENT_SCENARIOS.md`: Detailed scenarios (can be removed)
- `RETURNING_CLIENT_README.md`: Returning client guide (can be removed)

## Test Results (2026-03-12)

### Summary: 19/25 PASSED (76%)

**We found real bugs before customers did.**

### ✅ PASSING (19 tests)
- Scale Tests: 12/12 PASS
- Travel Agent Tests: 7/13 PASS

### ❌ FAILING (6 tests)

**1. Session Retrieval After Storage (CRITICAL)**
- Test: `test_retrieve_and_search_weekend_trip`
- Error: 404 Not Found when retrieving session after storage
- Impact: User stores conversation, all data is gone

**2. Concurrent Agent Updates Lost (CRITICAL)**
- Test: `test_multiple_agents_updating_booking`
- Error: Only 1 of 4 concurrent updates survives
- Impact: 75% of concurrent updates disappear

**3. Summarization Session Not Persisted (HIGH)**
- Test: `test_summarization_with_greece_trip`
- Error: 404 Not Found after summarization
- Impact: Long conversations disappear

**4. Multi-Trip Sessions Not Retrievable (CRITICAL)**
- Test: `test_three_trips_same_client`
- Error: 404 Not Found on returning client sessions
- Impact: Returning client feature broken

**5. Wrong API Method Name (TEST BUG)**
- Test: `test_long_term_memory_creation`
- Error: `create_long_term_memories` vs `create_long_term_memory`
- Impact: None - test bug

**6. Context Switching Session Not Found (CRITICAL)**
- Test: `test_context_switching_in_conversation`
- Error: 404 Not Found when retrieving previous session
- Impact: Context switching impossible

### Root Cause

**Pattern 1: Session Persistence Failure** (5 failures)
- PUT succeeds, GET returns 404
- Sessions not persisting to Redis

**Pattern 2: Concurrent Update Data Loss** (1 failure)
- Race condition or last-write-wins

### Bottom Line

- ✅ Good: 76% pass rate, basic functionality works
- ❌ Bad: Session persistence is broken
- 🚫 Blocker: Cannot test returning client until fixed

**Mission accomplished: Found bugs before customers did.**

## Next Steps

1. ✅ Run all tests and generate report
2. ✅ Identify what breaks
3. Fix critical issues (session persistence)
4. Re-run and validate
5. Establish performance baselines
6. Integrate into CI/CD

