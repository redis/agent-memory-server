# System Testing for Production Readiness

This document provides an overview of the system testing harness built to validate the Agent Memory Server's production readiness, specifically for the **Long Conversation Memory** use case.

## Overview

System tests validate end-to-end behavior at production-like scale. They complement unit and integration tests by:

- Testing complete user workflows
- Validating performance at scale
- Verifying behavior under concurrent load
- Ensuring correctness after summarization
- Measuring real-world latencies

## Quick Start

### Prerequisites

1. **Running server** on port 8001
2. **Redis** running and accessible
3. **API keys** set (OPENAI_API_KEY or ANTHROPIC_API_KEY)

### Run Tests

```bash
# Quick smoke test (2-3 minutes)
make test-system-quick

# Standard test (5-10 minutes)
make test-system

# Production-scale test (15-30 minutes)
make test-system-production
```

## What's Being Tested

Based on `long_conversation_memory.md`, the tests validate:

### ✅ Storage Performance
- **O(1) latency**: Conversation storage doesn't degrade with length
- **Consistent performance**: Latency remains stable across operations
- **Parallel sessions**: Multiple sessions don't interfere

### ✅ Summarization
- **Automatic triggering**: Summarization occurs when context window fills
- **Summary quality**: Older messages are properly condensed
- **Context preservation**: Important information is retained

### ✅ Message Integrity
- **Recent messages**: Always preserved regardless of summarization
- **Chronological order**: Messages stay in correct sequence
- **No data loss**: All updates are captured

### ✅ Functionality
- **Session reads**: Work correctly after summarization
- **Memory prompts**: Include relevant context
- **Concurrent updates**: Handled without conflicts

## Test Structure

```
tests/system/
├── test_long_conversation_scale.py  # Main test suite
├── README.md                        # Detailed documentation
├── GETTING_STARTED.md              # Quick start guide
├── run_scale_tests.sh              # Convenience script
└── __init__.py
```

### Test Classes

1. **TestLongConversationPrepare**: Create conversations of various sizes
2. **TestLongConversationRun**: Test operational scenarios
3. **TestLongConversationCheck**: Validate correctness
4. **TestScaleMetrics**: Comprehensive reporting

See the [architecture diagram](#system-test-architecture) for visual overview.

## Configuration

Control test scale with environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SCALE_SHORT_MESSAGES` | 10 | Messages in short conversations |
| `SCALE_MEDIUM_MESSAGES` | 50 | Messages in medium conversations |
| `SCALE_LONG_MESSAGES` | 200 | Messages in long conversations |
| `SCALE_PARALLEL_SESSIONS` | 5 | Concurrent sessions to create |
| `SCALE_CONCURRENT_UPDATES` | 10 | Simultaneous updates to test |

## Example Output

```
✅ Short conversation (10 msgs) stored in 0.234s
   Latency per message: 23.40ms

✅ Medium conversation (50 msgs) stored in 0.891s
   Latency per message: 17.82ms

✅ 5 parallel sessions created
   Total time: 1.234s
   Average session latency: 0.247s

✅ Summarization test completed
   Summary created: True
   Messages retained: 23 (started with 100)
   Context percentage used: 68.5%

✅ Message order preserved
   All messages in chronological order: ✓

========================================
✅ SCALE TEST COMPLETE
========================================
```

## Success Criteria

### Performance Benchmarks

- **Short conversations**: < 100ms per message
- **Medium conversations**: < 50ms per message
- **Long conversations**: < 20ms per message
- **Update operations**: < 200ms average
- **Parallel sessions**: Complete without timeouts

### Correctness Requirements

- ✅ All messages in chronological order
- ✅ Recent messages always preserved
- ✅ Summarization triggers when needed
- ✅ Memory prompts include context
- ✅ No data loss during concurrent updates

## Integration with CI/CD

### Pre-Deployment Checklist

1. ✅ Run `make test-system-production`
2. ✅ Verify all tests pass
3. ✅ Review performance metrics
4. ✅ Compare to baseline
5. ✅ Document any regressions
6. ✅ Get approval for deployment

### Continuous Monitoring

After deployment, monitor:
- Message storage latency
- Summarization frequency
- Session read performance
- Update operation latency

Compare production metrics to test baselines.

## Documentation

- **[tests/system/README.md](tests/system/README.md)**: Comprehensive documentation
- **[tests/system/GETTING_STARTED.md](tests/system/GETTING_STARTED.md)**: Quick start guide
- **[long_conversation_memory.md](long_conversation_memory.md)**: Requirements specification

## Troubleshooting

### Common Issues

**Server not reachable**
```bash
uv run agent-memory api --port 8001
```

**No API keys**
```bash
export OPENAI_API_KEY=sk-...
```

**Tests timeout**
- Reduce scale parameters
- Check server/Redis performance
- Review logs for bottlenecks

**Summarization not triggering**
- Increase message count/size
- Reduce context_window_max
- This may be expected behavior

## Next Steps

1. **Review** the test output and architecture
2. **Run** quick smoke test to validate setup
3. **Customize** scale parameters for your use case
4. **Establish** baseline metrics for your environment
5. **Integrate** into your CI/CD pipeline
6. **Monitor** production against baselines

## Support

For detailed information:
- See `tests/system/README.md` for full documentation
- Review test code in `tests/system/test_long_conversation_scale.py`
- Check server logs for debugging
- Consult `long_conversation_memory.md` for requirements

---

**Built with**: Python, pytest, agent-memory-client  
**Based on**: long_conversation_memory.md user story  
**Purpose**: Production readiness validation

