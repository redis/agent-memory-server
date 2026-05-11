# 🏫 AI Tutor

**File**: [`examples/ai_tutor.py`](https://github.com/redis/agent-memory-server/blob/main/examples/ai_tutor.py)

A functional tutoring system that demonstrates episodic memory for learning tracking and semantic memory for concept management.

## Core Features

- **Quiz Management**: Runs interactive quizzes and stores results
- **Learning Tracking**: Stores quiz results as episodic memories with timestamps
- **Concept Tracking**: Tracks weak concepts as semantic memories
- **Progress Analysis**: Provides summaries and personalized practice suggestions

## Memory Patterns Used

```python
# Episodic: Per-question results with event dates
{
    "text": "User answered 'photosynthesis' question incorrectly",
    "memory_type": "episodic",
    "event_date": "2024-01-15T10:30:00Z",
    "topics": ["quiz", "biology", "photosynthesis"]
}

# Semantic: Weak concepts for targeted practice
{
    "text": "User struggles with photosynthesis concepts",
    "memory_type": "semantic",
    "topics": ["weak_concept", "biology", "photosynthesis"]
}
```

## Usage Examples

```bash
cd examples

# Interactive tutoring session
python ai_tutor.py

# Demo with sample quiz flow
python ai_tutor.py --demo

# Custom student session
python ai_tutor.py --user-id student123 --session-id bio_course
```

## Key Commands

- **Practice**: Start a quiz on specific topics
- **Summary**: Get learning progress summary
- **Practice-next**: Get personalized practice recommendations based on weak areas
