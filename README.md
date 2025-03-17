# Redis Memory Server

A service that provides memory management for AI applications using Redis.

## Features

- Short-term memory management with configurable window size
- Long-term memory with semantic search capabilities
- Automatic context summarization using LLMs
- Support for multiple model providers (OpenAI and Anthropic)
- Configurable token limits based on selected model
- Topic extraction using BERTopic
- Named Entity Recognition using BERT

## Configuration

The service can be configured using environment variables:

- `REDIS_URL`: URL for Redis connection (default: `redis://localhost:6379`)
- `LONG_TERM_MEMORY`: Enable/disable long-term memory (default: `True`)
- `WINDOW_SIZE`: Maximum number of messages to keep in short-term memory (default: `20`)
- `OPENAI_API_KEY`: API key for OpenAI
- `ANTHROPIC_API_KEY`: API key for Anthropic
- `GENERATION_MODEL`: Model to use for text generation (default: `gpt-4o-mini`)
- `EMBEDDING_MODEL`: Model to use for text embeddings (default: `text-embedding-3-small`)
- `PORT`: Port to run the server on (default: `8000`)
- `TOPIC_MODEL`: BERTopic model to use for topic extraction (default: `MaartenGr/BERTopic_Wikipedia`)
- `NER_MODEL`: BERT model to use for named entity recognition (default: `dbmdz/bert-large-cased-finetuned-conll03-english`)
- `ENABLE_TOPIC_EXTRACTION`: Enable/disable topic extraction (default: `True`)
- `ENABLE_NER`: Enable/disable named entity recognition (default: `True`)

## Supported Models

### OpenAI Models

- `gpt-3.5-turbo`: 4K context window
- `gpt-3.5-turbo-16k`: 16K context window
- `gpt-4`: 8K context window
- `gpt-4-32k`: 32K context window
- `gpt-4o`: 128K context window
- `gpt-4o-mini`: 128K context window

### Anthropic Models

- `claude-3-opus-20240229`: 200K context window
- `claude-3-sonnet-20240229`: 200K context window
- `claude-3-haiku-20240307`: 200K context window
- `claude-3-5-sonnet-20240620`: 200K context window

### Topic and NER Models

- Topic Extraction: Uses BERTopic with the specified model (default: Wikipedia-trained model)
- Named Entity Recognition: Uses BERT model fine-tuned on CoNLL-03 dataset

**Note**: Embedding operations always use OpenAI models, as Anthropic does not provide embedding API.

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables (see Configuration section)
4. Run the server: `python main.py`

## Usage

### Add Messages to Memory

```
POST /sessions/{session_id}/memory
```

Request body:
```json
{
    "messages": [
        {
            "role": "user",
            "content": "Hello, how are you?"
        }
    ],
    "context": "Optional context for the conversation"
}
```

Response:
```json
{
    "status": "ok"
}
```

### Get Memory

```
GET /sessions/{session_id}/memory
```

Response:
```json
{
    "messages": [
        {
            "role": "user",
            "content": "Hello, how are you?",
            "topics": ["greeting", "well-being"],
            "entities": []
        }
    ],
    "context": "Optional context for the conversation",
    "tokens": 123
}
```

### List Sessions

```
GET /sessions/
```

Response:
```json
[
    "session-1",
    "session-2"
]
```

### Delete Session

```
DELETE /sessions/{session_id}/memory
```

Response:
```json
{
    "status": "ok"
}
```

## Development

To run tests:

```
python -m pytest
```

## License
TBD
