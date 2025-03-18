# Redis Memory Server

A service that provides memory management for AI applications using Redis. This server helps manage both short-term and long-term memory for AI conversations, with features like automatic topic extraction, entity recognition, and context summarization.

## Features

- **Short-term Memory Management**
  - Configurable window size for recent messages
  - Automatic context summarization using LLMs
  - Token limit management based on model capabilities

- **Long-term Memory**
  - Semantic search capabilities
  - Automatic message indexing
  - Configurable memory retention

- **Advanced Features**
  - Topic extraction using BERTopic
  - Named Entity Recognition using BERT
  - Support for multiple model providers (OpenAI and Anthropic)
  - Namespace support for session isolation

## Get Started

### Docker Compose

To start the API using Docker Compose, follow these steps:

1. Ensure that Docker and Docker Compose are installed on your system.

2. Open a terminal in the project root directory (where the docker-compose.yml file is located).

3. (Optional) Set up your environment variables (such as OPENAI_API_KEY and ANTHROPIC_API_KEY) either in a .env file or by modifying the docker-compose.yml as needed.

4. Build and start the containers by running:
   docker-compose up --build

5. Once the containers are up, the API will be available at http://localhost:8000. You can also access the interactive API documentation at http://localhost:8000/docs.

6. To stop the containers, press Ctrl+C in the terminal and then run:
   docker-compose down

Happy coding!


## API Reference

### API Docs

API documentation is available at:  http://localhost:8000/docs.

### Endpoint Preview

#### List Sessions
```http
GET /sessions/
```

Query Parameters:
- `page` (int): Page number (default: 1)
- `size` (int): Items per page (default: 10)
- `namespace` (string, optional): Filter sessions by namespace

Response:
```json
[
    "session-1",
    "session-2"
]
```

#### Get Memory
```http
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

#### Add Messages to Memory
```http
POST /sessions/{session_id}/memory
```

Request Body:
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

Query Parameters:
- `namespace` (string, optional): Namespace for the session

Response:
```json
{
    "status": "ok"
}
```

#### Delete Session
```http
DELETE /sessions/{session_id}/memory
```

Query Parameters:
- `namespace` (string, optional): Namespace for the session

Response:
```json
{
    "status": "ok"
}
```

## Configuration

You can configure the service using environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `REDIS_URL` | URL for Redis connection | `redis://localhost:6379` |
| `LONG_TERM_MEMORY` | Enable/disable long-term memory | `True` |
| `WINDOW_SIZE` | Maximum messages in short-term memory | `20` |
| `OPENAI_API_KEY` | API key for OpenAI | - |
| `ANTHROPIC_API_KEY` | API key for Anthropic | - |
| `GENERATION_MODEL` | Model for text generation | `gpt-4o-mini` |
| `EMBEDDING_MODEL` | Model for text embeddings | `text-embedding-3-small` |
| `PORT` | Server port | `8000` |
| `TOPIC_MODEL` | BERTopic model for topic extraction | `MaartenGr/BERTopic_Wikipedia` |
| `NER_MODEL` | BERT model for NER | `dbmdz/bert-large-cased-finetuned-conll03-english` |
| `ENABLE_TOPIC_EXTRACTION` | Enable/disable topic extraction | `True` |
| `ENABLE_NER` | Enable/disable named entity recognition | `True` |

## Supported Models

### Large Language Models

Redis Memory Server supports using OpenAI and Anthropic models for generation, and OpenAI models for embeddings.

### Topic and NER Models
- **Topic Extraction**: BERTopic with Wikipedia-trained model
- **Named Entity Recognition**: BERT model fine-tuned on CoNLL-03 dataset

> **Note**: Embedding operations use OpenAI models exclusively, as Anthropic does not provide an embedding API.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/redis-memory-server.git
cd redis-memory-server
```

2. Install dependencies:
```bash
pip install -e ".[dev]"
```

3. Set up environment variables (see Configuration section)

4. Run the server:
```bash
python -m redis_memory_server
```

## Development

### Running Tests
```bash
python -m pytest
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project derives from original work from the Motorhead project:
https://github.com/getmetal/motorhead/

The original code is licensed under the Apache License 2.0:
https://www.apache.org/licenses/LICENSE-2.0

Modifications made by Redis, Inc. are also licensed under the Apache License 2.0.
