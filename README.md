# Redis Memory Server

A Python memory server for agents and LLM applications. This application
provides memory features for LLM conversations, including short-term memory
(message history) and long-term memory (vector embeddings for semantic search).

## Features

- Short-term memory storage for conversation history
- Optional long-term memory with semantic search capabilities
- Automatic context summarization to handle long conversations
- Integration with OpenAI API (more coming soon)
- Redis-based storage with vector search

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   ```
   # Required
   REDIS_URL=redis://localhost:6379
   
   # Optional
   PORT=8000
   LONG_TERM_MEMORY=true
   MAX_WINDOW_SIZE=12
   MODEL=gpt-3.5-turbo
   
   # For OpenAI
   OPENAI_API_KEY=your_openai_api_key
  ```

## Usage

Start the server:

```
python main.py
```

## API Endpoints

- `GET /health`: Health check endpoint
- `GET /sessions`: Get a list of session IDs
- `GET /sessions/{session_id}/memory`: Get memory for a session
- `POST /sessions/{session_id}/memory`: Add messages to a session
- `DELETE /sessions/{session_id}/memory`: Delete a session's memory
- `POST /sessions/{session_id}/retrieval`: Perform semantic search on session memory

## License

