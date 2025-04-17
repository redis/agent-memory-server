## Long-Term Memory Compaction Feature

Write logic to compact duplicate memories:
0. Use the function `compact_long_term_memories` in agent_memory_server/long_term_memory.py. This will be a Docket background task.
1. Use a stable hash on new memory data, so that every memory gets a hash.
2. When compacting, first find all memories that share the same hash and use an LLM prompt to compact them into one memory.
3. Memories are only the same if their text, user ID, and session ID all match. Other data, such as topics, should be merged into the final memory.
4. After handling hash-based compaction, next iterate through batches of memories and find all memories very similar to them using semantic search with a VectorQuery.
5. Use the same logic for distinct memories as above (user ID, session ID, text is unique). This time, use an LLM to merge the memories by sending all dupes plus the original memory with a prompt. Merge the final result's topics, etc., and give it a new stable hash.

### Unknowns
- When this background task will run is unknown -- this project does not have a scheduler
- That's ok, just make the task (function) and test it for now
