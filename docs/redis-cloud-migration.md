# Redis Cloud Agent Memory migration

Redis Cloud Agent Memory Service and a local `agent-memory-server` deployment both store long-term memories in Redis hashes, but the current local server does not index Redis Cloud export keys directly.

## Schema difference

Redis Cloud Agent Memory Service exports observed in Redis use this shape:

```text
memory:<store_id>:ltm:<memory_id>
```

with hash fields:

```text
id
text
text_vector
owner_id
session_id
namespace
memory_type
topics
created_at
updated_at
```

The local RedisVL backend uses the configured `redisvl_index_prefix`, defaulting to:

```text
memory_idx:<memory_id>
```

with canonical hash fields:

```text
id_
text
vector
user_id
session_id
namespace
memory_type
topics
entities
memory_hash
discrete_memory_extracted
pinned
access_count
created_at
updated_at
last_accessed
```

The local RediSearch index is created from `agent_memory_server/memory_vector_db_factory.py` and configured by these settings in `agent_memory_server/config.py`:

```text
REDIS_MEMORY_REDISVL_INDEX_NAME      default: memory_records
REDIS_MEMORY_REDISVL_INDEX_PREFIX    default: memory_idx
REDIS_MEMORY_REDISVL_VECTOR_DIMENSIONS default: 1536
```

Setting `REDIS_MEMORY_REDISVL_INDEX_PREFIX` to the Cloud prefix is not enough: the server's query code expects the local field names (`id_`, `user_id`, `vector`) and search queries use `vector` as the vector field.

## Migration command

Use the built-in migration command to copy Cloud-shaped records into local AMS-shaped records. Source keys are never deleted.

Dry run first:

```bash
agent-memory migrate-cloud-long-term-memory \
  --store-id <redis-cloud-agent-memory-store-id>
```

Apply:

```bash
agent-memory migrate-cloud-long-term-memory \
  --store-id <redis-cloud-agent-memory-store-id> \
  --apply
```

Then rebuild the local RediSearch index:

```bash
agent-memory rebuild-index
```

The migration performs these transformations:

```text
memory:<store_id>:ltm:<id>   -> memory_idx:<id>
id                           -> id_
owner_id                     -> user_id
text_vector                  -> vector, downcast from float64 to RedisVL FLOAT32 when needed
created_at / updated_at      -> seconds epoch if Cloud exported milliseconds
missing memory_hash          -> generated from text/user/session/namespace/type
missing local metadata       -> safe defaults
```

Useful options:

```text
--source-pattern <pattern>   Override the source SCAN pattern.
--target-prefix <prefix>     Override the local target prefix.
--batch-size <n>             Tune scan/write batch size.
--overwrite                  Replace existing target keys. Default is skip.
```

## Cloud to local workflow

1. Export/copy Redis data from Cloud into local Redis.
2. Start local `agent-memory-server` pointed at that Redis.
3. Run the migration dry-run and check `eligible`, `skipped_*`, and sample keys.
4. Run with `--apply`.
5. Run `agent-memory rebuild-index`.
6. Verify with `/v1/long-term-memory/search` or the SDK/client.

## Local to Cloud workflow

The inverse direction should avoid direct key copying into Redis Cloud service internals unless the service contract explicitly supports it. Prefer the public Agent Memory API: read local `memory_idx:*` hashes, map local fields back to API `MemoryRecord` payloads, and write them through the Cloud endpoint so the service owns key layout and indexing.
