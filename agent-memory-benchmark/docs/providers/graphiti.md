# Graphiti

Open-source temporal knowledge graph ([getzep/graphiti](https://github.com/getzep/graphiti)) on Neo4j.

Docs:

- [Graphiti overview](https://help.getzep.com/graphiti/getting-started/overview)
- [Quick start](https://help.getzep.com/graphiti/getting-started/quick-start)
- [Neo4j configuration](https://help.getzep.com/graphiti/configuration/neo-4-j-configuration)

CLI id: `graphiti`. Extra: `graphiti`.

## Install

```bash
uv sync --extra graphiti
```

Neo4j 5.26+ with APOC is required. Graphiti documents Docker as:

```bash
docker run \
  --name neo4j-graphiti \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/REPLACE_ME \
  -e NEO4J_PLUGINS='["apoc"]' \
  neo4j:5.26-community
```

## Auth

```bash
OPENAI_API_KEY=replace-me
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=REPLACE_ME
```

## Provision

Start Neo4j. Confirm bolt on `7687`. The wrapper builds Graphiti indices on first use and deletes its group between examples, so point it at a dedicated or empty database.

## Run

```bash
uv run memory-bench run \
  --provider graphiti \
 --split oracle \
  --limit 1 \
  --run-name smoke-graphiti \
  --provider-param model=gpt-4o \
  --provider-param search_limit=20

uv run memory-bench judge --experiment smoke-graphiti --judge-model gpt-4o
```

## Cleanup

Stop and remove the container. This deletes local graph data.

```bash
docker rm -f neo4j-graphiti
```

On a shared Neo4j, drop the benchmark graph instead of removing the server.

## Notes

- Episode ingest is sequential so temporal invalidation can run.
- Graph construction uses Graphiti's installed defaults. Answers use `model`.
- Local Neo4j plus many LLM extract calls. Budget both compute and API spend.
