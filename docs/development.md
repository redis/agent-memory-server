# Development

## Running the servers locally

The easiest way to run the API server(s) is with Docker Compose:

```
$ docker-compose up
```

## Running Tests

```bash
uv run pytest
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Releasing Agent Memory Server

Releases are triggered manually via GitHub Actions workflow dispatch.

### Steps to Release

1. Update the version in `agent_memory_server/__init__.py`
2. Commit and push the version change to main
3. Go to GitHub Actions → "Release Docker Images" workflow
4. Click "Run workflow"
5. Choose options:
   - **Version**: Leave empty to use version from `__init__.py`, or specify a custom version
   - **Push latest tag**: Check to also tag as `latest` (recommended for stable releases)
6. Click "Run workflow"

This will:
- Build Docker images for linux/amd64 and linux/arm64
- Push to Docker Hub: `redislabs/agent-memory-server:<version>`
- Push to GitHub Container Registry: `ghcr.io/redis/agent-memory-server:<version>`
- Optionally tag as `latest` on both registries
- Create a GitHub release with the version tag

Docker Hub: https://hub.docker.com/r/redislabs/agent-memory-server


## Releasing Agent Memory Client

For the client, the workflow is different. First, merge your PR to main.
Then tag a commit (from main) and push to a tag based on the format
`client/vx.y.z-test` or `client/vx.y.z`:

- Test PyPI: Use `-test` in the *version tag*. For example:
```
$ git tag client/v0.9.0-b2-test
$ git push client/v0.9.0-b2-test
```

- Production PyPI: Do not include `-test` in the version tag:
```
$ git tag client/v0.9.0b2
$ git push client/v0.9.0b2
