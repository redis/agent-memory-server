# Development

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

Merging a PR to the main branch will trigger building and pushing a new image
to Docker Hub based on the commits in main (including the version number).
Currently, that image pushes to a test project:

https://hub.docker.com/r/andrewbrookins510/agent-memory-server


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
