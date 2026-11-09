# AWS Bedrock AgentCore Memory

Managed short-term events and long-term strategies (semantic, summary, user preference).

Docs:

- [Add memory](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/memory.html)
- [Memory strategies](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/memory-strategies.html)
- [Memory organization](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/memory-organization.html)
- [Overview blog](https://aws.amazon.com/blogs/machine-learning/amazon-bedrock-agentcore-memory-building-context-aware-agents/)

CLI id: `bedrock-agentcore`. Extra: `aws`.

## Install

```bash
uv sync --extra aws
```

## Auth

Use a standard AWS credential chain (env vars, profile, or role). The principal needs AgentCore Memory create, event, and retrieve permissions in the target region.

```bash
OPENAI_API_KEY=replace-me
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=replace-me
AWS_SECRET_ACCESS_KEY=replace-me
# or AWS_PROFILE=your-profile
```

Prefer a short-lived role over long-lived access keys.

## Provision

Create a Memory resource and configure one or more public AgentCore memory
strategies by following the AWS documentation. The wrapper deliberately does
not choose a strategy for you. Record the memory id and the namespace path
template produced by that strategy.

The runner isolates examples with a unique `actorId`. Pass a namespace path
containing `{user_id}` so each example retrieves only its own records, for
example the exact actor-scoped path documented for your configured strategy.
Confirm AgentCore Memory is available in your region.

IAM is your responsibility. Scope the policy to this experiment account.

## Run

```bash
uv run memory-bench run \
  --provider bedrock-agentcore \
 --split oracle \
  --limit 1 \
  --run-name smoke-agentcore \
  --provider-param model=gpt-4o \
  --provider-param region=us-east-1 \
  --provider-param memory_id=YOUR_MEMORY_ID \
  --provider-param namespace_path='/YOUR/STRATEGY/PATH/{user_id}/' \
  --provider-param top_k=10

uv run memory-bench judge --experiment smoke-agentcore --judge-model gpt-4o
```

## Cleanup

The wrapper deletes session events and retrieved memory records for the
example `actorId` during `reset`. Delete the AgentCore memory resource in the
AWS console or with the AgentCore control-plane `DeleteMemory` API when you no
longer need it. Check CloudWatch and billing for leftover extraction jobs.

## Notes

- Long-term extraction is asynchronous. The wrapper polls until records appear.
- AWS extraction cost is not in OpenAI `TokenUsage`.
- `namespace_path` must match the strategy configured on the supplied memory.
- This recipe comes from public AWS docs. It was not run against a live AgentCore memory.
