import pytest

from agent_memory_server.migrations import migrate_normalize_tag_separators_4


class FakePipeline:
    def __init__(self, execute_result):
        self.execute_result = execute_result
        self.hgetall_calls = []
        self.hset_calls = []

    def hgetall(self, key):
        self.hgetall_calls.append(key)
        return self

    def hset(self, key, mapping):
        self.hset_calls.append((key, mapping))
        return self

    async def execute(self):
        return self.execute_result


class FakeRedis:
    def __init__(self, scan_results, read_results):
        self.scan_results = iter(scan_results)
        self.read_results = iter(read_results)
        self.pipeline_transactions = []
        self.pipelines = []

    async def scan(self, cursor=0, match=None, count=100):
        return next(self.scan_results)

    def pipeline(self, transaction=True):
        self.pipeline_transactions.append(transaction)
        execute_result = next(self.read_results, [])
        pipe = FakePipeline(execute_result)
        self.pipelines.append(pipe)
        return pipe


class TestMigrations:
    @pytest.mark.asyncio
    async def test_migrate_normalize_tag_separators_rewrites_legacy_values(self):
        key = "memory_idx:test-memory"
        redis = FakeRedis(
            scan_results=[(0, [key])],
            read_results=[
                [
                    {
                        b"topics": b"cooking|italian",
                        b"entities": b"pasta|rome",
                        b"extracted_from": b"msg-1|msg-2",
                    }
                ],
                [],
            ],
        )

        await migrate_normalize_tag_separators_4(redis=redis)

        assert redis.pipeline_transactions == [False, False]
        read_pipe, write_pipe = redis.pipelines
        assert read_pipe.hgetall_calls == [key]
        assert write_pipe.hset_calls == [
            (
                key,
                {
                    "topics": "cooking,italian",
                    "entities": "pasta,rome",
                    "extracted_from": "msg-1,msg-2",
                },
            )
        ]

    @pytest.mark.asyncio
    async def test_migrate_normalize_tag_separators_skips_canonical_values(self):
        key = "memory_idx:test-memory"
        redis = FakeRedis(
            scan_results=[(0, [key])],
            read_results=[
                [
                    {
                        b"topics": b"cooking,italian",
                        b"entities": b"pasta,rome",
                        b"extracted_from": b"msg-1,msg-2",
                    }
                ],
                [],
            ],
        )

        await migrate_normalize_tag_separators_4(redis=redis)

        assert redis.pipeline_transactions == [False, False]
        read_pipe, write_pipe = redis.pipelines
        assert read_pipe.hgetall_calls == [key]
        assert write_pipe.hset_calls == []
