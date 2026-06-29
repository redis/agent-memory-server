from agent_memory_server.utils import redis as redis_utils


def test_is_redis_cluster_url_detects_supported_forms():
    assert redis_utils.is_redis_cluster_url("redis+cluster://redis-cluster:6379")
    assert redis_utils.is_redis_cluster_url("rediss+cluster://redis-cluster:6379")
    assert redis_utils.is_redis_cluster_url("redis://redis-cluster:6379?cluster=true")
    assert redis_utils.is_redis_cluster_url("redis://host1:6379,host2:6379")
    assert redis_utils.is_redis_cluster_url("redis://:pass@host1:6379,host2:6379")
    assert not redis_utils.is_redis_cluster_url("redis://localhost:6379")
    assert not redis_utils.is_redis_cluster_url("redis://:pass,word@localhost:6379")


def test_redis_url_for_docket_uses_cluster_scheme():
    assert (
        redis_utils.redis_url_for_docket("redis://redis-cluster:6379?cluster=true")
        == "redis+cluster://redis-cluster:6379"
    )
    assert (
        redis_utils.redis_url_for_docket(
            "rediss://redis-cluster:6379?cluster=true&ssl_cert_reqs=required"
        )
        == "rediss+cluster://redis-cluster:6379?ssl_cert_reqs=required"
    )
    assert (
        redis_utils.redis_url_for_docket("redis+cluster://redis-cluster:6379")
        == "redis+cluster://redis-cluster:6379"
    )


def test_redis_url_for_redisvl_uses_cluster_flag():
    assert (
        redis_utils.redis_url_for_redisvl("redis+cluster://redis-cluster:6379")
        == "redis://redis-cluster:6379?cluster=true"
    )
    assert (
        redis_utils.redis_url_for_redisvl(
            "rediss+cluster://redis-cluster:6379?ssl_cert_reqs=required"
        )
        == "rediss://redis-cluster:6379?ssl_cert_reqs=required&cluster=true"
    )


def test_docket_stream_key_hash_tags_cluster_names():
    assert (
        redis_utils.docket_stream_key(
            "memory-server", "redis+cluster://redis-cluster:6379"
        )
        == "{memory-server}:stream"
    )
    assert (
        redis_utils.docket_stream_key("memory-server", "redis://localhost:6379")
        == "memory-server:stream"
    )


def test_redis_url_for_async_redis_strips_cluster_scheme():
    assert (
        redis_utils.redis_url_for_async_redis("redis+cluster://redis-cluster:6379")
        == "redis://redis-cluster:6379"
    )
    assert (
        redis_utils.redis_url_for_async_redis(
            "rediss://redis-cluster:6379?cluster=true&ssl_cert_reqs=required"
        )
        == "rediss://redis-cluster:6379?ssl_cert_reqs=required"
    )
