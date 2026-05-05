"""
Test cloudpickle serialization of docket task arguments and exceptions.

Docket uses cloudpickle to serialize:
  1. Task arguments (when scheduling: cloudpickle.dumps(args))
  2. Task results (on success: cloudpickle.dumps(result))
  3. Task exceptions (on failure: cloudpickle.dumps(exception))

litellm exception classes have mandatory __init__ args (message, model,
llm_provider) that cloudpickle doesn't preserve during deserialization.
This means cloudpickle.loads() calls ExceptionClass.__init__() without
the required positional args, raising TypeError. This prevents the docket
worker from storing or reporting task errors.

The fix is in agent_memory_server.litellm_pickle_compat, which patches
litellm exceptions with __reduce__ methods for safe pickle round-tripping.
"""

import threading

import cloudpickle
import httpx
import litellm
import pytest

from agent_memory_server.models import MemoryMessage, MemoryRecord


# Factory to create litellm exceptions with correct constructor args
def _make_litellm_exc(exc_class, **overrides):
    """Create a litellm exception instance with the right constructor args."""
    mock_response = httpx.Response(
        status_code=500,
        request=httpx.Request(method="POST", url="https://test.example.com"),
    )
    kwargs = {"message": "test error", "model": "test-model", "llm_provider": "test"}
    kwargs.update(overrides)

    # Classes that require response as a non-optional positional arg
    if exc_class in (
        litellm.exceptions.PermissionDeniedError,
        litellm.exceptions.UnprocessableEntityError,
    ):
        kwargs.setdefault("response", mock_response)

    # APIError requires status_code as first positional arg
    if exc_class is litellm.exceptions.APIError:
        kwargs.setdefault("status_code", 500)

    return exc_class(**kwargs)


class TestTaskArgumentSerialization:
    """Verify that task arguments can be serialized by docket."""

    def test_memory_record_serializes(self):
        """MemoryRecord is the primary task argument for extract_memory_structure."""
        record = MemoryRecord(
            id="test-123",
            text="User prefers dark mode",
            user_id="alice",
            namespace="test",
        )
        data = cloudpickle.dumps(record)
        restored = cloudpickle.loads(data)
        assert restored.id == "test-123"
        assert restored.text == "User prefers dark mode"

    def test_memory_message_serializes(self):
        """MemoryMessage has a ClassVar threading.Lock for deprecation warnings."""
        msg = MemoryMessage(role="user", content="Hello")
        data = cloudpickle.dumps(msg)
        restored = cloudpickle.loads(data)
        assert restored.role == "user"
        assert restored.content == "Hello"

    def test_list_of_memory_records_serializes(self):
        """index_long_term_memories takes a list of MemoryRecord."""
        records = [MemoryRecord(id=f"test-{i}", text=f"Memory {i}") for i in range(5)]
        data = cloudpickle.dumps(records)
        restored = cloudpickle.loads(data)
        assert len(restored) == 5


class TestExceptionSerializationBaseline:
    """Verify baseline serialization behavior (non-litellm exceptions)."""

    def test_plain_exception_serializes(self):
        exc = ValueError("something went wrong")
        data = cloudpickle.dumps(exc)
        restored = cloudpickle.loads(data)
        assert str(restored) == "something went wrong"

    def test_threading_lock_does_not_serialize(self):
        lock = threading.Lock()
        with pytest.raises(TypeError, match="cannot pickle"):
            cloudpickle.dumps(lock)

    def test_httpx_client_does_not_serialize(self):
        client = httpx.Client(timeout=10.0)
        with pytest.raises(TypeError, match="cannot pickle"):
            cloudpickle.dumps(client)
        client.close()

    def test_httpx_connect_error_serializes(self):
        exc = httpx.ConnectError("connection failed")
        data = cloudpickle.dumps(exc)
        restored = cloudpickle.loads(data)
        assert isinstance(restored, httpx.ConnectError)

    def test_httpx_timeout_error_serializes(self):
        exc = httpx.ReadTimeout("Connection timed out")
        data = cloudpickle.dumps(exc)
        restored = cloudpickle.loads(data)
        assert isinstance(restored, httpx.ReadTimeout)

    def test_exception_from_memory_message_validation_serializes(self):
        try:
            MemoryMessage(role=123, content=456)  # type: ignore
        except Exception as e:
            try:
                data = cloudpickle.dumps(e)
                cloudpickle.loads(data)
            except TypeError as pickle_err:
                pytest.fail(
                    f"MemoryMessage validation exception cannot be pickled: {pickle_err}"
                )

    def test_exception_with_traceback_from_locked_class_serializes(self):
        class ServiceWithLock:
            _lock = threading.Lock()

            def do_work(self):
                with self._lock:
                    raise RuntimeError("LLM call failed")

        svc = ServiceWithLock()
        try:
            svc.do_work()
        except RuntimeError as e:
            try:
                data = cloudpickle.dumps(e)
                cloudpickle.loads(data)
            except TypeError as pickle_err:
                pytest.fail(
                    f"Exception with lock in traceback cannot be pickled: {pickle_err}"
                )

    def test_chained_exception_with_httpx_context_serializes(self):
        connect_err = httpx.ConnectError("connection failed")

        with pytest.raises(RuntimeError) as excinfo:
            try:
                raise connect_err
            except Exception as err:
                raise RuntimeError("LLM extraction failed") from err

        e = excinfo.value
        assert e.__cause__ is not None
        assert isinstance(e.__cause__, httpx.ConnectError)
        try:
            data = cloudpickle.dumps(e)
            cloudpickle.loads(data)
        except TypeError as pickle_err:
            pytest.fail(
                f"Chained exception with httpx context cannot be pickled: {pickle_err}"
            )


class TestLiteLLMExceptionBugProof:
    """
    Prove the underlying bug: litellm exception __init__ requires positional
    args that cloudpickle doesn't preserve.

    We demonstrate this by calling __init__ without the required args,
    which is what cloudpickle does internally during deserialization.
    """

    @pytest.mark.parametrize(
        "exc_class",
        [
            litellm.exceptions.APIConnectionError,
            litellm.exceptions.RateLimitError,
            litellm.exceptions.Timeout,
            litellm.exceptions.ServiceUnavailableError,
            litellm.exceptions.BadRequestError,
            litellm.exceptions.AuthenticationError,
            litellm.exceptions.NotFoundError,
            litellm.exceptions.ContentPolicyViolationError,
        ],
        ids=lambda c: c.__name__,
    )
    def test_litellm_init_requires_positional_args(self, exc_class):
        """
        litellm exceptions cannot be constructed without message, model,
        and llm_provider. This is why cloudpickle deserialization fails:
        it calls __init__() with no args.

        docket/worker.py line ~1001 calls cloudpickle.dumps(e) on every
        failed task. Without a __reduce__ patch, the deserialized exception
        would fail to reconstruct.
        """
        with pytest.raises(TypeError, match="missing.*required"):
            exc_class()


class TestLiteLLMExceptionPatched:
    """
    Verify the fix: with litellm_pickle_compat imported, all litellm
    exceptions roundtrip through cloudpickle successfully.

    The patch adds __reduce__ methods that bypass __init__ on deserialization,
    using Exception.__new__() and restoring __dict__ directly.
    """

    @classmethod
    def setup_class(cls):
        """Ensure the pickle compat patch is applied."""
        import agent_memory_server.litellm_pickle_compat  # noqa: F401

    @pytest.mark.parametrize(
        "exc_class",
        [
            litellm.exceptions.APIConnectionError,
            litellm.exceptions.RateLimitError,
            litellm.exceptions.Timeout,
            litellm.exceptions.ServiceUnavailableError,
            litellm.exceptions.BadRequestError,
            litellm.exceptions.AuthenticationError,
            litellm.exceptions.NotFoundError,
            litellm.exceptions.ContentPolicyViolationError,
            litellm.exceptions.InternalServerError,
            litellm.exceptions.BadGatewayError,
            litellm.exceptions.PermissionDeniedError,
            litellm.exceptions.UnprocessableEntityError,
            litellm.exceptions.APIError,
            litellm.exceptions.APIResponseValidationError,
            litellm.exceptions.ContextWindowExceededError,
        ],
        ids=lambda c: c.__name__,
    )
    def test_patched_litellm_exception_roundtrips(self, exc_class):
        """All litellm exceptions roundtrip through cloudpickle after patching."""
        exc = _make_litellm_exc(exc_class)
        data = cloudpickle.dumps(exc)
        restored = cloudpickle.loads(data)
        assert isinstance(restored, Exception)
        assert "test error" in restored.message
        assert restored.model == "test-model"
        assert restored.llm_provider == "test"

    def test_patched_exception_preserves_status_code(self):
        """Status codes survive the roundtrip."""
        exc = _make_litellm_exc(litellm.exceptions.RateLimitError)
        data = cloudpickle.dumps(exc)
        restored = cloudpickle.loads(data)
        assert restored.status_code == 429

    def test_patched_exception_preserves_str_representation(self):
        """str() on the restored exception contains the error message."""
        exc = _make_litellm_exc(
            litellm.exceptions.APIConnectionError,
            message="connection refused",
        )
        data = cloudpickle.dumps(exc)
        restored = cloudpickle.loads(data)
        assert "connection refused" in str(restored)

    def test_patched_exception_preserves_chaining(self):
        """Exception chaining (__cause__) survives the roundtrip."""
        cause = ValueError("upstream failure")
        exc = _make_litellm_exc(
            litellm.exceptions.APIConnectionError,
            message="connection failed",
        )
        exc.__cause__ = cause
        data = cloudpickle.dumps(exc)
        restored = cloudpickle.loads(data)
        assert restored.__cause__ is not None
        assert isinstance(restored.__cause__, ValueError)
        assert str(restored.__cause__) == "upstream failure"

    def test_patch_is_idempotent(self):
        """Calling patch() multiple times is safe."""
        import agent_memory_server.litellm_pickle_compat as compat

        compat.patch()
        compat.patch()

        exc = _make_litellm_exc(litellm.exceptions.Timeout, message="timed out")
        data = cloudpickle.dumps(exc)
        restored = cloudpickle.loads(data)
        assert "timed out" in restored.message
