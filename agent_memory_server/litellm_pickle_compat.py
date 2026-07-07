"""Monkey-patch litellm exception classes to be pickle-safe for docket.

Problem: litellm exceptions have mandatory __init__ args (message, model,
llm_provider) that cloudpickle doesn't preserve during deserialization.
When a docket worker task fails with a litellm exception, the worker calls
cloudpickle.dumps(e) to serialize the error for the result queue. The dumps
succeeds, but cloudpickle.loads() later calls ExceptionClass.__init__()
without the required positional args, raising TypeError. This prevents the
worker from storing or reporting the error.

Fix: Add __reduce__ methods that bypass __init__ on deserialization by
reconstructing the exception via Exception.__new__() and restoring __dict__.
"""

import litellm.exceptions


_LITELLM_EXCEPTION_CLASSES = [
    litellm.exceptions.AuthenticationError,
    litellm.exceptions.NotFoundError,
    litellm.exceptions.BadRequestError,
    litellm.exceptions.UnprocessableEntityError,
    litellm.exceptions.Timeout,
    litellm.exceptions.PermissionDeniedError,
    litellm.exceptions.RateLimitError,
    litellm.exceptions.ContextWindowExceededError,
    litellm.exceptions.ContentPolicyViolationError,
    litellm.exceptions.ServiceUnavailableError,
    litellm.exceptions.BadGatewayError,
    litellm.exceptions.InternalServerError,
    litellm.exceptions.APIError,
    litellm.exceptions.APIConnectionError,
    litellm.exceptions.APIResponseValidationError,
]


def _reconstruct_litellm_exception(cls, state):
    """Reconstruct a litellm exception without calling __init__.

    Uses Exception.__new__ to create the instance, then restores __dict__
    and sets Exception.args for proper str() representation.
    Also restores exception chaining attributes (__cause__, __context__).
    """
    exc = Exception.__new__(cls)

    # Extract special exception attributes before updating __dict__
    exc_args = state.pop("_exc_args", (state.get("message", ""),))
    exc_cause = state.pop("_exc_cause", None)
    exc_context = state.pop("_exc_context", None)

    exc.__dict__.update(state)
    exc.args = exc_args
    exc.__cause__ = exc_cause
    exc.__context__ = exc_context
    return exc


def _litellm_reduce(self):
    """Custom __reduce__ for litellm exceptions.

    Captures __dict__ state, filtering out any attributes that themselves
    cannot be pickled (e.g. httpx connection pools from real responses).
    Also preserves exception chaining attributes (__cause__, __context__).
    """
    import pickle

    state = {}
    for key, value in self.__dict__.items():
        try:
            pickle.dumps(value)
            state[key] = value
        except Exception:
            # Fall back to string representation for any unpicklable attributes
            state[key] = repr(value)

    # Preserve exception chaining and args
    state["_exc_args"] = self.args

    # Only preserve chaining if the chained exceptions are themselves picklable
    for attr, key in (("__cause__", "_exc_cause"), ("__context__", "_exc_context")):
        chained = getattr(self, attr, None)
        if chained is not None:
            try:
                pickle.dumps(chained)
                state[key] = chained
            except Exception:
                pass

    return (_reconstruct_litellm_exception, (type(self), state))


def patch():
    """Apply pickle-safety patches to all litellm exception classes."""
    for cls in _LITELLM_EXCEPTION_CLASSES:
        if not hasattr(cls, "_pickle_patched"):
            cls.__reduce__ = _litellm_reduce
            cls._pickle_patched = True


# Auto-patch on import
patch()
