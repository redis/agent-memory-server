import builtins
import importlib
import sys

import pytest

from agent_memory_server.utils.tag_codec import (
    decode_tag_values,
    encode_tag_values,
    sanitize_tag_values,
    validate_no_commas_in_tags,
)


def test_decode_tag_values_accepts_legacy_and_canonical_formats():
    assert decode_tag_values(None) == []
    assert decode_tag_values("") == []
    assert decode_tag_values(["a", "b"]) == ["a", "b"]
    assert decode_tag_values("a,b") == ["a", "b"]
    assert decode_tag_values("a|b") == ["a", "b"]
    assert decode_tag_values("a, b | c") == ["a", "b", "c"]


def test_encode_tag_values_uses_canonical_comma_separator():
    assert encode_tag_values(None) == ""
    assert encode_tag_values([]) == ""
    assert encode_tag_values(["a", "b"]) == "a,b"
    assert encode_tag_values([" a ", "", "b "]) == "a,b"


def test_encode_tag_values_raises_on_embedded_commas():
    with pytest.raises(ValueError, match="contains a comma"):
        encode_tag_values(["Austin, TX"])
    with pytest.raises(ValueError, match="contains a comma"):
        encode_tag_values(["ok", "bad, value"])


# --- validate_no_commas_in_tags ---


def test_validate_no_commas_passes_clean_values():
    assert validate_no_commas_in_tags(["a", "b"], "topics") == ["a", "b"]


def test_validate_no_commas_passes_none():
    assert validate_no_commas_in_tags(None, "topics") is None


def test_validate_no_commas_passes_empty_list():
    assert validate_no_commas_in_tags([], "topics") == []


def test_validate_no_commas_rejects_commas():
    with pytest.raises(ValueError, match=r"topics\[0\] contains a comma.*Austin, TX"):
        validate_no_commas_in_tags(["Austin, TX"], "topics")


def test_validate_no_commas_rejects_second_element():
    with pytest.raises(ValueError, match=r"entities\[1\] contains a comma"):
        validate_no_commas_in_tags(["ok", "bad, value"], "entities")


def test_validate_no_commas_imports_without_client_helper(monkeypatch):
    module_name = "agent_memory_server.utils.tag_codec"
    original_module = sys.modules.get(module_name)
    sys.modules.pop(module_name, None)

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "agent_memory_client.utils.tag_codec":
            raise ModuleNotFoundError(
                "No module named 'agent_memory_client.utils'",
                name="agent_memory_client.utils",
            )
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    try:
        module = importlib.import_module(module_name)
        assert module.__file__ is not None
        assert module.validate_no_commas_in_tags(["a", "b"], "topics") == ["a", "b"]
        with pytest.raises(
            ValueError, match=r"topics\[0\] contains a comma.*Austin, TX"
        ):
            module.validate_no_commas_in_tags(["Austin, TX"], "topics")
    finally:
        sys.modules.pop(module_name, None)
        if original_module is not None:
            sys.modules[module_name] = original_module


# --- sanitize_tag_values ---


def test_sanitize_tag_values_replaces_commas():
    assert sanitize_tag_values(["Austin, TX"]) == ["Austin TX"]


def test_sanitize_tag_values_handles_none():
    assert sanitize_tag_values(None) is None


def test_sanitize_tag_values_handles_empty():
    assert sanitize_tag_values([]) is None


def test_sanitize_tag_values_strips_and_filters():
    assert sanitize_tag_values(["  a , b  ", "", "  "]) == ["a b"]


def test_sanitize_tag_values_multiple():
    result = sanitize_tag_values(["cooking, italian style", "normal"])
    assert result == ["cooking italian style", "normal"]


def test_sanitize_tag_values_handles_non_string_elements():
    result = sanitize_tag_values([None, 42, "ok, value", True])
    assert result == ["42", "ok value", "True"]
