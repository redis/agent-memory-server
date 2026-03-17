from agent_memory_server.utils.tag_codec import decode_tag_values, encode_tag_values


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
