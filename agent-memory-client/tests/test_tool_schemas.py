"""
Test file for memory tool schemas.

Tests that tool schemas are correctly structured and that the 'message' memory type
is not exposed to LLM tools (it should only be used server-side).
"""

from agent_memory_client import MemoryAPIClient


class TestToolSchemaStructure:
    """Tests for tool schema structure and completeness."""

    def test_get_memory_search_tool_schema(self):
        """Test memory search tool schema structure."""
        schema = MemoryAPIClient.get_memory_search_tool_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search_memory"
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]
        assert schema["function"]["parameters"]["type"] == "object"
        assert "query" in schema["function"]["parameters"]["properties"]
        assert "query" in schema["function"]["parameters"]["required"]

    def test_get_add_memory_tool_schema(self):
        """Test add_memory_to_working_memory tool schema structure."""
        schema = MemoryAPIClient.get_add_memory_tool_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "add_memory_to_working_memory"
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]

        params = schema["function"]["parameters"]
        assert params["type"] == "object"
        assert "text" in params["properties"]
        assert "memory_type" in params["properties"]
        assert "text" in params["required"]
        assert "memory_type" in params["required"]

    def test_create_long_term_memory_tool_schema(self):
        """Test create_long_term_memory tool schema structure."""
        schema = MemoryAPIClient.create_long_term_memory_tool_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "create_long_term_memory"
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]

        params = schema["function"]["parameters"]
        assert params["type"] == "object"
        assert "memories" in params["properties"]
        assert "memories" in params["required"]

        # Check nested structure
        memory_items = params["properties"]["memories"]["items"]
        assert "text" in memory_items["properties"]
        assert "memory_type" in memory_items["properties"]
        assert "text" in memory_items["required"]
        assert "memory_type" in memory_items["required"]

    def test_edit_long_term_memory_tool_schema(self):
        """Test edit_long_term_memory tool schema structure."""
        schema = MemoryAPIClient.edit_long_term_memory_tool_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "edit_long_term_memory"
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]

        params = schema["function"]["parameters"]
        assert params["type"] == "object"
        assert "memory_id" in params["properties"]
        assert "memory_id" in params["required"]
        assert "text" in params["properties"]
        assert "memory_type" in params["properties"]

    def test_delete_long_term_memories_tool_schema(self):
        """Test delete_long_term_memories tool schema structure."""
        schema = MemoryAPIClient.delete_long_term_memories_tool_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "delete_long_term_memories"
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]

        params = schema["function"]["parameters"]
        assert params["type"] == "object"
        assert "memory_ids" in params["properties"]
        assert "memory_ids" in params["required"]

    def test_get_all_memory_tool_schemas(self):
        """Test getting all memory tool schemas."""
        schemas = MemoryAPIClient.get_all_memory_tool_schemas()

        # Should have multiple tools
        assert len(schemas) > 0

        # Check that all expected tools are present
        function_names = {schema["function"]["name"] for schema in schemas}
        expected_tools = {
            "search_memory",
            "get_or_create_working_memory",
            "add_memory_to_working_memory",
            "update_working_memory_data",
            "get_long_term_memory",
            "create_long_term_memory",
            "edit_long_term_memory",
            "delete_long_term_memories",
            "get_current_datetime",
        }
        assert expected_tools.issubset(function_names)


class TestMemoryTypeEnumExclusion:
    """Tests that 'message' memory type is NOT exposed in creation/editing tool schemas.

    Note: search_memory CAN include 'message' in its filter enum since it's for
    searching/reading existing memories, not creating new ones. The restriction
    only applies to tools that create or modify memories.
    """

    def test_add_memory_excludes_message_type(self):
        """Test that add_memory_to_working_memory excludes 'message' type."""
        schema = MemoryAPIClient.get_add_memory_tool_schema()

        params = schema["function"]["parameters"]
        memory_type_prop = params["properties"]["memory_type"]

        # Should only have episodic and semantic
        assert memory_type_prop["enum"] == ["episodic", "semantic"]
        assert "message" not in memory_type_prop["enum"]

    def test_create_long_term_memory_excludes_message_type(self):
        """Test that create_long_term_memory excludes 'message' type."""
        schema = MemoryAPIClient.create_long_term_memory_tool_schema()

        params = schema["function"]["parameters"]
        memory_items = params["properties"]["memories"]["items"]
        memory_type_prop = memory_items["properties"]["memory_type"]

        # Should only have episodic and semantic
        assert memory_type_prop["enum"] == ["episodic", "semantic"]
        assert "message" not in memory_type_prop["enum"]

    def test_edit_long_term_memory_excludes_message_type(self):
        """Test that edit_long_term_memory excludes 'message' type."""
        schema = MemoryAPIClient.edit_long_term_memory_tool_schema()

        params = schema["function"]["parameters"]
        memory_type_prop = params["properties"]["memory_type"]

        # Should only have episodic and semantic
        assert memory_type_prop["enum"] == ["episodic", "semantic"]
        assert "message" not in memory_type_prop["enum"]

    def test_search_memory_allows_message_type_filter(self):
        """Test that search_memory DOES allow 'message' type for filtering.

        This is intentional - search tools should be able to filter by message type
        to find conversation history, but creation/editing tools should not be able
        to create or modify message-type memories.
        """
        schema = MemoryAPIClient.get_memory_search_tool_schema()

        params = schema["function"]["parameters"]
        memory_type_prop = params["properties"]["memory_type"]

        # Search should include all types including message
        assert "episodic" in memory_type_prop["enum"]
        assert "semantic" in memory_type_prop["enum"]
        assert "message" in memory_type_prop["enum"]

    def test_creation_and_editing_tools_exclude_message_type(self):
        """Test that creation and editing tools (not search) exclude 'message'."""
        all_schemas = MemoryAPIClient.get_all_memory_tool_schemas()

        # Tools that should NOT expose message type (creation/editing tools)
        restricted_tools = {
            "add_memory_to_working_memory",
            "create_long_term_memory",
            "edit_long_term_memory",
        }

        # Tools that CAN expose message type (search/read tools)
        allowed_tools = {
            "search_memory",
            "get_long_term_memory",
        }

        for schema in all_schemas:
            function_name = schema["function"]["name"]
            params = schema["function"]["parameters"]

            # Check direct memory_type property
            if "memory_type" in params["properties"]:
                memory_type_prop = params["properties"]["memory_type"]
                if "enum" in memory_type_prop:
                    if function_name in restricted_tools:
                        assert (
                            "message" not in memory_type_prop["enum"]
                        ), f"Creation/editing tool '{function_name}' should not expose 'message' memory type"
                    elif function_name in allowed_tools:
                        # These tools are allowed to have message in enum for filtering
                        pass

            # Check nested properties (like in create_long_term_memory)
            if "memories" in params["properties"]:
                items = params["properties"]["memories"].get("items", {})
                if (
                    "properties" in items
                    and "memory_type" in items["properties"]
                    and "enum" in items["properties"]["memory_type"]
                    and function_name in restricted_tools
                ):
                    memory_type_prop = items["properties"]["memory_type"]
                    assert (
                        "message" not in memory_type_prop["enum"]
                    ), f"Creation/editing tool '{function_name}' should not expose 'message' memory type in nested properties"


class TestAnthropicSchemas:
    """Tests for Anthropic-formatted tool schemas."""

    def test_get_memory_search_tool_schema_anthropic(self):
        """Test memory search tool schema in Anthropic format."""
        schema = MemoryAPIClient.get_memory_search_tool_schema_anthropic()

        assert schema["name"] == "search_memory"
        assert "description" in schema
        assert "input_schema" in schema
        assert schema["input_schema"]["type"] == "object"
        assert "query" in schema["input_schema"]["properties"]
        assert "query" in schema["input_schema"]["required"]

    def test_create_long_term_memory_tool_schema_anthropic(self):
        """Test create_long_term_memory tool schema in Anthropic format."""
        schema = MemoryAPIClient.create_long_term_memory_tool_schema_anthropic()

        assert schema["name"] == "create_long_term_memory"
        assert "description" in schema
        assert "input_schema" in schema
        assert schema["input_schema"]["type"] == "object"
        assert "memories" in schema["input_schema"]["properties"]

    def test_edit_long_term_memory_tool_schema_anthropic(self):
        """Test edit_long_term_memory tool schema in Anthropic format."""
        schema = MemoryAPIClient.edit_long_term_memory_tool_schema_anthropic()

        assert schema["name"] == "edit_long_term_memory"
        assert "description" in schema
        assert "input_schema" in schema
        assert schema["input_schema"]["type"] == "object"
        assert "memory_id" in schema["input_schema"]["properties"]

    def test_delete_long_term_memories_tool_schema_anthropic(self):
        """Test delete_long_term_memories tool schema in Anthropic format."""
        schema = MemoryAPIClient.delete_long_term_memories_tool_schema_anthropic()

        assert schema["name"] == "delete_long_term_memories"
        assert "description" in schema
        assert "input_schema" in schema
        assert schema["input_schema"]["type"] == "object"
        assert "memory_ids" in schema["input_schema"]["properties"]

    def test_anthropic_schemas_exclude_message_type_for_creation(self):
        """Test that Anthropic creation/editing schemas exclude 'message' type."""
        all_schemas = MemoryAPIClient.get_all_memory_tool_schemas_anthropic()

        # Tools that should NOT expose message type (creation/editing tools)
        restricted_tools = {
            "add_memory_to_working_memory",
            "create_long_term_memory",
            "edit_long_term_memory",
        }

        # Tools that CAN expose message type (search/read tools)
        allowed_tools = {
            "search_memory",
            "get_long_term_memory",
        }

        for schema in all_schemas:
            function_name = schema["name"]
            params = schema["input_schema"]

            # Check direct memory_type property
            if "memory_type" in params["properties"]:
                memory_type_prop = params["properties"]["memory_type"]
                if "enum" in memory_type_prop:
                    if function_name in restricted_tools:
                        assert (
                            "message" not in memory_type_prop["enum"]
                        ), f"Anthropic creation/editing tool '{function_name}' should not expose 'message' memory type"
                    elif function_name in allowed_tools:
                        # These tools are allowed to have message in enum for filtering
                        pass

            # Check nested properties
            if "memories" in params["properties"]:
                items = params["properties"]["memories"].get("items", {})
                if (
                    "properties" in items
                    and "memory_type" in items["properties"]
                    and "enum" in items["properties"]["memory_type"]
                    and function_name in restricted_tools
                ):
                    memory_type_prop = items["properties"]["memory_type"]
                    assert (
                        "message" not in memory_type_prop["enum"]
                    ), f"Anthropic creation/editing tool '{function_name}' should not expose 'message' memory type in nested properties"
