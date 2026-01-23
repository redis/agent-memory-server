"""
Test file for memory tool schemas.

Tests that tool schemas are correctly structured and that the 'message' memory type
is not exposed to LLM tools (it should only be used server-side).
"""

from agent_memory_client import MemoryAPIClient, ToolSchema, ToolSchemaCollection


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
        """Test lazily_create_long_term_memory tool schema structure."""
        schema = MemoryAPIClient.get_add_memory_tool_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "lazily_create_long_term_memory"
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]

        params = schema["function"]["parameters"]
        assert params["type"] == "object"
        assert "text" in params["properties"]
        assert "memory_type" in params["properties"]
        assert "text" in params["required"]
        assert "memory_type" in params["required"]

    def test_create_long_term_memory_tool_schema(self):
        """Test eagerly_create_long_term_memory tool schema structure."""
        schema = MemoryAPIClient.create_long_term_memory_tool_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "eagerly_create_long_term_memory"
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
            "lazily_create_long_term_memory",
            "update_working_memory_data",
            "get_long_term_memory",
            "eagerly_create_long_term_memory",
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
        """Test that lazily_create_long_term_memory excludes 'message' type."""
        schema = MemoryAPIClient.get_add_memory_tool_schema()

        params = schema["function"]["parameters"]
        memory_type_prop = params["properties"]["memory_type"]

        # Should only have episodic and semantic
        assert memory_type_prop["enum"] == ["episodic", "semantic"]
        assert "message" not in memory_type_prop["enum"]

    def test_create_long_term_memory_excludes_message_type(self):
        """Test that eagerly_create_long_term_memory excludes 'message' type."""
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
            "lazily_create_long_term_memory",
            "eagerly_create_long_term_memory",
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

            # Check nested properties (like in eagerly_create_long_term_memory)
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
        """Test eagerly_create_long_term_memory tool schema in Anthropic format."""
        schema = MemoryAPIClient.create_long_term_memory_tool_schema_anthropic()

        assert schema["name"] == "eagerly_create_long_term_memory"
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
            "lazily_create_long_term_memory",
            "eagerly_create_long_term_memory",
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


class TestToolSchemaCustomization:
    """Tests for ToolSchema customization methods."""

    def test_tool_schema_set_description(self):
        """Test setting custom description on a tool schema."""
        schema = MemoryAPIClient.get_memory_search_tool_schema()
        assert isinstance(schema, ToolSchema)

        original_desc = schema.get_description()
        custom_desc = "My custom search description for LLM"

        # Test fluent API returns self
        result = schema.set_description(custom_desc)
        assert result is schema

        # Test description was updated
        assert schema.get_description() == custom_desc
        assert schema["function"]["description"] == custom_desc
        assert original_desc != custom_desc

    def test_tool_schema_set_name(self):
        """Test setting custom name on a tool schema."""
        schema = MemoryAPIClient.get_memory_search_tool_schema()

        original_name = schema.get_name()
        custom_name = "my_custom_search"

        result = schema.set_name(custom_name)
        assert result is schema

        assert schema.get_name() == custom_name
        assert schema["function"]["name"] == custom_name
        assert original_name != custom_name

    def test_tool_schema_set_parameter_description(self):
        """Test setting custom parameter description."""
        schema = MemoryAPIClient.get_memory_search_tool_schema()

        custom_param_desc = "The search query to find relevant memories"
        result = schema.set_parameter_description("query", custom_param_desc)
        assert result is schema

        params = schema["function"]["parameters"]["properties"]
        assert params["query"]["description"] == custom_param_desc

    def test_tool_schema_to_dict(self):
        """Test converting ToolSchema to dict."""
        schema = MemoryAPIClient.get_memory_search_tool_schema()
        schema.set_description("Custom description")

        result = schema.to_dict()
        assert isinstance(result, dict)
        assert result["function"]["description"] == "Custom description"
        # Ensure it's a new dict, not the same reference
        assert result is not schema._schema

    def test_tool_schema_copy(self):
        """Test copying a ToolSchema."""
        schema = MemoryAPIClient.get_memory_search_tool_schema()
        original_desc = schema.get_description()

        copy = schema.copy()
        assert isinstance(copy, ToolSchema)

        # Modify copy
        copy.set_description("Modified copy")

        # Original should be unchanged
        assert schema.get_description() == original_desc
        assert copy.get_description() == "Modified copy"

    def test_tool_schema_backwards_compatibility(self):
        """Test that dict-like access still works for backwards compatibility."""
        schema = MemoryAPIClient.get_memory_search_tool_schema()

        # Test __getitem__
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search_memory"

        # Test nested access
        assert "query" in schema["function"]["parameters"]["properties"]

    def test_anthropic_schema_customization(self):
        """Test customization works for Anthropic schemas."""
        schema = MemoryAPIClient.get_memory_search_tool_schema_anthropic()
        assert isinstance(schema, ToolSchema)

        custom_desc = "Custom Anthropic search description"
        schema.set_description(custom_desc)

        # Anthropic format has description at top level
        assert schema.get_description() == custom_desc
        assert schema["description"] == custom_desc


class TestToolSchemaCollectionCustomization:
    """Tests for ToolSchemaCollection customization methods."""

    def test_collection_returns_tool_schema_collection(self):
        """Test that get_all_memory_tool_schemas returns ToolSchemaCollection."""
        collection = MemoryAPIClient.get_all_memory_tool_schemas()
        assert isinstance(collection, ToolSchemaCollection)

    def test_collection_get_by_name(self):
        """Test getting a specific tool by name from collection."""
        collection = MemoryAPIClient.get_all_memory_tool_schemas()

        search_tool = collection.get_by_name("search_memory")
        assert search_tool is not None
        assert isinstance(search_tool, ToolSchema)
        assert search_tool.get_name() == "search_memory"

        # Test non-existent tool
        non_existent = collection.get_by_name("non_existent_tool")
        assert non_existent is None

    def test_collection_set_description(self):
        """Test setting description on a tool in the collection."""
        collection = MemoryAPIClient.get_all_memory_tool_schemas()

        custom_desc = "Bulk customized search description"
        result = collection.set_description("search_memory", custom_desc)
        assert result is collection

        search_tool = collection.get_by_name("search_memory")
        assert search_tool.get_description() == custom_desc

    def test_collection_to_list(self):
        """Test converting collection to list of dicts."""
        collection = MemoryAPIClient.get_all_memory_tool_schemas()
        collection.set_description("search_memory", "Custom desc")

        result = collection.to_list()
        assert isinstance(result, list)
        assert all(isinstance(item, dict) for item in result)

        # Find search_memory in list
        search_dict = next(
            (item for item in result if item["function"]["name"] == "search_memory"),
            None,
        )
        assert search_dict is not None
        assert search_dict["function"]["description"] == "Custom desc"

    def test_collection_iteration(self):
        """Test iterating over collection."""
        collection = MemoryAPIClient.get_all_memory_tool_schemas()

        count = 0
        for schema in collection:
            assert isinstance(schema, ToolSchema)
            count += 1

        assert count == len(collection)
        assert count > 0

    def test_collection_len(self):
        """Test length of collection."""
        collection = MemoryAPIClient.get_all_memory_tool_schemas()
        assert len(collection) == 9  # All 9 memory tools

    def test_anthropic_collection_customization(self):
        """Test customization works for Anthropic collection."""
        collection = MemoryAPIClient.get_all_memory_tool_schemas_anthropic()
        assert isinstance(collection, ToolSchemaCollection)

        custom_desc = "Custom Anthropic collection desc"
        collection.set_description("search_memory", custom_desc)

        search_tool = collection.get_by_name("search_memory")
        assert search_tool.get_description() == custom_desc

    def test_collection_indexing(self):
        """Test indexing into collection."""
        collection = MemoryAPIClient.get_all_memory_tool_schemas()

        first_tool = collection[0]
        assert isinstance(first_tool, ToolSchema)

    def test_collection_names(self):
        """Test getting all tool names from collection."""
        collection = MemoryAPIClient.get_all_memory_tool_schemas()

        names = collection.names()
        assert isinstance(names, list)
        assert "search_memory" in names
        assert "eagerly_create_long_term_memory" in names
        assert len(names) == 9
