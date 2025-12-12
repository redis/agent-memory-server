"""
Tool schema classes for customizing memory tool descriptions.

This module provides wrapper classes that allow users to customize
tool descriptions and other properties before passing them to LLMs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from typing_extensions import Self


class ToolSchema:
    """
    Wrapper for tool schema dictionaries that provides a fluent API
    for customizing tool descriptions and other properties.

    Supports both OpenAI and Anthropic schema formats.

    Example:
        ```python
        schema = MemoryAPIClient.get_memory_search_tool_schema()
        schema.set_description("Custom description for my use case")
        schema.set_name("my_custom_search_tool")

        # Pass to LLM
        tools = [schema.to_dict()]
        ```
    """

    def __init__(
        self,
        schema: dict[str, Any],
        schema_format: Literal["openai", "anthropic"] = "openai",
    ):
        """
        Initialize a ToolSchema wrapper.

        Args:
            schema: The raw schema dictionary
            schema_format: The schema format ("openai" or "anthropic")
        """
        self._schema = schema
        self._format = schema_format

    @property
    def format(self) -> Literal["openai", "anthropic"]:
        """Get the schema format."""
        return self._format

    def set_description(self, description: str) -> Self:
        """
        Set a custom description for the tool.

        Args:
            description: The new description text

        Returns:
            Self for method chaining
        """
        if self._format == "openai":
            self._schema["function"]["description"] = description
        else:  # anthropic
            self._schema["description"] = description
        return self

    def set_name(self, name: str) -> Self:
        """
        Set a custom name for the tool.

        Args:
            name: The new tool name

        Returns:
            Self for method chaining
        """
        if self._format == "openai":
            self._schema["function"]["name"] = name
        else:  # anthropic
            self._schema["name"] = name
        return self

    def set_parameter_description(self, param_name: str, description: str) -> Self:
        """
        Set a custom description for a specific parameter.

        Args:
            param_name: The name of the parameter to update
            description: The new description for the parameter

        Returns:
            Self for method chaining
        """
        if self._format == "openai":
            props = self._schema["function"]["parameters"]["properties"]
        else:  # anthropic
            props = self._schema["input_schema"]["properties"]

        if param_name in props:
            props[param_name]["description"] = description
        else:
            raise KeyError(
                f"Parameter '{param_name}' does not exist in the schema properties: {list(props.keys())}"
            )
        return self

    def get_description(self) -> str:
        """Get the current tool description."""
        if self._format == "openai":
            return self._schema["function"]["description"]
        return self._schema["description"]

    def get_name(self) -> str:
        """Get the current tool name."""
        if self._format == "openai":
            return self._schema["function"]["name"]
        return self._schema["name"]

    def get_parameter_description(self, param_name: str) -> str | None:
        """
        Get the description for a specific parameter.

        Args:
            param_name: The name of the parameter

        Returns:
            The parameter description, or None if not found
        """
        if self._format == "openai":
            props = self._schema["function"]["parameters"]["properties"]
        else:  # anthropic
            props = self._schema["input_schema"]["properties"]

        if param_name in props:
            return props[param_name].get("description")
        return None

    def to_dict(self) -> dict[str, Any]:
        """Return the schema as a dictionary for LLM consumption."""
        import copy

        return copy.deepcopy(self._schema)

    def copy(self) -> ToolSchema:
        """Create an independent copy of this schema."""
        import copy

        return ToolSchema(copy.deepcopy(self._schema), self._format)

    # Dict-like access for backwards compatibility
    def __getitem__(self, key: str) -> Any:
        return self._schema[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._schema[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._schema

    def __repr__(self) -> str:
        return f"ToolSchema(name={self.get_name()!r}, format={self._format!r})"


class ToolSchemaCollection:
    """
    Collection of tool schemas with bulk customization support.

    Example:
        ```python
        all_tools = MemoryAPIClient.get_all_memory_tool_schemas()
        all_tools.set_description("search_memory", "Custom search description")

        # Get specific tool
        search_tool = all_tools.get_by_name("search_memory")
        search_tool.set_parameter_description("query", "Custom query description")

        # Use all tools
        tools = all_tools.to_list()
        ```
    """

    def __init__(self, schemas: list[ToolSchema]):
        """
        Initialize a ToolSchemaCollection.

        Args:
            schemas: List of ToolSchema objects
        """
        self._schemas = schemas

    def get_by_name(self, name: str) -> ToolSchema | None:
        """
        Get a specific tool schema by name.

        Args:
            name: The tool name to find

        Returns:
            The ToolSchema if found, None otherwise
        """
        for schema in self._schemas:
            if schema.get_name() == name:
                return schema
        return None

    def set_description(self, name: str, description: str) -> Self:
        """
        Set description for a specific tool by name.

        Args:
            name: The tool name to update
            description: The new description

        Returns:
            Self for method chaining
        """
        schema = self.get_by_name(name)
        if schema:
            schema.set_description(description)
        return self

    def set_name(self, old_name: str, new_name: str) -> Self:
        """
        Rename a specific tool.

        Args:
            old_name: The current tool name
            new_name: The new tool name

        Returns:
            Self for method chaining
        """
        schema = self.get_by_name(old_name)
        if schema:
            schema.set_name(new_name)
        return self

    def to_list(self) -> list[dict[str, Any]]:
        """Return all schemas as a list of dictionaries."""
        return [s.to_dict() for s in self._schemas]

    def copy(self) -> ToolSchemaCollection:
        """Create an independent copy of this collection."""
        return ToolSchemaCollection([s.copy() for s in self._schemas])

    def names(self) -> list[str]:
        """Get all tool names in the collection."""
        return [s.get_name() for s in self._schemas]

    def __iter__(self):
        return iter(self._schemas)

    def __len__(self) -> int:
        return len(self._schemas)

    def __getitem__(self, index: int) -> ToolSchema:
        return self._schemas[index]

    def __repr__(self) -> str:
        names = [s.get_name() for s in self._schemas]
        return f"ToolSchemaCollection({names!r})"
