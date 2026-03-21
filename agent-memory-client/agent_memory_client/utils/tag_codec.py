"""Tag validation utilities for the client library."""


def validate_no_commas_in_tags(
    values: list[str] | None, field_name: str
) -> list[str] | None:
    """Validate that no tag value contains a comma.

    Raises ``ValueError`` with a descriptive message when a value contains a
    comma.  Returns *values* unchanged so it can be used directly as a Pydantic
    ``@field_validator`` helper.
    """
    if not values:
        return values
    for idx, value in enumerate(values):
        if "," in value:
            raise ValueError(
                f"{field_name}[{idx}] contains a comma: {value!r}. "
                "Commas are not allowed because they are used as "
                "delimiters in storage."
            )
    return values