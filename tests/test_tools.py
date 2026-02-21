"""Tests for agent tools."""

import pytest
from backend.agent.tools import TOOL_SCHEMAS, TOOLS_BY_NAME


class TestToolSchemas:
    """Tests for tool schema definitions."""

    def test_tool_schemas_are_list(self):
        """TOOL_SCHEMAS is a list of dicts."""
        assert isinstance(TOOL_SCHEMAS, list)
        assert len(TOOL_SCHEMAS) > 0

    def test_all_schemas_have_required_fields(self):
        """Every tool schema has type, function.name, and function.parameters."""
        for schema in TOOL_SCHEMAS:
            assert schema["type"] == "function"
            assert "function" in schema
            func = schema["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func

    def test_tool_names_match_registry(self):
        """All schema names have matching entries in TOOLS_BY_NAME."""
        for schema in TOOL_SCHEMAS:
            name = schema["function"]["name"]
            assert name in TOOLS_BY_NAME, f"Tool {name} not in TOOLS_BY_NAME"

    def test_known_tools_exist(self):
        """Core tools are registered."""
        expected = [
            "search_slides",
            "get_slide_content",
            "generate_quiz_question",
            "evaluate_student_answer",
            "get_student_progress",
            "extract_slide_image",
            "lookup_prerequisite",
        ]
        for tool_name in expected:
            assert tool_name in TOOLS_BY_NAME, f"Missing tool: {tool_name}"

    def test_schemas_have_valid_parameter_types(self):
        """Tool parameters have valid JSON Schema types."""
        valid_types = {"string", "integer", "number", "boolean", "array", "object"}
        for schema in TOOL_SCHEMAS:
            params = schema["function"]["parameters"]
            properties = params.get("properties", {})
            for prop_name, prop_def in properties.items():
                if "type" in prop_def:
                    assert prop_def["type"] in valid_types, (
                        f"Invalid type for {schema['function']['name']}.{prop_name}"
                    )
