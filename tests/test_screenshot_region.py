"""Tests for regional screenshot and multi-monitor support (A3 + A4)."""

from __future__ import annotations

from agentra.tools.computer import ComputerTool


class TestExtractRegion:
    def test_returns_tuple_when_all_provided(self) -> None:
        kwargs = {"region_x": 10, "region_y": 20, "region_width": 100, "region_height": 50}
        result = ComputerTool._extract_region(kwargs)
        assert result == (10, 20, 100, 50)

    def test_returns_none_when_partial(self) -> None:
        kwargs = {"region_x": 10, "region_y": 20}
        assert ComputerTool._extract_region(kwargs) is None

    def test_returns_none_when_empty(self) -> None:
        assert ComputerTool._extract_region({}) is None


class TestListMonitors:
    def test_returns_list(self) -> None:
        tool = ComputerTool()
        monitors = tool.list_monitors()
        assert isinstance(monitors, list)
        assert len(monitors) >= 1
        assert monitors[0]["index"] == 0
        assert "width" in monitors[0]
        assert "height" in monitors[0]

    def test_first_monitor_is_all_combined(self) -> None:
        tool = ComputerTool()
        monitors = tool.list_monitors()
        assert monitors[0]["label"] == "all combined"


class TestSchemaIncludes:
    def test_schema_has_region_fields(self) -> None:
        tool = ComputerTool()
        props = tool.schema["properties"]
        assert "region_x" in props
        assert "region_y" in props
        assert "region_width" in props
        assert "region_height" in props

    def test_schema_has_monitor_field(self) -> None:
        tool = ComputerTool()
        props = tool.schema["properties"]
        assert "monitor" in props

    def test_schema_has_list_monitors_action(self) -> None:
        tool = ComputerTool()
        actions = tool.schema["properties"]["action"]["enum"]
        assert "list_monitors" in actions
