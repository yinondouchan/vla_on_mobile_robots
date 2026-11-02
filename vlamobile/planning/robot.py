"""Thin client for the robot MCP server, for direct tool calls from the harness code."""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic_ai import BinaryImage
from pydantic_ai.mcp import MCPToolset


@dataclass
class Observation:
    """Parsed result of the `get_observation` tool."""

    task: str
    state: list[float]  # [base_yaw, lift_state, grip_r, grip_l]
    images: dict[str, BinaryImage]  # camera name -> JPEG


def parse_observation(content: list) -> Observation:
    """Parse ``["task: ...", "state: [...]", "camera: <name>", <image>, ...]``."""
    task, state, *cameras = content
    return Observation(
        task=task.removeprefix("task: "),
        state=json.loads(state.removeprefix("state: ")),
        images={label.removeprefix("camera: "): image for label, image in zip(cameras[::2], cameras[1::2])},
    )


class Robot:
    """Direct calls to the robot's MCP tools, outside of any agent's tool loop.

    Calls reuse one MCP session only while `server` is entered (e.g. through
    ``async with planner_agent:`` when the planner's toolset wraps `server`);
    otherwise each call opens its own connection.
    """

    def __init__(self, url: str):
        self.server = MCPToolset(url)

    async def get_subtasks(self) -> list[str]:
        return await self.server.direct_call_tool("get_subtasks", {})

    async def observe(self, image_size: int) -> Observation:
        content = await self.server.direct_call_tool(
            "get_observation", {"include_state": True, "image_size": image_size}
        )
        return parse_observation(content)

    async def execute_subtask(self, subtask: str) -> str:
        """Start `subtask` open-ended; returns immediately while the VLA keeps driving."""
        return await self.server.direct_call_tool("execute_subtask", {"subtask": subtask})

    async def stop_execution(self) -> str:
        return await self.server.direct_call_tool("stop_execution", {})
