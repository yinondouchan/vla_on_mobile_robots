"""Run one episode of the harness against the sim's MCP server and print the result.

Usage::

    python -m vlamobile.planning.run --task="Pick up the cube and place it on the red platform"

Without ``--task`` the planner takes the task from the sim (`get_observation`).
Any `HarnessConfig` field can be overridden, e.g. ``--monitor_image_size=128``.
"""

from __future__ import annotations

import asyncio

import fire
import logfire
from pydantic_ai import UsageLimits

from vlamobile.planning.config import HarnessConfig
from vlamobile.planning.planner import planner_agent
from vlamobile.planning.robot import Robot
from vlamobile.planning.types import EpisodeResult, HarnessDeps


async def run_episode(config: HarnessConfig, task: str | None = None) -> EpisodeResult:
    robot = Robot(config.mcp_url)
    prompt = f"Composite task: {task}" if task else "No composite task was given; take it from `get_observation`."
    # One MCP session for the whole episode, shared by the planner's tools and the monitoring loop.
    async with robot.server:
        result = await planner_agent.run(
            prompt,
            model=config.planner_model,
            deps=HarnessDeps(robot=robot, config=config),
            usage_limits=UsageLimits(request_limit=config.request_limit),
            capabilities=config.skills(),
        )
    return result.output


def main(task: str | None = None, **config_overrides) -> None:
    logfire.configure(send_to_logfire="if-token-present")
    logfire.instrument_pydantic_ai()
    result = asyncio.run(run_episode(HarnessConfig(**config_overrides), task))
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    fire.Fire(main)
