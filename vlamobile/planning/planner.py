"""Planner agent: decomposes the task, runs subtasks through the monitoring loop and reacts to reports."""

from __future__ import annotations

import logfire
from pydantic_ai import AbstractToolset, Agent, ModelRetry, RunContext, ToolFailed, ToolReturn

from vlamobile.planning import monitor
from vlamobile.planning.prompts import PLANNER_SYSTEM_PROMPT
from vlamobile.planning.types import EpisodeResult, HarnessDeps

# `run_subtask` owns starting and stopping subtasks, so the planner can't leave one running unmonitored.
HIDDEN_ROBOT_TOOLS = {"execute_subtask", "stop_execution"}


def finish(ctx: RunContext[HarnessDeps], success: bool, summary: str) -> EpisodeResult:
    """End the episode: whether the composite task was achieved, and a short summary."""
    return EpisodeResult(success=success, summary=summary, attempts=ctx.deps.attempts)


planner_agent = Agent(
    deps_type=HarnessDeps,
    output_type=finish,
    instructions=PLANNER_SYSTEM_PROMPT,
    model_settings={"thinking": False},
    name="planner",
)


@planner_agent.toolset(per_run_step=False)
def robot_tools(ctx: RunContext[HarnessDeps]) -> AbstractToolset[HarnessDeps]:
    return ctx.deps.robot.server.filtered(lambda ctx, tool_def: tool_def.name not in HIDDEN_ROBOT_TOOLS)


@planner_agent.tool
async def set_plan(ctx: RunContext[HarnessDeps], steps: list[str]) -> str:
    """Record the ordered plan; every step must be a prompt from `get_subtasks`, copied verbatim."""
    subtasks = await ctx.deps.robot.get_subtasks()
    unknown = [step for step in steps if step not in subtasks]
    if unknown:
        raise ModelRetry(f"unknown subtasks {unknown}; use only these exact prompts: {subtasks}")
    ctx.deps.plan = steps
    logfire.info("plan: {steps}", steps=steps)
    return f"plan recorded with {len(steps)} steps"


@planner_agent.tool
async def run_subtask(ctx: RunContext[HarnessDeps], subtask: str) -> ToolReturn:
    """Run one step of the plan until it succeeds, fails, stalls or times out, then stop the robot.

    Returns a report and the final camera frames.
    """
    if subtask not in ctx.deps.plan:
        raise ModelRetry(f"{subtask!r} is not in the plan {ctx.deps.plan}; update it with `set_plan` first")
    failures = 0
    for report in reversed(ctx.deps.attempts):
        if report.subtask != subtask or report.outcome == "succeeded":
            break
        failures += 1
    if failures > ctx.deps.config.max_subtask_retries:
        raise ToolFailed(
            f"{subtask!r} failed {failures} times in a row; change approach "
            "(a corrective step or a new plan) or give up"
        )
    logfire.info(
        "step {step}/{steps}: {subtask}",
        step=ctx.deps.plan.index(subtask) + 1,
        steps=len(ctx.deps.plan),
        subtask=subtask,
    )
    return await monitor.run_subtask(ctx, subtask)
