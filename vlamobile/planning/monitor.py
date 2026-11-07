"""Monitor agent and the monitoring loop that runs one subtask until it ends."""

from __future__ import annotations

import time

import anyio
from pydantic_ai import Agent, RunContext, ToolReturn, UsageLimits

from vlamobile.planning.prompts import MONITOR_SYSTEM_PROMPT, render_monitor_prompt
from vlamobile.planning.types import HarnessDeps, MonitorVerdict, SubtaskReport

monitor_agent = Agent(
    output_type=MonitorVerdict,
    instructions=MONITOR_SYSTEM_PROMPT,
    model_settings={"thinking": False},
    name="monitor",
)


def stalled(history: list[MonitorVerdict], checks: int) -> bool:
    """Whether none of the last `checks` verdicts got past the progress of the verdict before them."""
    return len(history) > checks and max(v.progress for v in history[-checks:]) <= history[-checks - 1].progress


async def run_subtask(ctx: RunContext[HarnessDeps], subtask: str) -> ToolReturn:
    """Run `subtask` open-ended, judge it every tick, stop the robot and report with the final frames."""
    robot, cfg = ctx.deps.robot, ctx.deps.config
    history: list[MonitorVerdict] = []
    start = time.monotonic()
    try:
        await robot.execute_subtask(subtask)
        while True:
            tick_start = time.monotonic()
            if tick_start - start > cfg.subtask_timeout_s:
                outcome = "timed_out"
                break
            if stalled(history, cfg.stall_checks):
                outcome = "stalled"
                break
            obs = await robot.observe(image_size=cfg.monitor_image_size)
            result = await monitor_agent.run(
                [render_monitor_prompt(subtask, history), *obs.images.values()],
                model=cfg.monitor_model,
                usage=ctx.usage,
                # Without it, run() would apply its default request limit to the shared episode usage.
                usage_limits=UsageLimits(request_limit=cfg.request_limit),
                capabilities=cfg.skills(),
            )
            verdict = result.output
            ctx.deps.events.emit(subtask, verdict)
            history.append(verdict)
            if verdict.status != "running":
                outcome = verdict.status
                break
            await anyio.sleep(tick_start + cfg.min_tick_period_s - time.monotonic())
    finally:
        # Shielded so the robot is also stopped when the run is cancelled.
        with anyio.CancelScope(shield=True):
            await robot.stop_execution()

    report = SubtaskReport(
        subtask=subtask,
        outcome=outcome,
        duration_s=round(time.monotonic() - start, 1),
        checks=len(history),
        final_progress=history[-1].progress if history else 0.0,
        summary=" ".join(v.describe() for v in history[-3:]) or "no verdicts",
    )
    ctx.deps.attempts.append(report)
    obs = await robot.observe(image_size=cfg.monitor_image_size)
    frames = [item for name, image in obs.images.items() for item in (f"camera: {name}", image)]
    return ToolReturn(return_value=report, content=["Final camera frames, robot stopped:", *frames])
