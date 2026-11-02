"""Policy that exposes the simulated robot as an MCP server.

Drop-in replacement for :class:`vlamobile.inference.LerobotInference`: the sim
calls ``predict`` every frame, and an MCP client (e.g. an LLM agent) drives the
robot through the ``get_controls``, ``get_observation``, ``control`` and ``stop_execution`` tools,
served over streamable HTTP at ``http://<host>:<port>/mcp``. When given a VLA
policy, the ``get_subtasks`` and ``execute_subtask`` tools let the client run
the subtasks it was trained on.
"""

from __future__ import annotations

import io
import threading
from dataclasses import dataclass, field

import anyio
import numpy as np
from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from PIL import Image as PILImage

from vlamobile.inference import LerobotInference

ACTION_NAMES = ["forward", "turn", "grip", "lift"]
ACTION_LOW = np.array([-1, -1, 0.0, 0.0])
ACTION_HIGH = np.array([1, 1, 1.0, 1.0])


def _jpeg(image: np.ndarray, size: int) -> bytes:
    img = PILImage.fromarray(image)
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@dataclass
class _Chunk:
    """Steps to execute, taken from ``actions`` or, when ``subtask`` is set, from the VLA."""

    num_steps: int | None  # None: run until interrupted, without recording frames
    obs_every: int
    image_size: int
    actions: list[np.ndarray] | None = None
    subtask: str | None = None
    step: int = 0
    frames: list[tuple[int, np.ndarray, dict[str, bytes]]] = field(default_factory=list)
    interrupted: bool = False
    done: threading.Event = field(default_factory=threading.Event)

    def record(self, state: np.ndarray, images: dict[str, np.ndarray]) -> None:
        jpegs = {name: _jpeg(image, self.image_size) for name, image in images.items()}
        self.frames.append((self.step, state, jpegs))


class McpPolicy:
    """Serve the sim robot over MCP; actions come from ``control`` and ``execute_subtask`` tool calls."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        fps: float = 30.0,
        chunk_timeout_s: float = 120.0,
        vla: LerobotInference | None = None,
    ):
        self.fps = fps
        self.chunk_timeout_s = chunk_timeout_s
        self._vla = vla
        self._lock = threading.Lock()
        self._state: np.ndarray | None = None
        self._images: dict[str, np.ndarray] | None = None
        self._task: str | None = None
        self._chunk: _Chunk | None = None
        self._last_action = np.zeros(4)

        self._server = MCPServer("mobile-robot")
        self._register_tools()
        if vla is not None:
            self._register_subtask_tools()
        threading.Thread(
            target=self._server.run,
            kwargs={"transport": "streamable-http", "host": host, "port": port},
            daemon=True,
        ).start()
        print(f"MCP server listening on http://{host}:{port}/mcp", flush=True)

    def reset(self) -> None:
        """Interrupt the running action chunk, if any, and zero the held action."""
        with self._lock:
            self._cancel_chunk()
            self._last_action = np.zeros(4)

    def predict(
        self,
        state: np.ndarray,
        images: dict[str, np.ndarray],
        task: str | None = None,
    ) -> np.ndarray:
        """Return the next action of the running chunk, or hold the last action when idle.

        Called once per sim frame. ``chunk.step`` counts the actions already
        applied, so the frame passed here shows the result of step ``chunk.step - 1``.
        """
        with self._lock:
            # Keep the latest frame for `get_observation`.
            self._state = np.array(state[0])
            self._images = {name: image.copy() for name, image in images.items()}
            self._task = task

            chunk = self._chunk
            if chunk is None:
                # Idle: no tool call is driving the robot, keep applying the last action.
                return self._last_action[None]

            if chunk.step == chunk.num_steps:
                # All actions were applied; this frame is the result of the last one.
                # Record it, wake up the tool call waiting on the chunk, and go idle.
                chunk.record(self._state, images)
                chunk.done.set()
                self._chunk = None
                if chunk.subtask is not None:
                    # The client never chose the VLA's last action, so don't keep
                    # driving with it. Grip and lift are held so a carried cube isn't dropped.
                    self._last_action = np.array([0.0, 0.0, *self._last_action[2:]])
                return self._last_action[None]

            # Record every `obs_every` steps for the tool result, starting with the
            # frame before the first action. Chunks that run until interrupted
            # return immediately, so nothing would read their frames.
            if chunk.num_steps is not None and chunk.step % chunk.obs_every == 0:
                chunk.record(self._state, images)

            if chunk.subtask is None:
                # `control` chunk: actions were given up front by the client.
                self._last_action = chunk.actions[chunk.step]
            else:
                # `execute_subtask` chunk: one action per frame from the VLA's queue
                # (lerobot's `select_action` refills it with a new action chunk when empty).
                if chunk.step == 0:
                    # Drop actions the VLA queued for a previous subtask prompt.
                    self._vla.reset()
                self._last_action = self._vla.predict(state, images, task=chunk.subtask).reshape(-1)
            chunk.step += 1
            return self._last_action[None]

    def _cancel_chunk(self) -> None:
        if self._chunk is not None:
            self._chunk.interrupted = True
            self._chunk.done.set()
            self._chunk = None

    def _start_chunk(self, chunk: _Chunk) -> None:
        """Make ``chunk`` the running chunk, replacing one that runs until interrupted."""
        with self._lock:
            if self._chunk is not None:
                if self._chunk.num_steps is not None:
                    raise ToolError("another action chunk is still running")
                self._cancel_chunk()
            self._chunk = chunk

    async def _run_chunk(self, chunk: _Chunk, include_state: bool) -> list:
        """Run ``chunk`` to completion and return its status and recorded observations."""
        if chunk.obs_every < 1:
            raise ToolError("obs_every must be >= 1")
        self._start_chunk(chunk)

        finished = await anyio.to_thread.run_sync(chunk.done.wait, self.chunk_timeout_s)
        if not finished:
            self.reset()

        status = f"executed {chunk.step}/{chunk.num_steps} steps"
        if chunk.interrupted:
            status += " (interrupted)"
        out: list = [status]
        for step, state, jpegs in chunk.frames:
            out.append(f"step {step}" + (f", state: {state.tolist()}" if include_state else ""))
            for name, data in jpegs.items():
                out += [f"camera: {name}", Image(data=data, format="jpeg")]
        return out

    def _register_tools(self) -> None:
        @self._server.tool()
        def get_controls() -> dict:
            """Action layout and limits. One action is applied per simulation step at `fps` steps per second.

            Between `control` calls the last action is held, so the robot keeps moving
            until a chunk ends with a zero forward/turn action.
            """
            return {
                "names": ACTION_NAMES,
                "low": ACTION_LOW.tolist(),
                "high": ACTION_HIGH.tolist(),
                "fps": self.fps,
            }

        @self._server.tool(structured_output=False)
        def get_observation(include_state: bool = True, image_size: int = 256):
            """Current task, camera images (longest side `image_size` px) and optionally the robot state."""
            with self._lock:
                if self._images is None:
                    raise ToolError("the simulation has not produced an observation yet")
                images, state, task = self._images, self._state, self._task

            out: list = [f"task: {task}"]
            if include_state:
                out.append(f"state: {state.tolist()}")
            for name, image in images.items():
                out += [f"camera: {name}", Image(data=_jpeg(image, image_size), format="jpeg")]
            return out

        @self._server.tool(structured_output=False)
        async def control(
            actions: list[list[float]],
            obs_every: int = 10,
            image_size: int = 128,
            include_state: bool = True,
        ):
            """Execute an action chunk, one [forward, turn, grip, lift] action per simulation step.

            Blocks until the chunk finishes or is interrupted (`stop_execution`, env
            reset or joystick override). The last action keeps being applied after the chunk ends.
            Returns the observation every `obs_every` steps plus the one after the
            last action, with images resized to `image_size` px. Stops a subtask
            started by `execute_subtask` without `num_frames`.
            """
            chunk = _Chunk(
                num_steps=len(actions),
                obs_every=obs_every,
                image_size=image_size,
                actions=[np.clip(np.asarray(a, dtype=np.float64), ACTION_LOW, ACTION_HIGH) for a in actions],
            )
            return await self._run_chunk(chunk, include_state)

        @self._server.tool()
        def stop_execution() -> str:
            """Stop the running `control` chunk or subtask and halt the robot.

            The interrupted call returns what it executed so far. Forward/turn are
            zeroed while grip/lift are held, so a carried cube isn't dropped.
            """
            with self._lock:
                chunk = self._chunk
                self._cancel_chunk()
                self._last_action = np.array([0.0, 0.0, *self._last_action[2:]])
            if chunk is None:
                return "nothing was running; robot halted"
            name = f"subtask {chunk.subtask!r}" if chunk.subtask is not None else "control chunk"
            return f"stopped {name} after {chunk.step} steps"

    def _register_subtask_tools(self) -> None:
        subtasks = self._vla.tasks

        @self._server.tool()
        def get_subtasks() -> list[str]:
            """Subtask prompts the VLA policy was trained on, to be run with `execute_subtask`."""
            return subtasks

        @self._server.tool(structured_output=False)
        async def execute_subtask(
            subtask: str,
            num_frames: int | None = None,
            obs_every: int = 10,
            image_size: int = 256,
            include_state: bool = True,
        ):
            """Let the VLA policy drive the robot on `subtask`, one of the `get_subtasks` prompts.

            With `num_frames`, runs for that many simulation steps and returns like
            `control`; afterwards forward/turn are zeroed while grip/lift are held.
            Without it, returns immediately and the subtask runs until replaced by a
            `control` or `execute_subtask` call, or interrupted (`stop_execution`,
            env reset or joystick override); watch it with `get_observation`.
            """
            if subtask not in subtasks:
                raise ToolError(f"unknown subtask {subtask!r}; expected one of {subtasks}")
            if num_frames is not None and num_frames < 1:
                raise ToolError("num_frames must be >= 1")
            chunk = _Chunk(
                num_steps=num_frames,
                obs_every=obs_every,
                image_size=image_size,
                subtask=subtask,
            )
            if num_frames is None:
                self._start_chunk(chunk)
                return [f"started subtask {subtask!r}"]
            return await self._run_chunk(chunk, include_state)
