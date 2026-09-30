"""Prompt templates for the three planning VLM calls: decompose, assess, decide.

Each builder returns a ``(system, user)`` pair; callers assemble chat-template
messages (system + image user turn) for ``HuggingFaceChat.chat`` /
``chat_json``. All prompts demand a single strict JSON object as the reply
(no prose, no code fences) so the tolerant extractor in :mod:`.vlm` can parse
it reliably. Images (the current camera frames) are attached to the user turn
by the caller; the prompts reference them but never embed them.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Decompose: composite task -> ordered subtask list
# ---------------------------------------------------------------------------

DECOMPOSE_SYSTEM = (
    """
    You are a task planner for a robot. You are given an description of the environment and the robot's structure.
    You are shown the robot's current camera views. Break a composite task into a short ordered list of concrete subtasks,
    each independently executable by a low-level visuomotor policy (e.g. picking up an object, placing it somewhere, moving to a location).
    Subtasks must be phrased as short imperative commands and must be in execution order.

    <environment>
    The environment is described as follows:
    {environment}
    </environment>

    <robot_structure>
    The robot's structure is described as follows:
    {robot_structure}
    </robot_structure>
    """ )

_DECOMPOSE_USER_TEMPLATE = """
    <images>
    The images show the robot's current camera views.
    </images>

    <task>
    Composite task: {task}
    </task>

    <notes>
    {notes}
    </notes>

    <instructions>
    Break this task into at most {max_subtasks} subtasks.{vocabulary_block}
    </instructions>

    <format>
    Reply with ONLY a single valid JSON object. No markdown, no code fences, no explanation outside the JSON.
    Answer with JSON of this exact shape:
    {{"subtasks": ["<subtask 1>", "<subtask 2>", ...]}}
    </format>
"""

_VOCABULARY_TEMPLATE = """
<vocabulary_instructions>
The low-level policy only understands the following commands. Every subtask
MUST be copied verbatim from this list (same wording, same casing):
{vocabulary}
</vocabulary_instructions>"""


def build_decompose_prompt(
    environment: str,
    robot_structure: str,
    task: str,
    max_subtasks: int,
    predefined_tasks: list[str] | None = None,
    notes: str = "",
) -> tuple[str, str]:
    """Prompt pair for the initial plan decomposition.

    When ``predefined_tasks`` is given, the model is constrained to compose
    the plan only from that vocabulary (the caller additionally validates and
    snaps the parsed output to the list).
    """
    vocabulary_block = ""
    if predefined_tasks:
        vocabulary = "\n".join(f"- {t}" for t in predefined_tasks)
        vocabulary_block = _VOCABULARY_TEMPLATE.format(vocabulary=vocabulary)
    user = _DECOMPOSE_USER_TEMPLATE.format(
        task=task,
        max_subtasks=max_subtasks,
        vocabulary_block=vocabulary_block,
        notes=notes,
    )
    return DECOMPOSE_SYSTEM.format(environment=environment, robot_structure=robot_structure), user


# ---------------------------------------------------------------------------
# Assess: current frames + history -> progress / status of current subtask
# ---------------------------------------------------------------------------

ASSESS_SYSTEM = """
    You are monitoring a robot's subtask execution.
    You are shown the robot's current camera views, the plan, and a text log 
    of what happened so far. Judge only the CURRENT subtask from the visual 
    evidence in the images; the history is context, not proof of success. Be 
    conservative.

    <environment>
    The environment is described as follows:
    {environment}
    </environment>

    <robot_structure>
    The robot's structure is described as follows:
    {robot_structure}
    </robot_structure>
"""

_ASSESS_USER_TEMPLATE = """The current subtask is: {subtask}. Based on the above image, is the current subtask complete? Answer with 0 or 1. The answer is:"""

