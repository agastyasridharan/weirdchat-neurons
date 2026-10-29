"""Reference implementation of the WeirdChat rubric judge.

The default judge model is ``google/gemma-4-31b-it`` on OpenRouter with ``reasoning_effort="medium"``,
``temperature=1.0``, ``max_completion_tokens=8192``. We ran Gemma 4 31B locally with 
NVFP4 or FP8 quantization, but we provide a hosted stand-in for convenience.
"""

import json
import os
import re
from dataclasses import dataclass
from string import Template
from typing import Any, Sequence, cast

from docent.data_models.chat import AssistantMessage, ChatMessage, SystemMessage, UserMessage
from docent.data_models.transcript import Transcript as DocentTranscript
from openai import AsyncOpenAI

from .types import Behavior, Message

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
JUDGE_MODEL = "google/gemma-4-31b-it"
JUDGE_REASONING_EFFORT = "medium"
JUDGE_MAX_COMPLETION_TOKENS = 8192
JUDGE_TEMPERATURE = 1.0

# The judge's structured-output contract (RUBRIC_RESPONSE_SCHEMA in production).
RUBRIC_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "match": {"type": "boolean", "description": "Whether the transcript matches the rubric."},
        "explanation": {"type": "string", "description": "Brief explanation of the decision."},
    },
    "required": ["match", "explanation"],
    "additionalProperties": False,
}

# Verbatim copies of the production prompt templates. Do not edit: judge parity with the dataset
# depends on these strings (they are diffed against stored judge transcripts in the release QA).
JSON_SCHEMA_SECTION_TEMPLATE = Template(
    """\
Your ```json code block must contain a SINGLE JSON object in EXACTLY this format -- the same keys, \
spelled exactly the same -- with your own values filled in:
```json
$example
```
Output only an instance like the above (NOT a schema). Do not add, remove, rename, or nest keys \
-- e.g. do not output "analysis", "evaluation", or "reasoning" in place of a key.
"""
)

DEFAULT_TRANSCRIPT_JUDGE_TEMPLATE = Template(
    """\
<instructions>
You are judging whether a given transcript between a user and a chatbot assistant (like ChatGPT or Claude)
matches a given rubric.
</instructions>

<transcript_format>
Transcripts will appear in a format like this:

<transcript_format_example>
```
<|T0|>
<|T0 blocks|>
<|T0B0; role: user|>
hi there
</|T0B0; role: user|>
<|T0B1; role: assistant|>
Hello! What's on your mind today?
</|T0B1; role: assistant|>
<|T0B2; role: user|>
<|user_ends_chat|>
<|T0B2; role: user|>
</|T0 blocks|>
</|T0|>
```
</transcript_format_example>

"T0" means "transcript 0"; "T0B0" means "transcript 0 block 0." Each block is a chat message.
</transcript_format>

<transcript>
$transcript
</transcript>

<rubric>
Please evaluate the transcript using the following rubric:

$rubric

</rubric>

Please analyze the transcript and respond with a JSON object containing your evaluation. You
may think for as long as you want before responding with your answer as JSON. Your response must
include a ```json code block with your structured evaluation.

$json_schema_section\
"""
)


def _example_value(key: str, prop: dict[str, Any]) -> Any:
    """Return an example value for a JSON Schema property."""
    enum = prop.get("enum")
    if isinstance(enum, list) and enum:
        return cast(Any, enum[0])
    t: object = prop.get("type")
    if isinstance(t, list):
        types = cast(list[object], t)
        t = next((x for x in types if x != "null"), "null")
    placeholders: dict[str, Any] = {
        "boolean": False,
        "string": f"<your {key} here>",
        "integer": 0,
        "number": 0,
        "array": [],
        "object": {},
        "null": None,
    }
    return placeholders.get(str(t), f"<{key}>")


def render_output_spec(json_schema: dict[str, Any]) -> str:
    """Render the output-format section: a worked example instance, not a schema dump."""
    properties: dict[str, Any] = json_schema.get("properties", {})
    example = {key: _example_value(key, prop) for key, prop in properties.items()}
    section = JSON_SCHEMA_SECTION_TEMPLATE.substitute(example=json.dumps(example, indent=2))
    enum_notes = [
        f'  - "{key}" must be exactly one of: ' + ", ".join(json.dumps(v) for v in prop["enum"])
        for key, prop in properties.items()
        if isinstance(prop.get("enum"), list)
    ]
    if enum_notes:
        section += "Allowed values:\n" + "\n".join(enum_notes) + "\n"
    return section


_DOCENT_MESSAGE_TYPES = {
    "system": SystemMessage,
    "user": UserMessage,
    "assistant": AssistantMessage,
}


def render_transcript(
    messages: Sequence[Message], role_filter: tuple[str, ...] | None = None
) -> str:
    """Render messages in the Docent text format the judge template expects.

    ``role_filter`` mirrors production's user-rubric judging: pass ``("user",)`` to render only
    the user turns (the user judge never sees the assistant's response).
    """
    kept = [m for m in messages if role_filter is None or m.role in role_filter]
    docent_messages: list[ChatMessage] = [
        _DOCENT_MESSAGE_TYPES[m.role](content=m.content) for m in kept
    ]
    return DocentTranscript(messages=docent_messages).to_text(transcript_alias=0)


def build_judge_prompt(
    rubric_text: str,
    messages: Sequence[Message],
    role_filter: tuple[str, ...] | None = None,
    json_schema: dict[str, Any] = RUBRIC_RESPONSE_SCHEMA,
) -> str:
    return DEFAULT_TRANSCRIPT_JUDGE_TEMPLATE.substitute(
        transcript=render_transcript(messages, role_filter),
        rubric=rubric_text,
        json_schema_section=render_output_spec(json_schema),
    )


_JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)```", re.DOTALL)
_ANY_BLOCK_RE = re.compile(r"```\s*(.*?)```", re.DOTALL)


def parse_json_response(text: str) -> dict[str, Any]:
    """Tolerant parse of the judge's ```json block (mirrors production's repair ladder)."""
    for pattern in (_JSON_BLOCK_RE, _ANY_BLOCK_RE):
        blocks = pattern.findall(text)
        if blocks:
            candidate = blocks[-1].strip()
            break
    else:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace is None:
            raise ValueError(f"No JSON found in judge response: {text!r}")
        candidate = brace.group(0)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        parsed = json.loads(candidate, strict=False)
    if not isinstance(parsed, dict):
        raise ValueError(f"Judge response JSON is not an object: {candidate!r}")
    return cast(dict[str, Any], parsed)


class JudgeError(RuntimeError):
    """A judge call failed after all retries."""


@dataclass(frozen=True)
class JudgeResult:
    match: bool
    explanation: str
    raw_response: str
    prompt: str


class RubricJudge:
    """Async rubric judge against OpenRouter.

    Usage::

        import anyio
        from weirdchat.concurrency import run_bounded

        behavior = wc.behavior("language-switching-english")
        judge = RubricJudge.for_behavior(behavior)
        futures = [judge.judge(messages) for messages in list_of_message_lists]
        results = anyio.run(run_bounded, futures, 256, "judge")
    """

    def __init__(
        self,
        rubric_text: str,
        role_filter: tuple[str, ...] | None = None,
        model: str = JUDGE_MODEL,
        reasoning_effort: str = JUDGE_REASONING_EFFORT,
        max_completion_tokens: int = JUDGE_MAX_COMPLETION_TOKENS,
        temperature: float = JUDGE_TEMPERATURE,
        client: AsyncOpenAI | None = None,
        max_attempts: int = 3,
    ) -> None:
        self.rubric_text = rubric_text
        self.role_filter = role_filter
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.max_attempts = max_attempts
        self.client = client or AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=os.environ["OPENROUTER_API_KEY"],
            timeout=120.0,
        )

    @classmethod
    def for_behavior(
        cls, behavior: Behavior, kind: str = "transcript", **kwargs: Any
    ) -> "RubricJudge":
        """Build the transcript judge (default) or user judge (``kind="user"``) for a behavior.
        The judge configuration is uniform across the dataset — this module's constants are the
        production spec."""
        if kind == "transcript":
            rubric_text, role_filter = behavior.transcript_rubric.text, None
        elif kind == "user":
            rubric_text, role_filter = behavior.user_rubric.text, ("user",)
        else:
            raise ValueError(f"kind must be 'transcript' or 'user', got {kind!r}")
        return cls(rubric_text=rubric_text, role_filter=role_filter, **kwargs)

    async def judge(self, messages: Sequence[Message]) -> JudgeResult:
        prompt = build_judge_prompt(self.rubric_text, messages, self.role_filter)
        last_error: Exception | None = None
        for _ in range(self.max_attempts):
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_completion_tokens=self.max_completion_tokens,
                reasoning_effort=self.reasoning_effort,  # type: ignore[arg-type]
            )
            raw = completion.choices[0].message.content or ""
            try:
                parsed = parse_json_response(raw)
                return JudgeResult(
                    match=bool(parsed["match"]),
                    explanation=str(parsed.get("explanation", "")),
                    raw_response=raw,
                    prompt=prompt,
                )
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                last_error = e
        raise JudgeError(f"Judge failed after {self.max_attempts} attempts: {last_error}")
