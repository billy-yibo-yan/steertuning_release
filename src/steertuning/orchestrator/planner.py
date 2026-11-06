"""Decompose a language command with Gemini ER 1.6."""

from __future__ import annotations

import io
from collections.abc import Mapping
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from steertuning.remote import gemini_client

PROMPT = """You plan tasks for a mobile manipulation robot. The image is the robot's current view.

Skills:
- navigate: drive to an object. The target is one short object name for visual search, such as
  "blue stapler" or "desk lamp". No verbs and no location phrases.
- manipulate: run one trained task at the robot's current location. The target is a task name
  copied exactly from this list:
{catalog}

Split the command into the fewest ordered steps. Navigate to an object before manipulating it.
Use only these two skills. Return only the structured plan.

Command: {command}
"""


class Step(BaseModel):
    skill: Literal["navigate", "manipulate"]
    target: str = Field(min_length=1)


class Plan(BaseModel):
    steps: list[Step] = Field(min_length=1)


def build_prompt(command: str, tasks: Mapping[str, str]) -> str:
    catalog = "\n".join(f'  * "{name}": {description}' for name, description in tasks.items())
    return PROMPT.format(catalog=catalog, command=command)


def parse_plan(text: str, tasks: Mapping[str, str]) -> Plan:
    """The plan in Gemini's JSON response. Every manipulate step must name a task in the catalog."""
    plan = Plan.model_validate_json(text)
    unknown = [step.target for step in plan.steps if step.skill == "manipulate" and step.target not in tasks]
    if unknown:
        raise ValueError(f"Plan uses tasks outside the catalog: {unknown}")
    return plan


def plan_command(
    command: str,
    image: np.ndarray,
    tasks: Mapping[str, str],
    model: str,
    temperature: float,
    thinking_budget: int,
    client=None,
) -> Plan:
    """Decompose `command` given one RGB image and a catalog of trained tasks {name: description}."""
    from google.genai import types

    client = client or gemini_client()
    jpeg = io.BytesIO()
    Image.fromarray(image).save(jpeg, format="JPEG")
    response = client.models.generate_content(
        model=model,
        contents=[
            build_prompt(command, tasks),
            types.Part.from_bytes(data=jpeg.getvalue(), mime_type="image/jpeg"),
        ],
        config=types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=Plan,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        ),
    )
    return parse_plan(response.text, tasks)
