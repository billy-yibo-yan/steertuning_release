"""Language annotation with Qwen3-VL-8B and instruction paraphrases."""

from __future__ import annotations

import json

import torch
from PIL import Image

INSTRUCTION_PROMPT = """You are given a short video clip.
Answer the question: What is the main action that occurs across the entire clip?

Infer ONE action that occurs across the entire clip.

Rules:
- Do NOT describe individual frames
- Do NOT mention hands, arms or people
- Do NOT list multiple actions
- Focus on motion and what changes over time

Output:
- One short action (max 8 words)
- Lead with an -ing verb and keep the phrase verb-first (e.g., "placing bread in toaster")
- No explanation
- Do not use complex vocabulary, to be straightforward and easy to understand

Answer:

These images are ordered in time from earliest to latest."""

PARAPHRASE_PROMPT = """You paraphrase short manipulation instructions. Each paraphrase must describe the EXACT same physical task as the original: the same action, the same object(s), and the same source or destination. You may only change surface wording: verb synonyms, word order and articles.

Rules:
1. Keep the object words exactly as written in the original. Do not add, remove or change any word attached to the object.
2. Keep the same physical motion, at the same force. Use only plain, everyday verbs that a person would accept as meaning the same thing.
3. Keep the object as the thing being moved or acted on. Do not swap the roles of the words.
4. Do not introduce new steps, tools, objects, locations, colors, materials, sizes or quantities.
5. Use the short verb-object-preposition style: lead with an -ing verb, at most 8 words (e.g., "placing bread in toaster").

Original instruction: "{instruction}"

Write {count} distinct paraphrases. Do not repeat the original or each other. If fewer good paraphrases exist, return fewer. Return only a JSON array of strings."""


def sample_frames(start: int, end: int, max_frames: int) -> list[int]:
    """Up to `max_frames` frames evenly spaced over an inclusive interval, both ends included."""
    length = end - start + 1
    count = min(max_frames, length)
    if count == 1:
        return [start]
    return [start + round(i * (length - 1) / (count - 1)) for i in range(count)]


def _is_verb_first(text: str) -> bool:
    words = text.split()
    return bool(words) and words[0].lower().endswith("ing")


def clean_instruction(reply: str) -> str:
    """A model reply without surrounding quotes or trailing punctuation."""
    return reply.strip().lstrip("\"'`").rstrip(" \"'`.,;:!?")


def parse_paraphrases(reply: str, original: str, count: int) -> list[str]:
    """Distinct paraphrases in the verb-first style from a model reply, without the original."""
    try:
        candidates = json.loads(reply[reply.find("[") : reply.rfind("]") + 1])
    except ValueError:
        candidates = reply.splitlines()
    seen = {original.lower().strip()}
    paraphrases = []
    for candidate in candidates:
        text = str(candidate).strip(" \t-*\"',.")
        if _is_verb_first(text) and len(text.split()) <= 8 and text.lower() not in seen:
            seen.add(text.lower())
            paraphrases.append(text)
    return paraphrases[:count]


class QwenVL:
    """Qwen3-VL-8B in 4-bit, for snippet instructions and text-only paraphrases."""

    def __init__(self, model: str, device: str) -> None:
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model, size={"shortest_edge": 65536, "longest_edge": 200704})
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model,
            device_map={"": device},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
        )

    def _generate(self, content: list[dict]) -> str:
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=512)
        return self.processor.tokenizer.decode(
            generated[0, inputs["input_ids"].size(1) :], skip_special_tokens=True
        ).strip()

    def describe(self, images: list[Image.Image]) -> str:
        """A short, verb-first instruction for time-ordered frames of one snippet."""
        content = [{"type": "text", "text": INSTRUCTION_PROMPT}]
        content += [{"type": "image", "image": image} for image in images]
        instruction = clean_instruction(self._generate(content))
        if not instruction:
            raise RuntimeError("Qwen3-VL returned an empty instruction")
        return instruction

    def paraphrase(self, instruction: str, count: int) -> list[str]:
        """Paraphrases that keep the action and match the short verb-object-preposition style, perhaps none."""
        prompt = PARAPHRASE_PROMPT.format(instruction=instruction.lower(), count=count)
        return parse_paraphrases(self._generate([{"type": "text", "text": prompt}]), instruction, count)


def paraphrase_table(language: QwenVL, instructions: list[str], description: str, count: int) -> dict:
    """The contents of `text_prompt_augmentations.json` for a task demonstration.

    Each snippet instruction draws from itself, its paraphrases, the typed task description and that
    description's paraphrases, so fine-tuning trains on the prompt used at deployment.
    """
    task_wording = [description, *language.paraphrase(description, count)]
    by_original = {description: task_wording}
    for instruction in dict.fromkeys(instructions):
        if instruction != description:
            by_original[instruction] = [instruction, *language.paraphrase(instruction, count), *task_wording]
    return {"task_description": description, "by_original": by_original}
