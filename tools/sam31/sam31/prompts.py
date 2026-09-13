"""Prompt parsing: multiple text concepts per run.

A run takes a list of text prompts (one COCO category each).  Prompts can be
given in three ways (merged and de-duplicated):

- repeatable ``--prompt`` flags,
- a ``--prompts-file`` with one prompt per line (``#`` comments allowed),
- several prompts inside a single string separated by ``|`` or ``;``,
  e.g. ``--prompt "car | bus | truck"``.

Commas are NOT separators — many concepts naturally contain commas.
"""

from __future__ import annotations

import re
from pathlib import Path

from .config import MAX_PROMPT_CHARS, MAX_PROMPTS, PROMPT_SEPARATORS

_SPLIT_RE = re.compile("[" + re.escape("".join(PROMPT_SEPARATORS)) + "]")


def split_prompt(text: str) -> list[str]:
    """Split a single string into prompts on ``|`` or ``;``."""
    return [part.strip() for part in _SPLIT_RE.split(text)]


def parse_prompts(
    prompt_args: list[str] | None,
    prompts_file: str | None = None,
) -> list[str]:
    """Merge all prompt sources into an ordered, de-duplicated list."""
    raw: list[str] = []
    for p in prompt_args or []:
        if not isinstance(p, str) or not p.strip():
            continue
        raw.extend(split_prompt(p))

    if prompts_file:
        path = Path(prompts_file)
        if not path.is_file():
            raise FileNotFoundError(f"Prompts file not found: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            raw.extend(split_prompt(line))

    prompts: list[str] = []
    seen: set[str] = set()
    for p in raw:
        if not p or p in seen:
            continue
        if len(p) > MAX_PROMPT_CHARS:
            raise ValueError(
                f"Prompt too long ({len(p)} > {MAX_PROMPT_CHARS} chars): {p[:80]!r}"
            )
        seen.add(p)
        prompts.append(p)

    if not prompts:
        raise ValueError("No text prompts given (use --prompt and/or --prompts-file).")
    if len(prompts) > MAX_PROMPTS:
        raise ValueError(f"Too many prompts: {len(prompts)} (max {MAX_PROMPTS}).")
    return prompts
