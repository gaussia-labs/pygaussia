"""Prompt templates used by the GEPA optimizer for candidate generation."""

GENERATION_SYSTEM_PROMPT = """\
You are an expert prompt engineer specializing in optimizing system prompts for AI assistants.
Your task is to analyze failing examples and generate improved system prompts that fix the failures \
while preserving the original intent.\
"""

GENERATION_USER_PROMPT = """\
Objective: {objective}

Current system prompt:
---
{current_prompt}
---

The following examples show where the current prompt produces unsatisfactory results:

{failing_examples}

Generate exactly {n} improved system prompt(s). Each must be a complete, standalone system prompt — \
not a comment, not a diff, not an explanation. Address the failures directly.\
"""
