"""Prompt templates used by the MIPROv2 optimizer for instruction proposal."""

INSTRUCTION_PROPOSAL_SYSTEM = """\
You are an expert prompt engineer. Your task is to generate diverse system prompt variants from a seed prompt.
Each variant must preserve the original objective but emphasize a different aspect of good behavior.\
"""

INSTRUCTION_PROPOSAL_USER = """\
Seed prompt: {seed_prompt}
Objective: {objective}

Generate exactly {n} system prompt variants. Each variant must emphasize the corresponding focus area:
{tips}

Rules:
- Each variant must be a complete, standalone system prompt.
- Do not include the focus area label or number in the variant itself.
- Vary the phrasing and emphasis across variants — avoid generating near-identical prompts.\
"""
