"""Humanity metric schemas."""

from .metrics import BaseMetric


class HumanityMetric(BaseMetric):
    """
    Humanity metric for evaluating emotional human-likeness.
    """

    humanity_assistant_emotional_entropy: float
    humanity_ground_truth_spearman: float
    humanity_assistant_anger: float
    humanity_assistant_anticipation: float
    humanity_assistant_disgust: float
    humanity_assistant_fear: float
    humanity_assistant_joy: float
    humanity_assistant_sadness: float
    humanity_assistant_surprise: float
    humanity_assistant_trust: float
    qa_id: str
