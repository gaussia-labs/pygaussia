"""Guardian abstract base class for bias detection."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pygaussia.utils.logging import VerboseLogger

if TYPE_CHECKING:
    from pygaussia.schemas.bias import GuardianBias, ProtectedAttribute


class Guardian(ABC):
    """
    An abstract base class that serves as a framework for implementing bias detection in LLM interactions.

    This class is designed to be implemented by different bias detection mechanisms that can analyze
    question-answer pairs for potential biases towards specific attributes. It provides a standardized
    interface for bias detection across different implementations.

    The class uses a nested BiasInfer model to structure the bias detection results in a consistent format.
    """

    def __init__(self, **kwargs):
        """
        Initialize the Guardian with a VerboseLogger for detailed logging of bias detection operations.
        """
        self.logger = VerboseLogger()

    @abstractmethod
    def is_biased(
        self, question: str, answer: str, attribute: "ProtectedAttribute", context: str | None = None
    ) -> "GuardianBias":
        """
        Analyze a question-answer interaction for potential bias towards specific attributes.

        This abstract method must be implemented by concrete Guardian classes to define their
        specific bias detection logic. The implementation should analyze the given question and
        answer pair for potential bias towards the specified attributes.

        Args:
            question (str): The question being analyzed
            answer (str): The answer being analyzed
            attribute (ProtectedAttribute): The attribute to check for bias
            context (Optional[str]): Additional context that might be relevant for bias detection

        Returns:
            GuardianBias: A GuardianBias object containing:
                - is_biased: Whether bias was detected
                - attribute: The specific attribute that showed bias
                - certainty: Optional confidence score for the detection

        Raises:
            NotImplementedError: If the concrete class does not implement this method
        """
        raise NotImplementedError("You should implement this method.")
