"""Base metric schemas for Gaussia."""

from pydantic import BaseModel


class BaseMetric(BaseModel):
    """
    Base class for all metric results.

    All metrics should include at minimum the session and assistant identifiers.
    """

    session_id: str
    assistant_id: str
