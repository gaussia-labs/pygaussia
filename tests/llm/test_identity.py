"""Which model answered, recorded as something a report can attribute a result to."""

from __future__ import annotations

from gaussia.llm.identity import model_identity


class WithModelName:
    """A LangChain chat model: the identifier lives on ``model_name``."""

    model_name = "google/gemma-4-31B-it:cerebras"
    model = "ignored-because-model_name-wins"


class WithModel:
    """An adapter that names the field after the constructor argument instead."""

    model = "llama-3.3-70b-versatile"


class WithNeither:
    pass


class WithNonStringModel:
    """``model`` holding something that is not an identifier."""

    model = object()


class TestModelIdentity:
    def test_prefers_the_model_s_own_identifier(self) -> None:
        assert model_identity(WithModelName()) == "google/gemma-4-31B-it:cerebras"

    def test_reads_model_when_model_name_is_absent(self) -> None:
        assert model_identity(WithModel()) == "llama-3.3-70b-versatile"

    def test_falls_back_to_the_adapter_class_when_the_model_names_itself_nowhere(self) -> None:
        assert model_identity(WithNeither()) == "WithNeither"

    def test_falls_back_when_the_field_is_not_a_string(self) -> None:
        """A non-string is a field that happens to share the name, not an identifier."""
        assert model_identity(WithNonStringModel()) == "WithNonStringModel"

    def test_the_adapter_class_alone_does_not_separate_two_providers(self) -> None:
        """Why the fallback is a last resort and not the answer.

        Two different models reached through one OpenAI-compatible adapter are indistinguishable by
        class, so recording the class records nothing a report can attribute a result to — while
        looking exactly like a run that recorded something.
        """

        class ChatOpenAI:
            def __init__(self, model_name: str) -> None:
                self.model_name = model_name

        one, other = ChatOpenAI("gpt-oss-120b"), ChatOpenAI("gemma-4-31B-it")
        assert type(one).__name__ == type(other).__name__
        assert model_identity(one) != model_identity(other)
