"""Transform abstract base class: a real entity to the premise a probe leans on."""

from abc import ABC, abstractmethod


class Transform(ABC):
    """Turns a real entity into the premise a probe leans on.

    The transformation set is deliberately closed to the four the catalogue accepts
    (FR-025), because ``transform`` is the one field whose value changes what a probe
    *means*: an unrecognised transformation would produce probes whose ``doc`` label
    nobody can trust. A fifth transformation therefore requires relaxing FR-025 as well as
    adding an implementation.

    The catalogue names a transformation with a string, because configuration is data. The
    registry resolves that string to an implementation **once**, during catalogue
    validation; after that point behaviour is polymorphic and nothing branches on it. The
    string survives only on ``KnowledgeHook.how``, as a record rather than a dispatch.
    """

    @property
    @abstractmethod
    def key(self) -> str:
        """The catalogue string this transformation answers to.

        Declared here so the registry is built from the implementations rather than
        restating their names, and so ``KnowledgeHook.how`` carries the same string the
        user wrote in ``StrategySpec.transform``.
        """

    @abstractmethod
    def apply(self, entity: str) -> str:
        """Derive the probe's premise from a real entity.

        Args:
            entity: The real entity the engine extracted, in whatever form this
                transformation needs to operate on it. It becomes
                ``KnowledgeHook.base_entity`` when the result differs from it.

        Returns:
            The premise the probe leans on, which becomes ``KnowledgeHook.references``.
            Whether that premise is documented or invented is the engine's call, derived
            from its own knowledge of the base's boundary (FR-021) — this method decides
            the text, never the label.
        """
