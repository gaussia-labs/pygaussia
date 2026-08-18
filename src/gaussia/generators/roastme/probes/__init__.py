"""Probe generation: the catalogue, the transformations, the engines and the library.

Each engine is imported directly from its own module, so installing only what an engine needs is
enough (FR-037). Re-exporting one here would pull an embedder and a graph library into
``from gaussia.generators.roastme import ProbeLibrary``, which is exactly the import SC-011
requires keep working with the ``roastme`` extra uninstalled:

    from gaussia.generators.roastme.probes.retrieval import RetrievalProbeEngine   # gaussia[roastme]
    from gaussia.generators.roastme.probes.graph import GraphProbeEngine           # gaussia[roastme]
    from gaussia.generators.roastme.probes.grag import MultiHopProbeEngine         # gaussia[roastme]
    from gaussia.generators.roastme.probes.enumeration import EnumerationProbeEngine

The enumeration engine needs no extra — only an ``EntityEnumerator`` the user writes (spec D14).
The grounded engine needs none either: LangChain is a base dependency, so a ``FactTwister`` driven by
the user's model pulls nothing of the extra and ``GroundedProbeEngine`` sits on the subsystem facade
beside the enumeration one (FR-042, FR-037).
"""
