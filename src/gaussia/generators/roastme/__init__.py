"""Roast Me — profile-then-exploit adversarial evaluation.

A generator subsystem rather than a metric (spec D8, D15): the Probe Library, the Profiler
and the Exploiter produce the Roast Dataset, and that dataset is what enters the metric
pipeline afterwards.

This package deliberately re-exports nothing heavy. ``gaussia.generators.__init__`` imports
eagerly, so anything registered here would make ``import gaussia.generators`` require the
``roastme`` extra (FR-037). Import the components from their own modules.
"""
