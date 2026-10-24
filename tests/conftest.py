"""Shared test configuration.

Hypothesis is derandomized so that every test run explores the same examples —
results must be reproducible (CLAUDE.md: deterministic everywhere).
"""

from hypothesis import settings

settings.register_profile("default", derandomize=True, deadline=None)
settings.load_profile("default")
