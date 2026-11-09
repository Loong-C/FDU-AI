"""Extensible Jev integration for optional search experiments."""

from .core import (
    JevHeuristicBase,
    OpenRouterJevClient,
    TransientJevError,
    TypeSafeJevClient,
)
from .q2 import (
    Q2JevDirectValueHeuristic,
    Q2JevNodeHeuristic,
    Q2JevPreScoreTieBreaker,
    Q2JevTieBreaker,
)
from .q6 import (
    Q6JevNodeHeuristic,
    Q6JevScoreHeuristic,
)

def __getattr__(name):
    # Keep package-level Q9 imports compatible without preloading python -m jev.q9.
    if name in {"Q9JevEvaluator", "Q9_JEV_CACHE_ENABLED", "Q9_JEV_ENABLED",
                "Q9_JEV_MODE", "Q9_JEV_WEIGHT"}:
        from . import q9
        return getattr(q9, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "JevHeuristicBase",
    "OpenRouterJevClient",
    "Q2JevDirectValueHeuristic",
    "Q2JevNodeHeuristic",
    "Q2JevPreScoreTieBreaker",
    "Q2JevTieBreaker",
    "Q6JevNodeHeuristic",
    "Q6JevScoreHeuristic",
    "Q9JevEvaluator",
    "Q9_JEV_CACHE_ENABLED",
    "Q9_JEV_ENABLED",
    "Q9_JEV_MODE",
    "Q9_JEV_WEIGHT",
    "TransientJevError",
    "TypeSafeJevClient",
]
