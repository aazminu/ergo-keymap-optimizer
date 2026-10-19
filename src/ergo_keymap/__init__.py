"""Corpus-driven ergonomic keymap optimizer for split columnar mechanical keyboards."""

from .optimizer import (
    CorpusProfile,
    Finger,
    Hand,
    KeyboardGeometry,
    KeyCoord,
    KeymapLayout,
    LayoutScorer,
    OptimizationResult,
    ScoreBreakdown,
    ScoringWeights,
    SimulatedAnnealingOptimizer,
    export_qmk_keymap,
    export_zmk_keymap,
)

__all__ = [
    "CorpusProfile",
    "Finger",
    "Hand",
    "KeyboardGeometry",
    "KeyCoord",
    "KeymapLayout",
    "LayoutScorer",
    "OptimizationResult",
    "ScoreBreakdown",
    "ScoringWeights",
    "SimulatedAnnealingOptimizer",
    "export_qmk_keymap",
    "export_zmk_keymap",
]

__version__ = "0.1.0"
