"""Centralized presets for unsupervised frame deduplication.

This module defines preset configurations for the unsupervised deduplication
algorithm, offering different trade-offs between preservation and compression.
"""

from __future__ import annotations

from typing import Dict, TypedDict


class UnsupervisedPreset(TypedDict):
    """Type definition for unsupervised deduplication preset parameters."""

    hash_threshold: int
    ordinal_footrule_threshold: float
    feature_similarity: float
    flow_static_threshold: float
    flow_low_ratio: float
    pan_orientation_std: float
    safety_keep_seconds: float


# Preset configurations for unsupervised frame deduplication
UNSUPERVISED_PRESETS: Dict[str, UnsupervisedPreset] = {
    "gentle": {
        "hash_threshold": 6,
        "ordinal_footrule_threshold": 220.0,
        "feature_similarity": 0.30,
        "flow_static_threshold": 0.08,
        "flow_low_ratio": 0.98,
        "pan_orientation_std": 0.60,
        "safety_keep_seconds": 1.0,
    },
    "balanced": {
        "hash_threshold": 8,
        "ordinal_footrule_threshold": 260.0,
        "feature_similarity": 0.26,
        "flow_static_threshold": 0.09,
        "flow_low_ratio": 0.97,
        "pan_orientation_std": 0.65,
        "safety_keep_seconds": 1.5,
    },
    "aggressive": {
        "hash_threshold": 12,
        "ordinal_footrule_threshold": 320.0,
        "feature_similarity": 0.22,
        "flow_static_threshold": 0.12,
        "flow_low_ratio": 0.94,
        "pan_orientation_std": 0.80,
        "safety_keep_seconds": 2.5,
    },
}


def get_preset(name: str) -> UnsupervisedPreset:
    """Get a preset configuration by name.

    Args:
        name: Preset name ('gentle', 'balanced', or 'aggressive')

    Returns:
        Dictionary containing preset parameters

    Raises:
        KeyError: If preset name is not found
    """
    if name not in UNSUPERVISED_PRESETS:
        available = ", ".join(UNSUPERVISED_PRESETS.keys())
        raise KeyError(f"Unknown preset '{name}'. Available presets: {available}")

    return UNSUPERVISED_PRESETS[name].copy()


def list_presets() -> list[str]:
    """Get list of available preset names.

    Returns:
        List of preset names
    """
    return list(UNSUPERVISED_PRESETS.keys())
