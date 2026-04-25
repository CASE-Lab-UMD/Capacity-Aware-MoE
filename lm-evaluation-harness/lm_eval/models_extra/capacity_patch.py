"""Backward-compatible import path for capacity-aware MoE patching."""

from lm_eval.capacity_aware.capacity_patch import apply_capacity_aware_moe_patch

__all__ = ["apply_capacity_aware_moe_patch"]
