# Capacity-Aware Core

This directory contains the primary, full implementation of the generic capacity-aware MoE routing patch.

Core file:
- `capacity_patch.py`

Runtime integration in `lm_eval` currently uses:
- `lm-evaluation-harness/lm_eval/capacity_aware/capacity_patch.py`

When changing patch behavior, keep both copies consistent.
