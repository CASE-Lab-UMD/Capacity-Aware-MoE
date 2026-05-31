# Capacity-Aware Core

This directory contains the primary implementation of the generic capacity-aware MoE routing patch.

Core file:
- `capacity_patch.py`

Runtime integration in `lm_eval` uses a thin compatibility wrapper:
- `lm-evaluation-harness/lm_eval/capacity_aware/capacity_patch.py`

When changing patch behavior, edit only the top-level `capacity_aware/capacity_patch.py`.
