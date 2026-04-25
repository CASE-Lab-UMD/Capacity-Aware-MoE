import math
import types
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F


def _compute_capacity(
    total_tokens: int,
    num_experts: int,
    top_k: int,
    capacity_factor: Optional[float],
) -> Optional[int]:
    if capacity_factor is None:
        return None
    if total_tokens <= 1:
        return None
    if num_experts <= 0:
        return None
    cap = int(math.ceil(float(capacity_factor) * top_k * (total_tokens / num_experts)))
    return max(cap, 1)


def _compute_indices(scores_sub: torch.Tensor, mask_sub: torch.Tensor, expert_capacity: int) -> torch.Tensor:
    masked_scores = scores_sub.masked_fill(~mask_sub, float("-inf"))
    _, capacity_indices = torch.topk(masked_scores, k=expert_capacity, dim=0, sorted=False)
    return capacity_indices


def _token_drop_by_score(
    scores: torch.Tensor,
    expert_capacity: int,
    top_k: int,
    strategy: str,
    rounds: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    t, k = scores.shape
    top_k_all = top_k
    if "overselect" in strategy:
        top_k_all = min(k, max(top_k, int(math.ceil(top_k * max(rounds, 1)))))

    topk_weight, topk_idx = torch.topk(scores, k=top_k_all, dim=-1, sorted=False)
    mask_buffer = torch.zeros((t, k), dtype=torch.bool, device=scores.device)
    mask_buffer.scatter_(-1, topk_idx, True)

    current_usage = mask_buffer.sum(dim=0)
    cols = (current_usage > expert_capacity).nonzero(as_tuple=True)[0]
    if cols.numel() > 0:
        scores_sub = scores[:, cols]
        mask_sub = mask_buffer[:, cols]
        capacity_indices = _compute_indices(scores_sub, mask_sub, expert_capacity)
        mask_buffer[:, cols] = torch.zeros_like(mask_sub).scatter(0, capacity_indices, True)

    topk_weight = scores.gather(-1, topk_idx)
    top_mask = mask_buffer.gather(-1, topk_idx)
    topk_weight = topk_weight * top_mask
    topk_idx = topk_idx.masked_fill(~top_mask, k)
    return topk_weight, topk_idx


def _token_drop_random(scores: torch.Tensor, expert_capacity: int, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    t, k = scores.shape
    topk_weight, topk_idx = torch.topk(scores, top_k, dim=-1, sorted=False)
    mask_buffer = torch.zeros((t, k), dtype=torch.bool, device=scores.device)
    mask_buffer.scatter_(-1, topk_idx, True)

    current_usage = mask_buffer.sum(dim=0)
    cols = (current_usage > expert_capacity).nonzero(as_tuple=True)[0]
    if cols.numel() > 0:
        scores_sub = torch.rand_like(scores[:, cols])
        mask_sub = mask_buffer[:, cols]
        capacity_indices = _compute_indices(scores_sub, mask_sub, expert_capacity)
        mask_buffer[:, cols] = torch.zeros_like(mask_sub).scatter(0, capacity_indices, True)

    topk_weight = scores.gather(-1, topk_idx)
    top_mask = mask_buffer.gather(-1, topk_idx)
    topk_weight = topk_weight * top_mask
    topk_idx = topk_idx.masked_fill(~top_mask, k)
    return topk_weight, topk_idx


def _token_drop_sequential(
    scores: torch.Tensor,
    expert_capacity: int,
    topk_idx: torch.Tensor,
    mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    t, k = scores.shape
    mask_buffer = torch.zeros((t, k), dtype=torch.bool, device=scores.device)
    mask_buffer.scatter_(-1, topk_idx, True)

    current_usage = mask_buffer.sum(dim=0)
    cols = (current_usage > expert_capacity).nonzero(as_tuple=True)[0]
    if cols.numel() > 0:
        mask_sub = mask_buffer[:, cols]
        if mode == "last":
            scores_sub = torch.cumsum(mask_sub, dim=-2)
        elif mode == "first":
            reversed_mask = torch.flip(mask_sub, dims=[-2])
            reversed_cumsum = torch.cumsum(reversed_mask, dim=-2)
            scores_sub = torch.flip(reversed_cumsum, dims=[-2])
        else:
            raise ValueError("Unsupported mode: should be 'first' or 'last'")

        capacity_indices = _compute_indices(scores_sub.float(), mask_sub, expert_capacity)
        mask_buffer[:, cols] = torch.zeros_like(mask_sub).scatter(0, capacity_indices, True)

    topk_weight = scores.gather(-1, topk_idx)
    top_mask = mask_buffer.gather(-1, topk_idx)
    topk_weight = topk_weight * top_mask
    topk_idx = topk_idx.masked_fill(~top_mask, k)
    return topk_weight, topk_idx


def _select_with_capacity(
    scores: torch.Tensor,
    top_k: int,
    expert_capacity: Optional[int],
    strategy: str,
    rounds: int,
) -> torch.Tensor:
    if expert_capacity is None:
        return torch.topk(scores, k=top_k, dim=-1, sorted=False).indices

    strategy = strategy or ""
    strategy_list = ["score", "last", "first", "random", "overselect"]
    if not any(s in strategy for s in strategy_list):
        return torch.topk(scores, k=top_k, dim=-1, sorted=False).indices

    if "random" in strategy:
        _, topk_idx = _token_drop_random(scores, expert_capacity, top_k)
    else:
        _, topk_idx = _token_drop_by_score(scores, expert_capacity, top_k, strategy, rounds)

    if "first" in strategy:
        _, topk_idx = _token_drop_sequential(scores, expert_capacity, topk_idx, mode="first")
    elif "last" in strategy:
        _, topk_idx = _token_drop_sequential(scores, expert_capacity, topk_idx, mode="last")

    num_experts = scores.shape[-1]
    dropped = topk_idx == num_experts
    if dropped.any():
        fallback = torch.topk(scores, k=top_k, dim=-1, sorted=False).indices
        topk_idx = torch.where(dropped, fallback, topk_idx)
    return topk_idx


def _patch_single_gate(moe_module, capacity_factor: float, strategy: str, rounds: int) -> bool:
    gate = _find_gate_module(moe_module)
    top_k = _find_top_k(moe_module)
    if gate is None or not hasattr(gate, "forward") or top_k <= 0:
        return False

    if getattr(gate, "_capacity_patch_applied", False):
        return False

    original_forward = gate.forward

    def patched_forward(self, hidden_states, *args, **kwargs):
        logits = original_forward(hidden_states, *args, **kwargs)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
            # DeepSeek-like gates return tuple(topk_idx, topk_weight, aux_loss) instead of logits.
            if (
                isinstance(logits, tuple)
                and len(logits) >= 2
                and isinstance(logits[0], torch.Tensor)
                and isinstance(logits[1], torch.Tensor)
                and logits[0].ndim == 2
                and logits[1].ndim == 2
                and hasattr(self, "weight")
            ):
                # hidden_states can be [B, S, H] for DeepSeek gates.
                hs = hidden_states
                if hs.ndim == 3:
                    hs = hs.view(-1, hs.shape[-1])
                if hs.ndim != 2:
                    return logits

                raw_logits = F.linear(hs, self.weight, None)
                num_experts = raw_logits.shape[-1]
                expert_capacity = _compute_capacity(
                    total_tokens=raw_logits.shape[0],
                    num_experts=num_experts,
                    top_k=top_k,
                    capacity_factor=capacity_factor,
                )
                if expert_capacity is None:
                    return logits

                scores = F.softmax(raw_logits, dim=-1, dtype=torch.float)
                selected_idx = _select_with_capacity(
                    scores=scores,
                    top_k=min(top_k, num_experts),
                    expert_capacity=expert_capacity,
                    strategy=strategy,
                    rounds=rounds,
                )
                selected_w = scores.gather(-1, selected_idx)

                norm_topk_prob = bool(getattr(self, "norm_topk_prob", False))
                if top_k > 1 and norm_topk_prob:
                    selected_w = selected_w / selected_w.sum(dim=-1, keepdim=True).clamp_min(1e-20)

                selected_w = selected_w.to(logits[1].dtype)
                tail = logits[2:] if len(logits) > 2 else ()
                return (selected_idx, selected_w, *tail)
            return logits

        num_experts = logits.shape[-1]
        expert_capacity = _compute_capacity(
            total_tokens=logits.shape[0],
            num_experts=num_experts,
            top_k=top_k,
            capacity_factor=capacity_factor,
        )
        if expert_capacity is None:
            return logits

        scores = F.softmax(logits, dim=-1, dtype=torch.float)
        selected_idx = _select_with_capacity(
            scores=scores,
            top_k=min(top_k, num_experts),
            expert_capacity=expert_capacity,
            strategy=strategy,
            rounds=rounds,
        )

        min_val = torch.finfo(logits.dtype).min
        masked_logits = torch.full_like(logits, min_val)
        selected_logits = logits.gather(-1, selected_idx)
        masked_logits.scatter_(-1, selected_idx, selected_logits)
        return masked_logits

    gate.forward = types.MethodType(patched_forward, gate)
    gate._capacity_patch_applied = True
    gate._capacity_patch_meta = {
        "capacity_factor": capacity_factor,
        "strategy": strategy,
        "rounds": rounds,
    }
    return True


def _first_positive_int(values: List[Optional[int]]) -> int:
    for v in values:
        if isinstance(v, int) and v > 0:
            return v
    return 0


def _find_top_k(moe_module) -> int:
    # Common names across HF MoE variants.
    return _first_positive_int(
        [
            getattr(moe_module, "top_k", None),
            getattr(moe_module, "topk", None),
            getattr(moe_module, "num_experts_per_tok", None),
            getattr(moe_module, "num_selected_experts", None),
        ]
    )


def _find_gate_module(moe_module):
    # Most common router names.
    for name in ["gate", "router", "router_gate", "gating", "gating_network"]:
        mod = getattr(moe_module, name, None)
        if mod is not None and hasattr(mod, "forward"):
            return mod
    return None


def _is_moe_like_module(module) -> bool:
    # MoE structural hints; used to avoid patching unrelated layers.
    if not any(hasattr(module, n) for n in ["experts", "expert", "moe_layer"]):
        return False
    if _find_gate_module(module) is None:
        return False
    if _find_top_k(module) <= 0:
        return False
    return True


def apply_capacity_aware_moe_patch(model, config) -> int:
    capacity_factor = getattr(config, "expert_capacity", None)
    if capacity_factor is None:
        return 0

    strategy = str(getattr(config, "strategy", "score") or "score")
    rounds = int(getattr(config, "rounds", 1) or 1)

    patched = 0
    for module in model.modules():
        if _is_moe_like_module(module):
            if _patch_single_gate(module, float(capacity_factor), strategy, rounds):
                patched += 1
    return patched
