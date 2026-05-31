import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


_PATCH_PATH = Path(__file__).resolve().parents[1] / "lm_eval" / "capacity_aware" / "capacity_patch.py"
_SPEC = importlib.util.spec_from_file_location("capacity_patch_under_test", _PATCH_PATH)
capacity_patch = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(capacity_patch)

_select_with_capacity = capacity_patch._select_with_capacity
_token_drop_by_device = capacity_patch._token_drop_by_device
apply_capacity_aware_moe_patch = capacity_patch.apply_capacity_aware_moe_patch


def test_lm_eval_wrapper_uses_repository_source_of_truth():
    assert "Capacity-Aware-MoE/capacity_aware/capacity_patch.py" in str(capacity_patch._ROOT_PATCH)


def test_score_capacity_selection_uses_sentinel_for_dropped_slots():
    scores = torch.tensor(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.1, 0.7],
        ]
    )

    selected = _select_with_capacity(
        scores=scores,
        top_k=1,
        expert_capacity=1,
        strategy="score",
        rounds=1,
    )

    assert selected.shape == (3, 1)
    assert selected.tolist() == [[0], [2], [1]]


def test_overselect_strategy_returns_requested_top_k_width():
    scores = torch.tensor(
        [
            [0.9, 0.8, 0.1],
            [0.7, 0.6, 0.2],
            [0.1, 0.2, 0.9],
        ]
    )

    selected = _select_with_capacity(
        scores=scores,
        top_k=1,
        expert_capacity=1,
        strategy="overselect",
        rounds=3,
    )

    assert selected.shape == (3, 1)


def test_device_capacity_matches_loop_reference_without_scalar_usage_sync():
    scores = torch.tensor(
        [
            [0.9, 0.8, 0.4, 0.3],
            [0.7, 0.6, 0.5, 0.2],
            [0.1, 0.2, 0.95, 0.85],
            [0.2, 0.1, 0.75, 0.65],
        ]
    )
    topk_weight, topk_idx = torch.topk(scores, k=2, dim=-1, sorted=False)
    expert_device = torch.tensor([0, 0, 1, 1])
    expected_mask = torch.zeros_like(scores, dtype=torch.bool)
    topk_mask = torch.zeros_like(scores, dtype=torch.bool)
    topk_mask.scatter_(1, topk_idx, True)
    for device_id in range(2):
        expert_mask = expert_device == device_id
        candidate_scores = scores[:, expert_mask].masked_fill(~topk_mask[:, expert_mask], float("-inf"))
        flat_scores = candidate_scores.reshape(-1)
        _, flat_idx = torch.topk(flat_scores, k=2, dim=0, sorted=False)
        keep_flat = torch.zeros_like(flat_scores, dtype=torch.bool)
        keep_flat.scatter_(0, flat_idx, True)
        expected_mask[:, expert_mask] = keep_flat.view_as(candidate_scores) & torch.isfinite(candidate_scores)

    weight, idx = _token_drop_by_device(
        scores=scores,
        device_capacity=2,
        top_k=2,
        strategy="score",
        rounds=1,
        num_devices=2,
    )
    actual_mask = torch.zeros_like(scores, dtype=torch.bool)
    valid = idx != scores.shape[-1]
    actual_mask.scatter_(1, idx.masked_fill(~valid, 0), valid)

    assert torch.equal(actual_mask, expected_mask)
    assert torch.equal(weight, scores.gather(1, idx.masked_fill(~valid, 0)).masked_fill(~valid, 0.0))


class TupleGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(2))
        self.norm_topk_prob = False

    def forward(self, hidden_states):
        flat = hidden_states.view(-1, hidden_states.shape[-1])
        scores = torch.softmax(flat @ self.weight.t(), dim=-1)
        weight, idx = torch.topk(scores, k=1, dim=-1, sorted=False)
        return idx, weight, None


class TupleMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.top_k = 1
        self.gate = TupleGate()
        self.experts = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])


def test_invalid_capacity_scope_raises_clear_error():
    model = TupleMoE()
    try:
        apply_capacity_aware_moe_patch(
            model,
            SimpleNamespace(expert_capacity=0.5, strategy="score", rounds=1, capacity_scope="node"),
        )
    except ValueError as exc:
        assert "capacity_scope" in str(exc)
    else:
        raise AssertionError("invalid capacity_scope should raise ValueError")


def test_tuple_gate_patch_zeroes_dropped_weights():
    model = TupleMoE()
    patched = apply_capacity_aware_moe_patch(
        model,
        SimpleNamespace(expert_capacity=0.5, strategy="score", rounds=1),
    )

    idx, weight, _ = model.gate(
        torch.tensor(
            [
                [[5.0, 0.0]],
                [[4.0, 0.0]],
                [[0.0, 3.0]],
            ]
        )
    )

    assert patched == 1
    assert idx.tolist() == [[0], [2], [1]]
    assert weight.tolist()[1] == [0.0]


class LogitsMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.top_k = 1
        self.gate = nn.Linear(2, 2, bias=False)
        self.experts = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
        with torch.no_grad():
            self.gate.weight.copy_(torch.eye(2))


def test_logits_gate_patch_masks_dropped_experts():
    model = LogitsMoE()
    patched = apply_capacity_aware_moe_patch(
        model,
        SimpleNamespace(expert_capacity=0.5, strategy="score", rounds=1),
    )

    logits = model.gate(
        torch.tensor(
            [
                [5.0, 0.0],
                [4.0, 0.0],
                [0.0, 3.0],
            ]
        )
    )
    assert patched == 1
    assert logits[0, 1] == torch.finfo(logits.dtype).min
    assert logits[2, 0] == torch.finfo(logits.dtype).min
    # Logits-return gates cannot encode a sentinel expert. If every selected
    # expert for a token is dropped, keep one fallback logit to avoid all -inf.
    assert logits[1, 0] == 4.0
    assert logits[1, 1] == torch.finfo(logits.dtype).min


class RouterTupleGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(2))
        self.top_k = 1

    def forward(self, hidden_states):
        flat = hidden_states.view(-1, hidden_states.shape[-1])
        scores = torch.softmax(flat @ self.weight.t(), dim=-1)
        weight, idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        weight = weight / weight.sum(dim=-1, keepdim=True)
        return scores, weight, idx


class RouterTupleMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = RouterTupleGate()
        self.experts = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])


def test_router_tuple_patch_reads_top_k_from_gate_and_keeps_indices_safe():
    model = RouterTupleMoE()
    patched = apply_capacity_aware_moe_patch(
        model,
        SimpleNamespace(expert_capacity=0.5, strategy="score", rounds=1),
    )

    scores, weight, idx = model.gate(
        torch.tensor(
            [
                [5.0, 0.0],
                [4.0, 0.0],
                [0.0, 3.0],
            ]
        )
    )

    assert patched == 1
    assert scores.shape == (3, 2)
    assert idx.tolist() == [[0], [0], [1]]
    assert weight.tolist()[1] == [0.0]
