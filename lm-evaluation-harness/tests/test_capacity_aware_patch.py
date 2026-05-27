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
apply_capacity_aware_moe_patch = capacity_patch.apply_capacity_aware_moe_patch


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
