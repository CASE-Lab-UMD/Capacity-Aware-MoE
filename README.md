# Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture of Experts

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2503.05066)
![Conference](https://img.shields.io/badge/ICLR-2026-blue)
![Python](https://img.shields.io/badge/Python-3.10+-green)

Official implementation of [**Capacity-Aware Inference**](https://arxiv.org/abs/2503.05066) (**ICLR 2026**).

This repository studies test-time load balancing for **Mixture of Experts (MoE)** inference and introduces practical routing strategies to reduce the straggler effect in expert-parallel execution.

<p align="center">
  <img src="Figures/straggler_effect.svg" alt="Straggler effect in MoE inference" width="52%">
</p>

## 📰 News
- Jan 2026: Capacity-Aware Inference accepted at **ICLR 2026**.
- Mar 2025: Paper and code released.

## ✨ Why This Repo
In sparse MoE models, only a subset of experts is activated per token. This improves parameter efficiency, but under expert parallelism, token routing can become highly imbalanced.

When a few experts are overloaded, all other devices wait for the slowest experts to finish, causing global latency inflation (the **straggler effect**).

This repo provides inference-time solutions that improve balance and throughput with minimal quality impact.

## 🔍 Core Methods
We implement two complementary strategies:

1. **Capacity-Aware Token Drop**
   - Enforces per-expert capacity constraints.
   - Drops overflow tokens routed to overloaded experts.
   - Target: lower tail latency with small performance loss.

2. **Capacity-Aware Expanded Drop**
   - Expands candidate routing to nearby lower-load experts before dropping.
   - Improves expert utilization and smooths load distribution.
   - Target: better throughput-efficiency tradeoff than direct dropping.

<p align="center">
  <img src="Figures/token_drop.svg" alt="Token Drop" width="45%">&nbsp;&nbsp;
  <img src="Figures/expanded_drop.svg" alt="Expanded Drop" width="45%">
</p>

## 📦 Repository Structure
- `modeling_hf/`: modified Hugging Face MoE modeling files.
- `lm-evaluation-harness/`: language evaluation pipeline and scripts.
- `VLMEvalKit/`: multimodal evaluation pipeline.
- `Figures/`: method and effect visualizations.
- `requirement.txt`: pinned root dependencies.
- `requirements.txt`: compatibility wrapper (`-r requirement.txt`).

## ⚙️ Installation
```bash
conda create -n capacity-moe python=3.10 -y
conda activate capacity-moe

pip install -r requirements.txt
```

## 🚀 Quick Start
### 1) Baseline Evaluation (Language)
```bash
cd lm-evaluation-harness
bash runs_prune/eval_baseline.sh
```

### 2) Capacity-Aware Evaluation
```bash
cd lm-evaluation-harness
bash runs_prune/eval_capacity.sh
```

Both scripts support environment variable overrides such as `CUDA_VISIBLE_DEVICES`, `PRETRAINED`, `OUTPUT_PATH`, `BATCH_SIZE`, `EXPERT_CAPACITY`, and `STRATEGY`.

### 3) Quick Demo
```bash
cd lm-evaluation-harness
PRETRAINED=./models/deepseek-moe-16b-base-temp \
EXPERT_CAPACITY=1.0 \
STRATEGY=score \
bash runs_prune/eval_capacity.sh
```

Expected outputs are written under:
`$PRETRAINED/expert_capacity-$EXPERT_CAPACITY/$STRATEGY/`

## 🧪 Evaluation Notes
- **Language benchmarks** are run through `lm_eval` in `lm-evaluation-harness`.
- **Multimodal benchmarks** are supported via `VLMEvalKit`.
- This repo focuses on **inference-time** routing/control under fixed checkpoints.

## 📈 Reported Effect
| Model / Setting | Method | Efficiency Impact | Quality Impact |
|---|---|---|---|
| OLMoE | Capacity-Aware Token Drop | ~30% speedup | ~0.9% degradation |
| Mixtral-8x7B-Instruct | Capacity-Aware Expanded Drop | ~1.85x speedup | ~0.2% gain |

For exact experimental setup and numbers, refer to the paper and evaluation scripts in this repo.

## ✅ Repro Checklist
- Environment: Python 3.10, dependency install via `pip install -r requirements.txt`.
- Model path: set `PRETRAINED` to a local checkpoint path.
- Hardware: set `CUDA_VISIBLE_DEVICES` and verify available GPU memory.
- Script: baseline with `runs_prune/eval_baseline.sh`, capacity-aware with `runs_prune/eval_capacity.sh`.
- Key knobs: `EXPERT_CAPACITY`, `STRATEGY`, `BATCH_SIZE`.

## 📄 Citation
If this repository is useful for your work, please cite:

```bibtex
@misc{he2025capacityawareinferencemitigatingstraggler,
  title={Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture of Experts},
  author={Shwai He and Weilin Cai and Jiayi Huang and Ang Li},
  year={2025},
  eprint={2503.05066},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2503.05066}
}
```
