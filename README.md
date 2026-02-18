# Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture of Experts

Official implementation of **Capacity-Aware Inference** (arXiv: [2503.05066](https://arxiv.org/abs/2503.05066)).
Accepted at **ICLR 2026**.

This repository studies test-time load balancing for **Mixture of Experts (MoE)** inference and introduces practical routing strategies to reduce the straggler effect in expert-parallel execution.

<p align="center">
  <img src="Figures/straggler_effect.svg" alt="Straggler effect in MoE inference" width="52%">
</p>

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

### 2) Capacity-Aware Evaluation (Language)
```bash
cd lm-evaluation-harness
bash runs_prune/eval_capacity.sh
```

Both scripts support environment variable overrides such as `CUDA_VISIBLE_DEVICES`, `PRETRAINED`, `OUTPUT_PATH`, `BATCH_SIZE`, `EXPERT_CAPACITY`, and `STRATEGY`.

## 🧪 Evaluation Notes
- **Language benchmarks** are run through `lm_eval` in `lm-evaluation-harness`.
- **Multimodal benchmarks** are supported via `VLMEvalKit`.
- This repo focuses on **inference-time** routing/control under fixed checkpoints.

## 📈 Reported Effect (from paper)
- Significant throughput gains in multiple MoE settings.
- Example outcomes include strong speedup with minor quality degradation, and in some settings quality-neutral or quality-positive behavior with improved efficiency.

For exact experimental setup and numbers, refer to the paper and evaluation scripts in this repo.

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
