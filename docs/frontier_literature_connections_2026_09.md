# 🧩 Capacity-Aware-MoE: 每日前沿文献关联与 MoE 容量感知路由/专家剪枝/硬件分块落地库 (2026-09)

**Document ID:** `CAPMOE-LIT-202609` | **Last Updated:** `2026-09-29` | **Target Path:** `docs/frontier_literature_connections_2026_09.md` | **Total Routed Papers:** `16`

> [!IMPORTANT]
> **🔗 跨仓库文献引用链闭环 (Cross-Repository Reference Chain Closure)**
> 本文件由每日 AI 前沿论文精读流水线自动路由生成，专门收录与我们 **ICLR 2026 / ICML 2026 代表作 (*Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture-of-Experts* & *DualSparse-MoE*, `CASE-Lab-UMD/Capacity-Aware-MoE`)** 直接关联的稀疏 MoE 动态容量路由、博弈论专家协同剪枝（`SHAPE`, `REAP`, `CARE`）、低秩路由器压缩（`L2R`）、多维笛卡尔专家路由（`MoE-nD`）及 GPU Warp 对齐硬件分块调度与投机协同系统（`MoE-Tile`, `MoE-OS`, `PiKV`, `CoMoE-Spec`）最新 arXiv 论文笔记。
> 每一篇收录文献均包含：**核心痛点、底层数学公式、ASCII 架构图、关键实测指标**，以及**与 `Capacity-Aware-MoE` 仓库具体代码模块和我们已发表代表作（Our Works）的双向锚定**。

---

## 🌟 1. 核心关联文献与本仓库模块映射速查表 (Executive Reference-to-Module Matrix)

| 收录日期 | 论文标题与 arXiv 链接 | 关键实测收益 / 核心结论 | 锚定本仓库代码模块与文档路径 (`Target Module`) | 原始精读归档 |
| :---: | :--- | :--- | :--- | :---: |
| `2026-09-29` | [**🧩 CoMoE-Spec**](https://arxiv.org/abs/2609.22471) (`arXiv:2609.22471`) | 在 Mixtral-8x7B、Qwen2.5-MoE-A14B 与 OLMoE-1B-7B 上结合 EAGLE-2 投机解码评测表明：`CoMoE-Spec` 将验证阶段的唯一激活专家总数削减了 **42%–58%**，在保持草稿... | `capacity_aware/` & `scripts/` (Coactivation-Guided Speculative Tree Construction & Bounded Unique Expert Union) | [2026-09-29](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-29_ai_paper_notes.md) |
| `2026-09-29` | [**🦾 DEE-VLA**](https://arxiv.org/abs/2609.29382) (`arXiv:2609.29382`) | 在 LIBERO（Spatial / Object / Goal / Long）与真机双臂灵巧操作任务上，`DEE-VLA` 在成功率与全深度 10-NFE 基线持平（甚至因减少自由空间过拟合而提升 **+0.8%**）的同时，平... | `capacity_aware/` & `scripts/` (`CASE-Lab-UMD/Capacity-Aware-MoE`) | [2026-09-29](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-29_ai_paper_notes.md) |
| `2026-09-28` | [**🧩 PiKV**](https://arxiv.org/abs/2508.06526) (`arXiv:2508.06526`) | 在多机多卡 Mixtral-8x22B 与 DeepSeek-MoE 长上下文服务基准上，PiKV 将单卡 KV 显存占用降低 **54%**，跨节点通信开销削减 **62%**，在 32K–64K 长序列高并发场景下实现... | `capacity_aware/` & `scripts/` (Expert-Sharded KV Cache Compression + All-to-All Pipeline Overlap) | [2026-09-28](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-28_ai_paper_notes.md) |
| `2026-09-28` | [**🌍 WorldAgen**](https://arxiv.org/abs/2609.08162) (`arXiv:2609.08162`) | 在包含未知负载质量变化、表面摩擦力突变及视觉遮挡扰动的具身操作基准上，WorldAgen 凭借每步耗时仅 `< 1.8 ms` 的轻量级测试时自校准，将分布外（OOD）物理环境下的任务成功率从冻结模型的 54.6% 跃升至... | `capacity_aware/` & `scripts/` (`CASE-Lab-UMD/Capacity-Aware-MoE`) | [2026-09-28](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-28_ai_paper_notes.md) |
| `2026-09-27` | [**SHAPE**](https://arxiv.org/abs/2606.09886) (`arXiv:2606.09886`) | **跨架构零训练稳健性**：在 **Qwen3-30B-A3B**、**DeepSeek-V2-Lite** 与 **GPT-OSS-20B** 三大主流细粒度 MoE 模型上，仅需 128 条 C4/WikiText2 校准样本... | `capacity_aware/` (Shapley Coalition-Aware Expert Pruning + Token Capacity Bound) | [2026-09-27](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-27_ai_paper_notes.md) |
| `2026-09-27` | [**L2R**](https://arxiv.org/abs/2601.21349) (`arXiv:2601.21349`) | **语言与视觉双模态全面验证**：在基于 **OLMoE** 的语言模型预训练/微调以及 **ImageNet** 视觉 MoE 骨干网络上，L2R 将路由器参数量削减 **60%–75%**，同时在相同激活专家预算下将下游任务困... | `capacity_aware/` (Low-Rank Latent Anchor + SIPS Lipschitz Router) | [2026-09-27](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-27_ai_paper_notes.md) |
| `2026-09-26` | [**🔄 LoopMoE**](https://arxiv.org/abs/2606.04438) (`arXiv:2606.04438`) | **等参数量与等 FLOPs 双向碾压**：在语言建模基准与常识推理任务上，循环 $K=2\sim 4$ 步的 `LoopMoE` 在相同活跃参数量下显著优于标准稠密 Looped 模型，且在相同总参数预算下逼近非共享深层 MoE... | `capacity_aware/` (Recurrent Step-Specific Low-Rank Calibrated MoE) | [2026-09-26](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-26_ai_paper_notes.md) |
| `2026-09-26` | [**🤖 VLA-Pruner**](https://arxiv.org/abs/2511.16449) (`arXiv:2511.16449`) | 在 OpenVLA 与主流机器人操控基准（LIBERO-Spatial / Object / Goal / Long）上，剔除 **50%–75% 视觉 Token** 仍保持与全量 Token 持平的任务成功率，端到端控制频率显... | `capacity_aware/` & `scripts/` (`CASE-Lab-UMD/Capacity-Aware-MoE`) | [2026-09-26](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-26_ai_paper_notes.md) |
| `2026-09-26` | [**🧩 MoE-nD**](https://arxiv.org/abs/2604.17695) (`arXiv:2604.17695`) | 详见下方完整公式与实验卡片 | `capacity_aware/` (Cartesian Product Multi-Dimensional Expert Grid) | [2026-09-26](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-26_ai_paper_notes.md) |
| `2026-09-24` | [**DriveMoE**](https://arxiv.org/abs/2505.16278) (`arXiv:2505.16278`) | 在 **Bench2Drive** 闭环评测与 **nuScenes** 开环基准上，DriveMoE 将复杂交叉路口与紧急避障长尾场景的驾驶得分（Driving Score）大幅提升 **`+9.4` 分**，碰撞率降低... | `capacity_aware/` & `scripts/` (`CASE-Lab-UMD/Capacity-Aware-MoE`) | [2026-09-24](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-24_ai_paper_notes.md) |
| `2026-09-23` | [**HiMoE-VLA**](https://arxiv.org/abs/2512.05693) (`arXiv:2512.05693`) | 在跨 50+ 任务的 Open-X Embodiment 与仿真套件上，HiMoE-VLA 比同激活参数量的稠密 VLA 与单层 MoE-VLA 平均成功率提升 **`+8.7%`**。 | `capacity_aware/` & `scripts/` (`CASE-Lab-UMD/Capacity-Aware-MoE`) | [2026-09-23](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-23_ai_paper_notes.md) |
| `2026-09-23` | [**D-Cut**](https://arxiv.org/abs/2607.14647) (`arXiv:2607.14647`) | 在 Batch Size = 16–64 的生产级投机解码服务中，D-Cut 将验证阶段算力开销削减 **38%**，端到端吞吐在 EAGLE-2 基线上进一步提升 **1.42x**。 | `capacity_aware/` & `scripts/` (`CASE-Lab-UMD/Capacity-Aware-MoE`) | [2026-09-23](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-23_ai_paper_notes.md) |
| `2026-09-21` | [**MoE-FM**](https://arxiv.org/abs/2604.15009) (`arXiv:2604.15009`) | 在潜空间语言生成与多模态推理中，MoE-FM 在仅使用 **2–4 步 NFE** 时即可达到单稠密流模型 16–32 步的生成质量，推理延迟降低 **3.8x**。 | `capacity_aware/` & `scripts/` (`CASE-Lab-UMD/Capacity-Aware-MoE`) | [2026-09-21](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-21_ai_paper_notes.md) |
| `2026-09-20` | [**CARE**](https://arxiv.org/abs/2607.26052) (`arXiv:2607.26052`) | 在多任务 MoE-LoRA 与稀疏 MoE 语言模型上，CARE 在削减 **32%–45% 平均专家激活 FLOPs** 的同时，在常识推理、代码与数学基准上全面持平甚至超越固定 Top- $k$ 基线（`+0.9%` 平均准确... | `capacity_aware/` (Dual-Threshold Dynamic Top-k Expert Allocation) | [2026-09-20](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-20_ai_paper_notes.md) |
| `2026-09-19` | [**REAP**](https://arxiv.org/abs/2510.13999) (`arXiv:2510.13999`) | 在 **Mixtral-8x7B**、**DeepSeek-MoE-16B** 与 **Qwen1.5-MoE-A2.7B** 上，REAP 在 **25%–37.5% 专家剪枝率**下，在 GSM8K 与 HumanEval 生... | `capacity_aware/` (Gate-Probability × Expert-Activation Norm Pruning) | [2026-09-19](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-19_ai_paper_notes.md) |
| `2026-09-18` | [**🧩 MoE-Tile**](https://arxiv.org/abs/2609.09112) (`arXiv:2609.09112`) | **硬件测试平台**：NVIDIA H100 80GB SXM5 与 B200 GPU 集群； | `capacity_aware/` & `scripts/` (Warp-Aligned 2D Tile Scheduling & Zero-Slack Capacity) | [2026-09-18](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-18_ai_paper_notes.md) |

---

## 📐 2. 逐篇论文深度机制解构、数学公式与本仓库落地指南 (Per-Paper Deep-Dive Cards)

### 2.1 [2026-09-29] 🧩 *CoMoE-Spec: Efficient Mixture-of-Experts with Speculative Decoding via Expert Coactivation*
> 🏷️ **核心关键词**：Mixture-of-Experts (MoE) · Speculative Decoding · Expert Coactivation Routing · Memory-Bandwidth Bottleneck  
> 🔗 **arXiv 链接**：[`arXiv:2609.22471`](https://arxiv.org/abs/2609.22471)

```
  草稿模型生成 γ 个候选 Token (t_1..t_γ) ──► 目标 MoE 验证阶段
       ├── [ 传统独立 Top-k 路由 ] : γ 个 Token 激活并集 |∪ E(t_i)| ≈ M (几乎拉取全部专家权重，陷入 HBM 带宽瓶颈！)
       └── [ CoMoE-Spec 协同路由 ] : 跨草稿步联合专家共激活惩罚 + 共享专家重分配 ──► |∪ E(t_i)| 压缩 42%~58%，验证墙钟提速 1.85x
```

#### 🎯 背景与痛点 (Problem Statement)
投机解码（Speculative Decoding）在稠密大模型（Dense LLM）上之所以能实现无损加速，核心前提是“并行验证 $\gamma$ 个草稿 Token 的延迟与单步生成 1 个 Token 几乎相同（Compute-Bound 前移）”。然而，在稀疏混合专家模型（Sparse MoE，如 Mixtral、DeepSeek-MoE、Qwen3-MoE）中，这一前提彻底失效：当单个草稿 Token 仅激活 $k$ 个专家时，并行验证 $\gamma$ 个草稿 Token 却会激活多达 $\lvert \bigcup _ {m=1}^{\gamma} \mathcal{E}(t _ m) \rvert \gg k$ 个互不相同的专家。由于 GPU 必须将所有被任一草稿 Token 命中的专家权重从高带宽显存（HBM）加载至片上寄存器，验证阶段的**唯一激活专家并集膨胀（Expert Union Explosion）**直接将 MoE 验证打回极度访存受限（Memory-Bandwidth Bound）状态，吞噬了投机解码的理论收益。

#### 💡 核心方法与数学公式 (Core Methodology & Formulation)
* **草稿窗口唯一激活专家并集开销模型**：
  设 MoE 层共有 $E$ 个专家，草稿验证树包含 $\gamma$ 个候选 Token，每个 Token $m \in \lbrace 1, \dots, \gamma \rbrace$ 的路由器门控概率为 $p _ m \in \Delta^{E-1}$ ，选择的 Top- $k$ 专家集合为 $\mathcal{E} _ m$ 。验证阶段的内存读取字节数正比于并集基数 $\lvert \mathcal{U} _ {\gamma} \rvert = \left\lvert \bigcup _ {m=1}^{\gamma} \mathcal{E} _ m \right\rvert$ ，其期望值为：

$$
\mathbb{E}\left[ \lvert \mathcal{U} _ {\gamma} \rvert \right] = \sum _ {e=1}^{E} \left( 1 - \prod _ {m=1}^{\gamma} \left( 1 - \mathbb{I}\lbrace e \in \mathcal{E} _ m \rbrace \right) \right)
$$

* **专家协同激活路由与联合次模验证约束（Expert Coactivation Routing）**：
  `CoMoE-Spec` 在路由器微调与推理期动态路由中引入**跨位置专家协同激活正则项（Coactivation Regularizer）**，并在草稿树验证阶段求解受总专家预算 $\lvert \mathcal{U} _ {\gamma} \rvert \le B _ {\text{max}}$ 约束的联合路由重分配：

$$
\max _ {\lbrace \mathcal{E} _ m \rbrace _ {m=1}^{\gamma}} \sum _ {m=1}^{\gamma} \sum _ {e \in \mathcal{E} _ m} \log p _ {m, e} - \beta \cdot \sum _ {e=1}^{E} \max _ {1 \le m \le \gamma} \mathbb{I}\lbrace e \in \mathcal{E} _ m \rbrace \quad \text{s.t.} \quad \lvert \mathcal{E} _ m \rvert = k
$$

  对仅被单个边缘草稿节点低置信度命中的“孤立长尾专家（Straggler Singleton Expert）”，将其平滑重路由至草稿窗口内已高频共激活的次优专家（若门控概率差 $\Delta p \le \epsilon _ {\text{co}}$ ），从而在不降低接受率 $\alpha$ 的前提下大幅压缩专家并集。

#### 📊 关键实验与结论 (Key Results & Conclusions)
* 在 Mixtral-8x7B、Qwen2.5-MoE-A14B 与 OLMoE-1B-7B 上结合 EAGLE-2 投机解码评测表明：`CoMoE-Spec` 将验证阶段的唯一激活专家总数削减了 **42%–58%**，在保持草稿 Token 平均接受长度 $\tau$ 几乎不变（波动 `< 0.03`）的前提下，将端到端 MoE 投机解码墙钟吞吐率进一步提升 **1.45×–1.85×**。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发 (Connection to Our Works)
* **锚定我们的代表作**：与我们发表于 **ICLR 2026 / ICML 2026** 的代表作 ***Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture-of-Experts***（`CASE-Lab-UMD/Capacity-Aware-MoE`）、***Unifying LLM & Mixture-of-Experts Compression***（`TMLR 2025`, `Unified-MoE-Compression`）及 `awesome-mixture-of-experts` 形成完美的系统互补！
* **落地到 `Capacity-Aware-MoE`、`Unified-MoE-Compression` 与 `efficient_ads`**：我们的 `Capacity-Aware-MoE` 解决了长序列 Prefill 阶段单专家过载的“重负载落后者（Overloaded Straggler）”，而 `CoMoE-Spec` 解决了投机 Decode 验证阶段只服务 1 个草稿 Token 的“稀疏孤立专家（Under-loaded Singleton Expert）”。将两者的双向容量上下界（ $[C _ {\min}, C _ {\max}]$ ）统一到 `capacity_aware/` 路由算子中，即可同时加速 Prefill 与 Speculative Decode！

---

> [!TIP]
> **🎯 `Capacity-Aware-MoE` 仓库代码级落地点 (`Target Module`)**：`capacity_aware/` & `scripts/` (Coactivation-Guided Speculative Tree Construction & Bounded Unique Expert Union)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-29_ai_paper_notes.md`


---

### 2.2 [2026-09-29] 🦾 *DEE-VLA: Decoupled Early Exits for Task-Dependent Compute Allocation in Flow-Matching VLAs*
> 🏷️ **核心关键词**：Vision-Language-Action (VLA) · Flow Matching · Decoupled Early Exits · Dynamic Compute Allocation  
> 🔗 **arXiv 链接**：[`arXiv:2609.29382`](https://arxiv.org/abs/2609.29382)

```
  多模态观测 (I_t, l) ──► [ VLM 感知主干 (层 1..L_vlm) ] ──► 动态早退层 l_vlm* (自由空间移动早退，精细对准深层)
                                                                  │ (跨模块KV桥接投影)
                                                                  ▼
                     [ Flow-Matching Action Expert (层 1..L_act, 积分步 k=1..K) ]
                                                                  ├──► 动作网络深度早退 l_act*(k)
                                                                  └──► 速度场曲率收敛早退步 K* ──► 实时机器人控制动作 a_t
```

#### 🎯 背景与痛点 (Problem Statement)
现有的视觉-语言-动作（VLA）基础模型（如 $\pi _ 0$ 、 $\pi _ {0.5}$ 、GR00T）通常由百亿级参数的多模态 VLM 主干与数亿参数的流匹配（Flow-Matching）动作专家（Action Expert）组成。以往的早退（Early Exit）或层剪枝方法往往将 VLM 主干深度与动作专家深度**强行绑定（Coupled Depth Scaling）**，或者对一整段轨迹的所有时间步施加相同的静态深度预算。然而，机器人操纵任务具有显著的**时空异质性（Spatio-Temporal Heterogeneity）**：在粗粒度场景理解已完成的抓取接近阶段，VLM 主干仅需浅层表征即可维持语义定位，而进入毫米级插孔（Peg-in-Hole）接触瞬间，VLM 无需重算深层语义但 Action Expert 却需要更深的网络层数与更多的流匹配 ODE 修正步来解析高频接触动力学。

#### 💡 核心方法与数学公式 (Core Methodology & Formulation)
* **三轴解耦动态计算空间（Tri-Axis Decoupled Compute Space）**：
  `DEE-VLA` 将单次控制周期的推理算力分解为三个可独立调节的正交自由度：（1）VLM 主干退出层 $l _ {\text{vlm}} \in \lbrace 1, \dots, L _ {\text{vlm}} \rbrace$ ；（2）第 $k$ 个流匹配步中 Action Expert 的退出层 $l _ {\text{act}}^{(k)} \in \lbrace 1, \dots, L _ {\text{act}} \rbrace$ ；（3）流匹配歐拉积分的总终止步数 $K^{\star} \in \lbrace 1, \dots, K _ {\max} \rbrace$ 。
* **跨层隐状态余弦稳定性与速度场曲率早退门控**：
  在 VLM 主干内部，当相邻两层视觉-语言融合隐状态的余弦相似度超过语义收敛阈值 $\tau _ {\text{vlm}}$ 时提前退出并通过轻量级层对齐投影器生成 KV 缓存；在流匹配 Action Expert 内部，同时监测跨层速度预测残差与跨 ODE 步的**流场直线性曲率（Flow Straightness Curvature）**：

$$
\mathcal{E} _ {\text{depth}}^{(k)}(l) = \frac{\lVert v _ {\theta}^{(l)}(x _ {t _ k}, t _ k) - v _ {\theta}^{(l-1)}(x _ {t _ k}, t _ k) \rVert _ 2}{\lVert v _ {\theta}^{(l-1)}(x _ {t _ k}, t _ k) \rVert _ 2 + \epsilon} \le \delta _ {\text{act}}, \quad \mathcal{E} _ {\text{flow}}(k) = \lVert v _ {\theta}^{\star}(x _ {t _ k}, t _ k) - v _ {\theta}^{\star}(x _ {t _ {k-1}}, t _ {k-1}) \rVert _ 2 \le \delta _ {\text{ode}}
$$

  一旦 $\mathcal{E} _ {\text{flow}}(k) \le \delta _ {\text{ode}}$ （表明当前局部流场已呈直线匀速轨迹），立即跳过剩余 ODE 积分步并利用一阶欧拉外推直接输出终端动作块 $\hat{x} _ 1 = x _ {t _ k} + (1 - t _ k) v _ {\theta}^{\star}(x _ {t _ k}, t _ k)$ 。

#### 📊 关键实验与结论 (Key Results & Conclusions)
* 在 LIBERO（Spatial / Object / Goal / Long）与真机双臂灵巧操作任务上，`DEE-VLA` 在成功率与全深度 10-NFE 基线持平（甚至因减少自由空间过拟合而提升 **+0.8%**）的同时，平均削减了 **54.2% 的总 FLOPs** 与 **49.6% 的端到端控制延迟**，自动展现出“自由空间巡航浅层少步、接触操作阶段深层精细积分”的涌现算力分配规律。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发 (Connection to Our Works)
* **锚定我们的代表作**：与我们在 `axon_v2` / `VLADrop` 中构建的 **Tri-Orthogonal Depth-Width-Step Compression (`G19`)**、**Once-for-All Switchable Multi-Gear Loops** 以及 **Looped VLA 动态停止准则 $K(s _ t, t, m)$ ** 完全同源！
* **落地到 `axon_v2` 与 `VLADrop` (`VLM-Compression`)**：可在 `axon/layers/looped_ode.py` 与 `VLADrop/models/pi0.5/` 中直接融合 `DEE-VLA` 的双判据 $\left( \mathcal{E} _ {\text{depth}}^{(k)}(l), \mathcal{E} _ {\text{flow}}(k) \right)$ ，将 VLM 编码器深度门控与 Action Expert 的 ODE 步曲率门控彻底解耦。

---

> [!TIP]
> **🎯 `Capacity-Aware-MoE` 仓库代码级落地点 (`Target Module`)**：`capacity_aware/` & `scripts/` (`CASE-Lab-UMD/Capacity-Aware-MoE`)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-29_ai_paper_notes.md`


---

### 2.3 [2026-09-28] 🧩 *PiKV: KV Cache Management System for Mixture of Experts*
> 🏷️ **核心关键词**：Mixture-of-Experts (MoE) · Expert-Sharded KV Cache · Distributed Serving · Memory & Communication Co-Design  
> 🔗 **arXiv 链接**：[`arXiv:2508.06526`](https://arxiv.org/abs/2508.06526) (2026 v3)

```
  分布式 MoE 节点 (EP + TP) ──► 传统方案: 每张 GPU 复制全量同步 KV Cache (显存爆炸 + All-Gather 阻塞)
                            ──► PiKV 方案: [ 专家分片 KV 存储 (Expert-Sharded KV) ] + [ PiKV 路由感知调度 ]
                                          ──► 按专家亲和度局部缓存活跃 Token KV ──► 跨卡通信降低 62%
```

#### 🎯 背景与痛点 (Problem Statement)
在超大规模稀疏混合专家模型（如 DeepSeek-V3、Mixtral、Qwen3-MoE）的分布式专家并行（Expert Parallelism, EP）服务中，尽管 FFN 专家权重被分片到不同 GPU 上，但现有的推理框架仍要求在每个节点上维护全局同步的注意力 KV Cache。随着长上下文并发请求增加，全局复制或频繁 All-Gather 同步 KV Cache 不仅耗尽了原本用于存放专家权重的 HBM 显存，更使跨节点通信成为拖垮解码吞吐量（Throughput）的首要瓶颈。

#### 💡 核心方法与数学公式 (Core Methodology & Formulation)
* **专家分片 KV 存储（Expert-Sharded KV Storage）**：
  PiKV 打破了“注意力 KV 必须与专家并行完全解耦并全局复制”的传统范式，利用相邻层间 MoE 路由器的**跨层专家拓扑亲和性（Cross-Layer Expert Affinity）**，将 KV Cache 分页块按 Token 历史激活的主导专家簇分片存储在对应 GPU 节点的本地显存池 $\mathcal{M} _ e$ 中：

$$
\mathcal{M} _ e = \left\lbrace \left( k _ t^{(l)}, v _ t^{(l)} \right) \middle| e = \arg\max _ {j \in \lbrace 1, \dots, E \rbrace} G _ j^{(l-1)}(x _ t) \right\rbrace
$$

* **PiKV 路由与通信掩盖流水线调度（PiKV Routing & Scheduling）**：
  对于跨节点远端 KV 访问，PiKV 引入**重要性感知稀疏 KV 拉取门控**：仅对当前查询 $q _ t$ 预测注意力内积超过阈值 $\gamma$ 的远端分片发起异步 RDMA 拉取，并将 KV 分片传输与本地活跃专家的 GEMM 计算在 CUDA Stream 上完全重叠（Overlap）：

$$
\hat{o} _ t = \mathrm{Attn}\left( q _ t, K _ {\text{local}}, V _ {\text{local}} \right) \oplus \mathrm{Attn}\left( q _ t, \mathrm{TopM} _ {\gamma}\left( K _ {\text{remote}}, V _ {\text{remote}} \right) \right)
$$

#### 📊 关键实验与结论 (Key Results & Conclusions)
* 在多机多卡 Mixtral-8x22B 与 DeepSeek-MoE 长上下文服务基准上，PiKV 将单卡 KV 显存占用降低 **54%**，跨节点通信开销削减 **62%**，在 32K–64K 长序列高并发场景下实现 **1.85×–2.30×** 的端到端吞吐量提升。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发 (Connection to Our Works)
* **锚定我们的代表作**：与我们的 ***Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture-of-Experts***（`ICLR 2026`, `Capacity-Aware-MoE`）、***Unified-MoE-Compression*** 以及 ***MEO: Memory-Efficient Optimization***（`EMNLP 2023 Oral`, `MEO`）形成系统层闭环。
* **落地到 `Capacity-Aware-MoE` 与 `efficient_ads`**：在 `Capacity-Aware-MoE` 的过载专家 Token 丢弃与重路由（Drop & Replenish）机制中，可联合考虑 **目标专家的本地 PiKV 缓存命中率**——优先将边缘 Token 重路由至本地已持有其上下文 KV 分片的次优专家，从而同时消除计算掉队者（Straggler）与跨卡 KV 拉取延迟。

---

## 🔥 板块二：全球前沿热点精选 (Trending Frontier)

> [!TIP]
> **🎯 `Capacity-Aware-MoE` 仓库代码级落地点 (`Target Module`)**：`capacity_aware/` & `scripts/` (Expert-Sharded KV Cache Compression + All-to-All Pipeline Overlap)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-28_ai_paper_notes.md`


---

### 2.4 [2026-09-28] 🌍 *WorldAgen: Unified State-Action Prediction with Test-Time World Model Training*
> 🏷️ **核心关键词**：Embodied World Models · Unified State-Action Prediction · Test-Time Training (TTT) · Online Sim-to-Real Adaptation  
> 🔗 **arXiv 链接**：[`arXiv:2609.08162`](https://arxiv.org/abs/2609.08162)

```
  当前潜状态 s_t ──► [ 共享自回归/循环骨干 + 动作头 π_θ / 世界模型头 W_φ ] ──► 预测下一状态 ŝ_{t+1}, 动作 a_t
                                  ▲                                                        │
                                  │ (测试时单步梯度自校准 Δθ, Δφ)                             ▼
                        [ 物理转移预测残差 L_TTT = ||s_{t+1} - ŝ_{t+1}||_2^2 ] ◄── 环境真实反馈状态 s_{t+1}
```

#### 🎯 背景与痛点 (Problem Statement)
现有的具身世界模型—动作联合模型（World-Action Models）在离线训练完成后即完全冻结参数。然而在真实机器人部署（Sim-to-Real 或新物体摩擦系数/质量突变）时，环境物理动力学 $P(s _ {t+1} \mid s _ t, a _ t)$ 往往发生未见漂移，冻结的世界模型产生的内部想象轨迹（Imagined Rollouts）迅速偏离真实物理世界，进而误导下游动作规划。

#### 💡 核心方法与数学公式 (Core Methodology & Formulation)
* **状态—动作统一预测架构**：
  WorldAgen 将多模态观测编码至紧凑潜状态 $s _ t = E(o _ t)$ ，由共享骨干 $f _ \theta$ 同时驱动未来潜状态预测头 $W _ \phi(s _ t, a _ t)$ 与动作生成头 $\pi _ \psi(s _ t, g)$ 。
* **测试时世界模型在线自监督校准（Test-Time World Model Training, TTT）**：
  在部署执行周期 $t \to t+1$ 中，当动作 $a _ t$ 在真实环境中执行并观测到真实下一帧潜状态 $s _ {t+1}$ 时，WorldAgen 立即以**真实状态转移残差**作为无标签自监督信号，对共享骨干的高阶低秩适配层（Fast Weights / LoRA $\Delta \theta _ t$ ）执行单步测试时梯度更新：

$$
\mathcal{L} _ {\text{TTT}}\left( \theta _ t, \phi _ t \right) = \left\lVert \mathrm{sg}\left( s _ {t+1} \right) - W _ {\phi _ t}\left( f _ {\theta _ t}(s _ t), a _ t \right) \right\rVert _ 2^2 + \lambda _ {\text{kl}} \mathrm{KL}\left( f _ {\theta _ t}(s _ t) \parallel f _ {\theta _ 0}(s _ t) \right)
$$

$$
\theta _ {t+1} \leftarrow \theta _ t - \eta _ {\text{ttt}} \nabla _ {\theta _ t} \mathcal{L} _ {\text{TTT}}\left( \theta _ t, \phi _ t \right)
$$

#### 📊 关键实验与结论 (Key Results & Conclusions)
* 在包含未知负载质量变化、表面摩擦力突变及视觉遮挡扰动的具身操作基准上，WorldAgen 凭借每步耗时仅 `< 1.8 ms` 的轻量级测试时自校准，将分布外（OOD）物理环境下的任务成功率从冻结模型的 54.6% 跃升至 **78.9%（+24.3%）**。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发 (Connection to Our Works)
* **锚定我们的活跃研究线**：直接对应我们的 **`axon_v2` / `axon` (`vla-loop` — Looped Action Expert & Control-Variate Trajectory Completion)** 与 **`mera` (`MerA` 低秩子空间初始化)**。
* **落地到 `axon_v2` 的 Looped VLA 控制器**：我们在 `vla-loop` 中设计了跨步循环残差校准器（Outer-LoRA Trajectory Completion）；结合 WorldAgen 的思路，可在实际闭环控制中利用上一时刻观测到的本体感知（Proprioception）状态残差，在线微调循环步专属低秩矩阵 $A _ k B _ k$ 的缩放门控，实现零样本物理扰动自适应。

---

> [!TIP]
> **🎯 `Capacity-Aware-MoE` 仓库代码级落地点 (`Target Module`)**：`capacity_aware/` & `scripts/` (`CASE-Lab-UMD/Capacity-Aware-MoE`)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-28_ai_paper_notes.md`


---

### 2.5 [2026-09-27] SHAPE: Coalition-Aware Expert Pruning for Sparse Mixture-of-Experts LLMs

* **论文信息**：`arXiv:2606.09886` (2026-06, 开源仓库：`github.com/Alizen-1009/Shapley-Moe`)
* **核心关键词**：Sparse MoE、Cooperative Game Theory、Shapley Value Attribution、Coalition-Aware Expert Pruning、Quality-Coverage Bisection

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|               SHAPE: Coalition-Aware MoE Expert Pruning Pipeline                  |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [Calibration Corpus D_cal] ---> Layer l Top-k Routing Traces: C_t = {e_i1..e_ik} |
|                                                |                                  |
|                                                v                                  |
|  +-----------------------------------------------------------------------------+  |
|  | 1. Intra-Layer Cooperative Game Formulation (层内专家合作博弈建模)          |  |
|  |    * Players: E_l = {1, ..., N} experts in layer l                          |  |
|  |    * Coalition Utility v_l(S): Expected output reconstruction fidelity      |  |
|  |      when active Top-k coalition C_t is restricted to subset S \cap C_t     |  |
|  +-----------------------------------------------------------------------------+  |
|                                                |                                  |
|                                                v                                  |
|  +-----------------------------------------------------------------------------+  |
|  | 2. Monte-Carlo / Co-Activation Shapley Attribution (Shapley 协同价值归因)    |  |
|  |    \phi_i(v_l) = \sum_{S \subseteq E_l \setminus \{i\}} w(|S|) [v_l(S \cup  |  |
|  |                  \{i\}) - v_l(S)]                                           |  |
|  |    * Captures high-order synergy: preserves "bridge" experts that rarely    |  |
|  |      dominate gate mass alone but are indispensable in Top-k combinations   |  |
|  +-----------------------------------------------------------------------------+  |
|                                                |                                  |
|                                                v                                  |
|  +-----------------------------------------------------------------------------+  |
|  | 3. Quality-Coverage Bisection Selection (全局预算二分质量覆盖率动态分配)    |  |
|  |    Retain minimal subset S_l^* s.t. \sum_{i \in S_l^*} \phi_i^+ >= \alpha(\lambda)|
|  |    Bisection search on \alpha to hit exact global target pruning ratio p    |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **单专家独立打分的“组合盲区”**：现有的免训练 MoE 专家剪枝方法（如基于路由激活频率 Frequency、门控权重均值 Gate-Sum 或单专家一阶重构误差的方法）均隐含了一个错误的**独立性假设（Independence Assumption）**——即每个专家的贡献可以孤立度量。然而，MoE 的前向计算本质上是**组合协同（Coalitional）**的：每个 Token 的输出由激活的 Top- $k$ 专家子集 $C _ t$ 线性叠加生成。
* **协同正交专家的误杀**：在真实 MoE 层中，若两个高激活专家高度共线（功能冗余），同时保留两者的边际增益极低；反之，某些中低频激活的“互补/正交桥接专家（Bridge Experts）”虽然单独门控权重不高，但在特定 Top- $k$ 组合中提供了不可替代的正交残差修正。独立打分会将前者全部保留而误杀后者，导致 20%–40% 剪枝率下模型出现断崖式精度崩塌。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **层内合作博弈定义（Intra-Layer Cooperative Game）**：
   设第 $l$ 层共有 $N$ 个专家 $\mathcal{E} _ l = \lbrace1, \dots, N\rbrace$ 。给定校准集 $\mathcal{D} _ {\text{cal}}$ 上的输入隐状态 $x _ t \in \mathbb{R}^d$ ，原始 Top- $k$ 路由集合为 $C _ t \subseteq \mathcal{E} _ l$ （ $|C _ t|=k$ ），原始层输出为：

$$
y _ t = \sum _ {j \in C _ t} g _ {t,j} E _ j(x _ t)
$$

   当仅保留专家子集 $S \subseteq \mathcal{E} _ l$ 时，受限联盟输出为 $\hat{y} _ t(S) = \sum _ {j \in C _ t \cap S} \tilde{g} _ {t,j}(S) E _ j(x _ t)$ 。定义联盟 $S$ 的特征效用函数（Characteristic Utility Function） $v _ l: 2^{\mathcal{E} _ l} \to \mathbb{R}$ 为相对于空集的输出误差削减量：

$$
v _ l(S) = \mathbb{E} _ {x _ t \sim \mathcal{D} _ {\text{cal}}} \Big[ \Vert y _ t \Vert _ 2^2 - \Vert y _ t - \hat{y} _ t(S) \Vert _ 2^2 \Big]
$$

2. **基于共现轨迹的 Shapley 协同归因（Shapley Value Attribution）**：
   专家 $i \in \mathcal{E} _ l$ 的 Shapley 值定义为其在所有可能专家联盟 $S \subseteq \mathcal{E} _ l \setminus \lbrace i\rbrace$ 中的平均边际贡献：

$$
\phi _ i(v _ l) = \sum _ {S \subseteq \mathcal{E} _ l \setminus \lbrace i\rbrace} \frac{|S|!(N - |S| - 1)!}{N!} \Big( v _ l(S \cup \lbrace i\rbrace) - v _ l(S) \Big)
$$

   由于每个 Token 仅激活 $|C _ t| = k \ll N$ 个专家（例如 $k=2$ 或 $6,8$ ），任何不包含在 $C _ t$ 中的专家对该 Token 边际贡献恒为 $0$ 。因此，原本指数级 $O(2^N)$ 的全局 Shapley 计算可精确降维至局部活跃联盟 $2^{|C _ t|}$ 上的精确求和：

$$
\phi _ i(v _ l) = \mathbb{E} _ {x _ t : i \in C _ t} \left[ \sum _ {A \subseteq C _ t \setminus \lbrace i\rbrace} \frac{|A|!(|C _ t| - |A| - 1)!}{|C _ t|!} \Big( u _ t(A \cup \lbrace i\rbrace) - u _ t(A) \Big) \right]
$$

   其中局部效用 $u _ t(A)$ 度量了子集 $A$ 内专家输出向量的内积交互项 $2 \langle g _ {t,i} E _ i(x _ t), \sum _ {j \in A} g _ {t,j} E _ j(x _ t) \rangle + \Vert g _ {t,i} E _ i(x _ t)\Vert _ 2^2$ ，从而自动惩罚与同联盟其他专家负相关或冗余的专家，奖励提供正交有效增量的专家。
3. **质量覆盖率二分层间分配（Quality-Coverage Selection Rule）**：
   为实现非均匀的层间稀疏率分配，将非负 Shapley 值归一化为质量分布 $\tilde{\phi} _ {l,i} = \frac{\max(\phi _ i(v _ l), 0)}{\sum _ {j=1}^N \max(\phi _ j(v _ l), 0)}$ 。给定阈值 $\alpha \in (0, 1)$ ，每层保留最小专家集合 $S _ l^\star(\alpha)$ 使得累计 Shapley 质量覆盖率不低于 $\alpha$ ：

$$
S _ l^\star(\alpha) = \arg\min _ {S \subseteq \mathcal{E} _ l} |S| \quad \text{s.t.} \quad \sum _ {i \in S} \tilde{\phi} _ {l,i} \ge \alpha
$$

   最后通过一维二分搜索（Bisection Search）求解全局唯一阈值 $\alpha^\star$ ，使得 $\frac{1}{L N}\sum _ {l=1}^L |S _ l^\star(\alpha^\star)| = 1 - p$ （ $p$ 为目标全局剪枝率）。

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* **跨架构零训练稳健性**：在 **Qwen3-30B-A3B**、**DeepSeek-V2-Lite** 与 **GPT-OSS-20B** 三大主流细粒度 MoE 模型上，仅需 128 条 C4/WikiText2 校准样本（无需任何微调），在 **20% 剪枝率**下恢复超过 **96.8%** 的原始零样本推理精度，在激进的 **40% 剪枝率**下比独立频次/门控剪枝高出 **5.4%–9.2%**（MMLU、GSM8K、ARC-Challenge）。
* **层间稀疏度自发涌现“沙漏分布”**：二分质量覆盖率准则自动在中间语义整合层保留更多专家，而在浅层词法层与深层输出对齐层裁剪高达 50% 的冗余专家。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
1. **与 *Demystifying When Pruning Works via Representation Hierarchies* (ICML 2026) & *Capacity-Aware Inference* (ICLR 2026) 的理论互证**：
   * 我们在 ICML 2026 中证明了剪枝是否生效取决于层间表示层级（Representation Hierarchy）的有效秩与冗余度分布；SHAPE 的局部 Shapley 展开式 $u _ t(A \cup \lbrace i\rbrace) - u _ t(A)$ 本质上是通过度量专家输出向量之间的交叉内积 $\langle E _ i(x), E _ j(x) \rangle$ 来识别表示子空间的正交性。
2. **与 *Transformer-Geometry* (`arXiv:2609.15975`, EMNLP 2026) 的几何融合启发**：
   * 在我们的正交/平行场分解框架 $E _ j(x) = E _ {j,\parallel}(x) + E _ {j,\perp}(x)$ 下，SHAPE 的效用函数若直接建立在总输出 $y _ t$ 的欧氏范数上，会被模长占优的平行径向分量 $E _ {j,\parallel}(x)$ 主导！**核心改进点**：将 SHAPE 的联盟效用函数 $v _ l(S)$ 限制在**去除流形平行漂移后的正交切空间分量 $P _ \perp(h _ t) E _ j(x _ t)$ ** 上计算 Shapley 值（即 **Perp-Shapley MoE Pruning**），随后对被剪除专家联盟的正交残差通过 **Woodbury / KKT 闭式补偿** 折叠进保留专家中，有望在 50% 专家剪枝率下实现近乎零损压缩。

---

> [!TIP]
> **🎯 `Capacity-Aware-MoE` 仓库代码级落地点 (`Target Module`)**：`capacity_aware/` (Shapley Coalition-Aware Expert Pruning + Token Capacity Bound)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-27_ai_paper_notes.md`


---

### 2.6 [2026-09-27] L2R: Low-Rank and Lipschitz-Controlled Routing for Mixture-of-Experts

* **论文信息**：Minghao Yang, Ren Togo, Guang Li, Takahiro Ogawa, Miki Haseyama (`arXiv:2601.21349`, 2026-01)
* **核心关键词**：MoE Routing Geometry、Low-Rank Latent Space、Lipschitz Continuity、Saturated Inner-Product Scoring (SIPS)、Multi-Anchor Routing

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|          L2R: Low-Rank & Lipschitz-Controlled MoE Routing Architecture            |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|                        Token Hidden State h \in R^d                               |
|                                     |                                             |
|                                     v                                             |
|        +---------------------------------------------------------+                |
|        | 1. Shared Low-Rank Latent Projection (低秩路由子空间映射)|                |
|        |    z = P h \in R^r   (r << d, orthogonalized P P^T = I_r)|                |
|        |    Filters out high-dimensional isotropic noise         |                |
|        +---------------------------------------------------------+                |
|                                     |                                             |
|                                     v                                             |
|        +---------------------------------------------------------+                |
|        | 2. Multi-Anchor Expert Prototypes (多锚点专家原型表示)   |                |
|        |    Each Expert e has M low-rank anchors: {u_{e,m}}_{m=1}^M               |
|        +---------------------------------------------------------+                |
|                                     |                                             |
|                                     v                                             |
|        +---------------------------------------------------------+                |
|        | 3. Saturated Inner-Product Scoring (SIPS Lipschitz 控制) |                |
|        |    s_{e,m}(z) = \tau \cdot \tanh( <z, u_{e,m}> / (\tau \|z\|_\gamma) )   |
|        |    Explicitly bounds || \nabla_h s_e(h) ||_2 <= L_lip    |                |
|        +---------------------------------------------------------+                |
|                                     |                                             |
|                                     v                                             |
|             SoftMax / Top-k Selection ---> Stable Expert Dispatch                 |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **高维线性路由的三大几何病态**：标准稀疏 MoE 普遍采用单层线性投影 $s(h) = W _ r h \in \mathbb{R}^N$ 作为路由器（Router）。作者从表示几何角度指出高维空间 $d \gg N$ 中的线性内积路由存在三大固有缺陷：
  1. **维度失配与噪声过拟合（Representation Mismatch）**：Token 隐状态 $h \in \mathbb{R}^d$ 包含了大量与任务路由无关的词法/位置高频噪声，全维内积导致路由决策极易受正交噪声方向干扰。
  2. **高维角度集中现象（Angular Concentration）**：随着层深增加，Transformer 隐状态落入狭窄的各向异性锥（Anisotropic Cone），不同专家路由向量与 $h$ 的余弦相似度高度趋同，导致门控分布扁平化或赢家通吃。
  3. **范数敏感与 Lipschitz 失控（Scale Sensitivity）**：当隐状态范数 $\Vert h\Vert _ 2$ 在深层或长序列中剧烈膨胀时，未受控的内积 $w _ e^\top h$ 会使 Softmax 进入指数饱和区，微小输入扰动即可引发离散 Top- $k$ 路由集合翻转（Routing Instability）。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **共享低秩潜空间路由投影（Low-Rank Latent Routing Space）**：
   引入行正交低秩投影矩阵 $P \in \mathbb{R}^{r \times d}$ （ $r \ll d$ ，例如 $d=2048, r=64$ ），将隐状态 $h$ 压缩至低秩判别子空间：

$$
z = P h \in \mathbb{R}^r, \qquad \mathcal{L} _ {\text{orth}} = \Vert P P^\top - I _ r \Vert _ F^2
$$

2. **饱和内积打分与显式 Lipschitz 边界控制（Saturated Inner-Product Scoring, SIPS）**：
   为消除隐状态径向范数 $\Vert h\Vert _ 2$ 暴涨导致的路由震荡，L2R 设计了带阻尼范数归一化与双曲正切饱和的打分算子：

$$
\phi _ {\text{SIPS}}(z, u _ e) = \tau \cdot \tanh\left( \frac{\langle z, u _ e \rangle}{\tau \left(\sqrt{\Vert z\Vert _ 2^2 + \epsilon^2}\right)^\gamma \left(\sqrt{\Vert u _ e\Vert _ 2^2 + \epsilon^2}\right)^\gamma} \right)
$$

   其中 $\tau > 0$ 控制饱和软边界， $\gamma \in [0, 1]$ 控制径向尺度不变性强度（当 $\gamma=1$ 时退化为受控余弦路由）。利用 $\text{sech}^2(x) \le 1$ 及正交投影 $\Vert P\Vert _ 2 = 1$ ，可严格证明打分函数对原始输入 $h$ 的梯度范数（即局部 Lipschitz 常数）存在显式解析上界：

$$
\left\lVert \nabla _ h \phi _ {\text{SIPS}}(P h, u _ e) \right\rVert _ 2 \le \Vert P\Vert _ 2 \cdot \frac{\Vert u _ e\Vert _ 2^{1-\gamma}}{\epsilon^\gamma} = L _ {\text{lip}}
$$

   从而从数学上保证了有界输入扰动 $\Vert\delta h\Vert _ 2 \le \delta$ 不会引发路由分数的剧烈跳变。
3. **多锚点专家表达（Multi-Anchor Routing）**：
   由于单个专家往往需要处理多模态或多子类语义簇，在低秩空间 $\mathbb{R}^r$ 中为每个专家分配 $M$ 个子锚点 $\lbrace u _ {e,m}\rbrace _ {m=1}^M \subset \mathbb{R}^r$ （参数量仅为 $N \times M \times r \ll N \times d$ ），通过 Log-Sum-Exp 软聚合计算专家总得分：

$$
s _ e(h) = \frac{1}{\beta} \log \sum _ {m=1}^M \exp\Big( \beta \cdot \phi _ {\text{SIPS}}(P h, u _ {e,m}) \Big)
$$

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* **语言与视觉双模态全面验证**：在基于 **OLMoE** 的语言模型预训练/微调以及 **ImageNet** 视觉 MoE 骨干网络上，L2R 将路由器参数量削减 **60%–75%**，同时在相同激活专家预算下将下游任务困惑度（PPL）降低 `0.42–0.68`，ImageNet Top-1 准确率提升 `+1.3%`。
* **路由稳定性与负载均衡双升**：在对抗性高斯扰动测试下，L2R 的 Top- $k$ 路由翻转率（Routing Flip Rate）比标准线性 Router 降低 **47%**，专家负载熵（Routing Entropy）更加接近理想均匀分布，无需强依赖破坏主任务梯度的大权重 Load-Balancing 辅助损失。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
1. **与 *Router-Tuning* (EMNLP 2025) & *Capacity-Aware Inference* (ICLR 2026) 的直接耦合**：
   * 我们在 *Router-Tuning* 中提出仅微调轻量路由器即可解锁深层稀疏网络潜力，但在极低资源或长上下文微调中，全维线性路由器容易过拟合表面范数特征。将 L2R 的 **SIPS + 低秩多锚点路由** 作为 *Router-Tuning* 的参数化形式，不仅能将可训练参数再降一个数量级，还能利用 Lipschitz 边界防止微调过程中的路由坍缩。
2. **与 *Transformer-Geometry* (`arXiv:2609.15975`, EMNLP 2026) & `MerA` SVD 初始化的深刻同构**：
   * L2R 发现的“径向范数敏感性（Scale Sensitivity）”与我们在 *Transformer-Geometry* 及 `ads-rsi`（定律 ADS-RSI-1：Scale-Cancellation）中揭示的**“深层残差流径向范数 $\Vert h\Vert _ 2$ 掩盖切向语义方向 $h / \Vert h\Vert _ 2$ ”**完全一致！此外，在将稠密模型或预训练线性路由器 $W _ r \in \mathbb{R}^{N \times d}$ 转化为 L2R 路由器时，无需随机初始化 $P$ ，可直接调用我们的 **`MerA` 数据感知激活协方差 SVD（Activation-Covariance SVD）** 提取前 $r$ 个主奇异方向初始化 $P$ ，实现零冷启动抖动的低秩 Lipschitz 路由升级。

---

> [!TIP]
> **🎯 `Capacity-Aware-MoE` 仓库代码级落地点 (`Target Module`)**：`capacity_aware/` (Low-Rank Latent Anchor + SIPS Lipschitz Router)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-27_ai_paper_notes.md`


---

### 2.7 [2026-09-26] 🔄 *LoopMoE: Unifying Iterative Computation with Mixture-of-Experts for Language Modeling*
> **聚焦领域**：Looped Transformers · Mixture of Experts (MoE) · Iterative Depth Scaling · Weight Sharing  
> **arXiv**：[`arXiv:2606.04438`](https://arxiv.org/abs/2606.04438)

```
  输入表征 h^{(0)} ──► [ 循环步 t = 1..K : IterAdaLN(h, t) 轮次特征调制 ]
                                       │
                                       ▼
                     [ 共享 MoE 路由层: Top-k 稀疏专家激活 + 跨循环容量均衡 ]
                                       │
                                       ▼
                     [ 解耦总参数量 P 与单 Token 算力 FLOPs (同参数量 PPL 显著降低) ]
```

#### 🎯 背景与痛点剖析 (Problem Statement)
* **权重复用与轮次角色分化的矛盾**：在 Looped Transformer 中，直接将同一组 Transformer 块重复循环 $K$ 次，虽然能以 $O(1)$ 参数开销换取 $O(K)$ 的等效推理深度，但会导致两个严重退化：（1）不同循环步 $t \in \lbrace1, \dots, K\rbrace$ 缺乏步间身份区分，引发梯度震荡与隐状态平行分量 $\Delta h _ \parallel$ 爆炸；（2）若将循环架构直接与 MoE 结合，不同循环步会争抢同一批头部 Expert，导致严重的跨循环路由坍缩（Cross-Loop Routing Collapse）。

#### 💡 核心方法与底层数学实现 (Mathematical Formulations)
1. **迭代步自适应层归一化 (Iteration-Adaptive LayerNorm, `IterAdaLN`)**：
   - 为第 $t$ 次循环引入轻量级步间嵌入向量 $e _ t \in \mathbb{R}^d$ ，对共享主干的归一化层施加轮次特异性的仿射缩放与偏移调制：

$$
\text{IterAdaLN}(h^{(t)}, t) = \big(1 + \gamma(e _ t)\big) \odot \frac{h^{(t)} - \mu}{\sigma} + \beta(e _ t)
$$

   - 通过仅占总参数量 $<0.1$ % 的步间条件调制参数，赋予共享 MoE 块在不同循环深度下截然不同的几何变换角色。
2. **跨循环容量感知负载均衡 (Iteration-Aware Capacity Balancing)**：
   - 设第 $t$ 步第 $i$ 个专家的路由门控概率为 $p _ i^{(t)}(x)$ ，论文将辅助负载均衡损失扩展至循环时间轴与批次维度的联合分布上，防止特定专家在连续多次循环中被重复饱和激活。

#### 📊 关键实验与结论 (Experiments & Findings)
* **等参数量与等 FLOPs 双向碾压**：在语言建模基准与常识推理任务上，循环 $K=2\sim 4$ 步的 `LoopMoE` 在相同活跃参数量下显著优于标准稠密 Looped 模型，且在相同总参数预算下逼近非共享深层 MoE 模型的困惑度（PPL）上限。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发 (Relevance & Synergy with Our Works)
* **🎯 锚定代表作与在研主线**：
  * [Paper #16: *Disentangling Representation Evolution in Transformers through Directional Decomposition* (EMNLP 2026, `arXiv:2609.15975`)]
  * [Paper #11: *Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture of Experts* (ICLR 2026)]
  * [Paper #10: *Router-Tuning for Dynamic Mixture of Experts* (EMNLP 2025)]
  * [Active Line: *Physical AI / VLA-Loop (Stage-Wise Multi-LoRA Residual Boost & Adaptive Layer Looping)*]
* **🔬 机理对比与技术演进**：
  * `LoopMoE` 采用 `IterAdaLN`（逐通道对角缩放 $\gamma(e _ t)$ ）来区分不同循环轮次；而我们在 `VLA-Loop`（见 W39 研发笔记 9/22–9/23）中提出**用极小秩的 Stage-Wise LoRA 去编辑共享主干的每一次循环**，并进一步推进到了**逐层自适应决定是否 Loop**；
  * 从我们 *Transformer-Geometry (EMNLP 26)* 的正交方向分解视角来看，`IterAdaLN` 仅在归一化后施加坐标轴缩放，主要调节平行缩放分量 $\Delta h _ \parallel$ ；而我们的 **共享主干 + 轮次轻量 LoRA ( $\Delta W _ t = B _ t A _ t$ )** 则能直接在子空间中引入低秩正交旋转分量 $\Delta h _ \perp$ ，在表达能力上严格包含 `IterAdaLN`！
* **💡 下一阶段研究（Next Research Directions）落地启发**：
  * 在撰写 `Physical AI` (MLSys) 论文的 Loop 章节时，可将 `LoopMoE` 的 `IterAdaLN` 作为轻量轮次调制的文献对照基准，用实验展示我们 **“共享主干 + MERA 初始化的轮次小 LoRA + 逐层自适应 Loop 路由”** 相比单纯 LayerNorm 调制的显著几何表达优势。

---

> [!TIP]
> **🎯 `Capacity-Aware-MoE` 仓库代码级落地点 (`Target Module`)**：`capacity_aware/` (Recurrent Step-Specific Low-Rank Calibrated MoE)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-26_ai_paper_notes.md`


---

### 2.8 [2026-09-26] 🤖 *VLA-Pruner: Temporal-Aware Dual-Level Visual Token Pruning for Efficient Vision-Language-Action Inference*
> **聚焦领域**：Vision-Language-Action (VLA) · Embodied AI · Visual Token Pruning · Temporal Consistency  
> **arXiv**：[`arXiv:2511.16449`](https://arxiv.org/abs/2511.16449)

```
  连续控制帧视觉流 ──► [ 层级一 (Prefill): 跨模态指令-视觉语义重要度评估 ]
                                           │
                                           ▼
                       [ 层级二 (Decode): 时域指数平滑动作相关性追踪 S_t = λS_{t-1} + (1-λ)A_t ]
                                           │
                                           ▼
                       [ Combine-then-Filter 联合剪枝: 避免浅层误删关键操控锚点 ]
```

#### 🎯 背景与痛点剖析 (Problem Statement)
* **“语义显著性”与“动作控制必要性”的错位（Semantic-Action Gap）**：在机械臂精细操控任务（如 LIBERO）中，单帧静态视觉编码器认为显著的背景物体，未必是当前动作步（Action Chunk）夹爪需要接触的目标；反之，若在浅层仅凭静态视觉注意力盲目丢弃大量 Patch Token，会导致深层 Action Expert 丢失空间几何锚点，引发轨迹剧烈抖动。

#### 💡 核心方法与数学实现 (Mathematical Formulations)
1. **双层重要度融合准则 (Combine-then-Filter Dual-Level Criterion)**：
   - 同时提取语言指令在 Prefill 阶段对第 $i$ 个视觉 Token 的语义关注度 $I _ {\text{sem}}^{(i)}$ ，以及解码器生成动作 Token 时的交叉注意力得分 $I _ {\text{act}, t}^{(i)}$ ；
2. **跨时间步动作相关性平滑 (Temporal Action Smoothing)**：
   - 利用连续控制帧之间的时间连续性，引入历史动作注意力动量缓存：

$$
\tilde{I} _ {\text{act}, t}^{(i)} = \lambda \tilde{I} _ {\text{act}, t-1}^{(i)} + (1 - \lambda) I _ {\text{act}, t}^{(i)}
$$

   - 仅保留综合得分 $S _ t^{(i)} = I _ {\text{sem}}^{(i)} \cdot \tilde{I} _ {\text{act}, t}^{(i)}$ 最高的视觉 Token 子集。

#### 📊 关键实验与结论 (Experiments & Findings)
* 在 OpenVLA 与主流机器人操控基准（LIBERO-Spatial / Object / Goal / Long）上，剔除 **50%–75% 视觉 Token** 仍保持与全量 Token 持平的任务成功率，端到端控制频率显著提升。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发 (Relevance & Synergy with Our Works)
* **🎯 锚定代表作与在研主线**：
  * [Active Line: *Physical AI (`VLADrop` / `DTR` / `HiSTrim` Exclude-Self Value-Space Perp KV256)*]
  * [Paper #8: *Understanding and Harnessing Sparsity for Unified Multimodal Models* (TMLR 2026)]
  * [Paper #9: *Uncovering the Redundancy in Transformers via Layer Dropping* (TMLR 2025)]
* **🔬 机理对比与技术演进**：
  * 我们在 W38 周记（9/15–9/17）中深刻总结了两条核心定律：（1）**Layer 0（纯 ID Embedding、尚未经过上下文交互）绝不能直接做激进 Token Drop**，必须在表征充分上下文化之后再按浅层保守、深层激进的曲线压缩；（2）**VLA 的鲁棒性来源于三个时间尺度的“伤口愈合（Wound Healing）”纠错通道**（步内注意力、步间去噪、episode 内周期性视觉重锚）；
  * `VLA-Pruner` 的时域平滑动量 $\tilde{I} _ {\text{act}, t}$ 恰恰显式利用了我们指出的第三层“episode 内时域连续重锚”特性！
* **💡 下一阶段研究（Next Research Directions）落地启发**：
  * 在 `Physical AI` (MLSys) 论文中，可将 `VLA-Pruner` 纳入 Related Work 与对比讨论，突出我们 **全栈四维协同压缩（数据 DTR + Token `HiSTrim` + 层 `VLADrop/Loop` + 步数 `SnapFlow` 单步蒸馏）** 相比单一视觉 Token 剪枝在真实硬件延迟（Batch=1 访存带宽瓶颈）上的系统级代差优势。

---

## 🔥 板块二：全球前沿热点精选 (Trending Frontier)

---

> [!TIP]
> **🎯 `Capacity-Aware-MoE` 仓库代码级落地点 (`Target Module`)**：`capacity_aware/` & `scripts/` (`CASE-Lab-UMD/Capacity-Aware-MoE`)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-26_ai_paper_notes.md`


---

### 2.9 [2026-09-26] 🧩 *MoE-nD: Per-Layer Mixture-of-Experts Routing for Multi-Axis KV Cache Compression*
> **聚焦领域**：Multi-Axis KV Cache Compression · Per-Layer Routing · Heterogeneous Quantization  
> **arXiv**：[`arXiv:2604.17695`](https://arxiv.org/abs/2604.17695)

* **核心痛点**：Transformer 不同层对 Token 驱逐（Eviction）、低比特量化（Quantization）和低秩分解（Low-Rank Projection）的敏感度截然不同，全局采用单一压缩轴或统一压缩率必然在敏感层造成性能崩塌。
* **具体做法**：将多维 KV 压缩配置（如 $\left(k\text{-bits}, v\text{-bits}, \text{keep-ratio}\right)$ ）构建为离散专家池，利用轻量级逐层 MoE 路由器根据输入分布动态为每一层分配最优混合压缩算子，在满足全局显存上界约束的同时最大化输出保真度。
* **结论**：在长文本理解与代码生成任务上实现 ** $3\times\sim 20\times$ ** 极限显存压缩且几乎无损精度，验证了我们关于**“各层表征冗余度非均匀分布，压缩率应沿层自适应分配”**的核心判断。

---

> [!TIP]
> **🎯 `Capacity-Aware-MoE` 仓库代码级落地点 (`Target Module`)**：`capacity_aware/` (Cartesian Product Multi-Dimensional Expert Grid)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-26_ai_paper_notes.md`


---

### 2.10 [2026-09-24] DriveMoE: Mixture-of-Experts for Vision-Language-Action Model in End-to-End Autonomous Driving

* **论文信息**：`arXiv:2505.16278` (2025/2026)
* **核心关键词**：End-to-End Autonomous Driving、Scene-Specialized Vision MoE、Skill-Specialized Action MoE、Flow-Matching Planner

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|       DriveMoE: Dual-Stage Vision & Action MoE for Autonomous Driving VLA         |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  6-Camera Surround View ---> [Stage 1: Scene-Specialized Vision MoE]              |
|                              Routes camera views (Front/Rear/Corner) & weather    |
|                                        |                                          |
|                                        v                                          |
|  Navigation Command + Ego State ---> [Stage 2: Skill-Specialized Action MoE]      |
|                              Built on Flow-Matching Planner: routes to specialized|
|                              experts for Lane-Keep, Unprotected Turn, Avoidance   |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **端到端自动驾驶中的长尾机动动作被直行样本淹没**：在驾驶数据集中，90% 以上为简单直行跟车，稠密 VLA 规划器在面对无保护左转、施工改道紧急避障等长尾场景时因梯度被简单样本主导而表现迟钝。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **视角感知视觉 MoE + 驾驶技能流匹配动作 MoE**：
   动作生成头采用条件流匹配架构，每个专家 $e \in \lbrace1, \dots, E _ {\text{act}}\rbrace$ 对应特定驾驶技能子流形，由高阶导航意图 $c _ {\text{nav}}$ 与场景特征联合路由：

$$
v _ {\text{drive}}(a _ \tau, \tau \mid z _ {\text{scene}}, c _ {\text{nav}}) = \sum _ {e \in \text{Top-}k} g _ e(z _ {\text{scene}}, c _ {\text{nav}}) \cdot v _ e(a _ \tau, \tau \mid z _ {\text{scene}})
$$

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在 **Bench2Drive** 闭环评测与 **nuScenes** 开环基准上，DriveMoE 将复杂交叉路口与紧急避障长尾场景的驾驶得分（Driving Score）大幅提升 **`+9.4` 分**，碰撞率降低 **36%**。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **与我们 `ads-rsi`（长尾稀疏切片保护）及 `vla-distillation` 的直接联动**：在多技能流匹配动作头中通过技能感知 LoRA 专家隔离高频常规动作与长尾极限动作，可有效防止蒸馏过程中的长尾退化。

---

> [!TIP]
> **🎯 `Capacity-Aware-MoE` 仓库代码级落地点 (`Target Module`)**：`capacity_aware/` & `scripts/` (`CASE-Lab-UMD/Capacity-Aware-MoE`)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-24_ai_paper_notes.md`


---

### 2.11 [2026-09-23] HiMoE-VLA: Hierarchical Mixture-of-Experts for Generalist Vision-Language-Action Policies

* **论文信息**：`arXiv:2512.05693` (2025/2026)
* **核心关键词**：Hierarchical MoE、Generalist VLA Policy、Task-Skill Decoupled Routing、Gradient Conflict Mitigation

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|       HiMoE-VLA: Hierarchical Mixture-of-Experts for Generalist VLA Policies      |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Language Goal + Visual State ---> [Level-1: Task/Embodiment Router G_{\text{task}}]|
|                                           |                                       |
|                   Selects Domain Expert Group \mathcal{G}_m                       |
|                                           v                                       |
|         Proprioception + Local Patch ---> [Level-2: Skill Primitive Router G_{\text{skill}}]|
|                                           |                                       |
|                   Activates Fine-Grained Motor Primitives (Reach / Grasp / Place) |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **异构本体与多任务联合训练中的“扁平路由混淆”**：在跨机械臂本体、跨数十种操作任务的通用 VLA 训练中，单层扁平 MoE 路由器容易按表层视觉背景而非底层运动学技能聚类，导致不同任务间出现严重的负迁移。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **双层语义-运动解耦条件路由（Bi-Level Semantic-Kinematic Conditional Routing）**：
   高层路由器 $G _ {\text{task}}(c _ {\text{lang}}, I _ {\text{global}})$ 根据语言指令与全局视觉场景选择任务簇 $m \in \lbrace1, \dots, M\rbrace$ ，低层路由器 $G _ {\text{skill}}^{(m)}(s _ {\text{prop}}, I _ {\text{wrist}})$ 根据本体关节状态与腕部相机高频特征在簇内选择动作基元专家 $e \in \mathcal{E} _ m$ ：

$$
P(e \mid x) = \sum _ {m=1}^M G _ {\text{task}}(m \mid c _ {\text{lang}}, I _ {\text{global}}) \cdot G _ {\text{skill}}^{(m)}(e \mid s _ {\text{prop}}, I _ {\text{wrist}}) \cdot \mathbb{I}(e \in \mathcal{E} _ m)
$$

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在跨 50+ 任务的 Open-X Embodiment 与仿真套件上，HiMoE-VLA 比同激活参数量的稠密 VLA 与单层 MoE-VLA 平均成功率提升 **`+8.7%`**。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **与我们 `ads-rsi` 中的 GemTagger 分层路由与 *Router-Tuning* (EMNLP 2025) 高度契合**：将高层任务上下文路由与底层高频状态路由树状解耦，可大幅提升细粒度专家的专业化纯度。

---

> [!TIP]
> **🎯 `Capacity-Aware-MoE` 仓库代码级落地点 (`Target Module`)**：`capacity_aware/` & `scripts/` (`CASE-Lab-UMD/Capacity-Aware-MoE`)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-23_ai_paper_notes.md`


---

### 2.12 [2026-09-23] D-Cut: Adaptive Verification Depth Pruning for Batched Speculative Decoding

* **论文信息**：`arXiv:2607.14647` (2026-07)
* **核心关键词**：Speculative Decoding、Verification Depth Pruning、Cross-Request Budget Allocation

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|       D-Cut: Adaptive Verification Depth Pruning for Batched Speculative Decoding |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Batched Draft Trees {T_1, ..., T_B} with Draft Confidence Scores {c_1, ..., c_B} |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | Early-Layer Margin Verification (浅层置信度提前决断)                        |  |
|  |    At Intermediate Layer L_{\text{cut}} < L:                                |  |
|  |    If early logit margin \Delta z^{(L_{\text{cut}})} >> \tau_accept or << -\tau_reject:|
|  |    Drop verified/rejected draft tokens from remaining layers L_{\text{cut}}+1..L|
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **批量投机解码验证阶段的深层算力浪费**：在大 Batch 投机解码中，目标大模型需要同时并行验证每个请求的 $K$ 个草稿 Token。实际上，超过 70% 的简单正确草稿或明显错误的草稿在目标模型的前 60% 层就已经毫无悬念地分出胜负，继续让它们跑完后 40% 层纯属浪费。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **基于中间层 Logit 间隔的动态截断准则**：
   在中间探测层 $l _ {\text{probe}}$ ，通过轻量早期退出投影计算草稿 Token $y _ i$ 的对数概率边际 $\Delta _ {i}^{(l)} = \hat{\ell}^{(l)}(y _ i) - \max _ {v \neq y _ i} \hat{\ell}^{(l)}(v)$ 。当 $|\Delta _ {i}^{(l)}| > \gamma _ l$ 时，立即锁定接受/拒绝决策，并将该草稿及其后续依赖子树从第 $l+1 \dots L$ 层的批次张量中动态压缩移除。

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在 Batch Size = 16–64 的生产级投机解码服务中，D-Cut 将验证阶段算力开销削减 **38%**，端到端吞吐在 EAGLE-2 基线上进一步提升 **1.42x**。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **与我们 *Capacity-Aware Inference* (ICLR 2026) & *Layer Dropping* (TMLR 2025) 直接协同**：在多步推理或投机验证中引入中间层间隔早退门控，可显著提升高并发批次下的有效吞吐。

---

> [!TIP]
> **🎯 `Capacity-Aware-MoE` 仓库代码级落地点 (`Target Module`)**：`capacity_aware/` & `scripts/` (`CASE-Lab-UMD/Capacity-Aware-MoE`)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-23_ai_paper_notes.md`


---

### 2.13 [2026-09-21] MoE-FM: Towards Faster Language Model Inference Using Mixture-of-Experts Flow Matching

* **论文信息**：`arXiv:2604.15009` (2026-04)
* **核心关键词**：Mixture-of-Experts Flow Matching、Piecewise-Linear Vector Fields、Latent Flow Language Models

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|          MoE-FM: Mixture-of-Experts Flow Matching for Fast Inference              |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Latent State z_t at Time t ---> Time- & State-Conditioned Router G(z_t, t)       |
|                                        |                                          |
|            +---------------------------+---------------------------+              |
|            v                           v                           v              |
|  [Expert Field v_1(z_t,t)]   [Expert Field v_2(z_t,t)]   [Expert Field v_E(z_t,t)]|
|  (Local Straight Transport)  (Local Straight Transport)  (Local Straight Transport)|
|            +---------------------------+---------------------------+              |
|                                        |                                          |
|                                        v                                          |
|            Composite Velocity v(z_t, t) = \sum_{e \in Top-k} g_e(z_t,t) v_e(z_t,t)|
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **全局单一速度场拟合多峰分布时的轨迹弯曲（Trajectory Curvature）**：当使用单个稠密网络拟合高度多模态的语言或动作分布时，不同模式的流线在中间时刻发生交叉，迫使平均速度场严重弯曲，从而需要数十步 ODE 积分才能避免离散化截断误差。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **分片局部直线化的专家向量场分解**：
   将全局速度场 $v(z _ t, t)$ 分解为 $E$ 个局部专家速度场的稀疏组合，并加入专家内轨迹曲率惩罚以促使每个专家负责的局部区域保持直线传输：

$$
\mathcal{L} _ {\text{MoE-FM}} = \mathbb{E} _ {t, z _ 0, z _ 1} \left[ \left\lVert \sum _ {e \in \text{Top-}k} g _ e(z _ t, t) v _ e(z _ t, t) - (z _ 1 - z _ 0) \right\rVert _ 2^2 + \mu \sum _ {e \in \text{Top-}k} g _ e(z _ t, t) \big\Vert \partial _ t v _ e(z _ t, t) \big\Vert _ 2^2 \right]
$$

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在潜空间语言生成与多模态推理中，MoE-FM 在仅使用 **2–4 步 NFE** 时即可达到单稠密流模型 16–32 步的生成质量，推理延迟降低 **3.8x**。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **对我们 `vla-distillation` 多模态动作块（Action Chunk）生成的启发**：机器人操作往往存在“从左侧绕行”或“从右侧抓取”的多峰分叉模式，引入时间与状态联合门控的轻量级 LoRA 专家速度场可有效消除多峰平均导致的直线穿越障碍物问题。

---

> [!TIP]
> **🎯 `Capacity-Aware-MoE` 仓库代码级落地点 (`Target Module`)**：`capacity_aware/` & `scripts/` (`CASE-Lab-UMD/Capacity-Aware-MoE`)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-21_ai_paper_notes.md`


---

### 2.14 [2026-09-20] CARE: Spend Experts Where You Are Unsure — Confidence-Adaptive Routing for MoE-LoRA

* **论文信息**：`arXiv:2607.26052` (2026-07)
* **核心关键词**：Confidence-Adaptive Routing、MoE-LoRA、Nucleus Expert Activation、Router Uncertainty Entropy

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|            CARE: Confidence-Adaptive Routing for Mixture-of-Experts               |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Token Hidden State h_t ---> Router Probabilities p_t = Softmax(W_r h_t) \in \Delta^E|
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 1. Router Uncertainty Quantification (路由分布置信度/不确定性度量)          |  |
|  |    Sort probabilities: p_{t,(1)} >= p_{t,(2)} >= ... >= p_{t,(E)}           |  |
|  |    High confidence (peaked p_t) -> K_t = 1; High entropy -> K_t = K_{\max}  |  |
|  +-----------------------------------------------------------------------------+  |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 2. Nucleus & Margin-Gated Dynamic Top-K(t) Selection                        |  |
|  |    K_t = \min \{ k \in [K_{\min}, K_{\max}] : \sum_{i=1}^k p_{t,(i)} >= \tau_p|
|  |               \text{ or } p_{t,(k)} - p_{t,(k+1)} >= \tau_m \}              |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **静态 Top- $k$ 路由的算力错配**：标准 MoE 对序列中的每一个 Token（无论是标点符号、常见停用词，还是复杂的逻辑转折词）均无差别地激活固定数量 $k$ 个专家。对于路由器高度确信的简单 Token（例如 $p _ {t,(1)} > 0.85$ ），强制拉起第 $2 \dots k$ 个低概率专家不仅浪费算力，还会引入长尾噪声干扰；而对于处于知识边界的模糊 Token，固定 $k$ 个专家又不足以覆盖多维语义假设。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **累积概率核与边际跳变双门控（Nucleus & Margin Gated Dynamic $K _ t$ ）**：
   将排序后的专家门控概率记为 $p _ {t,(1)} \ge p _ {t,(2)} \ge \dots \ge p _ {t,(E)}$ 。CARE 为每个 Token $t$ 动态分配激活专家个数 $K _ t \in [K _ {\min}, K _ {\max}]$ ：

$$
K _ t = \min \left\lbrace k \in \lbrace K _ {\min}, \dots, K _ {\max}\rbrace \middle| \sum _ {i=1}^k p _ {t,(i)} \ge \tau _ {\text{nuc}} \lor \big(p _ {t,(k)} - p _ {t,(k+1)}\big) \ge \tau _ {\text{margin}} \right\rbrace
$$

2. **零训练即插即用温度校准（Temperature Calibration under Global FLOPs Target）**：
   给定目标平均激活专家预算 $\bar{K} _ {\text{target}}$ ，在校准集上通过单标量温度 $\beta$ 缩放路由 logits $p _ t(\beta) = \text{Softmax}(W _ r h _ t / \beta)$ ，满足 $\mathbb{E} _ t[K _ t(\beta)] = \bar{K} _ {\text{target}}$ 。

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在多任务 MoE-LoRA 与稀疏 MoE 语言模型上，CARE 在削减 **32%–45% 平均专家激活 FLOPs** 的同时，在常识推理、代码与数学基准上全面持平甚至超越固定 Top- $k$ 基线（`+0.9%` 平均准确率）。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **与我们 *Capacity-Aware Inference* (ICLR 2026) & *Router-Tuning* (EMNLP 2025) 的协同**：可将 CARE 的 Token 级置信度核门控（Nucleus Routing）与我们在 ICLR 2026 中提出的硬件容量感知丢弃/重路由（Capacity-Aware Dropping）级联，在软件置信度与硬件队列容量两个维度同时实现最优分配。

---

> [!TIP]
> **🎯 `Capacity-Aware-MoE` 仓库代码级落地点 (`Target Module`)**：`capacity_aware/` (Dual-Threshold Dynamic Top-k Expert Allocation)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-20_ai_paper_notes.md`


---

### 2.15 [2026-09-19] REAP: Router-Weighted Expert Activation Pruning for Sparse MoE Models

* **论文信息**：`arXiv:2510.13999` (2025/2026)
* **核心关键词**：MoE Expert Pruning、Router Gate Weighting、Expert Activation Norm、Generative Reasoning Preservation

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|            REAP: Router-Weighted Expert Activation Pruning Pipeline               |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Token x_t ---> Router Gate g_{t,e} = Softmax(W_r x_t)_e                          |
|            ---> Active Expert Output E_e(x_t) = W_down (SiLU(W_gate x_t) * W_up x_t)|
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | Joint Multiplicative Saliency Metric (门控权重 x 激活输出范数联合度量)       |  |
|  |    I_{\text{REAP}}(e) = \mathbb{E}_{x_t \in \mathcal{A}_e} [ g_{t,e} \cdot  |  |
|  |                         \| E_e(x_t) \|_2 ] \cdot \hat{P}(e \in \text{Top-}k)|  |
|  +-----------------------------------------------------------------------------+  |
|                                        |                                          |
|                                        v                                          |
|       Prune Lowest-I_{\text{REAP}} Experts ---> Gate Renormalization (Zero-Train) |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **仅凭路由频率或专家合并（Expert Merging）在生成任务上的失效**：传统 MoE 压缩常根据专家被选中的频次 $\hat{P}(e \in \text{Top-}k)$ 剪枝，或将相似专家权重线性平均（Merging）。作者发现：（1）在代码生成与数学推理等生成任务中，线性合并两个非线性 SwiGLU 专家的权重会破坏内部特征门控对齐，引起特征坍缩；（2）许多高频被选中的专家其输出向量范数 $\Vert E _ e(x _ t)\Vert _ 2$ 极小（充当空操作/恒等缓冲），而真正决定推理跃迁的专家则具有高门控权重乘以高输出激活范数。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **路由器加权激活范数重要性（Router-Weighted Activation Norm）**：
   由于 MoE 层的精确输出增量为 $\Delta h _ t = \sum _ {e \in \text{Top-}k(x _ t)} g _ {t,e} E _ e(x _ t)$ ，单个专家 $e$ 从激活集合中移除时引起的期望一阶残差上界正比于 $g _ {t,e} \Vert E _ e(x _ t)\Vert _ 2$ 。因此 REAP 定义专家 $e$ 的全局重要性为：

$$
\mathcal{I} _ {\text{REAP}}(e) = \frac{1}{|\mathcal{D} _ {\text{cal}}|} \sum _ {t=1}^{|\mathcal{D} _ {\text{cal}}|} \mathbb{I}\big(e \in \text{Top-}k(x _ t)\big) \cdot g _ {t,e} \cdot \big\Vert E _ e(x _ t) \big\Vert _ 2
$$

2. **保留集门控重归一化（Post-Pruning Gate Renormalization）**：
   裁剪掉得分最低的专家集合 $\mathcal{E} _ {\text{prune}}$ 后，对剩余专家集合 $\mathcal{E} _ {\text{keep}}$ 的门控权重执行保和重归一化 $\tilde{g} _ {t,e} = \frac{g _ {t,e}}{\sum _ {j \in \text{Top-}k(x _ t) \cap \mathcal{E} _ {\text{keep}}} g _ {t,j}}$ ，以补偿被移除专家的幅度损失。

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在 **Mixtral-8x7B**、**DeepSeek-MoE-16B** 与 **Qwen1.5-MoE-A2.7B** 上，REAP 在 **25%–37.5% 专家剪枝率**下，在 GSM8K 与 HumanEval 生成基准上大幅超越各类专家合并算法（HC-SMoE、M-SMoE）达 **`+8.5%` 至 `+14.2%`**。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **与 *Capacity-Aware Inference* (ICLR 2026) & *Transformer-Geometry* (EMNLP 2026) 的结合**：
  * REAP 揭示了 $\Vert g _ {t,e} E _ e(x _ t)\Vert _ 2$ 相比单纯门控概率 $g _ {t,e}$ 的优越性。结合我们的 *Transformer-Geometry*，我们可以进一步将 $\Vert E _ e(x _ t)\Vert _ 2$ 替换为正交切向范数 $\Vert P _ \perp(h _ t) E _ e(x _ t)\Vert _ 2$ ，避免那些仅沿当前残差方向做无效径向放大的专家占据高分。

---

> [!TIP]
> **🎯 `Capacity-Aware-MoE` 仓库代码级落地点 (`Target Module`)**：`capacity_aware/` (Gate-Probability × Expert-Activation Norm Pruning)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-19_ai_paper_notes.md`


---

### 2.16 [2026-09-18] 🧩 *MoE-Tile: Warp-Aligned Tensor Slicing for Zero-Overhead Dynamic Sparse Routing on Modern Accelerators*
> **聚焦领域**：Mixture of Experts (MoE) · GPU Kernel Optimization · Warp Divergence · Hardware-Aware Sparsity  
> **arXiv**：[`arXiv:2609.09112`](https://arxiv.org/abs/2609.09112)

```
  动态 Token 路由序列 ──► [ 块级非对齐碎片 (Warp 严重分化) ] ──► 计算利用率 38%
                                      │
                                      ▼
                      [ MoE-Tile 算子: Warp-Aligned 2D Slicing ]
                                      ├── 线程块内 128x128 Tile 对齐填充
                                      └── 零开销 TMA (Tensor Memory Accelerator) 异步流水
                                      ▼
                    [ 算子级 GEMM 吞吐达 89% 理论峰值 (提速 2.34x) ]
```

#### 🎯 背景与硬件级痛点剖析 (Hardware Bottleneck)
* **动态门控与 GPU 线程块的天然矛盾**：MoE 模型的 Top- $k$ 门控路由将不同数量的 Token 动态分发给不同 Expert。在 GPU 底层执行专家 FFN 矩阵乘（GEMM）时，每个 Expert 分配到的实际 Token 数（Batch $M _ e$ ）并非硬件友好的 128 或 256 的倍数，导致大量 Warp 处于空转分化（Warp Divergence）状态，且引发非连续非对齐的显存搬运（Uncoalesced Memory Access），Tensor Core 实际利用率极低。

#### 💡 核心方法与原文底层工程实现 (Detailed System Mechanism)
1. **Warp 对齐二维分块调度器 (Warp-Aligned 2D Tile Slicer)**：
   - 设第 $e$ 个 Expert 接收到的 Token 数量为 $M _ e$ ，隐藏维度为 $K$ 与 $N$ ；
   - 传统实现采用 Padding 将 $M _ e$ 补齐到固定上界（造成显存与计算浪费），或采用 Ragged Batch（引发线程分化）；
   - 原文提出跨 Expert 全局排队与 Tile 重映射机制：将所有专家的计算任务切分为固定大小的硬件微块 $\mathcal{T} _ {i,j} \in \mathbb{R}^{128 \times 128}$ ，将跨 Expert 的边界碎片（Tail Residuals）打包组合进统一的共享微块中执行。
2. **硬件 TMA 异步流水线重叠 (Asynchronous TMA Pipelining)**：
   - 利用现代 GPU（Hopper/Blackwell）的 Tensor Memory Accelerator（TMA），在 Shared Memory 与 Global Memory 之间构建三级流水线缓冲，将 Tile 索引寻址重排序开销完全隐藏在 FFN 计算延迟内部。

#### 📊 关键实验与结论 (Experiments & Findings)
* **硬件测试平台**：NVIDIA H100 80GB SXM5 与 B200 GPU 集群；
* **测试模型**：DeepSeek-V2/V3 (236B/671B)、Mixtral-8x22B；
* **实测性能**：
  * **算子级 GEMM 计算吞吐**：相比标准 Megatron-LM 与 vLLM MoE 算子，计算吞吐提升 **2.34 倍**，Tensor Core 利用率从 38.2% 提升至 **89.1%**；
  * **端到端端 Decode 延迟**：Token 生成阶段延迟降低 **43.5%**，完全消除了动态路由带来的硬件抖动。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发 (Relevance & Synergy with Our Works)
* **🎯 锚定代表作**：
  * [Paper #11: *Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture of Experts* (ICLR 2026)]
  * [Paper #10: *Router-Tuning for Dynamic Mixture of Experts* (EMNLP 2025)]
  * [Paper #5: *MEO: Mixture of Experts Optimization* (EMNLP 2023)]
* **🔬 机理对比与技术演进**：
  * 我们在 *Capacity-Aware Inference (ICLR 26)* 中从**分布式全局宏观调度层面**定义了动态 Capacity Factor 与 Token 溢出分配策略；
  * *MoE-Tile* 则在**单卡底层 CUDA 算子与微架构 Tile 粒度**上解决了非规整 Token 批处理的执行开销；
* **💡 下一阶段研究（Next Research Directions）落地启发**：
  * 可将我们 ICLR 26 的全局分布式调度器与 MoE-Tile 底层 Triton/CUDA 算子进行纵向打通：由我们算法在上层输出动态均衡的 Expert 负载约束，下层由 MoE-Tile 执行 128 对齐的极速计算，构建从分布式集群到单卡底层内核的端到端超高效 MoE 推理栈。

---

> [!TIP]
> **🎯 `Capacity-Aware-MoE` 仓库代码级落地点 (`Target Module`)**：`capacity_aware/` & `scripts/` (Warp-Aligned 2D Tile Scheduling & Zero-Slack Capacity)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-18_ai_paper_notes.md`


---
