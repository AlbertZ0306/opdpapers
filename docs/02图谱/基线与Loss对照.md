# LLM OPD Papers: Baselines / Sampled-Token OPD / Loss Map

更新时间：2026-06-22

来源：基于 `docs/02图谱/失败模式映射.md`、`docs/01筛选/论文筛选.md`、本地 PDF、以及已有问答记录重新整理。未进行 web search。

## 口径

这里把 **sampled-token OPD** 严格定义为：

> student 先生成 on-policy rollout；训练时只在 student 实际采样到的 token `y_t` 上查询 / 使用 teacher log-prob 或 log-ratio，把它作为 token-level advantage、reward、weight 或单样本 KL estimator。

典型形式是：

$$
y \sim \pi_\theta(\cdot \mid x)
$$

$$
A_t
=
\log \pi_T(y_t \mid x,y_{<t})
-
\log \pi_\theta(y_t \mid x,y_{<t})
$$

以及 policy-gradient / weighted likelihood：

$$
L
=
-
\mathbb{E}
\left[
\sum_t
A_t
\log \pi_\theta(y_t \mid x,y_{<t})
\right]
$$

不把下面几类简单归为 sampled-token OPD：

- **full-vocabulary KL / logit distillation**：每个 prefix 上对整个词表算 KL。
- **top-k / subset KL**：只在 teacher 或 student top-k token 集上算近似 KL。
- **response-level reward / rubric / DPO**：监督是 response / preference / rubric 级别，不是 teacher log-prob on sampled token。
- **offline expert-token KD / SFT**：训练 token 来自固定 teacher/expert 数据，不是 student rollout token。

## 总体结论

不是这些论文都使用 sampled-token OPD。

可以粗分为几类：

- **明确 sampled-token / 单样本 log-ratio OPD 为核心**：`2602.12125`, `2603.11137`, `2604.03128` 的 RLSD, `2604.13010`, `2604.13016` 的部分实验, `2605.03677`, `2605.06387` 的 OPD 分支, `2605.10781` 的 RLRT。
- **on-policy rollout 但不是严格 sampled-token**：`2601.18734`, `2601.20802`, `2602.12275`, `2603.07079`, `2604.04461`, `2604.10688`, `2604.14084`, `2605.01347`, `2605.06597`。这些通常使用 full-vocab KL、top-k KL、weighted branch loss、或多组件 self-distillation。
- **不是 token-level white-box OPD**：`2604.03873` SODA 是 DPO preference；`2605.07396` ROPD 是 rubric/GRPO response reward；`2604.00626` 是 survey。

2026-05+ 增量 59 篇里，严格以 sampled-token log-ratio / K1 / token-level advantage 为核心的论文约 **15 篇**；**18 篇**属于混合或部分 sampled-token，即比较多种 OPD 形式、只把 sampled-token 信号当权重，或只改 rollout / pipeline 但底层可接 sampled-token OPD；**23 篇**主要是 full-vocab / top-k / subset KL、trajectory / semantic signal 或 teacher-schedule 变体；**3 篇**是分析或综述型，不对应单一训练 loss。

## 2026-05+ 增量 59 篇 OPD 形式统计

### 统计口径

本节只统计 `docs/01筛选/论文筛选.md` 中“2026-05+ 增量筛选更新（仅五类）”保留的 59 篇论文。

`是否 sampled-token` 使用四档：

- **是**：student rollout 后，在实际采样 token 上使用 teacher/student log-prob、log-ratio、K1 estimator、token-level advantage 或 token-level reweighting。
- **否**：虽然可能在 student prefix 上训练，但 loss 是 full-vocabulary KL、top-k / subset KL、teacher top-1 CE、trajectory-level / chunk-level / rubric / semantic objective，或只改 teacher schedule / rollout source。
- **混合/部分**：论文比较多种 OPD 形式，或主方法把 sampled-token 信号和 full-vocab/top-k KL、GRPO、trajectory routing、freshness/cache 控制组合使用。
- **不适用/分析型**：综述、诊断或机制分析论文，没有单一训练 objective。

`loss 形式`按论文主方法或最核心实验记录，不把“on-policy token-level”自动等同于 sampled-token。

### 是否 sampled-token 汇总

| 类别 | 数量 | 论文 |
|---|---:|---|
| **是** | 15 | `2605.07865`, `2605.08737`, `2605.11739`, `2605.12483`, `2605.28396`, `2605.30833`, `2605.31490`, `2606.01039`, `2606.02684`, `2606.09304`, `2606.17199`, `2606.00172`, `2606.11709`, `2605.23493`, `2605.21851` |
| **混合/部分** | 18 | `2605.11182`, `2605.16826`, `2606.07082`, `2605.07804`, `2605.17862`, `2605.25582`, `2606.01249`, `2606.04036`, `2605.11609`, `2605.12652`, `2605.18529`, `2605.22263`, `2605.26844`, `2606.18810`, `2605.28791`, `2606.00755`, `2606.02530`, `2606.10385` |
| **否** | 23 | `2605.22731`, `2605.30070`, `2605.19433`, `2605.27028`, `2606.00305`, `2606.08432`, `2606.09471`, `2605.09725`, `2605.13230`, `2605.31159`, `2606.03532`, `2605.10194`, `2605.17497`, `2605.21606`, `2606.15576`, `2605.11458`, `2605.15113`, `2605.22675`, `2605.28014`, `2606.01476`, `2606.03089`, `2606.11627`, `2606.19327` |
| **不适用/分析型** | 3 | `2605.10889`, `2605.18141`, `2605.11613` |

### loss 形式聚合

下面几类是非互斥统计，因为不少论文同时有 OPD loss、RLVR loss、filter / gate / schedule。

| loss 形式 | 覆盖情况 | 代表论文 | 说明 |
|---|---:|---|---|
| sampled-token log-ratio / K1 / PG advantage | 核心 15 篇，另有若干混合 | `2605.07865`, `2605.28396`, `2606.01039`, `2606.02684`, `2606.17199`, `2606.09304`, `2606.00172`, `2606.11709`, `2605.23493` | 只在 student sampled token 上使用 teacher/student 概率差，常作为 policy-gradient reward、advantage 或权重。 |
| full-vocab / top-k / subset KL 显式 loss | 约 20+ 篇 | `2605.11182`, `2605.16826`, `2605.27028`, `2606.09471`, `2606.04036`, `2606.19327`, `2605.10194` | 每个 prefix 上匹配一个分布，可能是全词表、top-k、span/subset 或 tail-bucket。 |
| trajectory / horizon / prefix 控制 | 约 10 篇 | `2605.07804`, `2605.27028`, `2605.28396`, `2605.31490`, `2606.00305`, `2606.08432`, `2606.09471` | 主要改 rollout 长度、prefix reliability、near-future trajectory、termination 或 refinement，不一定改底层 KL。 |
| trust-region / clipping / estimator transform | 约 10 篇 | `2605.07865`, `2605.08737`, `2605.25582`, `2606.01039`, `2606.01249`, `2606.09304`, `2606.17199` | 重点是稳定 sampled estimator、outlier、clip-safe region、trust region 或 extrapolation。 |
| token / step / path reweighting 或 routing | 约 15 篇 | `2605.10194`, `2605.21606`, `2605.21851`, `2606.00172`, `2606.11709`, `2606.15576`, `2606.18810` | teacher signal 不一定直接作为 loss，可能只决定 token 权重、方向翻转、span routing 或 path credit。 |
| logit-free / semantic / rubric / feedback signal | 约 7 篇 | `2605.15113`, `2605.28014`, `2606.01476`, `2606.03089`, `2606.19327`, `2605.07396` | 用 language feedback、reflection、constitution、rubric、semantic chunk 等替代 teacher logits 或 CoT。 |

### 增量逐篇对照表

| arXiv | 论文 / 方法 | 类型 | 是否 sampled-token | loss 形式 / objective | 备注 |
|---|---|---|---|---|---|
| `2605.10889` | Unmasking OPD | 机制综述 / 统一视角 | **不适用/分析型** | offline gradient-alignment diagnostic；比较主要 distillation objective 的局部梯度 | 不是训练算法；评估 teacher signal 是否对齐 ideal success gradient。 |
| `2605.11182` | The Many Faces of OPD | 综述 / taxonomy + 机制 | **混合/部分** | full-vocab KL、sampled-token PG、TopK RKL、stop-gradient TopK 等设计空间 | 实证比较多种 OPD / OPSD loss；不是单一 loss。 |
| `2605.18141` | A Brief Overview: OPSD in LLMs | 综述 / taxonomy | **不适用/分析型** | OPSD taxonomy；概念上是 per-token divergence | 无单一实验 loss。 |
| `2605.22731` | Post-Training is About States, Not Tokens | 机制理论 / 统一视角 | **否** | continuation-based OPD；teacher continuation CE / local supervision | 强调 supervision state；不是 sampled-token estimator 论文。 |
| `2605.16826` | Decoupling KL and Trajectories | 机制理论 / 统一视角 | **混合/部分** | prefix source x KL direction framework；FKL/RKL on teacher/student prefixes | 主实验使用 exact full-vocab KL，也讨论 RKL 的 sampled-token PG 解释。 |
| `2605.12483` | Sparse-to-Dense Reward Principle | 机制 recipe | **是** | FKL warmup + OPD as sampled-token dense teacher reward | OPD stage 明确解释为 student-sampled token 上的 teacher-student likelihood-ratio reward。 |
| `2605.11739` | Learning to Foresee / EffOPD | 机制实验 + 加速 | **是** | sampled-token OPD immediate log-ratio gradient + directional extrapolation | 论文分析 OPD update geometry；实现命令使用 reverse-KL advantage。 |
| `2605.30070` | Predictive Law for OPSD from World Feedback | 机制预测 | **否** | per-token reverse-KL OPSD；top-k 20 distillation | 关注 student-self-teacher gap predictive law，不是 sampled-token estimator。 |
| `2606.07082` | Geometry of OPD | 几何分析 | **混合/部分** | OPD/SFT/RLVR parameter-space diagnostics；sampled-token gap 与 objective interpolation | 分析不同 OPD variants，不绑定单一 loss。 |
| `2605.07804` | Prune-OPD | 长轨迹 / prefix drift | **混合/部分** | OPD reward tensor 乘 prefix reliability weight；top-k overlap / action acceptance | 改 reliability 权重和 suffix pruning；底层可接 top-k reward tensor 或 sampled-token advantage。 |
| `2605.17862` | f-OPD | 长轨迹 / freshness 稳定化 | **混合/部分** | freshness-weighted distillation loss + rollout-anchored regularizer + adaptive refresh | 作用于异步 OPD buffer；核心是 freshness/cache drift control。 |
| `2605.19433` | MOTAB / Backtracking | 长轨迹 / exposure bias | **否** | trajectory monitoring + backtracking + teacher corrective suffix；对照 top-k OPD | 主方法偏数据合成 / trajectory correction，不是 sampled-token KL estimator。 |
| `2605.27028` | ESR / Early Stopping Rollout | 长轨迹 / rollout horizon | **否** | standard reverse-KL OPD restricted to early rollout window | 不改 OPD loss，只截断 supervised horizon；论文公式是 prefix 上的 distribution KL。 |
| `2605.28396` | ADWIN | 长轨迹 / adaptive window | **是** | sampled-token immediate log-ratio OPD + prefix-window / delayed full-rollout alignment | 底层是 sampled token local reward；方法核心是 adaptive horizon。 |
| `2605.30833` | LGR / Supervision Fidelity Decay | 长轨迹 / supervision decay | **是** | sampled-token RKL advantage + one-step-ahead confidence reward | 在 sampled student token 上加 confidence reward；高熵位置用 top-k candidate 估计。 |
| `2605.31490` | POPD / TOPD | 长轨迹 / horizon control | **是** | token-level OPD gradient on truncated / progressive rollout horizon | 明确用 immediate token-level log-ratio reward，只改 rollout horizon。 |
| `2606.00305` | Near-Future Guidance / TOPD | 长轨迹 / trajectory bridge | **否** | vanilla reverse-KL OPD + OT-aligned near-future KL target | 增加 trajectory-level soft target，不是 sampled-token log-ratio estimator。 |
| `2606.08432` | TRD / Trajectory-Refined Distillation | 长轨迹 / trajectory refinement | **否** | refined trajectory 上的 full-vocab forward KL；对照 raw rollout FKL/RKL/top-k | 先 refine trajectory 再做 per-token KL，主 loss 是分布匹配。 |
| `2606.09471` | KAT / KL Agreement Trap | 长轨迹 / KL agreement failure | **否** | pure OPD reverse KL + softmax vocabulary loss；KL-based suffix termination | 复用 full/top-k reverse KL，不改为 sampled-token estimator。 |
| `2605.07865` | vOPD / KL for a KL | 优化稳定性 / 方差 | **是** | sampled-token reverse-KL PG estimator + detached full/top-k KL control-variate baseline | 目标保持 single-sample estimator，无偏降方差。 |
| `2605.08737` | Extrapolation Cliff / ListOPD | 优化稳定性 / 理论边界 | **是** | base-relative sampled-token/listwise ExOPD reward extrapolation + IS clipping | 针对 structural sampled tokens 的 clip-safe threshold。 |
| `2605.09725` | BRTS | teacher rollout selection | **否** | student-context top-k RKL + teacher-context top-k FKL/RKL auxiliary branch | 两个 top-k aggregation loss；不是 sampled-token log-ratio。 |
| `2605.13230` | TGPO | large divergence / teacher guidance | **否** | teacher top-1 directional guidance CE + GRPO + reference KL | 在 student prefix 上模仿 teacher argmax token，不是 sampled sampled-token KL。 |
| `2605.25582` | Extreme Region Policy Distillation | off-policy reuse / trust region | **混合/部分** | weakly constrained off-policy optimization + KL-constrained distillation / token reward signal | 两阶段 policy distillation，含 token reward signal，但不是标准 sampled-token OPD。 |
| `2605.31159` | Trust-Region Behavior Blending | trust region / rollout behavior | **否** | per-prefix reverse-KL OPD loss unchanged；teacher-guided behavior policy warmup | 改 rollout behavior policy，不改 loss；loss 是 top-k reverse KL。 |
| `2606.01039` | OPD+ | advantage design | **是** | f-divergence OPD sampled-token estimator + corrected stop-gradient advantage | 重点是 sampled-token objective 与正确梯度。 |
| `2606.01249` | TrOPD | trust region / outlier handling | **混合/部分** | trust-region K1 RKL + outlier top-k FKL + off-policy guidance FKL | trust 区域用 sampled K1，outlier 和 guidance 用 top-k / teacher-prefix FKL。 |
| `2606.02684` | FiRe-OPD | filter / reweight | **是** | trajectory filtering + token-level soft weights on sampled-token OPD advantage | 先筛 trajectory，再把 sampled-token log-ratio advantage 按 teacher confidence / student confusion 加权。 |
| `2606.03532` | CGTR / When Should the Teacher Move? | teacher schedule | **否** | RL objective + top-k JSD distillation；state-aware hard teacher refresh | 研究 self-teacher refresh schedule，不是 sampled-token objective。 |
| `2606.04036` | SDPG | PG connection / KL regularized policy optimization | **混合/部分** | exact full-vocabulary reverse-KL OPD + equivalent centered log-ratio PG interpretation | 实现用 explicit full-vocab KL；sampled-token PG 是梯度解释和对照。 |
| `2606.09304` | SG-OPD | sign gating | **是** | sampled-token RKL advantage + sign-consistency gate + phased teacher sampling | 在 sampled token 上根据 verifier sign 过滤 teacher signal。 |
| `2606.17199` | PowerOPD | estimator transform | **是** | sampled-token log-ratio reward with bounded power / Box-Cox transform | 直接修正 unbounded log-ratio sampled-token reward。 |
| `2605.10194` | TRACE | token routing | **否** | span-routed FKL/RKL/skip + GRPO；per-vocabulary KL clipped and decayed | token/span subset KL，不是 sampled-token log-ratio。 |
| `2605.11609` | AntiSD via PMI | token credit / anti-self-distillation | **混合/部分** | sampled-token conditional PMI interpretation + JSD ascent / entropy gate | 使用 sampled-token 信号解释与反向处理，但主框架含 JSD/gating。 |
| `2605.11613` | Input-Specific Credit | credit 机制分析 | **不适用/分析型** | credit-assignment diagnostic；generic vs input-specific credit | 不是训练算法。 |
| `2605.12652` | Multi-Rollout OPD / MOPD | multi-rollout teacher context | **混合/部分** | peer-conditioned full/token distribution KL；math 用 RKL，其他用 JSD | student rollout 上训练，但 loss 是 teacher distribution divergence。 |
| `2605.17497` | SSOPD | correct-wrong witness | **否** | GRPO + pointwise-clipped forward KL on wrong prefixes using successful witness | full-vocab FKL auxiliary loss，不是 sampled-token log-ratio。 |
| `2605.18529` | AMR-SD | meta-reflective credit | **混合/部分** | meta-reflective self-distillation + token-level credit | 依赖 reflection 信号，文档中未归为严格 sampled-token KL。 |
| `2605.21606` | Position-Weighted OPSD | position / reliability weighting | **否** | position/reliability-weighted OPSD KL | weighted KL distribution，不是 sampled-token estimator。 |
| `2605.21851` | OPPO | Bayesian value recursion | **是** | GRPO direction anchored by token-level log-ratio / Bayesian value recursion | sampled token log-ratio 作为 Bayesian evidence，方向由 GRPO/verifier anchor。 |
| `2605.22263` | Direction-Adaptive Self-Distillation | direction adaptive | **混合/部分** | aptitude-conditioned direction routing between imitation and self-distillation | 不绑定单一 estimator。 |
| `2605.26844` | Token Teachability | teachability filtering | **混合/部分** | teachability-filtered token distillation / disagreement selection | token subset and disagreement filtering，不严格等同 sampled-token。 |
| `2606.00172` | CAST | asymmetric self-teaching | **是** | GRPO/DAPO objective with sampled-token teacher-student gap shaping and sign reversal | sampled rollout token 上计算 teacher-student gap，改 token advantage。 |
| `2606.11709` | RLCSD | contrastive privileged signal | **是** | sampled-token contrastive signal modulates verifier advantage in PPO-style loss | 明确区别于 full-logit distill；teacher signal 只调 token-level magnitude。 |
| `2606.15576` | HSD / Path-Conditioned Self-Distillation | path-conditioned credit | **否** | path-conditioned teacher distribution / endpoint context replacement | 重点是 peer path context 与 per-token distribution matching。 |
| `2606.18810` | SC-GRPO | self-conditioned credit | **混合/部分** | token-level KL used as multiplicative weight on GRPO gradient；additive KL loss为负例 | 不是把 KL 作为主 loss，而是用 KL 权重重分配 sampled token 更新。 |
| `2605.11458` | ATESD | adaptive teacher exposure | **否** | privileged-context exposure controller + KL distillation | 控制 teacher exposure，不是 sampled-token estimator。 |
| `2605.15113` | VPD | language feedback / variational distillation | **否** | variational EM self-distillation；feedback-aware teacher + KL / trust-region distillation | dense teacher distribution / variational objective。 |
| `2605.22675` | SPD | capability subspace | **否** | capability-selective subspace projection + SFT/self-generated outputs | 不是标准 OPD KL sampled-token。 |
| `2605.23493` | EDGE-OPD | evidence-guided privileged signal | **是** | sampled-token reverse-KL / K1 OPD with positive-evidence mask + KL anchor | 论文明确采用 sampled-token reverse-KL OPD recipe。 |
| `2605.28014` | ROSD | reflective OPSD | **否** | reflection-conditioned teacher + localized / top-k distillation | top-k distillation，不是 sampled-token。 |
| `2605.28791` | Skill-Gated Self-Distillation | skill-gated | **混合/部分** | skill-conditioned gate on self-distillation loss | 主要是 gate，不足以归为严格 sampled-token。 |
| `2606.00755` | Policy Reheater | entropy internalization | **混合/部分** | OPSD/RLVR post-stage to internalize temperature / entropy | 后处理式 reheating，loss 口径依具体 OPSD/RLVR 组件。 |
| `2606.01476` | OmniOPD | logit-free / semantic chunk | **否** | chunk-level Bayesian semantic loss + token-level reference KL anchor | logit-free chunk-level signal，不是 teacher log-prob on sampled token。 |
| `2606.02530` | SafeSteer | safety / localized OPD | **混合/部分** | localized OPD safety alignment on safety-relevant states | 以安全状态定位和局部监督为核心，loss 口径依具体实现。 |
| `2606.03089` | Constitutional On-Policy Safe Distillation | safety / constitution | **否** | constitution-conditioned semantic teacher signal + on-policy safe distillation | semantic teacher signal，不是 sampled-token KL。 |
| `2606.10385` | Anchored Residual Guidance | privileged residual | **混合/部分** | anchored residual guidance；distill residual correction instead of full privileged distribution | 可与 sampled/logit 信号组合；核心是 residual/anchor。 |
| `2606.11627` | When Context Returns / NCA | context-return robustness | **否** | no-context anchoring regularizer + robustness diagnostic | 分析 + consistency regularizer，不是 sampled-token OPD。 |
| `2606.19327` | RCSD | rubric-conditioned self-distillation | **否** | rubric-conditioned forward KL on student-generated trajectories；FKL/RKL/JSD ablation | on-policy token-level guidance，但主 loss 是 full distribution FKL。 |

## 逐篇对照表

说明：原表把完整公式直接放在 `论文主 loss / objective` 列里，包含大量裸 `|` 和 `||`。这些字符在不少 Markdown 渲染器中会被误识别为表格分隔符。因此本表只保留 objective 的短标签，完整公式移到下一节。

| arXiv | 论文 / 方法 | 主要 baselines | 是否 sampled-token OPD | 主 loss / objective 简写 |
|---|---|---|---|---|
| `2601.07155` | Veto / Stable OPD | Teacher SFT, Student SFT, Supervised KD, SKD, On-policy KD；ablation 比 FKL/RKL + Veto | **否，主方法是 on-policy rollout + full distribution target reformulation** | Veto 中间 target + FKL/RKL |
| `2601.18734` | OPSD / Self-Distilled Reasoner | SFT, GRPO；ablation 比 FKL/RKL/JSD、full-vocab vs sampled-token | **主实验否；主实验用 full-vocabulary logit distillation。sampled-token 是对比 ablation** | on-policy full-vocab OPSD；sampled-token ablation |
| `2601.20802` | SDPO / RL via Self-Distillation | improved GRPO, on-policy GRPO, off-policy self-distillation；ablation 比 logit-level / token-level / sequence-level SDPO | **主方法否；主方法是 logit-level/top-k self-distillation，token-level 是 ablation** | feedback-conditioned self-distillation KL |
| `2602.04942` | PI-Distill / PI-OPSD | standard RL, SFT w/ CoT, SFT w/o CoT, SFT+RL, OPSD；`alpha in {0,0.5,1}` 的 pi-Distill variants | **pi-Distill 不是；OPSD 变体接近 on-policy reverse-KL penalty** | PI teacher/student joint GRPO；PI-OPSD penalty |
| `2602.12125` | G-OPD / ExOPD | SFT, OPD, ExPO；strong-to-weak 中 SFT/OPD/ExOPD；multi-teacher 中 SFT/OPD/ExPO | **是，按论文推导和实现口径是 sampled-token log-ratio OPD / dense reward** | sampled-token OPD reward + reward extrapolation |
| `2602.12275` | OPCD | base / initial student, direct in-context injection, off-policy context distillation；teacher-student OPCD vs self-OPCD | **否，on-policy rollout 但 top-k reverse KL** | context-conditioned top-k reverse KL |
| `2602.22495` | RLAD / TRRD | GRPO, KDRL, SFT/offline distillation | **部分是。它在 sampled rollout token 上用 teacher ratio 改 GRPO importance ratio，不是独立 KL loss** | teacher-anchored GRPO clipped surrogate |
| `2603.07079` | EOPD | KD, OPD, GRPO；entropy baselines；GKD / adaptive KL baselines | **混合。RKL 项是 sampled-token clipped OPD，FKL 项是 teacher top-k KL** | reverse KL + entropy-gated forward KL |
| `2603.11137` | REOPOLD | SFT, vanilla RKL / OPD, GRPO；另与 GKD full-vocab / top-5 比较 | **是，核心是 sampled-token RKL as reward** | sampled-token RKL reward + clipping/sampling |
| `2604.00626` | OPD Survey | 非新算法，不适用；整理 GKD, DistiLLM, G-OPD, OPSD, AOPD, vOPD 等 | **不适用** | taxonomy，无单一 loss |
| `2604.03128` | RLSD / Self-Distilled RLVR | GRPO, OPSD variants, teacher top-1 / student top-1 leakage ablations | **是。RLSD 只在 student-sampled token 上用 privileged teacher evidence ratio 调 advantage magnitude** | RLVR direction + teacher evidence-ratio magnitude |
| `2604.03873` | SODA | Base, SeqKD/SFT, GAD | **否。black-box semi-on-policy preference distillation** | SFT warmup + DPO preference |
| `2604.04461` | DP-OPD | DP-SGD student-only, DPKD, DISTILDP | **否，on-policy trajectories + GKD generalized divergence on continuation tokens；不是 sampled-token log-ratio OPD** | GKD divergence + DP-SGD |
| `2604.10688` | SCOPE | GRPO, KD, OPD；ablation: opposite weights 等 | **部分是。错误轨迹 OPD 分支是 sampled-token log-ratio；正确轨迹是 MLE/likelihood branch** | correctness-routed MLE / OPD |
| `2604.12002` | SD-Zero | SFT, RFT, GRPO/DAPO, SDFT；Phase 1 SRT 与 Phase 2 self-distillation | **主 self-distillation 否；top-K distillation。SRT 是 supervised revision training** | SRT CE + top-k reviser distillation |
| `2604.13010` | Lightning OPD | SFT baseline, standard online OPD, naive/offline OPD variants；与 Rang et al. 区分 | **是。离线化的 sampled-token OPD / advantage PG** | offline sampled-token advantage PG |
| `2604.13016` | Rethinking OPD | 多 teacher/student controlled OPD；pure-OPD vs cold-start recipe；sampled-token vs top-k | **论文分析多粒度；重点实验包括 sampled-token OPD，并认为 sampled-token reward 已足够** | sampled-token / full-vocab / top-k objectives |
| `2604.14084` | TIP | all-token OPD baseline, entropy-only, Soft-OR, Q3-only, div-only/LATF-style | **否，核心是 token subset selection + reverse-KL OPD；不是只看 sampled token** | selected-token reverse KL |
| `2604.20244` | HPD | MS, SFT, SeqKD, RKLD, JSD/GKD；on-policy section 还比较 OPD pipeline | **混合。HPD 用 offline expert token + student-sampled non-expert token 的 reweighted likelihood，不是标准 sampled-token OPD** | hybrid reweighted likelihood |
| `2605.01347` | MAD-OPD / OPAD | Base, single-teacher OPD, MT-OPD, MT-SeqKD | **否。student on-policy state 上做 full-vocabulary multi-teacher divergence** | multi-teacher weighted full-vocab divergence |
| `2605.03677` | Uni-OPD | SFT, ExPO, ExOPD；MLLM 中 vanilla OPD | **是，基础 OPD 是 sampled-token log-ratio reward；Uni-OPD 加数据平衡和 margin calibration** | sampled-token OPD + margin calibration |
| `2605.06188` | OPSD Compresses What RLVR Teaches | pre-OPSD baseline, All-rollout, Correct-only, Incorrect-only, Split-direction, All-rollout JSD | **分析对象是 OPSD；可写成 sampled-token advantage，也可视为 per-token reverse-KL objective。论文重点不是新 baseline 算法** | OPSD reverse-KL diagnostic objective |
| `2605.06387` | AOPD | SeqKD, OPD, GKD, ExOPD | **混合。标准 OPD 部分是 sampled-token advantage；非正 advantage 区域切到 top-K teacher FKL/JSD guidance** | positive PG + non-positive FKL/JSD |
| `2605.06597` | UniSD | SFT, SDFT, GKD, SSD, OPSD | **否。on-policy self-distillation 但主框架是 weighted divergence + auxiliary losses，不限 sampled token** | weighted divergence + auxiliary losses |
| `2605.07396` | ROPD | Black-box: SFT, T-Judge, OVD, GAD；White-box: SFT, LOPD, ExOPD | **否。black-box response/rubric-level on-policy RL，不用 teacher logits** | rubric reward + GRPO |
| `2605.10781` | RLRT / Rebellious Student | GRPO, SDPO, SRPO, RLSD；ablation RLRT-all 等 | **是。student-sampled token 上反向使用 teacher-student ratio 作为 advantage weight** | reversed teacher ratio on correct rollouts |

## 论文主 Loss / Objective 详情

本节使用 display math，条件概率统一写作 `\mid`，KL 两侧统一写作 `\Vert`，避免破坏 Markdown 表格。

### `2601.07155` Veto / Stable OPD

Veto 在 logit space 构造 teacher-student 中间 target：

$$
Q_\beta(v \mid s)
=
\frac{
P_T(v \mid s) P_S(v \mid s)^\beta
}{
Z_\beta(s)
}
$$

训练时再对 `Q_beta` 做 forward 或 reverse KL：

$$
D_{\mathrm{KL}}(Q_\beta \Vert P_\theta)
\quad\text{or}\quad
D_{\mathrm{KL}}(P_\theta \Vert Q_\beta)
$$

### `2601.18734` OPSD / Self-Distilled Reasoner

主设定是在 student rollout prefix 上做 full-vocabulary divergence：

$$
L_{\mathrm{OPSD}}
=
\mathbb{E}_{x,y^\star}
\mathbb{E}_{\hat y \sim p_S(\cdot \mid x)}
\sum_n
D\!\left(
p_T(\cdot \mid x,y^\star,\hat y_{<n})
\Vert
p_S(\cdot \mid x,\hat y_{<n})
\right)
$$

sampled-token ablation 可写成：

$$
r_n
=
\log p_T(\hat y_n \mid x,y^\star,\hat y_{<n})
-
\log p_S(\hat y_n \mid x,\hat y_{<n})
$$

### `2601.20802` SDPO / Reinforcement Learning via Self-Distillation

teacher 是同一模型在 feedback-conditioned context 下的分布：

$$
L_{\mathrm{SDPO}}
=
\sum_t
D_{\mathrm{KL}}\!\left(
\pi_\theta(\cdot \mid x,y_{<t})
\Vert
\operatorname{sg}\!\left[
\pi_\theta(\cdot \mid x,f,y_{<t})
\right]
\right)
$$

实现中可用 top-k logits 近似。

### `2602.04942` PI-Distill / PI-OPSD

PI-Distill 联合优化 PI teacher 与 student：

$$
J
=
\alpha J_{\mathrm{Teacher}}
+
(1-\alpha)J_{\mathrm{Student}}
$$

PI-OPSD 变体在 student rollout 上加入 PI teacher reverse-KL penalty：

$$
R_{\mathrm{env}}
-
\beta
D_{\mathrm{KL}}\!\left(
\pi_S(\cdot \mid s)
\Vert
\pi_T(\cdot \mid s,I)
\right)
$$

### `2602.12125` G-OPD / ExOPD

标准 OPD 的 sampled-token dense reward 可写成：

$$
r_t^{\mathrm{OPD}}
=
\log \pi^\star(y_t \mid x,y_{<t})
-
\log \pi_{\mathrm{ref}}(y_t \mid x,y_{<t})
$$

G-OPD 引入 flexible reference model 和 reward scaling；ExOPD 使用 `lambda > 1` 放大奖励项，使 reward term 不再和 KL regularization 等权绑定。

### `2602.12275` OPCD

student rollout，teacher 额外看到 context：

$$
L_{\mathrm{OPCD}}
=
\mathbb{E}_{x,c,\,y\sim\pi_\theta(\cdot \mid x)}
\frac{1}{|y|}
\sum_t
D_{\mathrm{KL}}\!\left(
\pi_\theta(\cdot \mid x,y_{<t})
\Vert
\pi_T(\cdot \mid c,x,y_{<t})
\right)
$$

实现中把 KL sum 限制到 student top-k token。

### `2602.22495` RLAD / TRRD

TRRD 把 teacher ratio 写进 GRPO/PPO-style clipped surrogate：

$$
r_{i,t}^{\mathrm{TRRD}}
=
\left(
\frac{\pi_S(a_{i,t} \mid s_{i,t})}
{\pi_{\mathrm{old}}(a_{i,t} \mid s_{i,t})}
\right)^\alpha
\left(
\frac{\pi_S(a_{i,t} \mid s_{i,t})}
{\pi_T(a_{i,t} \mid s_{i,t})}
\right)^{1-\alpha}
$$

这个 ratio 替代普通 GRPO importance ratio，等价于把 trust-region anchor 从 old policy 改成 old/teacher mixture。

### `2603.07079` EOPD

EOPD 在基础 OPD/RKL 上，对高 teacher-entropy 位置添加 FKL：

$$
L_t^{\mathrm{EOPD}}
=
L_t^{\mathrm{OPD}}
+
\alpha
\mathbf{1}[H_T^t > \tau]
L_t^{\mathrm{FKL}}
$$

其中 `L_t^{OPD}` 是 clipped reverse-KL single-sample/token loss，`L_t^{FKL}` 用 teacher top-k 近似。

### `2603.11137` REOPOLD

sampled-token RKL 被解释成 token reward：

$$
J_{\mathrm{RKL}}
=
\mathbb{E}
\sum_t
\rho_{i,t}(\theta)
R_{i,t}(\theta)
$$

$$
R_{i,t}
=
\operatorname{sg}\!\left[
\log
\frac{
\pi_T(o_{i,t} \mid q,o_{i,<t})
}{
\pi_\theta(o_{i,t} \mid q,o_{i,<t})
}
\right]
$$

REOPOLD 再加入 reward clipping、token mask 和 stage schedule。

### `2604.00626` OPD Survey

综述没有单一训练 loss；它整理 sampled-token KL、full-vocab KL、top-k KL、sequence-level objective、black-box preference/rubric 等不同范式。

### `2604.03128` RLSD / Self-Distilled RLVR

privileged teacher 只调节 advantage magnitude，不决定 gradient direction：

$$
\Delta_t
=
\operatorname{sg}\!\left[
\log P_T(y_t)
-
\log P_S(y_t)
\right]
$$

$$
w_t
=
\exp(\operatorname{sign}(A)\Delta_t)
=
\left(
\frac{P_T(y_t)}{P_S(y_t)}
\right)^{\operatorname{sign}(A)}
$$

$$
\hat A_t
=
A \cdot
\operatorname{clip}(w_t,1-\epsilon_w,1+\epsilon_w)
$$

然后把 `hat A_t` 放进 GRPO/PPO-style policy-gradient update。

### `2604.03873` SODA

SODA 先做 SFT warmup，再构造 teacher response 与 static student response 的 preference pair：

$$
(x,y^+,y^-)
=
(x,y_{\mathrm{teacher}},y_{\mathrm{student\ base}})
$$

然后使用 DPO：

$$
L_{\mathrm{DPO}}
=
-
\mathbb{E}
\log \sigma
\left(
\beta
\log
\frac{
q_\theta(y^+ \mid x)
}{
q_w(y^+ \mid x)
}
-
\beta
\log
\frac{
q_\theta(y^- \mid x)
}{
q_w(y^- \mid x)
}
\right)
$$

### `2604.04461` DP-OPD

DP-OPD 在 student-generated continuation tokens 上计算 GKD generalized divergence：

$$
L_{\mathrm{GKD}}(z_S,z_T;\beta,\tau_d)
$$

然后用 DP-SGD 更新 student；teacher frozen，不消耗额外 privacy budget。

### `2604.10688` SCOPE

正确轨迹走 MLE / likelihood branch：

$$
L_{\mathrm{MLE}}(x,y_i;\theta)
=
-
\sum_t
\rho_{i,t}(\theta)
\log \pi_\theta(a_{i,t} \mid x,a_{i,<t})
$$

错误轨迹走 OPD branch：

$$
L_{\mathrm{OPD}}(x,y_i;\theta)
=
\sum_t
\rho_{i,t}(\theta)
\left[
\log \pi_{\bar\theta}(a_{i,t} \mid x,a_{i,<t})
-
\log \pi_T(a_{i,t} \mid x,a_{i,<t})
\right]
$$

总目标按 correct / wrong groups 加权：

$$
J_{\mathrm{SCOPE}}
=
\mathbb{E}
\left[
\sum_{i\in\Omega^c_x}
w_i^{\mathrm{stu}}L_{\mathrm{MLE}}(x,y_i)
+
\sum_{i\in\Omega^w_x}
w_i^{\mathrm{tea}}L_{\mathrm{OPD}}(x,y_i)
\right]
$$

### `2604.12002` SD-Zero

Phase 1 是 self-revision training：

$$
L_{\mathrm{SRT}}
=
L_{\mathrm{revision}}
+
L_{\mathrm{generation}}
$$

Phase 2 把 reviser-conditioned teacher 蒸馏回 generator：

$$
L_{\mathrm{SelfDistill}}
=
\mathbb{E}
\sum_t
D_{\mathrm{KL}}\!\left(
\pi_\theta(\cdot \mid x,y_{<t})
\Vert
\operatorname{sg}\!\left[
\pi_{\theta_{\mathrm{SRT}}}(\cdot \mid x,y,P_r,y_{<t})
\right]
\right)
$$

实现配置中使用 top-k distillation。

### `2604.13010` Lightning OPD

sampled-token advantage：

$$
A_t
=
\operatorname{sg}\!\left[
\log \pi_T(a_t \mid s_t)
-
\log \pi_\theta(a_t \mid s_t)
\right]
$$

在线与离线 objective：

$$
J_{\mathrm{on}}
=
\mathbb{E}_{x\sim\pi_\theta}
\sum_t A_t
$$

$$
J_{\mathrm{off}}
=
\mathbb{E}_{x\sim\pi_{\mathrm{ref}}}
\sum_t A_t
$$

Lightning OPD 在 teacher consistency 下预计算 teacher log-probs。

### `2604.13016` Rethinking OPD

论文比较三种粒度：

$$
L_{\mathrm{sample}}
=
\mathbb{E}
\sum_t
\left[
\log p_t(\hat y_t)
-
\log q_t(\hat y_t)
\right]
$$

$$
L_{\mathrm{full}}
=
\mathbb{E}
\sum_t
D_{\mathrm{KL}}(p_t \Vert q_t)
$$

top-k 版本是在 subset vocabulary 上近似 full-vocab KL。

### `2604.14084` TIP

TIP 仍用 reverse-KL OPD，但只在筛出的 token 子集上计算：

$$
L_{\mathrm{TIP}}
=
\sum_{t\in S_{\mathrm{TIP}}}
D_{\mathrm{KL}}\!\left(
P_S(\cdot \mid s_t)
\Vert
P_T(\cdot \mid s_t)
\right)
$$

`S_TIP` 由 student entropy 与 teacher-student divergence 两轴确定。

### `2604.20244` HPD

HPD 统一成 reweighted likelihood：

$$
L_{\mathrm{HPD}}
=
\mathbb{E}
\left[
-
w_t^\star
\log q_\theta(a_t^\star \mid s_t)
-
w_t
\log q_\theta(a_t \mid s_t)
\right]
$$

其中 expert token weight 与 sampled non-expert token weight 由 negative K1 / reverse-KL gap 决定。

### `2605.01347` MAD-OPD / OPAD

student on-policy state 上做多 teacher weighted divergence：

$$
L_{\mathrm{MAD}}
=
\mathbb{E}
\frac{1}{|\hat y|}
\sum_t
\sum_k
w_k
D\!\left(
p_{T_k}(\cdot \mid s,H,\hat y_{<t})
\Vert
p_S(\cdot \mid s,\hat y_{<t})
\right)
$$

agentic tasks 使用 JSD，code tasks 使用 reverse KL；主设定是 full-vocabulary logits。

### `2605.03677` Uni-OPD

基础 sampled-token OPD reward：

$$
r_t^{\mathrm{OPD}}
=
\log
\frac{
\pi_T(o_t \mid q,o_{<t})
}{
\pi_\theta(o_t \mid q,o_{<t})
}
$$

Uni-OPD 再加入 student data balancing 与 outcome-guided teacher margin calibration；多 teacher 时可写成：

$$
\sum_i
w_i
D_{\mathrm{KL}}(\pi_\theta \Vert \pi_{T_i})
$$

### `2605.06188` OPSD Compresses What RLVR Teaches

分析对象是 OPSD reverse-KL objective：

$$
L_{\mathrm{OPSD}}
=
\mathbb{E}_{x,\,y\sim\pi_S}
\frac{1}{|y|}
\sum_t
D_{\mathrm{KL}}\!\left(
\pi_S(\cdot \mid x,y_{<t})
\Vert
\pi_T(\cdot \mid x,c,y_{<t})
\right)
$$

局部 policy-gradient 形式：

$$
A_t^{\mathrm{OPSD}}
=
\log \pi_T(y_t \mid s_t,c)
-
\log \pi_S(y_t \mid s_t)
$$

论文重点比较 correct-only、incorrect-only、split-direction 等诊断分支。

### `2605.06387` AOPD

标准 OPD 部分：

$$
A_t
=
\operatorname{sg}\!\left[
\log P_T(y_t \mid c_t)
-
\log P_S(y_t \mid c_t)
\right]
$$

$$
L_{\mathrm{OPD}}
=
-
\mathbb{E}
\frac{1}{|y|}
\sum_t
A_t
\log P_S(y_t \mid c_t)
$$

AOPD 用 gate：positive advantage 区域保留 PG exploitation，non-positive advantage 区域切到 teacher top-k truncated FKL / JSD guidance。

### `2605.06597` UniSD

UniSD 的主框架是 weighted divergence 加 auxiliary losses：

$$
L
=
\mathbb{E}_{x,\hat y\sim\pi_\theta}
\left[
\sum_t
m_t w_t
D\!\left(
\pi_\theta(\cdot \mid x,\hat y_{<t})
\Vert
\pi_T^\star(\cdot \mid x,c,\hat y_{<t})
\right)
+
\lambda_{\mathrm{aux}}L_{\mathrm{aux}}
\right]
$$

组件包括 agreement weights、EMA teacher、contrastive loss、feature matching、divergence clipping。

### `2605.07396` ROPD

rubric reward：

$$
r_i
=
\frac{
\sum_k w_k v_{i,k}
}{
\sum_k w_k
}
$$

GRPO advantage：

$$
A_i
=
\frac{
r_i-\operatorname{mean}(r_j)
}{
\operatorname{std}(r_j)+\epsilon
}
$$

然后用 clipped policy objective 更新 student。

### `2605.10781` RLRT / Rebellious Student

在正确 rollout 上反向使用 teacher-student ratio：

$$
\hat D_t
=
\log P_S^t(y_t)
-
\log P_T^t(y_t)
$$

$$
w_t^{\mathrm{RLRT}}
=
\exp(\operatorname{sign}(A)\hat D_t)
=
\left(
\frac{P_S^t(y_t)}{P_T^t(y_t)}
\right)^{\operatorname{sign}(A)}
$$

$$
A_t^{\mathrm{RLRT}}
=
A
\left[
(1-\lambda)
+
\lambda
\operatorname{clip}(w_t,1-\epsilon_w,1+\epsilon_w)
\right]
$$

## 易混点

**1. “student rollout” 不等于 “sampled-token OPD”。**

很多论文都让 student 先生成自己的 response，但 teacher 信号的粒度不同：

- `2601.18734`：student rollout 上做 full-vocab KL。
- `2602.12275`：student rollout 上做 student top-k reverse KL。
- `2605.07396`：student rollout 上用 rubric/verifier 打 response-level reward。
- `2604.03873`：student 的 base response 只作为 DPO negative。

**2. “token-level” 也不一定是 sampled-token。**

token-level 可以是：

- sampled token 的 log-ratio：`log pi_T(y_t)-log pi_S(y_t)`；
- full-vocab token distribution KL：例如 `D_KL(pi_S(. given prefix), pi_T(. given prefix))`；
- top-k subset KL；
- token mask / token selection 后的 KL；
- token-level feature / contrastive auxiliary loss。

**3. sampled-token OPD 的优点和代价。**

优点是便宜，只需要 teacher 对 sampled token 的 log-prob；缺点是方差高、丢掉未采样 token 的分布信息。`2604.13016` 和 survey 都强调 sampled-token 是常见轻量实现；`2601.18734` 则报告 full-vocab logit distillation 在其设置下优于 sampled-token objective。

**4. response-level RL 方法不要误写成 OPD KL。**

`ROPD`、`SODA` 这类黑盒方法的训练信号来自 rubric/preference，不依赖 teacher token probabilities。它们可以是 on-policy 或 semi-on-policy，但不是传统 white-box sampled-token OPD。
