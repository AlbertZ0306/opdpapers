# Online Policy Distillation：研究综述与前沿进展

> 整理时间：2026-06-29  
> 文档用途：汇报用正式学术报告  
> 覆盖范围：OPD / OPSD 领域约 85 篇核心论文（含本地 PDF 精读）

---

## 一、什么是 Online Policy Distillation

### 1.1 背景：知识蒸馏的结构性缺陷

知识蒸馏（Knowledge Distillation, KD）是将大型教师模型（teacher）的能力迁移到小型学生模型（student）的标准范式。传统 KD 的训练范式是**离线的（off-policy）**：学生在教师生成的固定文本上做监督学习，每一步梯度都以教师生成的完美 prefix 为条件。

这种范式存在一个被称为**曝光偏差（Exposure Bias）**的结构性缺陷。学生在训练时看到的是教师无误的序列前缀，但推理时却沿自己的输出自回归展开——一旦出现小的偏差，后续状态便会偏离训练分布，误差以序列长度 $T$ 的平方倍累积，即 $O(\epsilon T^2)$ 而非理想的线性 $O(\epsilon T)$。对于需要多步推理的任务，这一问题尤为严重：一道需要 10 步推理、每步精度 95% 的数学题，其轨迹整体成功率仅约 60%，而 off-policy 训练可能使实际情况远比这更差。

### 1.2 OPD 的核心思想

**Online Policy Distillation（OPD，在线策略蒸馏）**的核心思路是**将训练分布从教师生成的固定语料切换到学生自身的在线生成轨迹**。

与 off-policy KD 不同，OPD 的训练循环为：

1. **学生自主生成（rollout）**：学生在输入问题上，按当前参数 $\pi_\theta$ 自回归采样完整的回答轨迹；
2. **教师在学生轨迹上打分**：教师模型 $\pi_T$ 看到学生生成的 prefix $y_{<t}$，给出该位置的 next-token 分布 $\pi_T(\cdot \mid x, y_{<t})$；
3. **在学生访问的状态上优化散度**：学生在自己的 rollout states 上最小化与教师分布的散度，将梯度信号集中在学生实际会走到的状态上。

这种设计借鉴了交互式模仿学习中的 DAgger 思路，将 off-policy 的 $O(\epsilon T^2)$ 误差累积降回 $O(\epsilon T)$ 的线性量级。在自然语言推理任务中，这意味着：训练时出现的错误状态也能被纠正，而不是在推理期才第一次暴露于陌生的错误前缀下。

现实中，OPD 已成为工业级 LLM 后训练管线的核心组成部分：Qwen3、DeepSeek-V4、MiMo-V2-Flash、GLM-5 等主流模型均将 OPD 纳入后训练流程。

---

## 二、OPD 的 Loss 与最优化目标

### 2.1 统一的 $f$-散度框架

OPD 的损失函数可以在 $f$-散度的统一框架下描述：

$$
\mathcal{L}_{\text{OPD}}(\theta) = \mathbb{E}_{y \sim \pi_{\text{mix}}} \left[ \sum_{t=1}^{|y|} D_f\!\left(\pi_T(\cdot \mid x, y_{<t}),\; \pi_\theta(\cdot \mid x, y_{<t})\right) \right]
$$

其中：
- $\pi_{\text{mix}} = \lambda \pi_\theta + (1-\lambda)\pi_{\text{data}}$ 是 on-policy 与离线数据的混合采样策略，$\lambda \to 1$ 即纯在线采样；
- $D_f$ 是 $f$-散度系列中的某种散度，$f : (0, \infty) \to \mathbb{R}$ 为凸函数且 $f(1) = 0$；
- 教师分布 $\pi_T(\cdot \mid x, y_{<t})$ 以学生生成的 prefix $y_{<t}$ 为条件，即**在学生访问的状态上查询教师**。

### 2.2 主要散度选择

| 散度 | 优化目标 | 特性 | 适用场景 |
|---|---|---|---|
| **Reverse KL** $D_{\text{KL}}(\pi_\theta \Vert \pi_T)$ | $\min_\theta \mathbb{E}_{y \sim \pi_\theta}[\log \pi_\theta(y) - \log \pi_T(y)]$ | **模式搜寻（mode-seeking）**：学生集中于教师最高置信的模式，忽略次要输出 | 答案唯一的任务（数学、代码）；精确推理 |
| **Forward KL** $D_{\text{KL}}(\pi_T \Vert \pi_\theta)$ | $\min_\theta \mathbb{E}_{y \sim \pi_T}[\log \pi_T(y) - \log \pi_\theta(y)]$ | **模式覆盖（mode-covering）**：学生覆盖教师所有输出模式，可能在模式间插值产生幻觉 | 多输出合理的任务（创作、开放问答）；保持多样性 |
| **JSD** | 对称形式，两种方向的加权和 | 平衡模式搜寻与覆盖，数值上有界 | 中等多样性任务；翻译等 |
| **$\alpha$-散度** | 参数族，连续插值 FKL 与 RKL | 精细控制 mode-seeking 与 mode-covering 的权衡 | 自适应蒸馏方法 |

现代 OPD 方法中，**Reverse KL 是最常用的基础目标**，因为它在推理任务中能给出精确的 mode-seeking 监督。

### 2.3 Token 级别与序列级别的目标

**Token 级别目标（GKD 范式）：**

在每个位置 $t$ 上对全词表计算散度：

$$
\mathcal{L}_{\text{token}} = -\mathbb{E}_{y \sim \pi_\theta} \left[ \sum_t D_{\text{KL}}\!\left(\pi_T(\cdot \mid x, y_{<t}) \,\Vert\, \pi_\theta(\cdot \mid x, y_{<t})\right) \right]
$$

特点：低方差，每个位置有 $|V|$ 维的梯度信号（$|V|$ 为词表大小），但在教师分布校准差时可能引入偏差。

**序列级别目标（MiniLLM 范式）：**

将 Reverse KL 提升到全序列，通过策略梯度定理推导梯度：

$$
\nabla_\theta \mathcal{L}_{\text{MiniLLM}} = -\mathbb{E}_{y \sim \pi_\theta} \left[ \sum_{t=1}^{|y|} (R_t - 1) \nabla_\theta \log \pi_\theta(y_t \mid y_{<t}) \right]
$$

其中 $R_t = \sum_{t'=t}^{|y|} \log \frac{\pi_T(y_{t'} \mid y_{<t'})}{\pi_\theta(y_{t'} \mid y_{<t'})}$ 是从第 $t$ 步到序列末尾的累积 log-ratio 回报。

其中 $-1$ 来自 Reverse KL 中的熵项 $-\log \pi_\theta$。这表明**序列级 Reverse KL 等价于以教师 log-prob 为 dense reward 的策略梯度 RL**，这建立了 OPD 与强化学习之间的正式桥梁。

### 2.4 Sampled-Token 近似与 Full-Vocabulary KL

**Full-vocabulary KL（完整词表 KL）：** 在每个 prefix 位置计算对完整词表的精确散度。梯度方差低、知识迁移更忠实，但需要传输整个教师 logit 向量（词表大小 $|V|$ 可达 100K+），工程成本高。DeepSeek-V4 采用此方案，并通过缓存教师隐层状态在线重建 logits。

**Sampled-Token OPD（单采样 KL 估计）：** 在学生实际采样到的 token $y_t$ 上，用单点 Monte Carlo 估计散度：

$$
A_t = \log \pi_T(y_t \mid x, y_{<t}) - \log \pi_\theta(y_t \mid x, y_{<t})
$$

$$
\mathcal{L} = -\mathbb{E}\left[\sum_t A_t \cdot \log \pi_\theta(y_t \mid x, y_{<t})\right]
$$

该形式轻量、易于与 RL 框架（GRPO/PPO）融合，每个位置只需一个教师 log-prob，但**单样本估计方差高**，是近期多篇方差控制研究的动机。

### 2.5 OPD 与 KL-Constrained RL 的等价关系

OPD 与强化学习在目标函数上有深刻的等价关系。设 $r(y) = r_{\text{task}}(y)$ 为任务奖励，则 KL 约束 RL 的优化目标为：

$$
\max_\theta \mathbb{E}_{y \sim \pi_\theta}\left[r(y)\right] - \beta \cdot D_{\text{KL}}\left(\pi_\theta \Vert \pi_{\text{ref}}\right)
$$

当 $r(y) = \sum_t \log \pi_T(y_t \mid y_{<t})$ 时，即以教师 log-prob 作为 dense reward，该目标退化为 OPD 的形式。这说明：

- OPD 可以视为一种以**教师分布提供 dense reward 信号**的 KL-约束 RL；
- 改变 reward scaling（让 reward weight $\lambda > 1$）可以实现"**reward extrapolation**"，让学生有机会超越教师上界（G-OPD / ExOPD）；
- RLVR 使用稀疏可验证奖励，OPD 使用 dense 教师信号——二者是互补而非对立的关系。

### 2.6 On-Policy Self-Distillation（OPSD）

OPSD 是 OPD 的特殊情形，其中教师与学生是**同一个模型**，但教师获得了学生推理时无法访问的**特权信息（Privileged Information, PI）**，例如参考答案、验证结果或工具反馈。

$$
\mathcal{L}_{\text{OPSD}} = D_{\text{KL}}\!\left(\pi_\theta(\cdot \mid x, r, y_{<t}) \,\Vert\, \pi_\theta(\cdot \mid x, y_{<t})\right)
$$

其中左侧是拿到参考答案 $r$ 的 privileged teacher，右侧是普通 student。OPSD 的目标是让模型**将特权信息内化为无条件参数能力**，使其在推理时不依赖外部 teacher 也能达到相近效果。

---

## 三、OPD 的优点

### 3.1 消除 Exposure Bias，降低错误累积

OPD 的核心贡献是将误差累积从 $O(\epsilon T^2)$ 降至 $O(\epsilon T)$：训练时学生在自己会访问的状态上接受监督，推理时状态分布与训练分布匹配。实验上，OPD 在数学推理等长序列任务上相比 off-policy SFT 有显著提升。

### 3.2 Dense Token-Level 监督信号

相比 RLVR 的稀疏 scalar 奖励（整条回答对或错），OPD 在每个 token 位置都提供来自教师分布的 dense 监督信号，大幅改善 credit assignment：学生可以在 token 粒度上接收"哪一步推理偏离了教师"的精细反馈，而无需从序列末尾反向推断。

### 3.3 计算效率高

相比 RL（如 GRPO），OPD 以教师 log-prob 替代了 verifier 的二值结果，不需要可验证的 ground-truth 答案，可用于更广泛的任务。实验证据表明，OPD 在数学推理基准上可以以 **GRPO 1/10 的计算量**达到相近精度。

### 3.4 支持 Self-Improvement，无需外部 Teacher

OPSD 范式下，模型可以利用自身在有/无特权信息条件下的分布差异进行自我蒸馏，无需更大、更昂贵的外部教师，形成持续自我提升的训练循环。这是 DeepSeek 系列、MiMo 等模型采用的核心机制之一。

### 3.5 与 RLVR 互补，形成 Sparse-to-Dense Pipeline

OPD 和 RLVR 处于同一 KL-约束 RL 框架的不同点上。近期研究（`2605.12483`）提出了 **sparse-to-dense 原则**：

- **RLVR / GRPO** 适合探索阶段，发现新能力（稀疏 outcome reward 驱动探索）；
- **OPD / OPSD** 适合压缩阶段，将已发现的能力内化与校准（dense teacher signal 驱动蒸馏）。

二者串联使用可获得优于单独使用任一方法的效果。

---

## 四、OPD 的缺点与局限

### 4.1 Teacher-Student Gap 与 Thinking-Pattern Mismatch

强教师不保证有效蒸馏。`2604.13016`（Rethinking OPD）的系统实验表明，OPD 成功的必要条件有两个：

1. **Thinking-pattern compatibility**：教师与学生的思考模式（top-$k$ token 分布的 overlap ratio）必须相容。初始 overlap 低时，训练无法推进，高分教师反而可能完全失效；
2. **Teacher novelty**：即使教师分数更高，若与学生使用相同训练数据和 recipe，则教师无法提供学生未见过的新知识。

这意味着 OPD 对**教师-学生配对的选择高度敏感**，并非无条件有效的"免费午餐"。

### 4.2 Privileged Information Leakage

OPSD 的 loss 存在结构性问题（`2604.03128`）。将其分解可得：

$$
\mathcal{L}_{\text{OPSD}} = \mathcal{L}^* + I(Y_t;\, R \mid X, Y_{<t})
$$

其中第二项是不可避免的互信息 gap：教师可以看到参考答案 $R$，而 student 不行。在逐样本梯度中，参考答案特有的偏差项会污染更新方向，把特权评估注入梯度，形成 **leakage bandwidth**。直接对 privileged teacher 分布做分布匹配会导致：知识泄漏、KL 停滞（KL stagnation）和后期性能下降。

### 4.3 长轨迹中的 Supervision Fidelity Decay

学生生成的 prefix 越长，越可能偏离教师熟悉的轨迹。`2605.30833`（Supervision Fidelity Decay）发现：**当学生 prefix 变长，教师的 next-token 置信度和判别性会系统性下降**——教师开始"不知道怎么说"，其 dense signal 的可靠性显著降低，对学生的纠错能力接近于零。

相关研究还发现：坏 prefix 上的低 KL agreement 并不意味着有效的纠错信号，而可能是教师**适应了坏状态的虚假一致**（KL Agreement Trap，`2606.09471`）。

### 4.4 需要 White-Box Teacher 访问权限

标准 OPD 需要教师在学生 prefix 上的 token-level 分布（完整 logits 或 top-k logits）。对于闭源 API 教师（如 GPT-4、Claude），这不可行。黑盒 teacher 只能给出 response，不能给出不确定性分布，这限制了 OPD 的应用范围。

### 4.5 计算成本高

OPD 的训练需要：
- **在线 rollout 生成**：学生每步采样完整轨迹，计算开销大；
- **在线 teacher 推理**：教师需要在每条学生 rollout 上做前向推断；
- 若使用 full-vocabulary KL，还需传输完整 logit 向量。

相比 off-policy SFT，OPD 的 wall-clock 训练时间通常显著更高。

### 4.6 OPSD 不能可靠修复失败轨迹

`2605.06188`（OPSD Compresses What RLVR Teaches）的实验表明：OPSD 对正确轨迹更像**压缩与校准**（把已有能力的长推理变短），而对错误轨迹并不能可靠地修复——在错误轨迹上训练甚至会损伤 accuracy。因此，OPSD 适合作为 RLVR 之后的**后处理压缩阶段**（post-RL compaction），而不是探索新能力的机制。

### 4.7 高方差的 Sampled-Token 估计

Sampled-token OPD 的单点 Monte Carlo 估计方差高、分布重尾，可能导致训练不稳定。`2605.07865`（vOPD）通过引入 control variate baseline 将方差降低，但仍是当前活跃研究的问题之一。

---

## 五、OPD 领域的主要研究方向

OPD 领域目前已有约 85 篇核心论文（含 2026-05+ 增量），研究重心可归纳为以下七条主要方向。

---

### 5.1 目标散度设计：KL 方向、自适应与估计器

**核心问题：** 用什么散度、在什么 token 上匹配、如何在计算成本与统计效率之间权衡？

**早期工作**建立了 Reverse KL（mode-seeking，适合推理）与 Forward KL（mode-covering，适合多样性任务）的基础对立，以及混合使用（JSD、$\alpha$-散度、HPD）的可行性。

**自适应散度**是进一步的改进方向：
- **Entropy-Aware OPD（EOPD）**：在教师高熵 token 上加入 Forward KL（保留多样性），在低熵 token 保留 Reverse KL（精确模仿）；分析表明标准 OPD 只保留了 6.8% 的高熵 token，而教师有 18.5%，造成明显的多样性坍塌。
- **Veto / Stable OPD**：当教师-学生分布差距过大时，Reverse KL 会产生有害梯度，Forward KL 会过度平铺；Veto 在 logit 空间构造中间 target $Q_\beta$，同时作为梯度 veto 和置信度控制，抑制低置信 token 上的坏梯度。
- **AOPD（Asymmetric OPD）**：正 advantage 区域保留 RL exploitation；非正 advantage 区域切换为 teacher distribution matching，而非无效的负强化。

**估计器稳定性**是 2026-05+ 的新热点：
- **vOPD**：将 sampled-token OPD 梯度视为 REINFORCE 估计，引入 control variate baseline（per-token negative reverse KL）降低方差，top-$k$ 近似可进一步降低计算成本；
- **PowerOPD**：用有界幂变换（bounded power transformation）稳定 sampled-token log-ratio 的重尾分布；
- **OPD+**：指出 stop-gradient advantage 的目标与实际梯度方向不一致，重新设计 advantage 使二者匹配；
- **SG-OPD**：用 sign-consistency gating 过滤与 verifier 方向不一致的 teacher token 信号。

---

### 5.2 教师信号来源：外部教师、Self-Distillation 与特权上下文

**核心问题：** 教师信号从哪里来？不同来源的信息不对称如何利用而不引发泄漏？

这是 OPD 研究中**最活跃的横向主题**，贯穿整个领域。

**异体教师（Cross-model teacher）**是经典设置：大模型教小模型。主要挑战是超越教师上界（G-OPD / ExOPD 通过 reward extrapolation 实现），以及多教师共识（MAD-OPD 用多教师辩论后的置信度加权）。

**On-Policy Self-Distillation（OPSD）**是 2025-2026 年最重要的新范式：
- **Self-Distilled Reasoner**（UCLA）：同一模型分别作为 privileged teacher（看到参考解）和普通 student（只看问题），在 student rollout 上做 per-token divergence；
- **SD-Zero**：同一模型分 Generator 和 Reviser，Reviser 看到初始回答和 binary reward 后生成改进版本，再蒸馏回 Generator——把稀疏 binary reward 转化为 dense token-level 自监督。

**特权信息（Privileged Information）**的治理已从"是否使用"演化为"如何控制"：
- **RLSD**：将 privileged teacher signal 从梯度方向降级为幅度调制——梯度方向由 RLVR verifier 决定，privileged teacher 的 evidence ratio 只作为 stop-gradient 幅度乘子，避免特权评估污染更新方向；
- **Adaptive Teacher Exposure**：自适应调节教师可见的参考推理程度，避免过强 PI 造成泄漏或 shortcut；
- **Anchored Residual Guidance**：只蒸馏相对 anchor 的 residual correction，而非完整 privileged 分布，保留可迁移部分，丢弃不可迁移的绝对信息；
- **When Context Returns**：发现 context 被内化后，如果推理时重新给回原 PI，student 可能退化——说明 OPD 应同时优化 no-context 表现和 context-return 鲁棒性。

---

### 5.3 长轨迹与轨迹采样控制

**核心问题：** 随着学生轨迹变长，teacher signal 何时可用、何时失效？如何在轨迹层面管控监督质量？

这是 **2026-05+ 增量中最集中的新主线**，论文数量超过 10 篇。

**Prefix Drift 与 Supervision Fidelity Decay：** 学生生成的 prefix 逐渐偏离教师熟悉的分布后，教师对后续 token 的置信度和判别性系统性下降。这意味着：
- 并非全长 rollout 的每个 token 都值得蒸馏；
- 轨迹后段的教师信号可能比前段差很多。

具体研究方向包括：

| 研究问题 | 代表方法 |
|---|---|
| 检测 prefix drift，截断或降权后续监督 | Prune-OPD（top-$k$ overlap 检测 drift，后续降权） |
| 直接量化 supervision fidelity | Supervision Fidelity Decay / LGR（基于 teacher 置信度的 one-step-ahead reward） |
| 是否需要 full rollout | ESR / POPD（早停 rollout，后段 token 只增加噪声） |
| 识别坏 prefix 上的虚假 KL agreement | KL Agreement Trap（低 KL 可能是 teacher 适应坏状态，而非纠错信号） |
| 轨迹偏离时回退与修复 | Backtracking When It Strays（检测 stray 后回退到 last safe state） |
| 先修复轨迹再蒸馏 | Trajectory-Refined Distillation（先 refine trajectory，再在更可靠轨迹上蒸馏） |
| 自适应监督窗口 | ADWIN（按样本动态调节 OPD 监督范围，避免全长无效） |
| 跨步引入近未来信息 | Near-Future Guidance（用 OT-aligned 短期未来 trajectory 补足 token-local KL 的短视） |
| 异步 OPD 的 freshness 控制 | f-OPD（将 objective discrepancy 分解为 rollout drift + supervision drift，用 freshness 管控） |

这一方向的核心认识是：**OPD 的基本训练单位正在从单个 token 扩展到 prefix、span、step、window 和 trajectory**，需要在不同粒度上判断监督信号是否可靠。

---

### 5.4 Token / Step Credit 与选择性监督

**核心问题：** OPD 中并非每个 token 都值得同等训练——哪些 token 真正携带有用的教师信号？如何分配 credit？

**Full-token OPD 的低效问题：** TIP（`2604.14084`）的分析表明，高价值 token 分布在两类区域：（1）学生不确定的高熵 token；（2）学生过度自信但与教师分歧大的错误 token。只在这两类 token 上训练，效果接近或超过全 token OPD。

**Token Teachability（可学性）：** `2605.26844` 提出 teachability 概念——teacher-student disagreement 不一定都是可学的，取决于当前学生能力是否能吸收该信号。盲目选择 disagreement token 会蒸馏进不可教的噪声。

**TRACE（Token Routing）：** 在 privileged OPSD 中，全 token KL 会把梯度浪费在冗余位置，并放大特权信息泄漏，导致熵升、reasoning 变短和 OOD 退化。TRACE 只在 critical span 上路由 self-OPD，减少泄漏和无效梯度。

**Path-Conditioned Credit：** 基于 trajectory 中路径分歧定位 credit，而非均匀作用于全轨迹（HSD / Path-Conditioned Self-Distillation）。

**Multi-Rollout 利用：** 同一 prompt 的多个 rollout 含有丰富的成功/失败对照信息：
- Multi-Rollout OPD：用 peer success/failure 提供 dense supervision；
- SSOPD：正确 rollout 作为 witness，错误 rollout 提供纠错 prefix；
- CAST：处理 GRPO 全对/全错 group 中 zero-advantage 的无梯度问题，用不对称自学习补足。

**Anti-Self-Distillation（PMI）：** OPSD 在数学推理中可能强化与答案无关或有害的相关性。用 pointwise mutual information 识别"teacher 过度自信但非任务关键"的 token，对其做反向或抑制处理——改变蒸馏方向而非强度。

---

### 5.5 优化稳定性、Trust Region 与 RLVR+OPD 结合

**核心问题：** OPD 与 RL 结合时如何保证训练稳定？teacher/student gap 大时如何防止坏更新？

**Trust Region 控制：** 当教师-学生分布差距大时，普通 KL 监督会给出噪声大、方向不稳的梯度。Trust Region OPD 和 Trust-Region Behavior Blending 引入信任域约束，限制每步策略偏移。

**Teacher Schedule 在 Self-OPD 中的关键性：** `2606.03532`（When Should the Teacher Move?）的系统实验揭示：self-OPD 中 teacher 的更新频率和 isolation period 是影响稳定性的关键因素——teacher 更新太频繁会引发不稳定，但"teacher 年龄"本身不是唯一因素，需要综合考虑训练阶段和 reward 改善信号。

**RLVR 与 OPD 的角色分工：** 综合多篇论文的结论：
- **RLVR（稀疏奖励）→ 探索新能力**：发现模型之前未能解决的问题类型；
- **OPD（dense 教师信号）→ 内化与压缩**：将已发现的能力高效地迁移或压缩到部署模型。

这一分工已在 DeepSeek-V4 和 MiMo-V2-Flash 等工业模型中得到体现，其中 OPD 作为 model consolidation（模型整合）的核心机制。

**RLSD（方向-幅度解耦）：** 将 RLVR 和 OPSD 的梯度角色解耦：verifier outcome 决定 token 更新方向（探索哪个答案路径），privileged teacher 的 evidence ratio 只作为 stop-gradient 幅度乘子（调节每个 token 的学习力度），避免特权信息决定优化方向。

---

### 5.6 从统一理论视角重新理解 OPD

**核心问题：** OPD、SFT、RLVR 的本质区别是什么？如何在统一框架下理解所有变体？

**State Distribution 视角（`2605.22731`）：** 三种方法的真正区别不是 loss 形式，而是监督施加在哪些 state 上：
- SFT：teacher-forced states（固定、理想 prefix）；
- RL：reward-selected states（最优 rollout 的状态）；
- OPD：student on-policy states（学生自己会走到的状态）。

**KL 与 Trajectory 解耦（`2605.16826`）：** 传统讨论把 on/off-policy trajectory source 和 KL direction 绑定在一起，但二者实际上是两个独立的设计轴。将其解耦可以统一理解 SFT、DAgger、offline RL 和 OPD 的差异，并解释"在哪里监督"比"用什么散度"更重要的根本原因。

**参数几何视角（`2606.07082`）：** 从参数空间的更新轨迹、方向、曲率分析 OPD vs SFT vs RLVR 的差异，解释 OPD 为何能更快解锁能力（`2605.11739`：functional redundancy avoidance 和 early directional stabilization）。

**Sparse-to-Dense Reward Principle（`2605.12483`）：** 将整个训练阶段的奖励信号密度设计为统一原则：稀疏 outcome reward 负责探索，dense teacher signal 负责压缩，二者应在不同训练阶段交替主导。

**OPD Success Predictor（开放问题）：** 目前各论文分散研究的 success condition 包括 thinking-pattern overlap、teacher novelty、teacher entropy/fidelity、initial student-teacher gap、early KL dynamics 等。将这些指标整合成可在训练前执行的 go/no-go predictor，是当前最清晰的开放研究问题之一。

---

### 5.7 黑盒与语义 OPD

**核心问题：** 当教师 logits 不可用时，如何用 response、rubric、verbal feedback、verification 等语义信号替代 KL 监督？

这一方向反映了 OPD 向**实用化**和**与 proprietary teacher 对接**的扩展。

**现有路线：**

| 方法 | 信号形式 | 核心机制 |
|---|---|---|
| SODA | teacher response vs student response | 构造 contrastive preference，半在线黑盒蒸馏 |
| ROPD（Rubric-based OPD） | prompt-specific semantic rubrics | 从 teacher-student 对比中归纳评分规则，用 rubric 替代 logit 给学生 rollout 打分 |
| OmniOPD | speculative verification | 用 MC semantic estimation 替代 teacher logits，实现 logit-free OPD |
| SafeSteer | localized safety OPD | 只在安全相关 token 上施加局部蒸馏，降低 alignment tax |
| Constitutional OPD | constitution 作为语义 teacher | 用原则/规范作为 privileged teacher context，适用于 safety 场景无唯一 reference 的情况 |
| VPD | language feedback → variational objective | 把文本形式的 feedback 转成 dense variational policy distillation |

**核心开放问题：** 语义信号能否恢复 white-box teacher 分布的 uncertainty 信息、多模态替代输出和 token 级别的局部偏好，目前仍不清楚。rubric / verifier 的质量控制和 judge bias 是主要瓶颈。

---

## 六、主要研究方向总结

```
off-policy KD Exposure Bias
  ↓（OPD 范式切换）
在 student-visited states 上施加监督
  ↓
KL 方向 / Strict Imitation / Teacher-Student Gap 失败
  ↓（散度自适应、Trust Region、Estimator 稳定化）
Reward Scaling / Adaptive Divergence / Control Variate
  ↓
Privileged Self-Distillation 信息不对称与 Leakage
  ↓（方向-幅度解耦、Exposure Control、Residual Guidance）
Direction-Magnitude Decoupling / OPSD Governance
  ↓
长轨迹 Prefix Drift / Supervision Fidelity Decay
  ↓（Rollout 截断、轨迹修复、自适应窗口）
Token Teachability / Path-Conditioned Credit / Selective Distillation
  ↓
Semantic / Logit-Free / Rubric / Safety OPD
  ↓
统一 OPD/RLVR 路由器 / Reliability Prediction / 理论框架
```

| 研究方向 | 已形成共识 | 核心开放问题 |
|---|---|---|
| 目标散度设计 | Reverse KL 为主流；自适应 entropy-based mixing 有效 | full-vocab / top-k / sampled-token 的统一偏差-方差理论 |
| 教师信号来源 | OPSD 已成独立范式；PI 需要治理而非简单使用 | PI leakage taxonomy；black-box distribution recovery |
| 长轨迹控制 | Prefix drift 存在；full rollout 不必要 | 统一 teacher-signal reliability estimator；supervision horizon 自动选择 |
| Token Credit | Full-token OPD 冗余；teachability 优于 disagreement | Token-level causal credit；semantic role-aware selection |
| 优化稳定性 | Trust region、teacher schedule 是关键因素 | 统一 teacher drift / filter / freeze 控制理论 |
| 与 RLVR 关系 | Sparse-to-Dense pipeline 形成共识 | 何时从 RLVR 切换到 OPD 的统一调度器 |
| 黑盒/语义 OPD | Rubric、verification 是可行替代 | 语义信号的 uncertainty recovery；rubric 质量控制 |
| 统一理论 | State distribution 视角、KL-trajectory 解耦 | OPD success predictor；distillation scaling law |

---

## 参考文献（部分核心论文）

| arXiv | 标题 | 贡献 |
|---|---|---|
| `2604.00626` | A Survey of OPD for LLMs | OPD 综述，$f$-散度统一框架 |
| `2604.13016` | Rethinking OPD | OPD 成功条件现象学分析与 recipe |
| `2603.07079` | Entropy-Aware OPD | 自适应 KL 方向，多样性坍塌分析 |
| `2601.07155` | Stable OPD / Veto | Teacher-student gap 下的中间 target |
| `2601.18734` | Self-Distilled Reasoner | OPSD 基础范式 |
| `2604.03128` | Self-Distilled RLVR / RLSD | PI leakage 机制分析，方向-幅度解耦 |
| `2602.12125` | G-OPD / ExOPD | OPD 作为 dense KL-constrained RL；reward extrapolation |
| `2605.06188` | OPSD Compresses What RLVR Teaches | OPSD = post-RL compaction，而非 correction |
| `2605.10781` | Rebellious Student / RLRT | 反向利用 teacher mismatch，强化 self-driven reasoning |
| `2605.07865` | vOPD / KL for a KL | Sampled-token OPD 方差控制，control variate baseline |
| `2605.07804` | Prune-OPD | Prefix drift 检测与轨迹截断 |
| `2605.30833` | Supervision Fidelity Decay | Teacher confidence 随 prefix 变长系统性衰减 |
| `2606.09471` | KL Agreement Trap | 低 KL 可能是坏状态适应而非纠错信号 |
| `2604.14084` | TIP | Token importance 两轴分类，选择性监督 |
| `2605.10194` | TRACE | Token routing，减少 leakage 和无效梯度 |
| `2605.26844` | Token Teachability | Disagreement 不等于 learnability |
| `2605.22731` | Post-Training is About States | State distribution 视角统一 SFT/RL/OPD |
| `2605.16826` | Decoupling KL and Trajectories | Prefix source 与 KL direction 独立设计轴 |
| `2605.12483` | Sparse-to-Dense Reward Principle | RLVR 探索 + OPD 压缩的分工原则 |
| `2606.07082` | Geometry of OPD | 参数空间更新轨迹分析 |
