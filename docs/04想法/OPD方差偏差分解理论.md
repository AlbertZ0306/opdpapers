# N1：OPD 的方差-偏差分解与 Control-Variate 理论

更新时间：2026-06-22

定位：本文是 [研究方向与候选想法.md](/data1/zhaoenyue/opdpapers/docs/04想法/研究方向与候选想法.md) 中 flagship 候选 **N1** 的完整展开，给出形式化推导、方差-偏差分解、可测量 $\mathrm{Cov}(\text{teacher},\text{advantage})$ 的实验协议，以及对 OPSD Compresses、RLSD、vOPD、capability ceiling 的对应证明。它把 [方向力度解耦想法.md](/data1/zhaoenyue/opdpapers/docs/04想法/方向力度解耦想法.md) 的 factorial design 升级为有理论支撑的可预测条件。

---

## 1. 研究问题与一句话主张

**问题**：OPD 相比 RLVR 的收益，到底来自 teacher 改变了更新方向，还是提供了更细的 token-level credit？现有论文（SCOPE、RLSD、AOPD、OPSD Compresses、vOPD）各给一块经验拼图，没有统一解释。

**主张（待证明）**：

```text
把 RLVR 与 OPD 写成同一目标的 policy-gradient credit estimator 后，
teacher 信号对 per-token credit 起 control variate 作用。
OPD 对梯度的作用可严格分解为：
  (1) 方差缩减项 —— 不改变 fixed point；
  (2) 偏差/方向项 —— teacher 偏离真实 reward 最优时，把 fixed point 拉向 teacher。
OPD 净有用 ⟺ 方差缩减 > 偏差²，
该条件由可测量 Cov(teacher 信号, 真实 advantage) 和 teacher bias δ 决定。
```

这把"方向 vs 力度"之争变成一个**可计算判据**，并预测 OPD 在 correct/wrong/PI 三个 regime 的不同行为。

---

## 2. 记号与统一估计量框架

prompt $x$，student policy $\pi_\theta$，每个 prompt 采 $G$ 条 rollout：

$$
y_i\sim\pi_\theta(\cdot\mid x),\quad i=1,\dots,G
$$

state 与 action：

$$
s_{i,t}=(x,y_{i,<t}),\qquad a_{i,t}=y_{i,t}
$$

优化目标是真实 reward（verifier outcome）下的期望回报：

$$
J(\theta)=\mathbb{E}_{y\sim\pi_\theta}\big[R(y)\big]
$$

其策略梯度为

$$
\nabla_\theta J
=\mathbb{E}\Big[\textstyle\sum_t A^{*}_{i,t}\,\nabla_\theta\log\pi_\theta(a_{i,t}\mid s_{i,t})\Big],
$$

其中 $A^{*}_{i,t}=Q^{*}(s_{i,t},a_{i,t})-V^{*}(s_{i,t})$ 是**真实 token-level advantage**（不可直接观测，是所有方法都在估计的目标量）。

### 2.1 各方法 = 对 $A^{*}$ 的不同 credit 估计

把任意训练梯度写成 $\sum_{i,t}\hat c_{i,t}\,\nabla_\theta\log\pi_\theta(a_{i,t}\mid s_{i,t})$，区别只在 per-token credit $\hat c_{i,t}$ 怎么估 $A^{*}$：

| 方法 | credit $\hat c_{i,t}$ | 性质 |
|---|---|---|
| GRPO / RLVR | $A^{\mathrm{RL}}_i=\dfrac{R_i-\mathrm{mean}(R)}{\mathrm{std}(R)+\epsilon}$（轨迹内对 $t$ 常数） | 近似无偏但**高方差、零 token 分辨率** |
| sampled-token OPD | $c^{\mathrm{OPD}}_{i,t}=\log\pi_T(a_{i,t}\mid s_{i,t})-\log\pi_\theta(a_{i,t}\mid s_{i,t})$ | dense、低方差，但**可能有偏** |
| full-vocab OPD | $-\nabla_\theta D_{\mathrm{KL}}(\pi_\theta\Vert\pi_T)$ 等价的 distribution-level credit | dense，无 action 采样方差，**fixed point = teacher** |

关键观察：**GRPO 的 credit 高方差、低分辨率；teacher 的 credit 低方差、可能有偏。** 这正是 control variate / bias-variance 取舍的标准结构。

### 2.2 混合估计量

研究对象是一族混合 credit（涵盖 RLSD、AOPD、sampled-OPD、方向力度解耦的 A1/A6）：

$$
\hat c_{i,t}(\alpha)=(1-\alpha)\,A^{\mathrm{RL}}_i+\alpha\,c^{\mathrm{OPD}}_{i,t},
\qquad \alpha\in[0,1].
$$

$\alpha=0$ 是纯 GRPO，$\alpha=1$ 是纯 sampled-token OPD，中间是各种 blending。乘性形式 $A^{\mathrm{RL}}_i\cdot\exp(\lambda c^{\mathrm{OPD}})$（RLSD-style）在一阶展开下同构，分析结论一致。

---

## 3. Control-Variate 视角

设 teacher credit 的条件期望相对真实 advantage 有**系统偏差** $\delta$：

$$
\mathbb{E}[\,c^{\mathrm{OPD}}_{i,t}\mid s_{i,t}\,]=A^{*}_{i,t}+\delta_{i,t},
\qquad
\mathbb{E}[\,A^{\mathrm{RL}}_i\mid s_{i,t}\,]=A^{*}_{i,t}+b^{\mathrm{RL}}_{i,t}.
$$

- $\delta$ = **teacher bias**：teacher 偏好方向与真实-reward-最优方向之差（privileged 信息、teacher 偏弱、teacher miscalibration 都进入 $\delta$）。
- $b^{\mathrm{RL}}$ = GRPO 的**分辨率误差**（把整条轨迹的 advantage 平摊到每个 token，token 级有偏但方差大）。
- $\mathrm{Var}(c^{\mathrm{OPD}})\ll\mathrm{Var}(A^{\mathrm{RL}})$：teacher 给定即确定，GRPO 受 Monte-Carlo + 轨迹平摊双重方差。

teacher 信号作为 **control variate**：当它与真实 advantage 正相关时，混入它能降低 credit 估计的方差，代价是引入 $\alpha\delta$ 的偏差。

---

## 4. 方差-偏差分解（核心定理）

以 credit 估计对真实 advantage 的 **MSE** 衡量质量（MSE 小 ⟹ 梯度信噪比高 ⟹ 优化更有效）：

$$
\mathrm{MSE}(\alpha)=\underbrace{\big[\mathrm{Bias}(\alpha)\big]^2}_{\text{偏差项}}+\underbrace{\mathrm{Var}(\alpha)}_{\text{方差项}}.
$$

代入 $\hat c(\alpha)$（略去下标）：

**偏差项**

$$
\mathrm{Bias}(\alpha)=(1-\alpha)\,b^{\mathrm{RL}}+\alpha\,\delta .
$$

**方差项**

$$
\mathrm{Var}(\alpha)=(1-\alpha)^2\mathrm{Var}(A^{\mathrm{RL}})+\alpha^2\mathrm{Var}(c^{\mathrm{OPD}})
+2\alpha(1-\alpha)\,\mathrm{Cov}(A^{\mathrm{RL}},c^{\mathrm{OPD}}).
$$

### 4.1 OPD 何时严格有用

考察在 $\alpha=0$（纯 GRPO）处引入微量 teacher 信号的方向导数：

$$
\left.\frac{d\,\mathrm{MSE}}{d\alpha}\right|_{\alpha=0}
=2\,b^{\mathrm{RL}}(\delta-b^{\mathrm{RL}})
+2\big[\mathrm{Cov}(A^{\mathrm{RL}},c^{\mathrm{OPD}})-\mathrm{Var}(A^{\mathrm{RL}})\big].
$$

当该导数 $<0$ 时，存在 $\alpha>0$ 使 $\mathrm{MSE}(\alpha)<\mathrm{MSE}(0)$，即 **OPD 净有用**。整理出判据：

$$
\boxed{\;\mathrm{Var}(A^{\mathrm{RL}})-\mathrm{Cov}(A^{\mathrm{RL}},c^{\mathrm{OPD}})\;>\;b^{\mathrm{RL}}(\delta-b^{\mathrm{RL}})\;}
$$

直观读法：

- **左边 = 方差缩减红利**。$A^{\mathrm{RL}}$ 方差越大、teacher 与之相关性结构越能抵消其噪声，红利越大。
- **右边 = 偏差代价**，由 teacher bias $\delta$ 主导。

### 4.2 最优混合系数

在固定 $\delta$ 下令 $d\,\mathrm{MSE}/d\alpha=0$，得闭式最优：

$$
\alpha^{*}=
\frac{\mathrm{Var}(A^{\mathrm{RL}})-\mathrm{Cov}(A^{\mathrm{RL}},c^{\mathrm{OPD}})-b^{\mathrm{RL}}(\delta-b^{\mathrm{RL}})}
{\mathrm{Var}(A^{\mathrm{RL}})+\mathrm{Var}(c^{\mathrm{OPD}})-2\,\mathrm{Cov}(A^{\mathrm{RL}},c^{\mathrm{OPD}})+(\delta-b^{\mathrm{RL}})^2},
$$

裁剪到 $[0,1]$。两个极端：

- $\delta\to0$（teacher 无偏，well-calibrated）：$\alpha^{*}\to1$，**应大量使用 teacher**（纯 OPD/压缩安全）。
- $\delta$ 大（privileged / 弱 teacher）：分子转负，$\alpha^{*}\to0$，**应几乎不用 teacher 决定 credit，仅可作纯方差缩减的 control variate（见 §5.2）**。

### 4.3 方向项 vs 力度项的严格对应

上面 $\hat c(\alpha)$ 保持 **sampled-token 方向**不变，只调 credit（力度）。full-vocab/distribution-matching OPD 则额外把梯度**方向**拉向 teacher，对应 $\alpha=1$ 且把 fixed point 移到 $\pi_T$——即 §4.2 中 $\delta$ 直接进入**收敛点**而非仅 credit。因此：

```text
力度收益  = 方差缩减项（不动 fixed point）
方向收益  = 把 fixed point 移向 teacher（仅当 δ<0，即 teacher 比 student 当前更优时为正收益）
```

这就是仓库 [方向力度解耦想法.md](/data1/zhaoenyue/opdpapers/docs/04想法/方向力度解耦想法.md) 中"假设 A（方向）/ 假设 B（力度）"的**理论判据化**：哪个假设成立由 $\delta$ 的符号与量级、以及 $\mathrm{Cov}(A^{\mathrm{RL}},c^{\mathrm{OPD}})$ 决定。

---

## 5. 对现有论文的对应证明

理论的说服力在于：**一个分解同时推出多篇论文的经验结论。**

### 5.1 OPSD Compresses（`2605.06188`）：correct/wrong 非对称

- **Correct rollout**：student 已走对，teacher（同体 privileged 或更强）在这些 token 上偏好与成功 token 一致 ⟹ $c^{\mathrm{OPD}}$ 与 $A^{*}$ 同向，$\delta\approx0$，$\mathrm{Cov}(A^{\mathrm{RL}},c^{\mathrm{OPD}})>0$。判据 §4.1 满足 ⟹ OPD 安全且方差缩减 ⟹ 表现为 **accuracy-preserving compression**。
- **Wrong rollout**：outcome 只说"整条错"，正确续法是 student/teacher 都没产生的 token；teacher 无法定位缺失的中间步 ⟹ $c^{\mathrm{OPD}}$ 与真正需要的 $A^{*}$ 弱相关甚至反向，$\delta$ 大。判据违背 ⟹ 偏差主导 ⟹ **训练在错误轨迹上伤 accuracy**。

> 理论**精确复现** OPSD Compresses 的核心非对称发现，且给出"为什么"：不是 OPSD 本性如此，而是 $\delta$ 在两类轨迹上量级不同。

### 5.2 RLSD（`2604.03128`）：privileged teacher 只能调力度

privileged teacher 看到 reference/trace，其分布编码 student 推理时**不可达**的信息 ⟹ 结构性 $\delta\neq0$（PI leakage 即 $\delta$ 注入 fixed point）。由 §4.2，$\delta$ 大时 $\alpha^{*}\to0$ 用于 credit 混合；但若把 teacher 仅当 **control variate**（减去其期望、只取与 $A^{\mathrm{RL}}$ 相关的波动），可在**不引入 $\delta$ 偏差**的前提下取方差缩减：

$$
\hat c_{i,t}=A^{\mathrm{RL}}_i-\beta\big(c^{\mathrm{OPD}}_{i,t}-\bar c^{\mathrm{OPD}}\big),
\quad \beta^{*}=\frac{\mathrm{Cov}(A^{\mathrm{RL}},c^{\mathrm{OPD}})}{\mathrm{Var}(c^{\mathrm{OPD}})}.
$$

此式中 teacher **只缩放幅度、不决定符号方向**（方向由 $A^{\mathrm{RL}}$ 锚定）——这正是 RLSD"RLVR 定方向、privileged teacher 只调 magnitude"的做法。

> 理论给出 RLSD 的**最优性解释**：当 teacher 有结构性偏差时，control-variate（纯力度）是唯一能取方差红利而不吃偏差的用法。

### 5.3 vOPD（`2605.07865`）：control variate 的特例

vOPD 显式给 sampled-token KL estimator 加 baseline 降方差。在本框架里它是 $\delta=0$（white-box self/teacher，无 PI bias）下的 §5.2 control-variate，$\beta$ 取最优 baseline ⟹ 纯方差缩减项、不动 fixed point。

> vOPD = 本理论的退化特例，验证框架自洽。

### 5.4 Capability ceiling / Extrapolation cliff（`2602.12125`, `2605.08737`）：偏差项

标准 OPD 把 fixed point 钉在 $\pi_T$，对应 §4.3 的方向项：student 收敛到 teacher ⟹ 受 teacher 边界约束（ceiling）。G-OPD/ExOPD 的 reward scaling $\lambda>1$ 等价于人为令 $\delta<0$（外推到比 teacher 更优），而 extrapolation cliff 就是 $\delta$ 外推过度使偏差项反噬。

> ceiling 与 cliff 都是偏差项 $\alpha\delta$ 的两侧表现。

### 5.5 与 SDPG（`2606.04036`）的差异（即新颖性）

SDPG 已指出 full-vocab OPD 与 policy-gradient dense reward 在目标上相连。**本文新增**：(i) 把它细化为方差-偏差**分解**；(ii) 给出 control-variate **最优系数** $\alpha^{*}/\beta^{*}$；(iii) 给出 **OPD 净有用的可测量判据**（§4.1）；(iv) 用单一分解**跨 correct/wrong/PI 三 regime 复现** §5.1–5.4 的经验结论。SDPG 是"它们相连"，N1 是"分解 + 判据 + 律"。

---

## 6. 可测量 $\mathrm{Cov}(\text{teacher},\text{advantage})$ 的实验协议

这一节是 N1 从"理论"变成"可验证论文"的关键：它把 §4.1 判据里的每个量都落到**能从训练日志或离线 rollout 真算出来的数**。

先把判据抄回来当锚：

$$
\underbrace{\mathrm{Var}(A^{\mathrm{RL}})-\mathrm{Cov}(A^{\mathrm{RL}},c^{\mathrm{OPD}})}_{\text{方差缩减红利（左边）}}\;>\;\underbrace{b^{\mathrm{RL}}(\delta-b^{\mathrm{RL}})}_{\text{偏差代价（右边）}}
$$

**目标**：把左右两边都算出来，就能在**真跑 OPD 之前**预测它对某个区域有没有用。整节要估的就是其中 5 个量：$\mathrm{Var}(A^{\mathrm{RL}})$、$\mathrm{Var}(c^{\mathrm{OPD}})$、$\mathrm{Cov}(A^{\mathrm{RL}},c^{\mathrm{OPD}})$、$b^{\mathrm{RL}}$、$\delta$。

> **与已有预训练诊断的关系（重要）**：已有一篇 Unmasking On-Policy Distillation（`2605.10889`）提出训练前判断 OPD 何时帮/伤的 training-free 诊断，因此"预训练 go/no-go 诊断"这个目标**本身已被占据**——本节不能宣称首创。但它用的是"理想成功梯度与蒸馏梯度的 **cosine 对齐**"，且需要**每题 45K–200K 条 rollout**。本节判据的差异化与成本优势是核心卖点：判据**左边可从训练日志免费算**（§6.0），不需大规模额外采样。完整对标见 §9.3。

### 6.0 关键洞察：大部分量"免费"，只有偏差项"贵"

这是落地时最重要的一点。把 5 个量按"是否需要昂贵的反事实续写"分两类：

| 量 | 是否可观测 | 怎么来 | 成本 |
|---|---|---|---|
| $A^{\mathrm{RL}}_i$ | ✅ 直接 | GRPO 的 group-normalized reward，训练本来就算 | 免费 |
| $c^{\mathrm{OPD}}_{i,t}$ | ✅ 直接 | teacher logprob $-$ student logprob，各前向一次 | 免费 |
| $\mathrm{Var}(A^{\mathrm{RL}}),\,\mathrm{Var}(c^{\mathrm{OPD}}),\,\mathrm{Cov}(A^{\mathrm{RL}},c^{\mathrm{OPD}})$ | ✅ 直接 | 两个**可观测量**的方差/协方差，一个 batch 内即可统计 | 免费 |
| $A^{*}_{i,t}$（真实 advantage） | ❌ 不可观测 | 必须用 proxy $\hat A^{*}$（§6.1） | 贵 |
| $\delta,\,b^{\mathrm{RL}}$（两个偏差项） | ❌ 间接 | 需要 $\hat A^{*}$ | 贵 |

**因此判据左边完全免费**（两个可观测量的二阶矩），**只有右边的偏差代价需要反事实估计**。实操策略：先用免费的左边对所有区域粗筛，只在判据接近临界（左 $\approx$ 右）的区域才花钱估右边。下面 §6.1 整节就是在解决"怎么估 $\hat A^{*}$"这一个难点。

### 6.1 估计真实 advantage 的 proxy $\hat A^{*}$

$A^{*}_{i,t}=Q^{*}(s_{i,t},a_{i,t})-V^{*}(s_{i,t})$ 不可直接观测——它是"在状态 $s_{i,t}$ 选了 student 这个 token $a_{i,t}$，相比该状态平均水平，最终多拿多少 reward"。用两种 proxy 估计，互为 robustness check。

**方式 1：短续写反事实（counterfactual lookahead）—— 原理性做法**

两项都用 Monte-Carlo 续写估计，全程在 student policy $\pi_\theta$ 下：

1. $\hat Q(s_{i,t},a_{i,t})$：**固定前缀 $+$ 这个 token**，往后续写到终止 $K$ 次，取 verifier reward 均值。
2. $\hat V(s_{i,t})$：**只固定前缀**（下一个 token 重新从 $\pi_\theta$ 采），续写 $K$ 次，取均值。
3. $\hat A^{*}_{i,t}=\hat Q-\hat V$。

直觉：若"锁定 student 这个 token 后成功率明显高于该状态平均"，说明它是因果上重要的好 token，$\hat A^{*}>0$。

三点注意：

- 全程在 student policy 下续写，所以 $\hat A^{*}$ 是 on-policy 的真实 advantage，**算 $\delta$ 只需 student 自己 token 的 $\hat A^{*}$**，不必强制 teacher token（强制 teacher token 是 §4.3 方向分析的扩展）。
- 成本 $\approx$ 每个被测 token $2K$ 条续写，因此**只在高 student-entropy 的分支点 $+$ 一个 prompt 子集**上做，离线一次性算完，不进训练 loop。与 [研究方向与候选想法.md](/data1/zhaoenyue/opdpapers/docs/04想法/研究方向与候选想法.md) Idea 1 复用同一 pipeline。
- 省算力技巧：student 实际那条 rollout 本身就是 $\hat Q$ 的一个样本，可复用；$\hat Q$ 与 $\hat V$ 共享前缀，可用公共随机数 / antithetic 降方差。

**方式 2：outcome 回归 proxy —— 零成本下界对照**

不做任何续写，直接用 rollout 正误当粗 proxy：正确 rollout 的 token 记 $\hat A^{*}\!\approx\!+$，错误 rollout 记 $\approx\!-$（或更细：与已知正确解对齐的 token 为正，偏离处为负）。它很糙（把 trajectory-level 信号当 token-level），但**零额外算力**，用来和方式 1 交叉验证：两种 proxy 算出的判据符号一致，则结论稳。

### 6.2 待测统计量（每个 setting / 每个分桶各算一遍）

设在某个分桶里收集了 $N$ 个 token 样本 $\{(A^{\mathrm{RL}}_i,\,c^{\mathrm{OPD}}_{i,t},\,\hat A^{*}_{i,t})\}$。

**(a) 免费的三个二阶矩**（只用前两列，不需 $\hat A^{*}$）：

$$
\widehat{\mathrm{Cov}}=\frac1N\sum_{i,t}\big(A^{\mathrm{RL}}_i-\bar A^{\mathrm{RL}}\big)\big(c^{\mathrm{OPD}}_{i,t}-\bar c^{\mathrm{OPD}}\big),
$$

$\widehat{\mathrm{Var}}(A^{\mathrm{RL}})$、$\widehat{\mathrm{Var}}(c^{\mathrm{OPD}})$ 同理。注意 $A^{\mathrm{RL}}_i$ 在一条轨迹内对 $t$ 是常数，方差来自**轨迹间**；$c^{\mathrm{OPD}}_{i,t}$ 逐 token 变化。

**(b) 两个偏差项**（用上 $\hat A^{*}$）：

$$
\hat\delta=\frac1N\sum_{i,t}\big(c^{\mathrm{OPD}}_{i,t}-\hat A^{*}_{i,t}\big),
\qquad
\hat b^{\mathrm{RL}}=\frac1N\sum_{i,t}\big(A^{\mathrm{RL}}_i-\hat A^{*}_{i,t}\big).
$$

**(c) 判据符号预测**：代入

$$
\widehat\Delta=\widehat{\mathrm{Var}}(A^{\mathrm{RL}})-\widehat{\mathrm{Cov}}(A^{\mathrm{RL}},c^{\mathrm{OPD}})-\hat b^{\mathrm{RL}}(\hat\delta-\hat b^{\mathrm{RL}}).
$$

$\widehat\Delta>0$ ⟹ 预测"OPD 在这个桶里有用"。同一批数代入 §4.2 闭式即得预测的 $\hat\alpha^{*}$。

### 6.2.1 为什么必须分桶

理论说判据成立与否**随区域变化**（$\delta$ 在 correct/wrong 上量级不同，正是 §5.1 OPSD 非对称的来源）。若不分桶、在全体上算一个平均判据，correct 桶的方差红利会被 wrong 桶的偏差代价稀释，**相变就被抹平、看不出来**。所以必须按以下轴各算一遍 $(\widehat\Delta,\hat\alpha^{*})$，合起来就是一张"OPD 在哪些区域有用"的相图：

- **correct vs wrong rollout** —— 最重要，直接对应 $\delta$ 大小；
- **high/low teacher entropy、high/low student entropy、high/low teacher-student divergence** —— 对应 §4.2 里 $\alpha^{*}$ 该大该小的区域；
- **early reasoning token vs late verification token** —— 长轨迹里 $\delta$ 随位置漂移（呼应 SFD / prefix drift 主线）。

### 6.3 验证设计（确认理论的预测力）

固定 Qwen3-1.7B/4B teacher-student pair + MATH/LiveCodeBench：

| 实验 | 做法 | 证明什么 | 怎么读结果 |
|---|---|---|---|
| **E1**（主结果） | 多组 setting（teacher 强弱、是否 PI、correct/wrong 子集）先用 §6.2 算 $\widehat\Delta$ 符号，再**真跑** OPD 看 Δaccuracy | 判据有**预测力** | 报预测符号 vs 实际增益的混淆矩阵 + $\widehat\Delta$ 与 Δaccuracy 的相关系数；相关性高 = 理论成立 |
| **E2** | 扫 $\alpha\in\{0,0.25,0.5,0.75,1\}$ | 闭式 $\alpha^{*}$ 是否落在实测最优附近 | $\hat\alpha^{*}$ 与经验最优 $\alpha$ 接近 = §4.2 对 |
| **E3** | 一套代码复现 §5.1（OPSD 非对称）、§5.2（RLSD PI 只调力度）、§5.4（ceiling/cliff） | 三个独立现象**同源** | 三者都由各自桶的 $\hat\delta$ 量级解释 |
| **E4** | 训练中记录诊断曲线 | 机制随训练演化可见 | 见 §6.4 |

E1 是论文核心卖点：**"训练前算一个标量 $\widehat\Delta$，就能预测 OPD 会不会有用，准确率 X%"** —— 正是现状文档点名缺的 go/no-go detector（#1/#2）。

### 6.4 主要指标

不能只看最终 accuracy（它是终点，看不出是方差缩减还是偏差在起作用），同时报：

- **credit MSE 对 $\hat A^{*}$** —— 直接量化"credit 估计离真值多远"，是理论的因变量；
- **gradient SNR** —— 方差缩减红利的直接体现；
- **$\cos(g^{\mathrm{RL}},g^{\mathrm{OPD}})$** —— 区分"方向相同（靠力度）"还是"方向不同（靠方向）"，呼应 §4.3；
- **$\hat\delta$ 随训练演化** —— teacher bias 是否随 student 变强而漂移；
- 配套任务指标：pass@1 / pass@k、verifier reward、policy entropy、KL-to-teacher、E1 的预测-实际混淆矩阵。

### 6.5 实操要点与坑

1. **先免费后付费**：判据左边免费，先全桶扫一遍；只在 $\widehat\Delta\approx0$ 的边界桶上才花反事实算右边。
2. **$\hat A^{*}$ 噪声是最大风险**：$K$ 太小则 $\hat Q,\hat V$ 抖。对策——主结果只依赖判据**符号**（robust）而非精确值；两种 proxy 交叉验证；$\hat Q,\hat V$ 用公共随机数降方差。
3. **logprob 口径要一致**：$c^{\mathrm{OPD}}$ 的 teacher/student logprob 必须同 tokenizer、同 temperature 约定，否则 log-ratio 失真。
4. **分桶样本量**：wrong-rollout 桶在强 student 上可能很少，需要足够 prompt 才有统计功效。

---

## 7. 预期结论与论文卖点

**最可能结论**（与现有证据一致）：OPD 收益**不是**单一来源，而由 $\delta$ 与 $\mathrm{Cov}$ 决定的**相变**：

```text
δ≈0 且 Cov 高（correct rollout / 强 calibrated teacher）
  → 方差缩减主导 → 大胆用 teacher（压缩 / sampled-OPD / full-vocab 皆可）

δ 大（PI teacher / 弱 teacher / wrong rollout）
  → 偏差主导 → teacher 仅作 control variate 调力度（RLSD），不能定方向

δ<0（teacher 真比 student 当前优，strong-to-weak）
  → 方向收益为正 → full/top-k OPD 决定方向有意义（但 extrapolation 过度会触 cliff）
```

**论文卖点**：

1. **统一**：一个方差-偏差分解复现 ≥4 篇独立论文的经验结论（§5）。
2. **可预测判据**：§4.1 给出训练前可算、能 go/no-go 的标量条件，回应现状文档点名的 #1/#2 空白。
3. **解决长期争论**：给"方向 vs 力度"严格答案（§4.3）。
4. **可行**：实验臂是已设计好的 factorial design，小模型 + 数学/代码即可，不需大规模训练（区别于 N2）。

**主要风险与对策**：

- $\hat A^{*}$ 估计噪声 → 用 §6.1 两种 proxy 交叉验证，且主结果只依赖判据**符号**而非精确值。
- 一阶展开（$\alpha=0$ 处导数）可能不足以刻画大 $\alpha$ → 同时报 §4.2 闭式 $\alpha^{*}$ 的全区间数值验证（E2）。
- 乘性 vs 加性 blending 的等价性 → 在附录做一阶等价证明 + 数值对照。

---

## 8. 与仓库其他文档的关系

- 上游问题与 8 候选排序：[研究方向与候选想法.md](/data1/zhaoenyue/opdpapers/docs/04想法/研究方向与候选想法.md)
- 实验臂（A0–A6 factorial、诊断指标、分组口径）：[方向力度解耦想法.md](/data1/zhaoenyue/opdpapers/docs/04想法/方向力度解耦想法.md)
- $\hat A^{*}$ 反事实 pipeline 复用：研究方向文档 Idea 1（Counterfactual Teachability）
- 失败模式归因与论文 ID 对照：[失败模式映射.md](/data1/zhaoenyue/opdpapers/docs/02图谱/失败模式映射.md)、[失败模式研究现状.md](/data1/zhaoenyue/opdpapers/docs/02图谱/失败模式研究现状.md)

---

## 9. 相关工作与差异化（2026-06-22 带引用 novelty 核查）

来源：一次 deep-research 系统 novelty 核查（5 个搜索角度、16 篇主源、66 条断言抽取、25 条经 3 票对抗式验证）。下列 arXiv 论文均经核查确认为**真实可考**（arXiv / HuggingFace 可解析）。**总裁决**：N1 的**机制**（teacher-as-control-variate 的偏差-方差分解 + 协方差 go/no-go 判据）未被任何已查论文占据、看起来真新；但它**统一/复现的现象多为已发表结论**，因此 novelty 必须收窄到"统一分解 + 廉价判据"，而**不**主张那些现象本身是发现。

### 9.1 已被现有工作占据的部分（写论文时一律措辞为"we re-derive the known result of [cite]"，不作为 finding）

| N1 子主张 | 已被谁发表 | 引用 | 核查票 |
|---|---|---|---|
| (e) PG 统一：RLVR 与 OPD = 同一 policy-gradient 模板，仅 advantage 估计器不同 | 措辞几乎一字不差 | Self-Distilled RLVR `2604.03128`、SRPO `2604.02288`、OPSD `2601.18734`、G-OPD `2602.12125` | 3-0 |
| (b) 方向-力度解耦：teacher 只调 magnitude，环境 reward 定 direction | **正是 RLSD 的核心方法** | `2604.03128`（摘要原文 "teacher's evidence ratio modulates only the magnitude"） | 3-0 |
| (a) correct/wrong 非对称：teacher 信号在错误轨迹上更有用，正确轨迹上变噪声 | 经验已记录 | Unmasking OPD `2605.10889`、SCOPE `2604.10688`、SRPO `2604.02288`、BRTS `2605.09725` | 3-0 |
| (d) capability ceiling / extrapolation cliff | G-OPD 已处理，连 λ=1.5 处的 cliff 都有 | `2602.12125` | 3-0 |
| 部分梯度分解（但为 **MI leakage** 视角，非 control-variate） | 最接近的结构类比，不构成 pre-empt | RLSD `2604.03128` Prop 1（marginal + r-specific deviation，方差随互信息单调增） | 3-0 |
| 经典监督蒸馏的偏差-方差权衡（非 RL） | 仅共享抽象思想 | `2102.00650`（CVPR'21, Rethinking Soft Labels） | 3-0 |

### 9.2 N1 真正新颖的部分（核查 3-0 未被占据）

- **teacher 作为 control variate**（RQ1）：6 篇主源全文搜 "control variate" **零命中**；OPSD `2601.18734` 被显式**反驳**（0-3）——其 teacher 是固定全分布目标，不是方差控制装置。
- **偏差-方差分解**，分离"不动 fixed point 的方差缩减"与"移动 fixed point 的偏差"（RQ2）：LLM-RL 语境下无人做过。
- **`Var(A_RL) − Cov(A_RL, c_OPD) > b_RL(δ − b_RL)` 判据**（RQ3）：无任何论文含 Var/Cov/bias 形式的 go/no-go 标量。

### 9.3 最接近的竞品与差异化：Unmasking OPD（`2605.10889`）

它**已经**提出训练前 training-free 的 OPD go/no-go 诊断，所以"预训练诊断"这一目标已被占据。N1 的差异化（须在 §6 与实验里正面对标）：

| 维度 | Unmasking OPD `2605.10889` | N1 |
|---|---|---|
| 判据形式 | 理想成功梯度 vs 蒸馏梯度的 **cosine 对齐** | **Var/Cov/bias 协方差标量** |
| 成本 | **每题 45K–200K 条 rollout** | 判据左边从训练日志**免费**算，仅边界区需少量反事实（§6.0） |
| 解释力 | 诊断"是否对齐" | 同一分解再**推出** correct/wrong、方向-力度、ceiling 等现象 |

### 9.4 收窄后的 novelty 一句话声明

> 我们不是首次发现方向-力度解耦、correct/wrong 非对称或 capability ceiling，也不是首个提出预训练 OPD 诊断；我们是**首个把 teacher 信号显式当作 control variate、给出 OPD 收益的偏差-方差分解、并由此推出一个可从训练日志近乎免费计算的协方差 go/no-go 判据，且用同一分解从单一原理统一解释上述已知现象**。

### 9.5 reviewer 会追问的风险（需提前在论文里防御）

1. **control variate 是经典 RL 技术**（Q-prop、Stein control variates、action-dependent baselines）。本次核查**未穷尽**这条 pre-LLM 经典线（见核查 openQuestion）。防御：证明"teacher-as-control-variate"给出经典 baseline 理论给不出的**非平凡预测**——尤其 correct/wrong 相变、$\delta$ 符号决定方向收益。
2. **"一个分解真能同时推出全部四个现象，还是每个要单独假设？"** —— 这是 N1 自身最大的未验证理论点（§7 风险已列），审稿必查。需在理论部分对四个现象**逐一给出可控假设下的推导**，不能只在 §5 给定性对应。
3. **时效性**：该领域极快，novelty 裁决基于"已查集合内的缺席"，无法证明全网无人做过；投稿前应重跑一次核查。
