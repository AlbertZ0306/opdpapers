# P5：On-Policy Distillation 的收益是知识迁移还是方差缩减？

更新时间：2026-06-23

定位：本文是 [研究方向与候选想法.md](/data1/zhaoenyue/opdpapers/docs/04想法/研究方向与候选想法.md) 中经 novelty 核查后**重排第一**的候选 **P5** 的完整展开。它是一篇 **deflationary（去魅型）经验归因论文**，把 [OPD方差偏差分解理论.md](/data1/zhaoenyue/opdpapers/docs/04想法/OPD方差偏差分解理论.md)（N1）唯一幸存的新资产——"teacher 信号 = control variate"的视角——从"统一理论"重定位为"一个可证伪的经验论断的分析工具"。N1 文档由此降级为本文的**理论附录**。

---

## 1. 一句话论点（可当标题）

> **Is On-Policy Distillation Knowledge Transfer or Variance Reduction?**
>
> 在标准数学/代码推理的 OPD 设置里，我们用控制实验证明：OPD 相比 RLVR 的大部分增益来自对稀疏 RL credit 的**方差缩减**，而非 teacher 分布带来的**新知识 / 新更新方向**；一个**完全不碰 teacher logits、不注入任何外部知识**的 teacher-free 方差缩减 RLVR，能复现其中 $\rho$ 比例的增益。我们刻画 $\rho$ 何时接近 1（OPD ≈ 高级 baseline）、何时接近 0（teacher 知识真有用），并给出可证伪的 regime 相图。

**关键卖点**：这是一个**两种结果都能发**的设计——$\rho$ 高是 deflationary 大新闻，$\rho$ 低是"teacher 方向确有新知识"的干净正面结论（且顺带反驳 N1 的偏差-方差假说，本身可发）。

---

## 2. 为什么这个问题值得做（在饱和领域里的站位）

到 2026 年中，OPD 子领域已高度饱和：PG 统一、方向-力度解耦、correct/wrong 非对称、capability ceiling、token teachability、estimator 方差、privileged context 治理、long-horizon、rubric/black-box、safety——**每条都已有论文**（见 [失败模式映射.md](/data1/zhaoenyue/opdpapers/docs/02图谱/失败模式映射.md)）。在这种领域里，"再提一个机制 / detector / router"几乎必然增量。

能进顶会的只剩三类动作：(i) 攻击全领域共享假设并给反直觉结论；(ii) 进入无人前沿；(iii) 用严格混淆分析证明别人宣称的收益来自别处。**P5 属于 (iii)**：整个 OPD 叙事默认 teacher 提供的是"student 学不到的知识/分布"，P5 质疑这个默认。

经 [带引用 novelty 核查](#9-与现有工作的差异化经核查) 确认：**没有论文**把 teacher 当 control variate、**没有论文**做 OPD 收益的"方差缩减 vs 知识迁移"归因、**没有论文**构造 teacher-free 方差缩减对照来量化 $\rho$。最接近的两篇也只到现象、不到归因（§9）。

---

## 3. 两个对立假说与可证伪预测

把 OPD 相对 RLVR 的增益拆成两种互斥来源：

**H-VR（方差缩减假说）**：OPD 好，是因为 teacher 的 dense per-token 信号把 RLVR 稀疏、轨迹级、高方差的 credit 估计变得低方差、高分辨率——**这不改变优化的 fixed point**，本质等价于一个更好的 baseline。

**H-KT（知识迁移假说）**：OPD 好，是因为 teacher 提供了 student 自己探索 + RL 到不了的**新方向 / 新分布**——**这移动 fixed point**，是真正的能力注入。

用 N1（§4）的语言，这正是"力度收益（方差缩减、不动 fixed point）"vs"方向收益（移动 fixed point）"之分。本文不证明 N1 理论，而是**用它生成可证伪预测，再用实验裁决**：

```text
若 H-VR 主导：
  - teacher-free 方差缩减 RLVR 应复现大部分增益（ρ→1）
  - OPD 不应解锁 GRPO + 方差缩减都解不开的新题（pass@k 覆盖集不扩大）
  - ρ 在 correct-rollout 占多 / calibrated 或 self-teacher / in-distribution 时最高

若 H-KT 主导：
  - teacher-free 复现率低（ρ→0）
  - OPD 解锁新题（覆盖集扩大）
  - ρ 在 strong-to-weak distillation / wrong-rollout 多 / OOD 时最低
```

这给出一张 **regime 相图**，无论结果偏哪边都是结构化结论。

---

## 4. 形式化：把增益分到"方差缩减"与"方向/知识"

沿用 N1 的统一记号（详见 [OPD方差偏差分解理论.md](/data1/zhaoenyue/opdpapers/docs/04想法/OPD方差偏差分解理论.md)）。任意方法的更新写成对真实 token-level advantage $A^{*}$ 的某个 credit 估计：

$$
\Delta\theta\propto\sum_{i,t}\hat c_{i,t}\,\nabla_\theta\log\pi_\theta(a_{i,t}\mid s_{i,t}).
$$

- GRPO：$\hat c_{i,t}=A^{\mathrm{RL}}_i$（轨迹级、高方差、零分辨率）。
- OPD：$\hat c_{i,t}$ 含 teacher log-ratio $c^{\mathrm{OPD}}_{i,t}=\log\pi_T-\log\pi_\theta$（dense、低方差、**可能有偏 $\delta$**）。

N1 的判据（§4.1）说 OPD 的净作用 = 方差缩减红利 $-$ 偏差代价。**P5 的操作化定义**：

- **方差缩减成分**：用任何**不注入外部知识**的手段（只用 policy 自身的信号）把 $A^{\mathrm{RL}}$ 的方差降下来所能带来的增益。
- **知识/方向成分**：teacher 把 fixed point 移向 $\pi_T$ 所带来的、上一项无法复现的剩余增益（$\delta<0$ 即 teacher 真更优时为正）。

核心可测量 = **复现率**：

$$
\rho=\frac{\mathrm{Acc}(\text{teacher-free VR})-\mathrm{Acc}(\text{GRPO})}{\mathrm{Acc}(\text{OPD})-\mathrm{Acc}(\text{GRPO})}.
$$

$\rho\to1$ 支持 H-VR；$\rho\to0$ 支持 H-KT；中间值给出两者的经验配比。

---

## 5. 关键设计：teacher-free 方差缩减 RLVR（实验的 wedge）

整篇论文成败系于一个对照基线：**它必须只用 policy 自身的信号降低 RL credit 方差，注入零外部知识。** 这样"它能复现 OPD 增益"才能合法推出"OPD 增益不是知识"。给出一族强度递增的设计（同时报，互为消融）：

**B1 学习型 token-level baseline（actor-critic 式 dense credit）**
在 policy 自身 rollout 上训一个 value head $V_\phi(s)$，用 $\hat c_{i,t}=R_i-V_\phi(s_{i,t})$ 做 dense、逐 token 的 credit。纯粹降方差、零外部知识。

**B2 control-variate 化的 $A^{\mathrm{RL}}$（N1 §5.2 的 teacher-free 版）**
用一个**廉价 token-level 预测器** $\hat m_{i,t}$（预测 $A^{\mathrm{RL}}$ 的可预测部分，仅基于 policy 自身特征）作 control variate：

$$
\hat c_{i,t}=A^{\mathrm{RL}}_i-\beta\big(\hat m_{i,t}-\bar m\big),\qquad
\beta^{*}=\frac{\mathrm{Cov}(A^{\mathrm{RL}},\hat m)}{\mathrm{Var}(\hat m)}.
$$

这是"纯方差缩减、不动 fixed point"的最干净实现——直接对应 N1 把 teacher 替换成 teacher-free 预测器。

**B3 self-distillation 但零新知识（同策略熵/一致性正则）**
用 policy **自己当前的分布**（非外部/非更强 teacher）做 dense 信号，例如 self-consistency / EMA-self 上的轨迹级一致性。它能降方差但定义上不引入新知识，作为"知识=0 但 dense"的上界对照。

**B4 CAST（`2606.00172`，收编现成 teacher-free dense 信号）**
直接采用 CAST 的 answer-free、stop-gradient self-teacher（observe prompt + 已生成 prefix）塑造 token-level advantage。它已是一个**经验上跑通的 teacher-free dense 基线**，但原文从未把它当方差缩减分析、不算 $\rho$。P5 把它**重新解释并投到 $\rho$ 谱系上**：这既省了从零造基线的工程量，又把 P5 与最危险的邻居（构造重叠）转成"我们给它做了它没做的归因"的正面区分。

> B1/B2/B3/B4 都满足"零外部知识"约束，区别只在降方差的强度与机制。若其中**任何一个**就复现了 OPD 大部分增益 → H-VR 强证据。

**对照组完整列表**：GRPO（$\rho$ 的分母基线）、OPD（$\rho$ 的分子目标）、B1/B2/B3/B4（teacher-free VR）。可选加 RLSD / vOPD 作为"已有方法落在 $\rho$ 谱系何处"的参照。B2 的 control variate 有一个**理论锚点**：vOPD 已证 OPD 的最优控制变量是 teacher 逐 token reverse-KL，B2 的 teacher-free 预测器越逼近这个量，$\rho$ 越能合法分离 VR 与 KT。

---

## 6. 正交的"知识解锁"探针（不依赖 $\rho$ 的第二条证据）

$\rho$ 比较的是平均 accuracy，可能被"两种成分恰好幅度相近"混淆。加一条**独立**判据：**OPD 是否解锁了 GRPO + teacher-free VR 都解不开的题。**

- 对每个方法测 **pass@k 覆盖集**（k 较大，逼近能力上界），统计可解题目集合 $S_{\text{method}}$。
- **H-VR 预测**：$S_{\text{OPD}}\subseteq S_{\text{GRPO}}\cup S_{\text{VR}}$（OPD 不扩大可达集，只是更稳更快达到）。
- **H-KT 预测**：$S_{\text{OPD}}\setminus(S_{\text{GRPO}}\cup S_{\text{VR}})\neq\varnothing$（存在只有 teacher 能解锁的题）。

覆盖集差是"知识迁移"最直接的指纹：方差缩减只会让你**更可靠地拿到本就够得着的解**，不会凭空扩大可达边界。

---

## 7. 实验设计（按 regime，结果导向）

固定 Qwen3-1.7B/4B + MATH/LiveCodeBench，沿用 [方向力度解耦想法.md](/data1/zhaoenyue/opdpapers/docs/04想法/方向力度解耦想法.md) 的分组口径。

**E1（主结果）：全局 $\rho$ 与覆盖集**
在标准 self-teacher OPD（最常见设置）上测 $\rho$ 和覆盖集差。这是论文的头条数字。

**E2（相图）：$\rho$ 随 regime 变化**
按以下轴各测一个 $\rho$，验证 §3 的相图预测：
- self-teacher（$\delta\approx0$）vs strong-to-weak（$\delta<0$，teacher 真更强）；
- correct-rollout 占多 vs wrong-rollout 占多；
- in-distribution vs OOD（student 够不着的难度）。
预测：$\rho$ 在第一类高、第二类低，且与 N1 判据的 $\hat\delta$ 量级负相关。

**E3（机制确认）：方差与 fixed point**
直接测 teacher-free VR 是否真把 credit 方差降到与 OPD 同量级（gradient SNR、credit 方差），以及 OPD 的 fixed point 是否真被拉向 teacher（KL-to-teacher 演化）。把 $\rho$ 高/低和"方差是否降够 / fixed point 是否移动"对上。

**E4（对照定位）**：把 RLSD、vOPD 投到 $\rho$ 谱系上，说明已有"力度类"方法本质落在方差缩减区。

---

## 8. 预期结论的两种形态（都可发）

**形态 A（deflationary，更轰动）**：$\rho$ 在主流 self-teacher 数学推理设置里高（如 $>0.7$），覆盖集不扩大。结论：**"主流 OPD 的推理增益主要是方差缩减，可用 teacher-free baseline 近乎复现；teacher 的 distribution 在这些设置里没带来可迁移的新知识。"** 这与 OPSD Compresses（`2605.06188`，"compression not correction"）方向一致但更进一步——给出**可 teacher-free 复现**的强归因。

**形态 B（正面，安全网）**：$\rho$ 低、覆盖集扩大，尤其在 strong-to-weak / OOD。结论：**"teacher 确实注入了新方向/知识，且仅出现在可预测的 regime（$\delta<0$）。"** 这把"OPD 何时真做知识迁移"钉死成一个可预测条件，本身是干净贡献，且实证反驳了 N1 的 H-VR 主导假设。

无论 A/B，**regime 相图 + 复现率 $\rho$ + 覆盖集探针**三件套都成立。

---

## 9. 与现有工作的差异化（经两轮带引用核查）

> 第二轮核查（2026-06-23，98 agent / 16 主源 / 25 条断言三票对抗核验）裁决：P5 的三件套核心**仍然 open、未被任何单篇预占**——VR-vs-KT 归因（3-0 未预占）、teacher-free VR 对照 + $\rho$（最稳新资产，3-0）、覆盖集解锁探针用于 teacher-vs-teacher-free（方法成立但无人这样用，3-0）。但周边已高度密集，护城河是"把分散组件拼成归因方法论"，成败取决于 **positioning** 而非 novelty 是否存在。下表已并入第二轮新发现的邻居。

| 现有工作 | 做了什么 | P5 多了什么 / 如何区分 |
|---|---|---|
| **CAST `2606.00172`（最危险邻居，新增）** | 已构造 **teacher-free dense 信号**：answer-free、stop-gradient 的 self-teacher 在 GRPO 上塑造 token-level advantage——与 P5 资产②的**构造**高度重叠 | 全文 0 次 "control variate / variance reduction / baseline"，框成 "credit assignment refinement"，**不做 VR-vs-KT 归因、不算 $\rho$、无覆盖集探针**。→ P5 应把 CAST **收编为一个 teacher-free VR 基线**（B 系列之一），而非假装无人做过 teacher-free dense 信号 |
| Kim et al. `2505.14216`（NeurIPS 2025） | KT-vs-not 归因**已存在**：accuracy(pass@1) vs capability(pass@k)，推测"只有引入新知识才提升 capability，否则蒸馏 pattern ≈ RLVR" | 但仅针对 **offline teacher 蒸馏**，非 on-policy、非 per-token dense reward、**无 teacher-free 对照、无 $\rho$、无相图**。P5 的二分法在 on-policy + 可量化复现上才成立 |
| Yue et al. `2504.13837`（NeurIPS 2025） | 立了 **pass@k 覆盖集方法学**（小 k RLVR 赢、大 k base 反超） | 用在 RLVR-vs-base，**没用在 teacher-vs-teacher-free 蒸馏对照**。P5 借其作为资产③的探针仪器，但用途全新 |
| OPSD Compresses `2605.06188` | 发现 OPSD 在正确轨迹是压缩、错误轨迹可能伤 accuracy（thinking-enabled 数学，2-1 通过含 scope 警告） | 没把"压缩"归因到方差缩减、没造 teacher-free 基线量化可复现性。P5 给出**可 teacher-free 复现**的强归因 |
| Unmasking OPD `2605.10889` | 训练前梯度-cosine 诊断 OPD 何时帮/伤（方向轴，非方差轴） | P5 不是诊断，而是**归因 + 去魅**；不依赖昂贵 per-question rollout |
| RLSD `2604.03128` | 方向-力度解耦的**方法**（teacher 只调 magnitude，方向锚定环境 reward） | P5 把"力度 = 方差缩减、方向 = 知识"**经验量化**，并问"力度成分能否完全 teacher-free" |
| vOPD `2605.07865` | OPD 的 control-variate 降方差，最优基线 = teacher 逐 token reverse-KL；**unbiased（不移动 OPD fixed point）**，但**基线 teacher-derived、不与 GRPO 比** | P5 用 **teacher-free** control variate 检验"整个 teacher 是否可被方差缩减替代"——更激进，且正面攻 OPD-vs-GRPO 的 gap |

**未被占据的核心**：teacher-free 方差缩减对照（含收编 CAST）+ 复现率 $\rho$ + 覆盖集解锁探针 这一**归因方法论**，以及"OPD ≈ 高级 baseline"这一**可证伪经验论断**。

**核查暴露的硬约束（写进风险）**：
- **"RL/OPD 受限于 base、不移动 fixed point"在文献里有争议**——核查中"RLVR 不能解锁新能力"被 **0-3 驳回**（ProRL `2505.24864` 证明长时 RL 能扩展 pass@k 边界）。→ P5 的覆盖集论证**不能**当作"RL 理论上解锁不了"的既定事实，必须 hedge 成 **regime/方法相关**，并把 ProRL 列为反例对照。
- vOPD 已证 OPD 最优控制变量是 teacher 的逐 token reverse-KL，这反而给 P5 的 B2（teacher-free control variate）一个**明确的逼近目标**：teacher-free 预测器越接近这个量，$\rho$ 越能合法分离 VR 与 KT。

---

## 10. 风险与防御

1. **基线不公平质疑**（"你 teacher-free VR 没调好，所以显得 OPD = VR"）。
   防御：报 B1/B2/B3 三个独立 VR 设计 + 各自超参 sweep；只要**任一**复现大部分增益即支持 H-VR；并用 E3 证明 VR 确实把方差降到 OPD 量级（否则不算公平复现）。
2. **"零外部知识"边界争议**（self-distillation/EMA-self 算不算引入知识？）。
   防御：B1/B2 完全不含任何 teacher/更强模型/更优分布，只用 policy 自身 value 与统计；B3 仅作上界、单独标注。
3. **$\rho$ 被两成分幅度巧合混淆**。
   防御：覆盖集探针（§6）是不依赖幅度的正交证据。
4. **领域/规模外推**（只在小模型数学上成立？）。
   防御：E2 已含 OOD 与 strong-to-weak；明确把结论限定为"在所测 regime"，并把相图当作主贡献而非单点结论。
5. **"RL 解锁不了新能力"是争议前提，不能当公理**（核查中此类断言被 ProRL `2505.24864` 0-3 驳回）。
   防御：覆盖集探针（§6）只作为**经验指纹**报告"在本设置下 OPD 是否扩大可达集"，绝不论证"RL 理论上无法解锁"；把 ProRL 列为"长时 RL 能扩边界"的反例对照，结论 hedge 成 regime/方法相关。
6. **与 CAST 构造撞车**（审稿人："teacher-free dense 信号 CAST already did it"）。
   防御：CAST 是**方法**、P5 是**归因方法论**；把 CAST 收编为 B4 基线，论文卖点是"我们对包括 CAST 在内的 teacher-free 信号做了它们都没做的 VR-vs-KT 归因 + $\rho$ + 覆盖集探针"。
7. **时效**：领域极快，投稿前重跑一次 novelty 核查。

---

## 11. Kill test（无现成 checkpoint 版，投入前必做）

> 前提更正：本仓库**没有**现成的 GRPO/OPD run 日志或 checkpoint，所以 kill test 必须从零造出 $\rho$ 的**分母**（OPD−GRPO 的增益），不能只补分子。由此带来两个新约束：(1) $\rho$ 是"差之比"，分母一旦接近 0 或被种子噪声淹没，$\rho$ 就爆炸/无意义；(2) 新增一道前置生死门——"在我够得着的规模上，OPD 到底有没有稳定可测、超过种子噪声的增益"。设计原则：**复用唯一不得不搭的 GRPO 训练器、把不需要训练的方向性代理前置、分层投入、对分母稳健性设硬阈值。**

**Tier 0 —— 零训练的方向性代理（最便宜，先做）**
不必先训 OPD 就能预判 $\rho$ 偏哪边（用资产①：teacher = control variate）。
- 只训（或借）**一条 GRPO**，取其 rollout；
- 选定 teacher，对这些 rollout 做**纯前向**，逐 token 算 $c^{\mathrm{OPD}}=\log\pi_T-\log\pi_\theta$ 与 RL 优势 $A^{\mathrm{RL}}$；
- 测 $\mathrm{Cov}(A^{\mathrm{RL}},c^{\mathrm{OPD}})$、$\mathrm{Var}(A^{\mathrm{RL}})$，及把 $A^{\mathrm{RL}}$ 对 $c^{\mathrm{OPD}}$ 回归的 $R^2$（= teacher 作为控制变量能吸收的方差预算）。
- 判读（方向性、非终判）：$R^2$/协方差**高** → teacher 形状就是个方差缩减控制变量，预测 **$\rho$ 高（H-VR）**；协方差**≈0** → teacher 信息与 RL credit **正交**，不可能靠降方差起作用，预测 **$\rho$ 低（H-KT）**。

**Tier 1 —— 前置生死门：OPD 增益是否存在**
同一模型、同一小任务、**严格相同的 rollout/更新预算**，各跑 GRPO 和 OPD，≥3 seed；算 $\Delta=\mathrm{Acc(OPD)}-\mathrm{Acc(GRPO)}$ 及其 seed 标准差 $\sigma$。
- $\Delta\gtrsim 3\sigma$ → 分母稳健，进 Tier 2；
- $\Delta$ 淹在噪声里 → 【新增 kill】现象在此规模不成立：最小幅度上调规模，或承认"主流 OPD 在够得着的规模无可归因增益"，**别硬测 $\rho$**（分母≈0 的 $\rho$ 全是幻觉）。
- regime 选择：优先 **self-teacher/同族**（头条靶）；若其 $\Delta$ 太小测不出，可先用 **strong-to-weak** 拿一个**大而干净的分母**把 pipeline 跑通（代价：该 regime 里 $\rho$ 低不算意外）。kill test 只需**一个分母干净的 regime**，全相图留正文 E2。

**Tier 2 —— 真正的 $\rho$：加一条 teacher-free VR**
仅在 Tier 1 过门后做。**复用 Tier 1 的 GRPO 训练器**，bolt 上**一个**最便宜的 teacher-free 基线即可（不必三个全上）：
- 首选 **B1（value head）**——边际成本最低、最稳；或 **B3（self-consistency dense 信号）**——不加新网络；或 **B4=CAST `2606.00172`**——若公开代码能 drop-in 则最省，且顺带把"最危险邻居"转成对照。
- 同模型/任务/预算/≥3 seed 跑这条 VR，算 $\rho$ 与 pass@k **覆盖集差**；
- **同时**用 Tier 0 口径确认这条 VR 真把 credit 方差降到 OPD 量级（否则 $\rho$ 高是"基线没调好"的假阳性，非 H-VR）。

**最小配置**：Qwen3-1.7B student（self-teacher=EMA/早期 ckpt，或 4B 作 strong teacher）；MATH/GSM8K 的 500–1000 子集（verifier 便宜）；GRPO/OPD/VR 三者 rollout 与更新步严格对齐；≥3 seed；全程**最多 3 条训练**（GRPO、OPD、1×VR）+ Tier 0 前向。

**判读（从零版，比旧版多两道门）**：
```text
Tier 0 协方差/R² 高 → 预测 H-VR，进 Tier 1（信心更足）
Tier 0 协方差 ≈ 0  → 预测 H-KT，仍可进，但预期切形态 B
─────────────────────────────────────────────
Tier 1 Δ ≥ 3σ      → 分母稳健，进 Tier 2
Tier 1 Δ 在噪声里   → 【新增 kill】现象不存在/太弱：调规模或停，别测 ρ
─────────────────────────────────────────────
Tier 2 ρ 高 + 覆盖不扩 + VR 方差确降到 OPD 量级 → 形态 A，全力投 P5
Tier 2 ρ 低 + 覆盖扩                          → 形态 B（正面）或转 P1 因果 credit 审计
Tier 2 ρ 高 但 VR 方差没降                     → 【假阳性 kill】基线没调好，回去修，别下结论
```
（[P1 见 研究方向与候选想法.md](/data1/zhaoenyue/opdpapers/docs/04想法/研究方向与候选想法.md)）

---

## 12. 与仓库其他文档的关系

- **主实验设计（E1–E4 + kill 闸门 + 先后顺序 + 选型）**：[OPD收益归因_实验设计.md](/data1/zhaoenyue/opdpapers/docs/04想法/OPD收益归因_实验设计.md)（本文 §7/§11 的可执行展开）
- 候选排序与 P1–P5 全景：[研究方向与候选想法.md](/data1/zhaoenyue/opdpapers/docs/04想法/研究方向与候选想法.md)
- 理论附录（control-variate 偏差-方差分解、判据、Cov 实验协议）：[OPD方差偏差分解理论.md](/data1/zhaoenyue/opdpapers/docs/04想法/OPD方差偏差分解理论.md)（N1，本文降级其为附录）
- 实验臂、诊断指标、分组口径：[方向力度解耦想法.md](/data1/zhaoenyue/opdpapers/docs/04想法/方向力度解耦想法.md)
- 现象与论文 ID 对照：[失败模式映射.md](/data1/zhaoenyue/opdpapers/docs/02图谱/失败模式映射.md)、[失败模式研究现状.md](/data1/zhaoenyue/opdpapers/docs/02图谱/失败模式研究现状.md)
