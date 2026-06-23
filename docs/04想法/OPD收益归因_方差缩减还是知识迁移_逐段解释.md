# OPD收益归因：方差缩减还是知识迁移 - 逐段解释

来源文档：[OPD收益归因_方差缩减还是知识迁移.md](/data1/zhaoenyue/opdpapers/docs/04想法/OPD收益归因_方差缩减还是知识迁移.md)

更新时间：2026-06-23

说明：本文按原文顺序解释每个逻辑段落。这里的“段”不严格等同于 Markdown 的每个空行，而是按原文中一个完整论证单元来划分；列表项和实验项也会单独解释，因为它们承担了论文设计中的独立功能。

---

## 0. 标题、时间与定位

### 段 0.1：标题

标题“OPD 的收益是知识迁移还是方差缩减？”把问题从“OPD 是否有效”推进到“OPD 为什么有效”。这是一种归因问题，不是单纯算法改进问题。

标题里有两个互相竞争的解释：

- **知识迁移**：teacher 提供 student 原本没有的新能力、新分布或新推理方向。
- **方差缩减**：teacher 只是让 RLVR 的稀疏、高噪声 credit 变得更密、更稳定，类似一个高级 baseline 或 control variate。

这两个解释对应完全不同的研究结论。如果是知识迁移，OPD 的核心价值在 teacher；如果是方差缩减，OPD 的核心价值可能可以被 teacher-free 的 RL 方差缩减方法替代。

### 段 0.2：更新时间

更新时间标注为 2026-06-23，说明这份文档是在已有大量 OPD 增量论文之后形成的研究想法。这个时间点重要，因为文档后面多次强调：到 2026 年中，OPD 方向已经很拥挤，普通“再提出一个路由器/检测器/稳定化 trick”的新颖性不足。

### 段 0.3：定位

定位段说明 P5 是当前候选想法中重排第一的研究方向。它不是要再发明一个 OPD 算法，而是要写一篇 **deflationary** 经验归因论文。

这里的 deflationary 可以理解为“去魅”或“降格解释”：很多论文默认 OPD 的收益来自 teacher 的知识，但 P5 要检验这种叙事是否被夸大。也许 OPD 并没有真正带来很多新知识，只是把 RLVR 的 credit assignment 做得更稳定。

这一段还说明 N1 文档的地位变化：原本 N1 想做统一理论，但经过 novelty 核查后，它最有价值的部分只剩下“teacher signal 可以看作 control variate”。因此 N1 不再作为主论文，而是降级为 P5 的理论附录。P5 的主贡献变成一个可证伪的实验归因框架。

---

## 1. 一句话论点

### 段 1.1：英文标题问题

英文标题“Is On-Policy Distillation Knowledge Transfer or Variance Reduction?”适合作为论文标题，因为它直接提出一个二选一式的核心科学问题。审稿人一眼能看出：这不是普通 OPD 方法论文，而是在挑战 OPD 子领域的默认解释。

这个标题的好处是清晰、尖锐、可实验检验。它没有承诺某个算法一定更强，而是承诺回答“OPD 的收益来源是什么”。

### 段 1.2：主论点

主论点是：在标准数学和代码推理的 OPD 设置里，OPD 相比 RLVR 的大部分增益可能来自对稀疏 RL credit 的方差缩减，而不是 teacher 分布注入了新知识或新方向。

这里的关键变量是复现率 `rho`。如果一个完全不访问 teacher logits、不使用外部知识的 teacher-free 方差缩减 RLVR 能复现 OPD 增益的很大比例，就说明 OPD 的收益未必来自 teacher 知识。

这段话里的“完全不碰 teacher logits、不注入任何外部知识”非常重要。它保证对照基线是干净的：如果这个基线也能达到 OPD 的效果，那么不能再把 OPD 增益简单归功于 teacher 的知识。

### 段 1.3：rho 的含义

`rho` 是 P5 的核心量化指标。它衡量 teacher-free 方差缩减方法能复现多少 OPD 相对 GRPO 的增益。

直观上：

```md
rho 接近 1：OPD 增益大多可以由方差缩减解释。
rho 接近 0：OPD 增益主要来自 teacher 的知识/方向。
rho 在中间：两种机制都存在，需要按 regime 拆分。
```

因此 P5 不是只给一个“OPD 是不是有用”的结论，而是把 OPD 增益分解成可估计的成分。

### 段 1.4：关键卖点

“两种结果都能发”的意思是：P5 的实验设计不是只押注一个结论。

如果 `rho` 高，论文可以主张“主流 OPD 其实主要是高级方差缩减 baseline”，这是反直觉的去魅结果。

如果 `rho` 低，论文可以主张“teacher 确实提供了 RLVR 无法替代的新方向”，这是对 OPD 价值的正面确认。

因此 P5 的稳健性来自问题设置本身：它不依赖某个固定方向的实验结果，而是无论结果支持 H-VR 还是 H-KT，都能形成清晰的科学结论。

---

## 2. 为什么这个问题值得做

### 段 2.1：OPD 子领域已经饱和

这一段列出 OPD 领域已经被研究过的失败模式和改进方向：PG 统一、方向-力度解耦、correct/wrong 非对称、capability ceiling、token teachability、estimator 方差、privileged context、long-horizon、rubric/black-box、safety 等。

它的作用是说明：如果只是再提出一个局部算法，很容易被认为增量不足。P5 必须在更高层次上提出一个能重解释整个领域的问题。

这里的“每条都已有论文”也是定位策略：P5 不与某一篇算法论文正面拼局部性能，而是要做跨论文的归因框架。

### 段 2.2：顶会论文的三类剩余机会

这段提出在饱和领域里还能有顶会价值的三类动作：

1. 攻击全领域共享假设并给出反直觉结论。
2. 进入尚无人研究的前沿。
3. 用严格混淆分析证明已有收益来自别处。

P5 属于第三类。它不是说 OPD 无效，而是问“OPD 被认为有效的原因是否被误归因”。这比提出一个新 trick 更有论文张力，因为它可能改写大家对 OPD 的理解。

### 段 2.3：P5 质疑的默认叙事

OPD 领域的默认叙事是：teacher 提供 student 学不到的知识或分布，因此 student 通过 distillation 变强。

P5 质疑这个默认叙事：在很多数学/代码推理设置里，teacher 的 dense signal 也许只是把 RLVR 原本稀疏的 outcome reward 分摊到 token 层面，从而降低 credit assignment 方差。

这个质疑很关键，因为如果成立，OPD 的核心工程目标就会变化：不一定需要更强 teacher，也许需要更好的 teacher-free credit estimator。

### 段 2.4：novelty 核查结论

这一段总结 novelty 核查：没有论文系统地把 teacher 当作 control variate，也没有论文做“方差缩减 vs 知识迁移”的 OPD 收益归因，更没有论文构造 teacher-free 方差缩减对照来计算 `rho`。

这并不意味着相关组件完全没人做过。后文会承认 CAST、vOPD、OPSD Compresses 等都有相邻思想。但 P5 的新意在于把这些组件组织成一个归因方法论：teacher-free VR baseline、`rho`、覆盖集探针、regime 相图。

---

## 3. 两个对立假说与可证伪预测

### 段 3.1：把 OPD 增益拆成两个来源

这一段把 OPD 相对 RLVR 的增益拆成两种互斥来源：方差缩减和知识迁移。

“互斥”在这里不是说现实中只能有一种，而是说在概念上要把它们分开测量。一个 OPD 实验的总收益可以同时包含两部分，但 P5 要问哪一部分占主导。

### 段 3.2：H-VR 方差缩减假说

H-VR 认为 OPD 好，是因为 teacher 的 dense per-token signal 让 RLVR 的 credit 更稳定。

RLVR 通常只有整条答案的 outcome reward，例如正确/错误或最终分数。这个信号稀疏，而且同一条轨迹里每个 token 很难区分责任。OPD 给每个 token 一个 teacher-student log-ratio 或 KL 信号，因此能降低 token credit 的噪声。

H-VR 的关键点是：这种收益不改变优化目标的 fixed point。换句话说，它帮助 student 更稳定地朝原本 reward 最优方向走，而不是把 student 拉向一个新的 teacher 目标。

### 段 3.3：H-KT 知识迁移假说

H-KT 认为 OPD 好，是因为 teacher 提供了 student 自己探索不到的新方向或新分布。

如果 teacher 真比 student 强，并且 teacher 的分布包含 student 当前策略无法有效发现的推理路径，那么 OPD 不只是降噪，而是在移动优化的 fixed point。这种情况下，student 最终可能达到 RLVR baseline 达不到的区域。

H-KT 是传统 OPD 叙事更偏好的解释：teacher 是知识来源，而不是噪声控制工具。

### 段 3.4：用 N1 语言连接两个假说

这一段把 P5 和 N1 的“方向 vs 力度”语言连接起来。

- 力度收益对应 H-VR：teacher 只改变每个 token 更新多少，不改变最终目标。
- 方向收益对应 H-KT：teacher 改变梯度方向，把模型拉向 teacher 分布。

P5 不试图在理论上证明 N1 一定正确，而是把 N1 当作生成实验预测的工具。理论的价值在于提出可测量的假说，最终由实验裁决。

### 段 3.5：H-VR 的预测

如果 H-VR 主导，应该看到三个现象：

1. teacher-free 方差缩减 RLVR 能复现大部分 OPD 增益，因此 `rho` 接近 1。
2. OPD 不会显著扩大可解题目集合，也就是不会解锁 GRPO 加 teacher-free VR 都解不开的新题。
3. 在 correct-rollout 多、teacher 校准好、self-teacher、in-distribution 等场景下，`rho` 最高。

这些场景的共同特点是 teacher 与 student 的目标方向差异小，teacher 更像一个低噪声 credit 信号，而不是新知识来源。

### 段 3.6：H-KT 的预测

如果 H-KT 主导，应该看到相反现象：

1. teacher-free 方差缩减复现率低，`rho` 接近 0。
2. OPD 能解锁新题，扩大 pass@k 覆盖集。
3. 在 strong-to-weak、wrong-rollout 多、OOD 难度高等场景下，`rho` 最低。

这些场景里 teacher 与 student 差距更大，teacher 可能真的提供了 student 自己探索不到的方向。

### 段 3.7：regime 相图

最后一句说明 P5 的目标不是给出单点结论，而是画出 regime 相图。

也就是说，P5 不需要证明“OPD 永远只是方差缩减”或“OPD 永远是知识迁移”。更合理的结论是：

```md
某些 regime 下 OPD 近似高级 baseline。
某些 regime 下 teacher 确实提供新知识。
关键是找出分界条件。
```

这使论文结论更稳健，也更符合已有 OPD 文献中出现的复杂现象。

---

## 4. 形式化：把增益分到方差缩减与方向/知识

### 段 4.1：统一记号

这一段沿用 N1，把所有方法的更新都写成 policy-gradient 形式：

```md
更新 = credit × score function
```

其中 score function 是 `∇ log πθ(a | s)`，表示“增加或减少当前 sampled token 概率”的方向；credit `c` 决定这个 token 更新多强、符号是什么。

这样写的好处是可以把 GRPO、sampled-token OPD、teacher-free VR 放到同一个框架里比较。

### 段 4.2：统一更新公式

公式

```md
Δθ ∝ sum c_hat_i,t ∇θ log πθ(a_i,t | s_i,t)
```

表达的是：每次参数更新是所有 rollout token 的梯度累加。每个 token 的贡献由两部分决定：

- `∇ log πθ` 给出提升该 token 概率的方向。
- `c_hat` 给出这个方向应该被放大、缩小、反向还是忽略。

因此“OPD 改变了什么”可以转化成“OPD 改变了 credit，还是改变了 fixed point”。

### 段 4.3：GRPO 的 credit

GRPO 的 token credit 通常来自整条轨迹的 reward/advantage。对同一条 rollout 内的所有 token，credit 可能接近相同。

这导致两个问题：

- **高方差**：单条轨迹的 outcome reward 噪声很大。
- **零 token 分辨率**：无法区分到底哪个中间 token 对成功或失败负责。

P5 正是怀疑 OPD 的主要收益来自修复这两个问题，而不是 teacher 真的带来新知识。

### 段 4.4：OPD 的 credit

OPD 的 sampled-token 形式会给每个 token 一个 teacher-student log-ratio：

```md
c_OPD = log π_T - log π_θ
```

这个信号是 dense 的，因为每个 token 都能算；通常方差比 outcome reward 低，因为它不是等到整条答案结束才给分。

但它可能有偏。teacher 喜欢的 token 不一定是真实 reward 最优的 token。这个偏差在 privileged teacher、弱 teacher、错误 rollout、OOD 场景里尤其重要。

### 段 4.5：N1 判据与 P5 操作化

N1 的观点是：OPD 的净收益等于方差缩减红利减去偏差代价。

P5 把这个理论观点转成实验定义：

- 方差缩减成分：任何不使用外部知识、只用 policy 自身信号降低 RL credit 方差后得到的增益。
- 知识/方向成分：OPD 超过上述 teacher-free 方差缩减之后剩下的增益。

这一步很重要，因为“知识迁移”本身很难直接观测。P5 用“teacher-free VR 无法复现的剩余增益”来操作化定义它。

### 段 4.6：复现率 rho

`rho` 的定义是：

```md
rho = (Acc(teacher-free VR) - Acc(GRPO)) / (Acc(OPD) - Acc(GRPO))
```

分母是 OPD 相对 GRPO 的总收益。分子是 teacher-free 方差缩减方法相对 GRPO 的收益。

如果 teacher-free VR 的增益几乎等于 OPD 的增益，那么 `rho` 接近 1，说明 OPD 的收益大多可以不用 teacher 解释。

如果 teacher-free VR 几乎没有追上 OPD，那么 `rho` 接近 0，说明 teacher 的知识或方向可能不可替代。

### 段 4.7：rho 的解释

`rho` 不是一个绝对性能指标，而是归因指标。它关心的是“OPD 的 gap 中有多少能被 teacher-free VR 吃掉”。

因此同一个 OPD accuracy 提升，在不同 baseline 下可能有不同解释。P5 必须把 GRPO、OPD、teacher-free VR 放在同一训练预算和调参强度下比较，否则 `rho` 会被基线强弱污染。

---

## 5. 关键设计：teacher-free 方差缩减 RLVR

### 段 5.1：为什么 teacher-free baseline 是整篇论文的 wedge

这一段强调：P5 的成败取决于 teacher-free 方差缩减 baseline 是否可信。

如果 baseline 偷偷用了 teacher、外部数据或更强模型，它就不能证明“OPD 增益不是知识”。只有当 baseline 严格只用 policy 自身信号时，复现 OPD 增益才有归因意义。

这就是文中说的 wedge：它是把 OPD 增益切开、区分 VR 与 KT 的实验楔子。

### 段 5.2：B1 学习型 token-level baseline

B1 是 actor-critic 式 dense credit。它在 policy 自己的 rollout 上训练一个 value head `V_phi(s)`，用 `R - V(s)` 作为 token-level credit。

这个设计的思想是：如果模型能预测某个 prefix 后续成功概率，那么就能把整条轨迹的 outcome reward 分解到 token/prefix 层面，从而降低方差。

B1 不需要 teacher。它只使用环境 reward 和 student 自己访问过的 state，所以是干净的 teacher-free VR baseline。

### 段 5.3：B2 control-variate 版 RLVR

B2 更直接对应 N1 的 control-variate 理论。它训练一个廉价 token-level 预测器 `m_hat`，预测 RL advantage 中可预测的部分，然后从原始 advantage 中减掉这个可预测噪声项。

公式中的

```md
c_hat = A_RL - beta (m_hat - mean(m))
```

意思是：保留 RL advantage 的无偏方向，但用一个相关的预测器抵消噪声。最优 `beta` 由协方差和方差决定。

这个设计是“纯方差缩减、不移动 fixed point”的最干净形式，因为它不让 `m_hat` 决定最终优化目标，只让它降低 credit estimator 的方差。

### 段 5.4：B3 self-distillation 但零新知识

B3 使用 policy 自己当前分布、EMA-self 或 self-consistency 作为 dense 信号。

它的边界比 B1/B2 稍微复杂，因为 self-distillation 是否完全“零知识”可能有争议。当前策略的 EMA 或一致性信号虽然不是外部 teacher，但也可能引入训练动态上的偏好。

所以文档把 B3 放在“上界对照”位置：它可以作为 teacher-free dense signal 的强 baseline，但解释时要单独标注，不如 B1/B2 那么干净。

### 段 5.5：B4 CAST

B4 把 CAST 收编为现成的 teacher-free dense baseline。CAST 已经构造了 answer-free、stop-gradient self-teacher，用来塑造 token-level advantage。

P5 的策略不是假装 CAST 不存在，而是把它纳入实验谱系：CAST 做了 teacher-free dense signal，但没有把它解释成方差缩减，也没有用它来做 OPD 的 VR-vs-KT 归因。

这是一种很重要的定位防御。面对最危险邻居时，P5 不与其争“谁先提出 teacher-free dense signal”，而是强调自己提出的是归因方法论。

### 段 5.6：B1-B4 的共同约束

这段说明 B1/B2/B3/B4 都满足“零外部知识”约束，只是降方差强度和机制不同。

如果其中任何一个能复现 OPD 大部分增益，就支持 H-VR。这里的“任何一个”使实验更稳健：P5 不依赖某一个 baseline 恰好成功，而是构造一族 teacher-free VR 方法。

### 段 5.7：完整对照组

完整对照包括：

- GRPO：作为 `rho` 的分母基线。
- OPD：作为要解释的目标方法。
- B1/B2/B3/B4：作为 teacher-free VR 对照。
- 可选 RLSD/vOPD：作为已有“力度类/方差类”方法的参照点。

这套对照的逻辑是：先测 OPD 相对 GRPO 的总收益，再看 teacher-free VR 吃掉多少，最后把已有 OPD 变体投到同一谱系中解释。

### 段 5.8：B2 与 vOPD 的理论锚点

vOPD 已经证明 teacher 的逐 token reverse-KL 可以作为 OPD 的控制变量来降方差。

P5 利用这一点给 B2 找到目标：如果 teacher-free 预测器能逼近 teacher reverse-KL 的降方差功能，就能更干净地说明 OPD 的一部分收益并非来自 teacher 知识，而来自控制变量作用。

这也解释了为什么 P5 不是凭空造 baseline，而是把已有理论中的 control-variate 结构 teacher-free 化。

---

## 6. 正交的知识解锁探针

### 段 6.1：为什么需要 rho 之外的证据

`rho` 比较的是平均 accuracy 增益。但平均值可能混淆两种情况：

- teacher-free VR 和 OPD 提升了同一批题，只是幅度不同。
- OPD 解锁了一批 teacher-free VR 完全解不开的新题。

因此 P5 需要一个不依赖平均增益幅度的第二证据：覆盖集探针。

### 段 6.2：pass@k 覆盖集

对每个方法，测较大 `k` 下能解出的题目集合 `S_method`。

pass@k 比 pass@1 更接近“能力边界”：如果模型多采样几次仍然从未解出某题，说明该题可能不在当前策略可达范围内；如果只是 pass@1 低但 pass@k 能解，说明更多是可靠性或采样效率问题。

### 段 6.3：H-VR 对覆盖集的预测

H-VR 预测 OPD 不会显著扩大可解集合：

```md
S_OPD 应该大致包含在 S_GRPO 与 S_VR 的并集中。
```

直观上，方差缩减让模型更稳定地找到本来就可达的解，而不是创造新的可达解。

### 段 6.4：H-KT 对覆盖集的预测

H-KT 预测会存在只有 OPD 能解开的题：

```md
S_OPD - (S_GRPO ∪ S_VR) 非空。
```

如果这些题稳定存在，就说明 teacher 提供了 teacher-free VR 无法替代的新方向或知识。

### 段 6.5：覆盖集差作为知识迁移指纹

这一段把覆盖集差称为“知识迁移的最直接指纹”。

需要注意的是，这不是在理论上断言 RLVR 永远不能扩展能力。文档后面会 hedge 这一点，因为 ProRL 等工作说明长时 RL 也可能扩展 pass@k 边界。

这里更准确的说法是：在同预算、同设置下，如果 OPD 扩大了 teacher-free VR 无法扩大的可达集合，那么这是 teacher 知识迁移的经验指纹。

---

## 7. 实验设计

### 段 7.1：实验设置

实验固定 Qwen3-1.7B/4B 和 MATH/LiveCodeBench，并沿用已有“方向力度解耦想法”的分组口径。

这样做有两个好处：

- 控制变量清楚，避免因为模型/数据选择过多导致归因混乱。
- 可以复用仓库里已经设计好的实验臂、诊断指标和分桶方式。

### 段 7.2：E1 主结果

E1 测全局 `rho` 和覆盖集差，是论文最重要的头条实验。

它回答两个问题：

```md
teacher-free VR 能复现多少 OPD 增益？
OPD 是否解锁 teacher-free VR 解不开的题？
```

如果论文只能展示一个核心表格，E1 就是那个表格。

### 段 7.3：E2 regime 相图

E2 按不同 regime 分别测 `rho`。

几个分组轴的含义是：

- self-teacher vs strong-to-weak：teacher 是否真的比 student 强。
- correct-rollout 多 vs wrong-rollout 多：teacher signal 是在压缩正确路径，还是试图纠错。
- in-distribution vs OOD：student 当前探索是否已经覆盖有效解法。

预测是：self-teacher、correct-rollout、in-distribution 下 `rho` 高；strong-to-weak、wrong-rollout、OOD 下 `rho` 低。

### 段 7.4：E3 机制确认

E3 不只看 accuracy，而是验证机制变量：

- teacher-free VR 是否真的降低了 credit 方差。
- gradient SNR 是否接近 OPD。
- OPD 是否把模型 fixed point 拉向 teacher。
- KL-to-teacher 是否随训练下降。

这一步防止出现“teacher-free VR accuracy 高，但不是因为方差缩减”的解释漏洞。

### 段 7.5：E4 对照定位

E4 把 RLSD、vOPD 放进 `rho` 谱系。

这样可以把已有工作重新解释为不同位置的机制实例：有些方法更像力度/方差缩减，有些方法更像方向/知识迁移。

E4 的价值不一定在主性能，而在于让 P5 的归因框架能解释已有文献。

---

## 8. 预期结论的两种形态

### 段 8.1：形态 A，deflationary 结论

形态 A 是更轰动的结果：在主流 self-teacher 数学推理设置里，`rho` 很高，覆盖集不扩大。

这意味着 OPD 的大部分推理收益可以由 teacher-free 方差缩减复现，teacher distribution 没有显著带来可迁移的新知识。

这个结论会把 OPD 从“知识迁移算法”重新解释成“高级 RL credit baseline”。它和 OPSD Compresses 的“compression not correction”方向一致，但更强，因为它给出了 teacher-free 复现证据。

### 段 8.2：形态 B，正面确认 teacher 知识

形态 B 是安全网结果：`rho` 低、覆盖集扩大，尤其在 strong-to-weak 或 OOD 场景里。

这说明 teacher 确实注入了新方向或知识。P5 仍然有贡献，因为它不是泛泛说“OPD 有用”，而是指出 OPD 真正做知识迁移的 regime。

这种结果还会反驳 N1 的 H-VR 主导假设，但这不是失败，而是一个清晰的实证裁决。

### 段 8.3：三件套不依赖结果方向

无论形态 A 还是 B，P5 的核心方法都成立：

- 用 `rho` 衡量 teacher-free VR 复现率。
- 用覆盖集探针检测知识解锁。
- 用 regime 相图刻画何时 VR 主导、何时 KT 主导。

这就是文档反复强调“两种结果都能发”的原因。

---

## 9. 与现有工作的差异化

### 段 9.1：第二轮 novelty 核查结论

这一段说明经过 98 agent、16 主源、25 条断言的对抗核验后，P5 的核心三件套仍然没有被单篇论文预占。

这里的三件套是：

- VR-vs-KT 归因。
- teacher-free VR 对照与 `rho`。
- 覆盖集解锁探针用于 teacher-vs-teacher-free 比较。

但这段也承认周边已经很密集。P5 的护城河不是“每个组件都无人做过”，而是把分散组件拼成归因方法论。

### 段 9.2：CAST

CAST 是最危险邻居，因为它已经有 teacher-free dense signal，并且会塑造 token-level advantage。

P5 的区分点是：CAST 把自己定位成 credit assignment refinement，不做 variance reduction vs knowledge transfer 的归因，不计算 `rho`，也没有覆盖集探针。

所以 P5 应该把 CAST 收编为 B4 baseline，而不是回避它。这样既避免 novelty 被攻击，也能把 CAST 变成 P5 实验体系的一部分。

### 段 9.3：Kim et al. 2505.14216

这篇工作已经讨论过 knowledge transfer vs not 的归因，并使用 pass@1 与 pass@k 区分 accuracy 和 capability。

P5 的区别是：它研究的是 on-policy distillation、per-token dense reward 和 teacher-free 对照，而不是 offline teacher distillation。

也就是说，Kim et al. 给了相邻的问题意识，但没有做 P5 的 on-policy、teacher-free、`rho` 归因框架。

### 段 9.4：Yue et al. 2504.13837

这篇工作提供了 pass@k 覆盖集方法学，说明小 k 和大 k 下能力判断可能不同。

P5 借用这个方法，但用途不同：不是比较 RLVR 与 base，而是比较 teacher OPD 与 teacher-free VR。

因此它是工具来源，不是问题预占。

### 段 9.5：OPSD Compresses

OPSD Compresses 发现 OPD 在正确轨迹上更像压缩，在错误轨迹上不一定能纠错，甚至可能伤 accuracy。

P5 在此基础上更进一步：如果压缩收益主要可以被 teacher-free VR 复现，那么“compression”背后的机制可能就是方差缩减，而非 teacher 知识。

所以 P5 与它的关系是现象到归因的推进。

### 段 9.6：Unmasking OPD

Unmasking OPD 做的是训练前梯度对齐诊断，关注 teacher signal 与理想 success gradient 的方向是否一致。

P5 关注的是 OPD 总收益来源，而不是只诊断方向是否对齐。它要回答“teacher 的作用是知识还是降噪”，并用 teacher-free 对照量化。

因此两者一个偏诊断，一个偏归因。

### 段 9.7：RLSD

RLSD 已经把方向和力度解耦：方向由环境 reward 锚定，teacher 只调 magnitude。

P5 的区别是把这种“力度”解释为方差缩减，并进一步问：这个力度成分是否可以完全 teacher-free。

因此 RLSD 是 P5 的重要理论邻居，但不是完整替代。

### 段 9.8：vOPD

vOPD 已经研究 OPD 的 control-variate 降方差，证明 teacher-derived reverse-KL baseline 可以降低方差且不移动 OPD fixed point。

P5 更激进：它不是在 OPD 内部用 teacher 做 baseline，而是问整个 OPD-vs-GRPO gap 是否可以被 teacher-free control variate 替代。

这个差异很关键。vOPD 是“让 OPD estimator 更好”，P5 是“检验 OPD 是否只是一个高级方差缩减器”。

### 段 9.9：未被占据的核心

这一段总结 P5 的新资产：teacher-free VR 对照、`rho`、覆盖集探针，以及“OPD 近似高级 baseline”的可证伪经验论断。

这里“可证伪”很重要。P5 不是预设 OPD 一定只是 baseline，而是设计实验让这个说法可以被数据支持或推翻。

### 段 9.10：硬约束一，RL/OPD fixed point 说法有争议

核查发现“RLVR 不能解锁新能力”这种说法被 ProRL 等工作反驳。长时 RL 可能扩展 pass@k 边界。

因此 P5 不能把“方差缩减不会扩大能力边界”当成理论公理。它只能把覆盖集差当作本实验设置下的经验指纹，并且承认 regime/方法相关。

这是防止论文被审稿人抓住过度断言的关键 hedge。

### 段 9.11：硬约束二，vOPD 给 B2 提供锚点

vOPD 的结果反而支持 P5 的 B2 设计：如果 teacher reverse-KL 是好的控制变量，那么 teacher-free 预测器可以尝试逼近这个控制变量的作用。

这让 B2 不只是一个 heuristic，而是有明确理论目标：尽量复制 teacher 信号中的降方差部分，同时排除 teacher 知识。

---

## 10. 风险与防御

### 段 10.1：风险一，基线不公平

审稿人可能质疑 teacher-free VR baseline 没有被充分调优，或者调得太强/太弱导致结论偏向某一方。

防御策略是同时报告 B1/B2/B3 多个独立设计和超参 sweep。只要其中任一个能复现 OPD 大部分增益，就能支持 H-VR；同时用方差指标证明它确实起到了方差缩减作用。

### 段 10.2：风险二，零外部知识边界争议

self-distillation 或 EMA-self 是否算“外部知识”可能有争议。

防御策略是把 B1/B2 作为最干净主证据，因为它们只使用 policy 自身 value 和统计；B3 作为上界对照，单独标注，不承担最强结论。

### 段 10.3：风险三，rho 被幅度巧合混淆

平均 accuracy 增益可能因为幅度巧合而误导。例如 teacher-free VR 和 OPD 的平均提升相近，但解决的是不同题。

防御策略是加入覆盖集探针。覆盖集不只看平均分，而看哪些题被解开，因此能提供正交证据。

### 段 10.4：风险四，领域和规模外推

如果实验只在小模型数学任务上成立，结论可能不能推广到更大模型、更多任务或 OOD 场景。

防御策略是明确把结论限定在所测 regime，并把 regime 相图作为主贡献。P5 不需要声称所有 OPD 都只是方差缩减。

### 段 10.5：风险五，RL 解锁能力争议

文档再次强调不能把“RL 解锁不了新能力”当作公理，因为已有工作说明长时 RL 可能扩展能力边界。

因此覆盖集探针只能作为经验测量：在本设置下 OPD 是否比 GRPO 和 teacher-free VR 解锁更多题。结论必须写成 regime/方法相关。

### 段 10.6：风险六，与 CAST 构造撞车

CAST 已经有 teacher-free dense signal，审稿人可能认为 P5 的 baseline 已经被做过。

防御策略是承认 CAST，并把它作为 B4 baseline。P5 的贡献不是发明 teacher-free dense signal，而是用包括 CAST 在内的 teacher-free 信号做 VR-vs-KT 归因。

### 段 10.7：风险七，时效

OPD 领域更新很快，投稿前可能出现新论文抢占 P5 的核心。

防御策略是投稿前重新做 novelty 核查，尤其检查是否已经有人做了 teacher-free VR 对照、`rho` 或覆盖集解锁探针。

---

## 11. Kill test

### 段 11.1：为什么要先做 kill test

Kill test 的目的是在投入完整训练前，用半天到一天判断 P5 是否值得继续。

它不要求写理论或跑完整新训练，而是基于现成 GRPO/OPD run 日志快速测试 teacher-free dense baseline 是否能复现 OPD 的大部分增益。

如果这个快速测试完全不支持任何清晰趋势，就说明要先解决实验口径，而不是直接写论文。

### 段 11.2：kill test 步骤

步骤包括：

1. 取已有 GRPO 和 OPD checkpoint/曲线。
2. 快速训练一个 B1 value head 或 B2 control variate RLVR。
3. 测 `rho` 与覆盖集差。

这三个步骤已经覆盖 P5 的两个核心证据：平均增益复现率和能力覆盖集。

### 段 11.3：结果一，rho 高且覆盖集不扩

如果 `rho` 高，且 OPD 没有扩大覆盖集，说明形态 A 很可能成立。

这时可以全力推进 P5 的 deflationary 叙事：主流 OPD 收益主要来自方差缩减，teacher-free baseline 可以复现。

### 段 11.4：结果二，rho 低且覆盖集扩

如果 `rho` 低，且 OPD 扩大覆盖集，说明 teacher 确实提供了 teacher-free VR 无法替代的知识或方向。

这时 P5 仍然成立，但叙事切换为形态 B：OPD 何时真的做知识迁移，以及这种迁移在哪些 regime 出现。

### 段 11.5：结果三，噪声大且 rho 不稳

如果结果噪声很大，`rho` 不稳定，说明当前 teacher-free VR baseline 或测量口径还不可靠。

这时不应急着写主论文，而应先解决：

- VR baseline 是否足够强。
- credit 方差和 gradient SNR 是否测对。
- 覆盖集统计是否有足够样本。
- OPD 与 GRPO 训练预算是否公平。

---

## 12. 与仓库其他文档的关系

### 段 12.1：候选排序文档

“研究方向与候选想法.md”提供 P1-P5 的全景和排序。P5 是其中经 novelty 核查后重排第一的方向。

这说明当前文档不是孤立想法，而是从多个候选中筛选出来的主攻方向。

### 段 12.2：N1 理论附录

“OPD方差偏差分解理论.md”提供 control-variate、偏差-方差分解、Cov 判据等理论工具。

在 P5 中，N1 不再是主论文，而是理论附录。它帮助生成假说和解释实验结果，但最终贡献由 P5 的归因实验承担。

### 段 12.3：方向力度解耦想法

“方向力度解耦想法.md”提供实验臂、诊断指标和分组口径。

P5 复用它的核心视角：方向对应知识迁移，力度对应 credit 调节或方差缩减。

### 段 12.4：失败模式映射和研究现状

“失败模式映射.md”和“失败模式研究现状.md”提供 OPD 现象与论文 ID 对照。

P5 需要这些文档来支撑两件事：

- 说明 OPD 领域已经高度饱和，普通局部算法新意有限。
- 把 P5 的 regime 相图与已有 failure mode 对齐，例如 correct/wrong 非对称、privileged context、long-horizon、capability ceiling 等。

---

## 总结：这篇 P5 文档到底在做什么

这篇文档的核心不是提出一个新 OPD loss，而是提出一个 OPD 收益归因框架。

它把 OPD 的收益拆成两种机制：

```md
方差缩减：teacher 像 control variate，让 RL credit 更稳。
知识迁移：teacher 提供 student 自己无法探索到的新方向。
```

然后用三件套来区分它们：

```md
1. teacher-free VR baseline
2. 复现率 rho
3. pass@k 覆盖集解锁探针
```

如果 teacher-free VR 能复现 OPD 大部分收益，P5 就证明主流 OPD 在该 regime 下更像高级方差缩减。如果不能复现，并且 OPD 扩大覆盖集，P5 就证明 teacher 在该 regime 下确实提供知识迁移。

因此 P5 的真正贡献是：把“OPD 为什么有效”从直觉争论变成可测量、可证伪、可按 regime 分解的实验问题。
