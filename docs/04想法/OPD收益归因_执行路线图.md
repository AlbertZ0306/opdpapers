# P5 执行路线图：从零到投稿，每一步做什么

更新时间：2026-06-23

定位：本文是 [OPD收益归因_实验设计.md](/data1/zhaoenyue/opdpapers/docs/04想法/OPD收益归因_实验设计.md) 的**时间有序执行视图**。设计文档讲"实验的逻辑结构"，本文讲"从什么都没做开始，第 1 步、第 2 步……具体干什么、产出什么、过不过得了闸门、过不了怎么办"。两份文档术语完全对齐（Phase −1…5、P−1…P−6、G−1…G5、C-GRPO/C-OPD/B1–B4）。

**当前状态：什么都还没做。第一步见 Step 1。**

---

## 0. 步骤总表（一页纸）

类型：🟢推理（无训练，便宜） / 🟡单次训练 / 🔴多次训练（重）。依赖列指必须先完成的 Step。

| Step | 阶段 | 做什么 | 类型 | 依赖 | 产物 | 闸门 |
|---|---|---|---|---|---|---|
| **1** | A 起步 | 站起 eval+verifier harness，产出 base pass@k 曲线 | 🟢 | — | harness + base 曲线 | — |
| 2 | A 起步 | P−2 verifier 审计，修等价形式，冻结 verifier | 🟢 | 1 | 冻结 verifier + 审计报告 | 子门 |
| 3 | A 起步 | P−3 eval 口径校准，定 $k_{\max}$，写死协议 | 🟢 | 1 | 冻结 eval 配置 | 子门 |
| 4 | B 预备 | P−4 teacher 有效性（锁 self-teacher 设定 + s2w gap） | 🟢 | 2,3 | 确认的 teacher 配置 | 子门 |
| 5 | B 预备 | P−5 算力对齐口径，pilot，写 knob 表 | 🟡 | — | 算力对齐约定 | 子门 |
| 6 | B 预备 | 搭/复用 GRPO 训练器，smoke test | 🟡 | 5 | 可跑的训练器 | smoke |
| **7** | C 现象 | GRPO ×≥3 seed → P−1 复现 + P−6 噪声地板 $\sigma$ | 🔴 | 2,3,6 | 复现对照 + $\sigma$ | **G−1 收口** |
| 8 | C 现象 | OPD-self 1 条 → 算 $\Delta$ | 🟡 | 4,7 | $\Delta$、$\sigma$ | **G1**：$\Delta\ge3\sigma$ |
| 9 | D 方向 | Phase 0 Tier 0：在 Step 7 rollout 上算 Cov/$R^2$ | 🟢 | 7 | 协方差代理 | **G0**：预测 $\rho$ 方向 |
| **10** | E 头条 | 训 teacher-free 基线 B1/B2/B3/B4 | 🔴 | 8 | VR checkpoints | — |
| **11** | E 头条 | 算 $\rho$ + 覆盖集解锁差 + 验 VR 方差降够 | 🟢 | 10 | **E1 头条数字** | **G2** |
| 12 | F 相图 | E2：δ / correct-wrong / ID-OOD 三轴各测 $\rho$ | 🔴 | 11 | regime 相图 | **G3** |
| 13 | G 机制 | E3：方差/SNR/KL-to-teacher，与 $\rho$ 对账 | 🟢 | 11 | 机制证据 | **G4** |
| 14 | H 定位 | E4：跑 RLSD/vOPD 投到 $\rho$ 谱系 | 🔴 | 11 | 对照定位 | — |
| 15 | H 泛化 | 代码域重跑 E1 核心 | 🔴 | 11 | 跨域结论 | **G5** |
| 16 | I 收尾 | 重跑 novelty 核查 + 成文 | 🟢 | 11+ | 投稿稿 | — |

**关键节点**：Step 1 = 唯一不依赖任何东西的起点；Step 7 = 第一次训练、收口 G−1；Step 11 = kill test 终点 + E1 头条，整个项目生死在此判明。

---

## 1. Stage A —— 起步：可信评测闭环（无训练）

> 目标：先有"能生成、能判对错、能算 pass@k"的闭环。这是后面一切测量的物理前提。

### Step 1 ★第一步★：站起 eval + verifier harness
- **做什么**：用 vLLM 把 **Qwen3-1.7B-Base** 在 **MATH500** 上跑起来（固定温度/top-p/max_len）；接一个数学 verifier（math_verify / sympy 等价）；每题采 N=256，用无偏估计 $1-\binom{N-c}{k}/\binom{N}{k}$ 算 pass@$k$（$k=1,4,16,64,256$）。
- **产物**：可复跑的 harness + base 模型的 pass@k 曲线（尤其 pass@1 与 pass@256、饱和位置）。
- **为什么第一**：零训练、最便宜；不依赖任何其它东西；一次产出 Step 2/3 所需的生成与曲线；立刻暴露"base 是否有 headroom、verifier 假阴是否在污染数字"。
- **失败处理**：跑不出合理 pass@k → 先查 decoding/max_len（thinking 被截断会塌 pass@k）。

### Step 2：P−2 verifier 审计
- **做什么**：在 Step 1 的生成上，用更强 oracle（sympy 符号等价 + 必要时 LLM judge）对照，抓假阳/假阴；重点修 `1/2` vs `0.5`、`\frac{\sqrt2}{2}` vs `1/\sqrt2`、集合顺序、单位、LaTeX 变体。
- **产物**：**冻结的 verifier** + 混淆计数报告。
- **判据（子门）**：假阳率、假阴率 < ~1–2%。
- **失败处理**：达不到 → 继续补 normalization 规则，必要时换 verifier 库。

### Step 3：P−3 eval 口径校准
- **做什么**：固定 decoding（温度/top-p/max_len/stop）；验 pass@k 随 k 单调且饱和；取饱和处定 **$k_{\max}$**（覆盖集探针用）；重采样验 avg@k 的 seed 稳定性。
- **产物**：写死的 eval 配置（温度/top-p/max_len/$k_{\max}$/采样预算）+ 无偏 pass@k 估计代码。
- **判据（子门）**：pass@k 在 $k_{\max}$ 前饱和；avg@k seed 方差可接受。

**Stage A 出口**：评测闭环可信（harness + 冻结 verifier + 固定 eval 配置 + $k_{\max}$）。

---

## 2. Stage B —— 预备口径（半推理 / 设计决策）

### Step 4：P−4 teacher 有效性
- **做什么**：
  - self-teacher：**锁定 privileged-context 设定**（OPSD 口径：teacher 额外看 reflection / verified trace），在若干 on-policy rollout 上算 $c^{\mathrm{OPD}}=\log\pi_T-\log\pi_\theta$，确认分布非退化。
  - ⚠️ 陷阱：冻结初始权重 + 相同 context 会让 $c^{\mathrm{OPD}}\approx0$（OPD≈no-op，分母塌缩）。必须用带 privileged context 的设定让信号在 init 就非平凡。
  - strong-to-weak：用 Stage A harness 测 Acc(teacher) 与 Acc(student)，确认 teacher 确更强且 student 有 headroom。
- **产物**：确认的 teacher 配置 + 实测 $c^{\mathrm{OPD}}$ 方差与 teacher/student gap。
- **判据（子门）**：self-teacher $c^{\mathrm{OPD}}$ 非平凡；s2w 中 Acc(teacher) ≫ Acc(student)。

### Step 5：P−5 算力对齐口径
- **做什么**：选 matched-budget 单位（推荐：对齐**梯度更新步 × prompt 数 × 每 prompt rollout 数 G × max_len**，把 OPD 多出的 teacher 前向开销单独透明上报，不抹平 wall-clock）；跑几百步 GRPO/OPD/B1 pilot 确认 knob 对得齐。
- **产物**：一张固定 knob 表，全 Phase 沿用。
- **判据（子门）**：约定可执行且 pilot 验证无崩。

### Step 6：搭 / 复用 GRPO 训练器
- **做什么**：基于 verl / OpenRLHF / TRL 之一，配好 DAPO-Math-17K + Qwen3-1.7B-Base + group rollout + Stage A 的 verifier 作 reward；跑几十步 smoke test。
- **产物**：可跑的 GRPO 训练器。
- **判据（smoke）**：reward 上升、loss 不爆、advantage 归一化正常。

**Stage B 出口**：teacher 配置确认 + 算力口径定 + 训练器跑通。

---

## 3. Stage C —— 建立现象与噪声地板（首批训练）

> 这批 run **一鱼三吃**：P−1 复现 + P−6 噪声地板 + 复用为 Step 9 协方差代理输入。Stage A+B+C 走完即收口 **G−1**（pre-flight 全过）。

### Step 7：GRPO ×≥3 seed —— P−1 复现 + P−6 噪声地板
- **做什么**：按 Step 5 预算跑 **GRPO × ≥3 seed**（头条想要 5）到参考论文步数；用冻结 harness 评测。
  - **P−1**：与公开锚点（`2602.12125` / OPSD `2605.06188` 的 Qwen3-1.7B 某点）比 avg@k，容差 ±2–3pt。偏了查 reward / advantage 归一化 / KL 系数 / lr / mask。
  - **P−6**：从 seed 间 Acc 算 $\sigma$；按计划 seed 数算最小可检测 $\Delta$。
- **产物**：复现对照表 + $\sigma$ 估计 + 所需 seed 数。
- **闸门（G−1 收口）**：复现落容差内 → pre-flight 全部达标。复现不了 → 停下修 rig，不要往后走。

### Step 8：OPD-self 1 条 —— 算 $\Delta$
- **做什么**：同预算跑 1 条 **C-OPD-self**（≥3 seed），算 $\Delta=\mathrm{Acc(OPD)}-\mathrm{Acc(GRPO)}$。
- **闸门 G1**：
  - $\Delta\gtrsim3\sigma$ → 分母稳健，进 Stage D；
  - $\Delta$ 淹在噪声里 → 最小幅度上调规模（Qwen3-4B）或换 regime（先用 s2w 拿干净分母）；仍不行 → 承认此规模无可归因增益，**停或转 [P1 因果 credit 审计](/data1/zhaoenyue/opdpapers/docs/04想法/研究方向与候选想法.md)**。

---

## 4. Stage D —— 判断方向（零额外训练）

### Step 9：Phase 0 Tier 0 协方差代理
- **做什么**：在 Step 7 的 GRPO rollout 上，对 Step 4 的 teacher 做纯前向，算 $\mathrm{Cov}(A^{\mathrm{RL}},c^{\mathrm{OPD}})$、$\mathrm{Var}(A^{\mathrm{RL}})$ 及回归 $R^2$。
- **闸门 G0**：$R^2$/协方差高 → 预测 $\rho$ 高（信心进 Stage E）；≈0 → 预测 $\rho$ 低（预期走形态 B）；取不到稳定信号 → 回查方差口径。
- **意义**：在烧 Stage E 的钱之前，先免费拿到 $\rho$ 的方向。

---

## 5. Stage E —— E1 头条：$\rho$ + 覆盖集（核心训练，kill test 终点）

### Step 10：训 teacher-free 基线
- **做什么**：复用训练器，bolt 上 **B1（value head）/ B2（control variate）/ B3（self-consistency）/ B4（CAST `2606.00172`）**，同模型/prompt/预算/≥3 seed。**kill 阶段可只先上最便宜的 B1 或 B4 跑通**，主实验四个全报、互为消融。
- **产物**：各 VR 条件 checkpoint。

### Step 11 ★生死点★：算 $\rho$ + 覆盖集
- **做什么**：
  1. 用最强基线 $B^{*}$ 算 $\rho=\frac{\mathrm{Acc}(B^{*})-\mathrm{Acc(GRPO)}}{\mathrm{Acc(OPD)}-\mathrm{Acc(GRPO)}}$（报 bootstrap CI）；
  2. 测 pass@$k$（到 $k_{\max}$）得各方法覆盖集 $S$，算解锁差 $\lvert S_{\text{OPD}}\setminus(S_{\text{GRPO}}\cup S_{\text{VR}})\rvert$；
  3. 用 Step 9 口径确认 B 系列把 credit 方差**真降到 OPD 量级**。
- **闸门 G2 + 分支决策**：
  - $\rho$ 高 + 覆盖不扩 + VR 方差降够 → **形态 A（deflationary），全力投 P5**；
  - $\rho$ 低 + 覆盖扩 → **形态 B（正面）**，换叙事仍能发，或转 P1；
  - $\rho$ 高但 VR 方差没降 → **假阳性**，回 Step 10 修基线；
  - $\rho$ 噪声大 → 回 G1 查分母 / 加 seed。
- **意义**：这是 §11 kill test 的终点，也是 E1 主结果。项目能不能成、走哪种叙事，在此判明。

---

## 6. Stage F/G/H —— 扩成顶会版

### Step 12：E2 regime 相图（G3）
沿三轴各测一组 $\rho$ + 解锁差 + $\hat\delta$：teacher 强度（s2w δ<0 vs self δ≈0）、rollout 正确性（wrong vs correct）、分布（OOD vs ID）。预测 $\rho$ 与 $\hat\delta$ 负相关。相图紊乱 → 回查 Step 11 控制变量。

### Step 13：E3 机制确认（G4，无新训练）
在已有 run 上加测 credit 方差 / gradient SNR / KL-to-teacher；对账：高 $\rho$ ⇔（方差降够 ∧ fixed point 未移动），低 $\rho$ ⇔（fixed point 被拉向 teacher ∧ 出现解锁题）。不自洽 → 重审"方差 vs 知识"操作化定义。

### Step 14：E4 对照定位
跑 R-RLSD、R-vOPD 测其 $\rho$，说明"力度类"方法落在方差缩减区（预期 $\rho$ 高）。

### Step 15：跨域泛化（G5）
代码域（Eurus-RL-Code / DAPO-code 训练，LiveCodeBench v6 / MBPP+ / HumanEval+ 评测）重跑 E1 核心，看 $\rho$ 与覆盖集结论是否保持。分歧 → 明确限定域，把"何处成立"当贡献。

---

## 7. Stage I —— 收尾

### Step 16：novelty 核查 + 成文
投稿前重跑一次 [带引用 novelty 核查](/data1/zhaoenyue/opdpapers/docs/04想法/OPD收益归因_方差缩减还是知识迁移.md)（领域极快，注意 CAST `2606.00172` 等新邻居）；按"$\rho$ + 覆盖集 + 机制 + 相图 + 定位"组织论文；与 P5 §9 差异化表对齐。

---

## 8. 里程碑、MVP 与"何时可以停"

| 里程碑 | 完成的 Step | 意义 |
|---|---|---|
| M0 评测闭环 | 1–3 | 能产出可信数字 |
| M1 pre-flight 达标（G−1） | 1–7 | rig/verifier/eval/σ 全部就位 |
| M2 现象确立（G1） | 8 | 有可归因的 OPD>GRPO 增益 |
| M3 **E1 头条（G2）** | 9–11 | **kill test 终点，项目生死判明** |
| M4 MVP 可发表 | + 13 | $\rho$+覆盖集+机制 三件套，一篇 deflationary 论文 |
| M5 完整顶会版 | + 12,14,15 | 升级为"OPD 收益归因 regime 地图" |

- **最小可发表切片（MVP）= Step 1–11 + Step 13**。
- **任一闸门触发"停/转"**：G−1 失败→修 rig；G1 失败→现象不存在，转 P1；G2 判 $\rho$ 低→切形态 B（仍可发）。
- **省钱顺序**：Stage A/B/D 全是 🟢 推理或轻 pilot，先把这些做满，把训练（🔴）尽量推迟到方向已明（G0/G1 通过）之后。

---

## 9. 与仓库其他文档的关系

- 实验逻辑结构（条件集、选型、指标、闸门定义）：[OPD收益归因_实验设计.md](/data1/zhaoenyue/opdpapers/docs/04想法/OPD收益归因_实验设计.md)
- 论点 / 假说 / 判据 / 风险 / kill test：[OPD收益归因_方差缩减还是知识迁移.md](/data1/zhaoenyue/opdpapers/docs/04想法/OPD收益归因_方差缩减还是知识迁移.md)（P5 正文）
- 理论附录（control-variate 分解、Cov 协议、$\hat\delta$ 估计）：[OPD方差偏差分解理论.md](/data1/zhaoenyue/opdpapers/docs/04想法/OPD方差偏差分解理论.md)（N1）
- 模型/数据/评测选型来源：[模型数据评测统计.md](/data1/zhaoenyue/opdpapers/docs/02图谱/模型数据评测统计.md)
- 候选全景与 P1–P5 排序：[研究方向与候选想法.md](/data1/zhaoenyue/opdpapers/docs/04想法/研究方向与候选想法.md)
