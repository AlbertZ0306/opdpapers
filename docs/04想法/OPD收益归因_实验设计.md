# P5 实验设计：OPD 收益是知识迁移还是方差缩减

更新时间：2026-06-23

定位：本文是 [OPD收益归因_方差缩减还是知识迁移.md](/data1/zhaoenyue/opdpapers/docs/04想法/OPD收益归因_方差缩减还是知识迁移.md)（P5）的**主实验设计文档**。P5 正文给出论点、假说与判据；本文把它落成**可执行、有先后顺序、有 go/no-go 闸门**的实验计划。本版**先主要覆盖所有主实验**（E1–E4 及其前置 kill 闸门），消融与鲁棒性实验留待后续补充。模型与数据集选型对齐 [模型数据评测统计.md](/data1/zhaoenyue/opdpapers/docs/02图谱/模型数据评测统计.md) 的主流口径（Qwen3 + DAPO-Math + AIME/AMC/HMMT/MATH500），以保证结果与现有 OPD 文献可比。

---

## 0. 一页纸总览

**要测的核心量**（两条独立证据链）：

1. **复现率** $\rho=\dfrac{\mathrm{Acc}(\text{teacher-free VR})-\mathrm{Acc}(\text{GRPO})}{\mathrm{Acc}(\text{OPD})-\mathrm{Acc}(\text{GRPO})}$ —— teacher 增益里有多少能被"零外部知识"的方差缩减复现。
2. **覆盖集解锁** $S_{\text{OPD}}\setminus\big(S_{\text{GRPO}}\cup S_{\text{VR}}\big)$ —— OPD 是否解锁了 GRPO + teacher-free VR 都解不开的题（pass@$k$ 大 $k$）。

**判读**：$\rho\to1$ 且覆盖集不扩 → H-VR（方差缩减，deflationary 形态 A）；$\rho\to0$ 且覆盖集扩 → H-KT（知识迁移，形态 B）。

**实验链（先后顺序，每阶段带闸门）**：

```text
Phase −1 预备与口径校准（pre-flight）          ── 闸门 G−1：rig/verifier/eval/噪声地板全部达标
   │
Phase 0  基建 + Tier 0 零训练协方差代理         ── 闸门 G0：teacher 信号是否像控制变量
   │
Phase 1  Tier 1：GRPO + OPD（self-teacher，最小规模）── 闸门 G1：Δ=Acc(OPD)−Acc(GRPO) ≥ 3σ
   │
Phase 2  Tier 2 = E1 主结果：加 teacher-free VR，测 ρ + 覆盖集 ── 闸门 G2：ρ 稳定且 VR 方差确降
   │
Phase 3  E2：regime 相图（δ 轴 / correct-wrong / ID-OOD）
   │
Phase 4  E3：机制确认（方差、gradient SNR、fixed point 漂移）
   │
Phase 5  E4：对照定位（RLSD / vOPD 投到 ρ 谱系）+ 第二域（代码）泛化
```

**Phase −1 是主实验前的必经预备**（详见 §4.0）：它不产出论文数字，只保证后面所有 Acc / Δ / ρ 的测量是可信的——rig 正确、verifier 干净、eval 口径固定、噪声地板已知、算力对齐口径已定。Phase 0–2 同时就是 P5 的 [§11 三层 kill test](/data1/zhaoenyue/opdpapers/docs/04想法/OPD收益归因_方差缩减还是知识迁移.md)；跑通即拿到头条数字。Phase 3–5 把单点结论扩成可发表的相图与机制证据。

---

## 1. 实验对象（方法 / 条件集）

所有条件**共享同一 backbone、同一 prompt 集、同一算力预算**，只改 credit 信号来源。这是 $\rho$ 合法的前提。

| 代号 | 条件 | credit 信号 | 注入外部知识？ | 角色 |
|---|---|---|---|---|
| **C-GRPO** | GRPO | 轨迹级 verifiable reward + group baseline | 否 | $\rho$ 分母基线 |
| **C-OPD-self** | self-teacher OPD | teacher log-ratio（teacher = 冻结初始权重 / EMA self） | 否（同模型） | **头条**：主流 OPD |
| **C-OPD-s2w** | strong-to-weak OPD | teacher log-ratio（teacher 真更强） | 是 | E2 的 δ<0 臂 |
| **B1** | 学习型 value head | $R_i-V_\phi(s_{i,t})$，dense per-token | 否 | teacher-free VR |
| **B2** | teacher-free control variate | $A^{\mathrm{RL}}-\beta^{*}(\hat m-\bar m)$，$\hat m$ 仅用 policy 自身特征 | 否 | teacher-free VR（最干净） |
| **B3** | self-consistency dense 信号 | policy 自身一致性/熵正则 dense 信号 | 否（定义上无新知识） | teacher-free VR 上界 |
| **B4** | CAST `2606.00172` | answer-free stop-gradient self-teacher 塑造 token advantage | 否 | 收编的现成 teacher-free 基线 |
| **R-RLSD** | RLSD `2604.03128` | 方向=RLVR，力度=teacher | 是 | E4 定位用 |
| **R-vOPD** | vOPD `2605.07865` | teacher reverse-KL 控制变量 | 是 | E4 定位用 |

B 系列原则：**只用 policy 自身的 value/统计/分布降方差，注入零外部知识**——这样"B 能复现 OPD 增益"才能合法推出"OPD 增益不是知识"。$\rho$ 取 B1–B4 中**最强的一个**（任一复现即支持 H-VR）；同时全报，互为消融。B2 有理论锚点：vOPD 已证 OPD 最优控制变量是 teacher 逐 token reverse-KL，B2 的 teacher-free 预测器越逼近此量越好。

---

## 2. 模型与数据集选型（对齐主流，保证可比）

### 2.1 模型（Qwen3 系列，OPD 数学推理默认骨架）

| 角色 | 选型 | 依据（模型数据评测统计.md） |
|---|---|---|
| 主 student | **Qwen3-1.7B-Base** | OPD 数学推理最密集骨架（`2602.12125`/`2604.13016`/`2605.10781`/`2606.00172` 等） |
| 规模检查 student | Qwen3-4B-Base | 同上，验证结论不只在 0.6B/1.7B 成立 |
| self-teacher | student 的**冻结初始权重**（OPSD 口径 `2605.06188`），或 EMA-self | 同模型自教师，δ≈0 |
| strong teacher | **Qwen3-8B**（算力够则 Qwen3-30B-A3B-Instruct-2507） | strong-to-weak 标准对照（`2602.12125`/`2605.03677`），δ<0 |

> 选 Base 而非 Instruct 作 student：与 `2602.12125`/`2605.10781` 一致，避免 instruct 后训练把 OPD/GRPO 的差异提前抹平。

### 2.2 数据集

| 用途 | 数据集 | 依据 |
|---|---|---|
| 主训练 prompt | **DAPO-Math-17K** | OPD/RLVR 数学最主流 prompt 源（极高频） |
| 高难度 / OOD-hard 训练 | DeepMath（difficulty ≥ 6 子集） | `2602.12125`/`2605.03677` 的难样本口径 |
| ID 评测 | **MATH500、AMC23、AIME24、AIME25**（avg@$k$） | OPD 数学主评测闭环 |
| 覆盖集评测（大 $k$ pass@$k$） | MATH500 + AIME24/25 固定子集，$k\in\{1,4,16,64,256\}$ | pass@$k$ 覆盖方法学（Yue et al. `2504.13837`） |
| OOD 评测（student 够不着） | OlympiadBench、GPQA-Diamond、MMLU-Pro | EOPD `2603.07079` 的 OOD 口径 |
| 第二域泛化（Phase 5） | 训练 Eurus-RL-Code / DAPO-code；评测 LiveCodeBench v6、MBPP+、HumanEval+ | 代码是 OPD 第二梯队标准域 |

**核心域 = 数学**（verifier 便宜、文献基线最厚、OPD>GRPO gap 最稳）；代码仅作 Phase 5 泛化臂。

---

## 3. 评测指标定义

| 指标 | 定义 | 用在 |
|---|---|---|
| Acc（avg@$k$） | 固定 $k$（如 avg@16）下平均正确率 | $\rho$ 的三个端点 |
| **$\rho$** | $\frac{\mathrm{Acc}(B^{*})-\mathrm{Acc}(\text{GRPO})}{\mathrm{Acc}(\text{OPD})-\mathrm{Acc}(\text{GRPO})}$，$B^{*}$=最强 teacher-free 基线 | E1 头条 |
| 覆盖集 $S_{\text{method}}$ | pass@$k_{\max}$ 下可解题目集合 | E1 第二证据 |
| 解锁差 | $\lvert S_{\text{OPD}}\setminus(S_{\text{GRPO}}\cup S_{\text{VR}})\rvert$ | E1 / E2 |
| credit 方差 | per-token credit 估计的方差 | E3 机制 |
| gradient SNR | $\lVert\mathbb{E}[g]\rVert / \mathrm{std}(g)$ | E3 机制 |
| KL-to-teacher | $\mathrm{KL}(\pi_\theta\Vert\pi_T)$ 随训练演化 | E3：fixed point 是否被拉向 teacher |
| $\hat\delta$ | teacher 偏差估计（N1 协议） | E2：与 $\rho$ 的负相关 |

---

## 4. 主实验逐项设计

### Phase −1 —— 预备与口径校准（pre-flight，主实验前必做）

**目的**：主实验的所有结论都建立在"Acc / Δ / ρ 测得准"之上。本阶段不产出论文数字，只逐项排除会**静默污染**整条实验链的隐患。任一项不达标，后面的 ρ 都不可信。每项都是带判据的小实验，不是纯工程搭建。

| 代号 | 预备实验 | 为什么必须在主实验前做 | 通过判据（G−1 的子门） |
|---|---|---|---|
| **P−1 rig 复现性** | 在一个**已知设置**上复现一篇公开 GRPO 与 OPD-self 结果（如 `2602.12125` / OPSD `2605.06188` 的某个点） | GRPO 实现若偏弱/有 bug，分母 Δ 与 ρ 全部失真；必须先证明 rig 能产出文献量级的数 | 复现 avg@$k$ 落在公开值的合理容差内（如 ±2–3pt） |
| **P−2 verifier 审计** | 在带标注的题目子集上测 reward verifier 的假阳/假阴率，修等价形式（如 $\frac{1}{2}$ vs 0.5、单位、LaTeX 变体） | verifier 错误会同时污染 Acc 和 advantage 信号 $A^{\mathrm{RL}}$，直接毒化 Phase 0 协方差与所有 ρ | 假阳/假阴率低于阈值（如 <1–2%）；等价形式匹配修好 |
| **P−3 eval 口径校准** | 固定 decoding（温度/top-p/max-len）；验证 pass@$k$ 随 $k$ 单调、avg@$k$ 跨 seed 稳定；**pilot 选定覆盖集的 $k_{\max}$**（pass@$k$ 饱和处） | 覆盖集探针强依赖 $k_{\max}$；$k$ 太小测不出解锁、太大烧钱。eval 口径不固定则跨方法不可比 | pass@$k$ 在 $k_{\max}$ 前饱和；avg@$k$ seed 方差可接受；口径写死成协议 |
| **P−4 teacher 有效性** | 检查 self-teacher 的 log-ratio 分布非退化（$c^{\mathrm{OPD}}$ 方差>0，否则 OPD≈no-op）；对 strong-to-weak，确认 teacher 在评测集上**确实更强**且 student 有 headroom | 若 self-teacher 信号近乎为零，OPD 与 GRPO 无差，分母塌缩；若 strong teacher 并不更强，δ<0 臂不成立 | self-teacher $c^{\mathrm{OPD}}$ 非平凡；s2w 中 Acc(teacher)≫Acc(student) |
| **P−5 算力对齐口径** | 定义并 pilot "matched budget" 的**单位**：按梯度更新步 / 按 student rollout token / 按 wall-clock；显式处理 OPD 多出的 teacher 前向开销 | §5 要求三条件算力对齐，否则 ρ 被"VR 训得久/算得多"混淆；OPD 与 GRPO 的算力构成不同，必须先定口径 | 形成一份可执行、已 pilot 验证的对齐约定 |
| **P−6 噪声地板 / power** | 用 GRPO 跑 ≥3 seed，测 Acc 的 seed 标准差 $\sigma$；据此估计可检测的最小 Δ，反推 G1 所需 seed 数 | 不知道 $\sigma$ 就无法判断 G1 的"$\Delta\ge3\sigma$"是否可达；可能整个研究在此规模上根本测不出现象 | 预期 Δ 在计划 seed 数下可检测；否则加 seed 或上调规模 |

**复用约定**：P−1 / P−6 都要跑 GRPO，这条 GRPO run **直接复用为 Phase 0 的协方差代理输入**，不重复训练。P−3 选定的 $k_{\max}$、P−5 定的算力单位、P−2 修好的 verifier 在后续所有 Phase 沿用。

**闸门 G−1**：六项子门全过 → 进 Phase 0；任一不过 → 先修该项（修 verifier / 调 eval / 换 teacher / 重定算力口径 / 加 seed / 修 rig），**不要带着坏口径进主实验**。

---

### Phase 0 —— 基建 + Tier 0 零训练协方差代理

**目的**：搭好三条件共用的 GRPO/OPD 训练器；在烧钱前用纯前向预判 $\rho$ 方向。

**做法**：
1. 实现/复用一个 GRPO 训练器（DAPO-Math-17K，Qwen3-1.7B-Base，group rollout）。跑**一条** GRPO 到收敛。
2. 取其 rollout，对选定 teacher 做**纯前向**，逐 token 算 $c^{\mathrm{OPD}}=\log\pi_T-\log\pi_\theta$ 与 RL 优势 $A^{\mathrm{RL}}$。
3. 测 $\mathrm{Cov}(A^{\mathrm{RL}},c^{\mathrm{OPD}})$、$\mathrm{Var}(A^{\mathrm{RL}})$，及 $A^{\mathrm{RL}}$ 对 $c^{\mathrm{OPD}}$ 回归的 $R^2$（teacher 作为控制变量能吸收的方差预算）。

**闸门 G0**：
- $R^2$/协方差**高** → teacher 形状就是方差缩减控制变量 → 预测 $\rho$ 高，进 Phase 1 信心更足；
- 协方差**≈0** → teacher 信息与 RL credit 正交 → 预测 $\rho$ 低，仍进但预期走形态 B；
- 取不到稳定 advantage 信号 → 先修训练器/方差口径。

**成本**：1 条 GRPO + 若干前向，无 OPD/VR 训练。

---

### Phase 1 —— Tier 1：建立 $\rho$ 的分母（现象是否存在）

**目的**：在最小规模上确认"OPD 相对 GRPO 有稳定可测、超过种子噪声的增益"——没有它就没有可归因对象。

**做法**：Qwen3-1.7B-Base，DAPO-Math-17K，**严格相同的 rollout 数 / 更新步 / 总 token 预算**下，各跑 **C-GRPO** 与 **C-OPD-self**，≥3 seed（头条建议 5 seed）。在 MATH500/AMC23/AIME24/25 上测 avg@16，算 $\Delta=\mathrm{Acc(OPD)}-\mathrm{Acc(GRPO)}$ 与 seed 标准差 $\sigma$。

**闸门 G1**：
- $\Delta\gtrsim 3\sigma$ → 分母稳健，进 Phase 2；
- $\Delta$ 淹在噪声里 → 最小幅度上调规模（→Qwen3-4B 或换更易出 gap 的 regime），或承认"此规模无可归因增益"并**停止**（别在分母≈0 上算 $\rho$）。
- regime 选择：优先 self-teacher（头条靶）；若其 $\Delta$ 太小，可先用 strong-to-weak 拿干净分母把 pipeline 跑通（代价：该 regime $\rho$ 低不算意外）。

---

### Phase 2 —— Tier 2 = E1 主结果：$\rho$ + 覆盖集

**目的**：P5 的头条数字。

**做法**：
1. 复用 Phase 1 的训练器，bolt 上 teacher-free 基线 **B1 / B2 / B3 / B4**（同模型/prompt/预算/≥3 seed）。kill 阶段可只先上最便宜的 B1 或 B4(CAST)；主实验四个全报。
2. 测 $\rho$（用最强基线 $B^{*}$）；
3. 测 pass@$k$（$k\in\{1,4,16,64,256\}$）得各方法覆盖集 $S$，算解锁差 $\lvert S_{\text{OPD}}\setminus(S_{\text{GRPO}}\cup S_{\text{VR}})\rvert$；
4. **同时**用 Phase 0 口径确认 B 系列把 credit 方差真降到 OPD 量级。

**闸门 G2**：
- $\rho$ 稳定（seed 间方差小）且 **B 的方差确降到 OPD 量级** → 结果可信，进 Phase 3；
- $\rho$ 高但 B 方差没降 → **假阳性**，B 只是恰好刷分，回去修基线，别下结论；
- $\rho$ 噪声大 → 回 G1 检查分母，或加 seed。

**预期产物**：一句话头条——"在主流 self-teacher 数学 OPD 上，teacher-free VR 复现了 $\rho$ 比例的增益，覆盖集（不）扩大"。

---

### Phase 3 —— E2：regime 相图

**目的**：把单点 $\rho$ 扩成"$\rho$ 何时高/低"的可证伪相图（P5 §3 预测）。沿三条轴各测一组 $\rho$ + 解锁差：

| 轴 | 低 $\rho$ 端（H-KT） | 高 $\rho$ 端（H-VR） | 实现 |
|---|---|---|---|
| teacher 强度 δ | strong-to-weak（C-OPD-s2w，δ<0） | self-teacher（C-OPD-self，δ≈0） | 换 teacher：Qwen3-8B/30B-A3B vs 冻结自教师 |
| rollout 正确性 | wrong-rollout 占多 | correct-rollout 占多 | 按 group 正确率分桶训练/评测 |
| 分布内外 | OOD（OlympiadBench/GPQA/MMLU-Pro） | ID（MATH500/AMC23） | 换评测集 |

**预测 & 检验**：$\rho$ 在第一类低、第二类高；且 $\rho$ 与 N1 判据的 $\hat\delta$ 量级**负相关**（同时测 $\hat\delta$）。

**闸门 G3**：相图若与预测一致 → 强结构化结论；若紊乱无规律 → 说明 $\rho$ 受未控混淆主导，回查 Phase 2 控制变量（预算/方差匹配）。

---

### Phase 4 —— E3：机制确认

**目的**：把"$\rho$ 高/低"与"方差是否真降 / fixed point 是否真移动"对上，排除"$\rho$ 由别的东西驱动"。

**做法**（在 Phase 2/3 的 run 上加测量，不新增训练）：
1. **方差侧**：测各方法 credit 方差与 gradient SNR；验证 teacher-free VR 是否把方差降到与 OPD 同量级（H-VR 的必要条件）。
2. **fixed point 侧**：测 $\mathrm{KL}(\pi_\theta\Vert\pi_T)$ 随训练演化——OPD 是否真把 fixed point 拉向 teacher，而 GRPO/VR 不。
3. **对账**：高 $\rho$ ⇔（VR 方差降够 ∧ fixed point 未显著移动）；低 $\rho$ ⇔（fixed point 被拉向 teacher ∧ 出现解锁题）。

**闸门 G4**：机制与 $\rho$ 自洽 → 因果叙事成立；不自洽（如 $\rho$ 高但 fixed point 大幅移动）→ 重新审视"方差缩减 vs 知识"的操作化定义。

---

### Phase 5 —— E4：对照定位 + 泛化

**目的**：把已有"力度类"方法钉到 $\rho$ 谱系上，并验证结论跨域。

**做法**：
1. 在同一套设置里跑 **R-RLSD**、**R-vOPD**，测其 $\rho$，说明它们本质落在方差缩减区（预期 $\rho$ 高）。
2. **第二域泛化**：在代码域（Eurus-RL-Code / DAPO-code 训练，LiveCodeBench v6 / MBPP+ / HumanEval+ 评测）重跑 E1 核心（GRPO/OPD-self/最强 B），看 $\rho$ 与覆盖集结论是否保持。

**闸门 G5**：跨域一致 → 结论稳健，可写"在所测 regime 普遍成立"；分歧 → 明确限定域，把"何处成立"本身当贡献。

---

## 5. 全局控制与公平性（贯穿所有 Phase）

1. **算力对齐**：所有条件固定相同的 prompt 数、每 prompt rollout 数 $G$、更新步数、总训练 token。否则 $\rho$ 被"VR 训得久"混淆。
2. **种子**：每条件 ≥3 seed（头条 $\rho$ 与相图关键点用 5 seed）；一切差值报 mean±std。
3. **$\rho$ 的"差之比"风险**：仅在 $\Delta\ge3\sigma$（G1）时才信 $\rho$；报 $\rho$ 的 bootstrap 置信区间。
4. **评测协议统一**：同 decoding 温度、同 $k$、同 verifier；覆盖集用同一固定题目子集。
5. **"零外部知识"边界**：B1/B2 完全不含任何 teacher/更强模型/更优分布；B3/B4 仅用 policy 自身分布，单独标注其"上界/收编"性质。
6. **时效**：投稿前重跑一次 [novelty 核查](/data1/zhaoenyue/opdpapers/docs/04想法/OPD收益归因_方差缩减还是知识迁移.md)（领域极快）。

风险与防御详见 P5 正文 §10（基线不公平、零知识边界、$\rho$ 幅度巧合、规模外推、ProRL 争议前提、CAST 撞车）。

---

## 6. 实验依赖图与最小可发表切片

```text
Phase −1 (G−1) ──► Phase 0 (G0) ──► Phase 1 (G1) ──► Phase 2 (G2) ──► Phase 3 (G3)
                                                          │                  │
                                                          └──► Phase 4 (G4) ◄┘
                                                                             │
                                                                      Phase 5 (G5)
```

- **Phase −1 是全链前置**：rig/verifier/eval/噪声地板未达标前不进 Phase 0；它不进论文，但决定后面一切测量是否可信。
- **最小可发表切片（MVP）**：Phase −1（预备）+ Phase 0–2（E1）+ Phase 4（E3 机制）。即"$\rho$ + 覆盖集 + 机制确认"三件套，已构成一篇 deflationary 经验论文。
- **完整顶会版**：再加 Phase 3（相图）+ Phase 5（定位 + 跨域），把单点结论升级为"OPD 收益归因的 regime 地图"。
- **任一闸门触发"停/转"**：G1 失败→现象不存在，转 [P1 因果 credit 审计](/data1/zhaoenyue/opdpapers/docs/04想法/研究方向与候选想法.md)；G2/G4 判 $\rho$ 低且覆盖扩→切 P5 形态 B 正面叙事（同样可发）。

---

## 7. 与仓库其他文档的关系

- **时间有序执行视图（从第 1 步到投稿，16 步逐步安排）**：[OPD收益归因_执行路线图.md](/data1/zhaoenyue/opdpapers/docs/04想法/OPD收益归因_执行路线图.md)（本文的"怎么按顺序做"配套）
- 论点 / 假说 / 判据 / 风险 / kill test：[OPD收益归因_方差缩减还是知识迁移.md](/data1/zhaoenyue/opdpapers/docs/04想法/OPD收益归因_方差缩减还是知识迁移.md)（P5 正文）
- 理论附录（control-variate 偏差-方差分解、Cov 协议、$\hat\delta$ 估计）：[OPD方差偏差分解理论.md](/data1/zhaoenyue/opdpapers/docs/04想法/OPD方差偏差分解理论.md)（N1）
- 分组口径、诊断指标、实验臂：[方向力度解耦想法.md](/data1/zhaoenyue/opdpapers/docs/04想法/方向力度解耦想法.md)
- 模型/数据/评测选型来源：[模型数据评测统计.md](/data1/zhaoenyue/opdpapers/docs/02图谱/模型数据评测统计.md)
- 候选全景与 P1–P5 排序：[研究方向与候选想法.md](/data1/zhaoenyue/opdpapers/docs/04想法/研究方向与候选想法.md)
