# 纠错-压缩 OPD 后续问答记录

更新时间：2026-06-19

来源：本地 PDF 抽取文本、`LLM_OPD论文筛选总结.md`、`LLM_OPD失败模式_机制归因_算法改进映射.md`、`LLM_OPD失败模式研究现状与开放问题.md`、`后续问答记录.md`。本次未进行 web search。

## Q: Correction-vs-Compression OPD 这个方向，已有论文是否做过类似实验分析？

用户提出的核心思路是：不要只做简单 correct/wrong routing，而是先判断当前 rollout 或 prefix 属于哪一类：

| rollout / prefix 类型 | 期望动作 |
| --- | --- |
| correct rollout | compression OPD |
| wrong but recoverable prefix | correction OPD |
| wrong and unrecoverable prefix | truncate / backtrack / RL exploration |
| style-only mismatch | skip or contrastive cancel |

关键是设计 detector，判断 teacher signal 是在纠错、压缩、强化风格，还是放大错误。

## A: 总体结论

**已有论文已经分别研究了这四类中的多个子问题，但还没有看到一篇论文统一实现“compression / correction / unrecoverable / style-only”四路 detector + router。**

最直接的证据来自 `2605.06188` **OPSD Compresses What RLVR Teaches**：它已经明确提出并实验验证，OPSD 在 thinking-enabled 数学推理中更像压缩正确推理，而不是修复错误轨迹。它做了 correct-only、incorrect-only、split-direction、JSD、teacher-context、context reinjection、longer training 等一系列诊断，结论是：**correct-only OPSD 是相对安全的压缩器，incorrect-only OPSD 往往损害准确率，改变 KL 方向或加更丰富 teacher context 也没有稳定转成 correction。**

后续和新增论文基本沿着四条线推进：

1. **压缩正确轨迹**：`2605.06188`、TRD、TRACE、Sparse-to-Dense 等说明 dense teacher signal 很适合压缩、缩短、内化已经可行的推理。
2. **纠错可恢复错误轨迹**：SCOPE、SSOPD、ROSD、TRD、HSD、TOPD、TRACE 都在尝试把 teacher signal 定位到错误前缀、错误 span、divergence point 或未来轨迹分叉。
3. **识别不可恢复或低价值 prefix**：TRD 的 prefix failure、KAT 的 low-KL agreement trap、Prune-OPD 的 prefix drift、SFD 的 supervision fidelity decay、MOTAB 的 backtracking 都说明有些 prefix 继续做 KL 是低效甚至有害的。
4. **过滤风格或泛化 shortcut mismatch**：RLCSD、CREDIT、AntiSD、TRACE、ROSD、FiRe-OPD 都在说明 teacher-student mismatch 可能只是 style、generic template、shortcut、privileged leakage，而不是真正 reasoning credit。

所以，这个研究方向已有很强的文献基础，但仍有空间：**现有论文多是单一 detector 或单一路由策略，还没有统一比较四类 detector 的在线决策质量，也没有把“纠错 vs 压缩 vs 截断 vs 风格取消”做成一个受控 ablation 框架。**

## 四类状态的已有覆盖

| 状态 | 已有代表 | 已有 detector / proxy | 已尝试动作 | 缺口 |
| --- | --- | --- | --- | --- |
| correct rollout -> compression OPD | `2605.06188`, TRD, TRACE, Sparse-to-Dense | verifier correctness、key-span annotator、trajectory refinement 后 correctness | correct-only KL、key-span FKL、trajectory-level refined distillation、dense transfer | 仍需区分“有益压缩”和“覆盖 student 自己成功路径”。RLRT 说明 correct rollout 上盲目模仿 teacher 可能压制 self-driven path。 |
| wrong but recoverable prefix -> correction OPD | SCOPE, SSOPD, ROSD, TRD, HSD, TOPD, TRACE | teacher PPL 低、correct sibling witness、reflection error quote、fail->pass refinement、shared-prefix divergence、high OT future divergence、error-span annotator | teacher-weighted OPD、witness-conditioned teacher、localized reflection KL、refined trajectory KL、path-conditioned KL、near-future guidance、error-span RKL | detector 仍分散，且很多依赖外部 annotator、reflection 或 correct sibling；缺少统一、低成本、在线 recoverability 估计。 |
| wrong and unrecoverable prefix -> truncate / backtrack / RL exploration | TRD, KAT, Prune-OPD, SFD, MOTAB / Backtracking | prefix failure、sustained low KL agreement、top-k overlap drop、teacher confidence decay、teacher likelihood below adaptive boundary | trajectory refinement、KL-trap truncation、dynamic rollout truncation、lookahead-guided replacement、backtrack and stitch teacher correction | 还缺与 correction OPD 的边界判定：什么时候应继续纠错，什么时候应截断、回退或让 RL 探索。 |
| style-only mismatch -> skip / contrastive cancel | RLCSD, CREDIT, AntiSD, TRACE, ROSD, FiRe-OPD | KL mass on discourse/style tokens、input-generic baseline、shortcut vs deliberation token sign、non-span mask、reflection localization、trajectory/filter confidence | contrastive cancellation、input-generic debiasing、reverse sign or entropy gate、no-KL non-spans、localized distillation、trajectory filtering | 风格、格式、泛化 shortcut、privileged leakage 常被分开处理，还没有和 recoverability detector 合成一个 OPD router。 |

## 逐篇审查

| 论文 | 与 correction-vs-compression 的关系 | 论文中的实验 / 分析证据 | 与用户方案的差距 |
| --- | --- | --- | --- |
| `2605.06188` OPSD Compresses What RLVR Teaches | **最直接**。明确问 OPSD 是 correction 还是 compression。结论是 OPSD 主要压缩已经正确的推理，不稳定修复错误轨迹。 | correct-only / incorrect-only / all-rollout 对比：correct-only 缩短响应且相对保准确率，incorrect-only 损害准确率；Split-direction 和 JSD 改 KL 方向仍不能稳定变成 correction；teacher context variants、privileged-context reinjection、500-step longer training 都只是沿 accuracy-length tradeoff 移动；multi-seed 和 question-level shift 也支持 correct-only 优于 incorrect-only。 | 已经做了 compression-vs-correction 诊断，但 detector 仍是 trajectory-level correctness，不判断 recoverable prefix、unrecoverable prefix、style-only mismatch。 |
| `2604.10688` SCOPE | 把 correct rollout 和 wrong rollout 分成两条路径：正确轨迹用于 exploitation / compression，错误轨迹用于 teacher-guided correction。 | flawed-prefix recovery analysis：低 teacher PPL 的错误前缀更容易被 teacher recover，高 PPL 前缀 recovery rate 急剧下降；DPAW 消融显示去掉或反转 teacher-guided weight 会损害性能，反转 teacher weight 在 AIME24 上尤其差；student-guided weight 对正确但低概率成功路径有帮助。 | 很接近用户方案中的“wrong but recoverable prefix -> correction OPD”。但它仍主要是 correct/wrong + PPL weighting，没有显式输出 style-only 或 unrecoverable 的多路动作。 |
| `2605.10781` RLRT / Rebellious Student | 纠正了“correct rollout 就应该模仿 teacher”的假设。它说明成功轨迹上 teacher mismatch 可能是 student 自己发现的有效路径，应反向利用。 | reward gate 消融：去掉 correct-only gate 后 RLRT-all 会训练崩溃，说明 reversed teacher signal 只能在正确轨迹上用；clip range 消融说明收益来自 teacher-gap reweighting；explore/exploit token marker 和 distribution shift 分析说明 RLRT 改变候选集，而不只是压缩。 | 它不是 correction detector，而是提醒 compression branch 也要再细分：correct rollout 中哪些 token 应压缩 teacher，哪些应保留或强化 student self-driven path。 |
| `2606.08432` TRD / Trajectory-Refined Distillation | 把 OPD 失败归因到 prefix failure：错误前缀已经走入几乎无法无回溯修复的路径，此时 token-level KL 在 frozen failed prefix 上纠错尺度不对。 | 理论分析 teacher distribution 在 prefix failure 下变成“继续错误前缀”和“转向纠正”的 bimodal mixture；实验证据包括 per-token KL、teacher-student PPL gap、epistemic-token mass；trajectory analysis 中 fail->pass cells 表示 refinement 纠错，pass->fail leakage 很低；subset ablation 比较 fail-only、succ-only、fail->succ 等 refined corpus。 | 很接近“wrong recoverable / unrecoverable”的前缀级思想，但 TRD 的动作是先生成 refined trajectory 再蒸馏，不是在线四路 detector；style-only mismatch 不在核心范围。 |
| `2605.17497` SSOPD | 用同一 GRPO group 内的 correct-wrong contrast 构造纠错监督：正确 completion 是 witness，错误 completion 的 prefix 是需要 correction 的位置。 | shortest-correct / longest-wrong rule 来自 stopping-time view；frontier weighting 只在同 prompt 同时有正确和错误分支时启用；selector ablation 显示 shortest correct + longest wrong 最好。 | 很接近“wrong but recoverable prefix -> correction OPD”，但要求 group 中存在正确 sibling；all-wrong 不可恢复、style-only mismatch 需要额外 detector。 |
| `2605.28014` ROSD | 反对全轨迹 OPSD 直接模仿 reference solution，提出 reflection-guided、error-localized correction。 | self-reflector 输出 corrective idea 和 first erroneous span；错误 quote 用于 mask valid prefix，只在错误后缀蒸馏；ablation 显示去掉 reflection 或 localized distillation 会损害 in-domain / OOD；error localization dynamics 支持定位越来越可靠。 | 非常接近“detector 判断 teacher signal 是纠错还是压缩”。但 detector 依赖 reflector，且主要处理 wrong rollout 的 localized correction；对 style-only / unrecoverable 的显式分类还不完整。 |
| `2606.15576` Path-Conditioned / HSD | 指出仅用 final answer 作为 teacher context 时，中间位置 credit 很弱；用同组 successful peer path 可把 credit 集中到 failed rollout 的 divergence point。 | per-token credit profile 显示 answer-conditioned teacher 在中间位置几乎沉默，而 path-conditioned teacher 在 shared prefix 后的 divergence position 出现 sharp peak；ablation 比较 demonstration choice 和 group coverage；还报告 credit mass 集中在 divergence 附近。 | 它是很强的 correction detector 证据：shared-prefix divergence 可定位错误分叉。但要求同组有成功 peer，且不处理不可恢复或风格 mismatch。 |
| `2606.00305` TOPD / Near-Future Guidance | 指出 token-level high loss 不一定等于 trajectory-level correction point，单 token 修正可能落入 token-by-token learning trap。 | near-future OT divergence analysis：高 loss token 平均更可能是 divergence，但约一部分 high-loss token 其实 low-divergence，是 false alarm；downweight low-OT high-loss token 有益，downweight high-OT high-loss token 有害；TOPD 注入 near-future trajectory guidance 优于 standard OPD。 | 直接支持用户提出的 detector：要判断 teacher signal 是真实纠错还是局部噪声。但它的 detector 是 near-future OT，计算成本和在线训练可用性仍需研究。 |
| `2606.09471` KAT / KL Agreement Trap | 处理 unrecoverable / low-value suffix：坏 prefix 上 teacher 和 student 可能低 KL 一致，但这种一致不是可用监督。 | 分析 low-KL agreement trap：持续低 KL 后 teacher supervision alignment 变弱，teacher top-token 词云和 stage-wise supervision ablation 说明 post-trap suffix 低价值；KAT 用滑窗 KL 和动态阈值截断 suffix，优于固定 prefix 截断。 | 很接近“wrong and unrecoverable prefix -> truncate”。但它主要用 low-KL trap 检测，不区分可纠错高 KL prefix 与 style-only mismatch。 |
| `2605.07804` Prune-OPD | 处理 prefix drift：student 生成越长，teacher signal 可能逐渐不可用，应该动态降权或截断。 | 使用 teacher-student compatibility / top-k overlap 检测 drift；drift 后 down-weight 后续 reward 并触发 dynamic rollout truncation；目标是把 compute 分配给 locally exploitable teacher supervision。 | 是 unrecoverable / unreliable prefix detector 的一类实现，但不是 compression-correction 全路由。 |
| `2605.30833` Supervision Fidelity Decay / LGR | 说明长 prefix 后 teacher confidence 和 discriminativeness 衰减，teacher-dependent corrective signal 变弱。 | controlled prefix-completion experiment 显示 prefix 越长，teacher completion accuracy、confidence、discriminativeness 越下降；LGR 用 lookahead-guided replacement 选择能诱导更高 teacher confidence 的 token。 | 提供 recoverability / teacher-fidelity proxy，但它更像 token replacement / generation guidance，不是四路 OPD detector。 |
| `2605.19433` Backtracking / MOTAB | 明确把不可恢复逻辑错误从可接受风格偏差中分离，越界时 backtrack 到安全前缀并 stitch teacher correction。 | 用 teacher likelihood、predictive entropy、adaptive boundary 检测 unsafe point；触发 backtracking，找安全 bifurcation state，再让 teacher 生成 pristine correction；case studies 展示 logical collapse 被回退修复。 | 很贴近“unrecoverable -> backtrack”。但方法偏数据构造 / step-level teacher intervention，不是标准 OPD 训练中的统一 loss router。 |
| `2606.11709` RLCSD | 处理 style-only mismatch：privileged gap 可能集中在 style tokens，而非 task-bearing tokens。 | token KL 排名分析显示 style-distant teacher 的高 KL 主要落在 discourse / delimiter / structural tokens，而 stylistically closer teacher 更偏 digits / operators / task-bearing tokens；contrastive hints 改善 OPSD/RLSD/RLCSD；`-AORM anchoring` 和 contrastive construction 消融证明信号清洗必要。 | 很接近“style-only mismatch -> contrastive cancel”。但主要解决 style drift，不判断 prefix recoverability。 |
| `2605.10194` TRACE | 直接把全 token OPD 改成 critical span routing：正确 rollout 的 key spans 做 FKL，错误 rollout 的 localized error spans 可做 RKL，non-spans 不做 KL。 | all-token self-OPD 出现长度 collapse、entropy 异常、OOD degradation；mask localization ablation 显示 all-token KL、random 25%、inverted 25% 都不如 span-localized routing；annotator quality ablation 显示 online-self annotator 也能保留部分收益。 | 很接近四路 router 的 token/span 版本，但它依赖 annotator span；unrecoverable prefix 主要通过不选 span和 decay 控制，不是显式 truncate/backtrack。 |
| `2605.11613` CREDIT | 处理“generic shortcut / input-generic mismatch”：self-distillation reward 可能奖励所有输入都通用的模板，而不是当前输入的因果 credit。 | 将 self-distillation token reward 分解为 input-specific 与 input-generic；batch-contrastive baseline 移除 generic component；token visualizations 显示 generic boilerplate 被压低、problem-specific token 被强化；lambda sweep 显示过度 debiasing 会损害信号。 | 对 style-only / generic mismatch 很相关，但不处理 correct-vs-wrong 或 recoverability。 |
| `2605.11609` AntiSD | 指出 privileged teacher 会奖励 shortcut tokens、压制 deliberation tokens；默认自蒸馏可能强化“答案已知后的捷径”。 | per-token signal 被解释为 conditional PMI；Figure 2 将 shortcut tokens 与 deliberation tokens 分开；默认 SD 消融下降明显；AntiSD 通过反向 per-token sign 和 entropy-triggered gate 避免 teacher entropy collapse 后继续错误监督。 | 是 style/shortcut detector 的机制证据，也提醒 correction OPD 不应盲目模仿 reference-conditioned teacher。 |
| `2606.02684` FiRe-OPD | 提出 trajectory-level hard filtering + token-level soft reweighting，和用户的 detector-router 思路很接近。 | component ablation 显示 token weighting 和 trajectory filtering 都有贡献；hard trajectory filtering + soft token weighting 最好；case study 和统计显示高权重 token 更偏 reasoning / numbers，低权重 token 更偏 routine / style。 | 是通用 filter-then-reweight，但没有显式把动作分成 compression、correction、truncate/backtrack、style cancel。 |
| `2605.12483` Sparse-to-Dense Reward Principle | 宏观上提出 sparse reward 用于探索，dense teacher signal 用于把能力压缩到部署 student。 | 四阶段 workflow 中 teacher RL、FKL warmup、OPD、student RL 各阶段消融都 load-bearing；RL-improved teacher 经 dense OPD bridge 比直接 student GRPO 更好。 | 支持“OPD 更擅长 compression / transfer”的宏观原则，但不是 prefix-level detector。 |

## 已经做过的关键实验类型

### 1. Outcome-filtered OPSD：直接检验压缩还是纠错

`2605.06188` 的实验最接近用户问题。它固定 OPSD loss，只改变哪些 rollout 进入训练：

$$
L_{\mathrm{OPSD}}
=
\sum_{(x,y)}
\sum_t
D_{\mathrm{KL}}
\left(
\pi_S(\cdot \mid x,y_{<t})
,\,
\pi_T(\cdot \mid x,c,y_{<t})
\right).
$$

然后按 rollout correctness 过滤：

$$
L_{\mathrm{correct}}
=
\sum_{R(y)=1} L_{\mathrm{OPSD}}(x,y),
\quad
L_{\mathrm{incorrect}}
=
\sum_{R(y)=0} L_{\mathrm{OPSD}}(x,y).
$$

如果 OPSD 是 correction，`incorrect-only` 应该明显提升；如果 OPSD 是 compression，`correct-only` 应该更安全。论文结果支持后者。

### 2. Recoverability detector：teacher 能否在当前 prefix 上恢复

SCOPE 的 flawed-prefix recovery analysis 很关键。它先拿 student 的错误轨迹，按 teacher PPL 分桶，再截断 prefix 让 teacher 继续生成：

$$
\mathrm{Recoverability}(y_{<t})
\approx
\Pr
\left[
\mathrm{Verify}
\left(
y_{<t} \oplus y^{T}_{\ge t}
\right)=1
\right].
$$

经验上，低 teacher PPL prefix 更容易被恢复，高 teacher PPL prefix 常是结构性坏上下文。因此 SCOPE 用：

$$
w_{\mathrm{wrong}}
\propto
\mathrm{PPL}_T(y \mid x)^{-1/\tau}
$$

给错误轨迹的 OPD 分支加权。这已经是一个 recoverable-prefix detector 的雏形。

### 3. Prefix failure / unrecoverable detector：继续做 KL 是否还有效

TRD、KAT、Prune-OPD、SFD、MOTAB 都指出：不是所有错误 prefix 都值得继续 KL。

典型信号包括：

- teacher-student top-k overlap 或 compatibility 下降；
- teacher PPL 或 entropy 异常；
- teacher confidence / discriminativeness 随 prefix length 衰减；
- sustained low KL agreement 出现在 degraded state；
- teacher likelihood 跌破 adaptive safety boundary。

这些都可以抽象成：

$$
\mathrm{RecoverableScore}(x,y_{<t})
=
f
\left(
\mathrm{PPL}_T,\,
D_{\mathrm{KL}}(\pi_S,\pi_T),\,
\mathrm{TopKOverlap},\,
H_T,\,
\mathrm{Conf}_T,\,
\mathrm{VerifierPrefixProxy}
\right).
$$

当分数太低时，动作不应是继续 OPD，而应是 truncate、backtrack、trajectory refinement 或 RL exploration。

### 4. Style-only / generic mismatch detector：teacher gap 是否真是 task credit

RLCSD、CREDIT、AntiSD 给出三种不同证据：

- RLCSD：高 KL token 如果集中在 discourse、delimiter、connective、format token，就更像 style drift；
- CREDIT：如果 teacher 对当前 token 的偏好在随机替换输入后仍然存在，就是 input-generic shortcut；
- AntiSD：privileged context 可能奖励 shortcut token、压制 Wait / Let / Maybe 等 deliberation token。

这类 detector 可以写成：

$$
\mathrm{TaskCredit}_{t}
=
\mathrm{TeacherGap}_{t}
-
\mathrm{GenericOrStyleBaseline}_{t}.
$$

如果 `TaskCredit_t` 很低但 teacher gap 很高，说明它更可能是 style-only mismatch，应 skip、cancel 或 debias，而不是做 correction OPD。

## 还没有被完整做过的实验

### 实验 1：四路 router 的统一 ablation

在同一批 rollout 上，对每个 prefix/token 输出四类动作：

$$
a_{i,t}
\in
\{
\mathrm{compress},
\mathrm{correct},
\mathrm{truncate/backtrack},
\mathrm{skip/cancel}
\}.
$$

统一目标可以写成：

$$
L
=
\sum_{i,t}
\mathbb{1}[a_{i,t}=\mathrm{compress}]\,L^{\mathrm{comp}}_{i,t}
+
\mathbb{1}[a_{i,t}=\mathrm{correct}]\,L^{\mathrm{corr}}_{i,t}
+
\mathbb{1}[a_{i,t}=\mathrm{skip}]\,0.
$$

如果 `a=truncate/backtrack`，则停止后续 token 训练，或生成 refined / stitched trajectory 再训练。

需要和下面 baselines 比较：

| baseline | 含义 |
| --- | --- |
| all-token OPD | 不做 detector |
| correct-only OPD | 只做 compression |
| incorrect-only OPD | 只做 correction |
| SCOPE-style correct/wrong routing | trajectory-level 二路 routing |
| TRD-style refinement | 先修 trajectory 再蒸馏 |
| TRACE-style span routing | token/span 级 routing |
| proposed four-way router | 四路 detector + 动作 |

### 实验 2：detector proxy 的正交比较

现有论文的 detector proxy 很多，但没有统一比较。可以同一训练框架中比较：

| proxy | 对应论文 | 可能识别的状态 |
| --- | --- | --- |
| outcome correctness | `2605.06188`, SCOPE | correct vs wrong |
| teacher PPL | SCOPE | wrong but recoverable |
| top-k overlap / compatibility | Prune-OPD | prefix drift / unreliable suffix |
| sliding-window low KL | KAT | low-KL agreement trap |
| teacher confidence / entropy decay | SFD, MOTAB | supervision fidelity / unsafe point |
| reflection error quote | ROSD | local error span |
| correct sibling divergence | SSOPD, HSD | recoverable failed prefix |
| near-future OT divergence | TOPD | real trajectory divergence vs false high-loss |
| KL token type / style rank | RLCSD | style-only mismatch |
| input-generic baseline | CREDIT | generic shortcut |
| annotator critical span | TRACE | key / error / non-span |

### 实验 3：同一错误轨迹上的动作选择

对同一个 wrong rollout，应该区分至少三段：

1. **valid prefix before first error**：不要全局压制，避免破坏正确铺垫；
2. **recoverable local error span**：做 correction OPD 或 RKL / reflection guidance；
3. **post-error unrecoverable suffix**：truncate、backtrack、或者只用于 RL exploration，不做 teacher KL。

ROSD、TRACE、HSD、TOPD 都证明这种局部区分有价值，但还没有形成统一评测。

### 实验 4：正确轨迹上的 compression vs rebellious preservation

正确轨迹也不能只做 compression。需要检测：

- routine / redundant tokens：可以 compression；
- critical key spans：可做 FKL / MLE 强化；
- student self-driven but teacher-disfavored tokens：RLRT 式保留或反向强化；
- pure style mismatch：skip / cancel。

这可以直接接 RLRT、TRACE、RLCSD：

$$
a_{t}
=
\begin{cases}
\mathrm{compress}, & R=1,\ \mathrm{routine\ but\ teacher\ aligned} \\
\mathrm{key\ reinforce}, & R=1,\ \mathrm{critical\ key\ span} \\
\mathrm{preserve\ student}, & R=1,\ \pi_S(a_t)>\pi_T(a_t)\ \mathrm{and\ path\ succeeds} \\
\mathrm{skip}, & \mathrm{style\ only}
\end{cases}
$$

## 最终判断

这个方向已经被多篇论文从不同侧面触及：

- `2605.06188` 已经直接证明 OPSD 更像 compression 而非 correction；
- SCOPE、SSOPD、ROSD、TRD、HSD、TOPD、TRACE 已经提出多种 correction-localization 机制；
- TRD、KAT、Prune-OPD、SFD、MOTAB 已经说明坏 prefix 或长 suffix 上 teacher signal 会衰减、失真、低价值，需要 truncate / backtrack / refinement；
- RLCSD、CREDIT、AntiSD 已经说明 teacher gap 可能只是 style、generic shortcut 或 privileged shortcut，需要 cancel / debias / skip。

但现有工作仍是碎片化的：**没有一篇论文把这些 detector 统一成四路 OPD router，并系统比较每个 detector 对最终性能、长度、entropy、OOD、修复率、压缩率和错误放大率的贡献。**

因此，Correction-vs-Compression OPD 仍然是一个值得探索的算法方向。最有价值的切入点不是再做简单 correct/wrong routing，而是构造一个 prefix/token 级 detector，明确输出：

1. 这里应该压缩；
2. 这里应该纠错；
3. 这里 teacher 已经帮不上，应截断或回退；
4. 这里只是风格/泛化 mismatch，应跳过或对消。
