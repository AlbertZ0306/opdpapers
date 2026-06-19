# OPD Papers Library

这个目录用于整理 OPD / OPSD / RLVR + distillation 相关论文、PDF、本地摘要、失败模式图谱和后续研究想法。文档优先使用中文命名，论文原文 PDF 保存在 `papers/` 下。

## 推荐阅读入口

| 入口 | 说明 |
| --- | --- |
| `docs/00索引/论文索引.md` | 全量论文索引，包含论文类别、来源链接和本地 PDF 路径。 |
| `docs/01筛选/论文筛选.md` | LLM OPD 论文筛选总结，区分保留、扩展、相邻和技术报告。 |
| `docs/01筛选/论文类型.md` | 按机制综述、长轨迹、稳定性、token credit、privileged context 等类型整理。 |
| `docs/02图谱/失败模式映射.md` | 逐篇总结 failure mode、机制归因、算法改进位置。 |
| `docs/02图谱/失败模式研究现状.md` | 汇总 OPD failure mode 的研究现状、已解决方向和开放问题。 |
| `docs/02图谱/基线与Loss对照.md` | 逐篇对照 baseline、loss/objective、是否 sampled-token OPD。 |
| `docs/02图谱/模型数据评测统计.md` | 统计论文使用的模型、数据集和 benchmark。 |

## 问答与想法

| 文件 | 内容 |
| --- | --- |
| `docs/03问答/早期问答.md` | 早期论文阅读问答记录。 |
| `docs/03问答/后续问答.md` | 后续逐篇论文解释、loss 解释、baseline 和方向讨论。 |
| `docs/03问答/方向力度解耦.md` | 审查 OPD 与 RLVR 的 update direction / magnitude 解耦，比较 RLSD、RLCSD、SC-GRPO、CAST、SG-OPD 等是否做过类似实验。 |
| `docs/03问答/纠错压缩OPD.md` | 审查 Correction-vs-Compression OPD，比较 OPSD Compresses、SCOPE、RLRT、TRD、ROSD、TRACE 等是否做过类似分析。 |
| `docs/04想法/方向力度解耦想法.md` | 关于 OPD/RLVR 更新方向和力度排列组合的研究想法。 |

## 元数据

| 文件 | 说明 |
| --- | --- |
| `docs/90元数据/papers.bib` | BibTeX 记录。 |
| `docs/90元数据/下载记录.json` | 已下载论文和官方页面的元数据。 |
| `docs/90元数据/增量候选.json` | 2026 年 5 月以来新增候选论文元数据。 |

## PDF 与页面布局

| 路径 | 说明 |
| --- | --- |
| `papers/llm_opd_methods/` | 核心 LLM OPD / OPSD / self-distillation / RLVR+distillation 方法、分析和综述论文。 |
| `papers/model_tech_reports/` | 模型技术报告中包含 OPD 或 OPD-style post-training 的论文。 |
| `papers/multimodal_and_adjacent/` | speech、VLA/robotics、VLM、flow matching、GUI 等 OPD 相邻论文。 |
| `papers/local_originals/` | 早期本地提供或手动读取的原始 PDF。 |
| `papers/duplicates/` | 预留目录，用于重复下载文件。 |
| `official_pages/` | 官方模型发布页面的本地 HTML/text 保存。 |

## 使用约定

- 后续新增论文优先更新 `docs/00索引/论文索引.md`、`docs/01筛选/论文筛选.md` 和相关图谱文档。
- 论文阅读中的问题和回答继续放入 `docs/03问答/` 下的中文命名文件。
- 研究想法和实验设计放入 `docs/04想法/`，避免混入逐篇论文事实总结。
- 本地已有 PDF 时，优先基于本地论文和已有文档回答，除非确实需要最新外部信息。
