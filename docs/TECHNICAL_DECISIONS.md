# Phase 1 技术决策

项目核心需求为根目录 `RiskSifter_Codex_Project_Brief.md`。本轮仅交付阶段 A 的垂直闭环，补充必要的审核、重复导入、失败恢复与导出；不实现 Self-Improving、RL、RSI。名称使用 RiskSifter（工作目录名称），尚未进行商标清查，不宣称名称排他性。

**实测后的决策更新（2026-10-15）**：最终默认模型改为 SmolVLM2-2.2B-Instruct，256M 保留为轻量选项。发现 image processor 默认放大/切图后，统一关闭 image splitting，以 2.2B 的 384px / 256M 的 512px 原生视频边长推理。修正后 256M 可快速生成简短描述；2.2B 在三个固定原始小样本上提及了 8/11 个有独立标签的类别，256M 为 3/11，因此选择前者作为 16GB 模式默认。样本太小，不声称普遍优越。完整指标、失败响应及硬件证据见 `reports/EXPERIMENTS.md`。下文保留初始路线比较过程，最终参数以代码和此更新为准。

## 开源底座核查（2026-10-14）

以下来自官方仓库/模型卡，不将论文报告数字视作本项目实测。不以 star 数判断质量；仅静态阅读了候选项目，没有逐个安装复现。

| 候选 | 许可/运行条件 | 对 Phase 1 的价值与取舍 |
|---|---|---|
| [AutoMine](https://github.com/xiaomi-research/automine) | Apache-2.0；AV2 3D tracking、RefAV、外部 Claude coder 工作流 | 2026 公开竞赛实现；依赖轨迹与生成代码，不能直接用于任意 MP4。参考筛选/复核思想，不 Fork。 |
| [K-Risk](https://github.com/benmagnifico/K-Risk) | MIT 代码；派生数据另有源数据条件 | 2026-10 有更新；基于轨迹的高风险事件集，TTC 等依赖运动数据。参考证据/审计结构，不借用无标定风险标签。 |
| [Simple-TAD](https://github.com/tue-mps/simple-tad) | 项目 LICENSE 为 CC BY-NC 4.0；VideoMAE/PyTorch；安装文档仍有早期 timm/deepspeed 固定版本 | 有模型及推理入口，可作为后续独立 proposal backend；本轮不引入研究训练依赖、不复制受非商业限制代码。 |
| [DashLab](https://github.com/Sherin-SEF-AI/DashLab) | MIT 项目；PyQt6、YOLO 依赖需分别审查许可 | 有工作台与本机训练经验；核心是检测器蒸馏/自监督，不直接提供本任务的事件标注闭环。避免把伪标签当真值。 |
| [DoTA](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly) | MIT 仓库不等于所有视频 MIT；作者授权分享帧；55GB/分卷下载 | 最契合后续异常时间定位评测。当前没有下载全量，不报告 DoTA 性能。 |
| [SmolVLM2 256M](https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct) | Apache-2.0；Transformers；英文模型 | 选择作为可实际下载的小型本地描述器，使用有时间戳的多图输入；不要求 decord/FlashAttention。模型输出始终为推断。 |

许可原文：[Simple-TAD LICENSE](https://github.com/tue-mps/simple-tad/blob/master/LICENSE)、[安装条件](https://github.com/tue-mps/simple-tad/blob/master/INSTALL.md)。这些仓库并非都经过本项目运行验证。

## 选择与边界

采用原创薄工作台 + 成熟组件组合。Python 3.11+ / FastAPI / SQLite / OpenCV / FFmpeg / 静态 HTML-CSS-JS，无 Node 构建步骤、无云 API、无数据库服务。核心安装不依赖 PyTorch。应用只监听 `127.0.0.1`，禁止跨站写入，处理任务单队列以控制峰值内存。

数据路径：原视频内容哈希 → 原件快照 → 静音 H.264 15fps、最大宽 960 的回放代理 → 2fps 运动索引 → 候选 NMS → 6 帧证据 → 可选 VLM → 待审核事件 → 修订历史 → MP4 + JSONL + manifest + SHA256 + ZIP。

* **候选算法**：160×96 灰度、裁掉天空/仪表板，Farneback 光流减去中位全局运动，结合外观差异排序。固定 6 秒窗、默认最多 4 个，tIoU > 0.15 抑制重叠。`0.008` 为未经道路数据调优的变化门槛。它只能挑选视觉变化，不能宣称识别危险；普通交通、相机抖动和检测框覆盖都可能触发。另提供同预算均匀采样基线。
* **描述器**：SmolVLM2 256M，固定模型 revision，6 张图按时间顺序输入，512 像素上限，greedy 生成最多 128 token。模型不生成结构化 TTC、距离或风险等级；原文保留，即使它违背 prompt 产生幻觉也不静默隐藏。
* **Query**：由实际模型描述加检索指令生成；未额外训练 Query 模型。标签是模型原文中的简单概念提取，不是独立检测器输出。
* **边界**：Phase 1 使用运动峰值附近窗口，允许人工精修。没有把固定窗口包装成 VLM 精确定位。
* **搜索**：保存描述/标签的词项匹配 + 明示的中英概念扩展，可重复查询不重新推理。不能处理复杂否定、因果或没有描述到的动作；跨模态向量检索放在 Phase 2。
* **可追溯性**：保留源/代理哈希、模型固定版本、prompt、全部实际输入证据时间/哈希、原始响应、策略配置、运行记录和每次人工修改。置信度为 `null`，因为没有校准集；proposal_score 不是概率。
* **去重/续跑**：相同文件 SHA256 去重；同配置的完整运行命中缓存。不同模型/策略产生独立运行，失败候选可重试；人工修订不可被重试覆盖。近重复视觉视频和跨运行事件合并未实现。
* **资源控制**：顺序处理、一批一模型、有限帧/候选预算。轻量模式无需权重；模型文件/依赖缺失时显式降级到未解释候选，其他错误直接失败。长视频仍需完整解码/代理转码，不声称小时级吞吐已经优化。

## 硬件路线

| 模式 | 配置 | 验证范围 |
|---|---|---|
| 轻量 | CPU，4–8GB 可作为起点；模型设为 none | 导入、索引、候选、回放、编辑、导出可用；无语义识别 |
| Mac 本地 VLM | 16GB M2 实测；256M/float16/MPS，CPU fallback | 真实运行指标见实验报告；8GB 机型未实测，减少预算、或 none |
| NVIDIA | 安装适配 CUDA 的 torch/torchvision；相同 annotator 设 device=cuda | 当前没有 4090，不能报告实测速度/峰值显存。可替换同系列 2.2B 权重尝试更强描述，需重新评估 |

没有选择 MLX 作为强依赖：先用同一 Transformers 代码复用 CPU/MPS/CUDA，降低维护面；量化/MLX 只有实际测得收益后再引入。没有引入向量数据库、分布式任务、Agent 框架或训练框架。

## 扩展接口

普通候选策略的 `propose` 方法、描述器的 `annotate` 方法、独立 `evaluation.evaluate_events` 和 `Store.review` 是可替换边界，无需额外 Protocol/插件框架。评估只接受外部参考事件；没有模型自评 reward 或自动反馈训练。将来可加入新策略实现，但必须保留独立测试、版本记录与人工修订。
