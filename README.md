# RiskSifter

本地视频事件挖掘与审核工作台。把驾驶视频变成可回放、可搜索、可修订、可追溯导出的候选事件数据。

**Phase 1 已实现端到端流程。自动结果是待审核伪标签，不是已验证的危险事件。** 没有 RL、RSI、Self-Improving、云模型 API 或自动训练。核心需求来自 [项目 Brief](RiskSifter_Codex_Project_Brief.md)。

## 立即运行

需要 Python **3.11–3.12**（本机验证 3.12）。FFmpeg 通过 imageio-ffmpeg wheel 提供，无需单独安装。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
risksifter doctor
risksifter serve
```

打开 **http://127.0.0.1:8765**。导入 MP4/MOV 文件或本机文件夹 → 选择视频 → 挖掘 → 查看候选与证据 → 调整边界、描述、Query → 确认/拒绝 → 导出 ZIP。

轻量模式立即可用，无需模型权重，输出运动变化候选及证据，**不生成交通语义判断**。初次依赖/权重/样例下载需要网络，此后的索引、模型推理、回放、审核和导出可离线运行。视频不会上传。

已有本地环境可直接运行 `.venv/bin/risksifter serve`；端口占用时使用 `--port 8766`。新克隆的仓库不包含模型、视频或运行数据库，请按下面步骤下载样例。

## 启用本地模型与官方真实样例

```bash
source .venv/bin/activate
pip install -e '.[vlm,test]'
python scripts/download_model.py
python scripts/download_sample.py
risksifter import data/public/comma2k19-driving.mp4
risksifter mine all --annotator smolvlm
risksifter serve
```

默认模型为 **SmolVLM2-2.2B-Instruct**，固定 revision；M2 16GB 已实测。下载约 8.4GiB 权重，加载为 float16。受限网络可在下载命令前显式设置 `HF_ENDPOINT=https://hf-mirror.com`。镜像只用于公开依赖下载；推理强制 `local_files_only=True`，不接受远程自定义模型代码。

低内存/更快的可选模式：

```bash
python scripts/download_model.py \
  --repo HuggingFaceTB/SmolVLM2-256M-Video-Instruct \
  --revision 067788b187b95ebe7b2e040b3e4299e342e5b8fd \
  --output models/smolvlm2-256m
risksifter serve --model-path models/smolvlm2-256m
```

256M 描述更简略，可能漏掉重要对象。资源不足时选轻量模式，减少候选预算；不默认声称 8GB 机器具备 16GB 测试机同等体验。CUDA 机器请先安装匹配驱动的 PyTorch/torchvision；`--device cuda` 可显式选择。**4090 尚未实测**。

## 已实现能力

* 本地文件、多文件上传、文件夹递归导入；SHA256 去重，保留原件与来源许可。
* 15fps 静音回放代理、2fps 运动索引、固定预算候选筛选与重叠抑制。
* 可选本地 VLM 查看每个候选的 6 张证据帧；保存时间、哈希、模型版本、prompt、原始响应、预处理及运行记录。
* 中文工作台：后台任务、视频时间线、候选卡片、证据跳转、边界编辑、Query/描述/标签编辑、确认/拒绝、乐观锁和修改历史。
* 基于已有描述的中英关键词检索；不重新分析视频。默认显示每个视频最新分析，可切换查看历史分析。
* 导出可播放 MP4、JSONL 标注、证据、来源、审核历史、运行配置和 SHA256 manifest，打包成版本化 ZIP。
* 失败保留/重试、相同配置缓存、重启后中断任务提示；不覆盖人工修改。缺少模型文件可明确降级，意外推理错误会让任务失败。

**尚未实现**：可靠危险类型分类、精确事件起止定位、向量语义检索、近重复视频去重、对象跟踪、自动匿名化、人工审核收益统计、大规模性能优化。运动分数不是碰撞概率；无校准信息时 TTC、距离、风险级别和置信度保持空值。

## 真实验证结果

详见 [实验报告](reports/EXPERIMENTS.md)，包含失败案例及原始 JSON/log。

* M2 / 16GB / macOS 14.5 / MPS：179 秒公开视频 → 8 个候选、48 张证据。最终 2.2B 运行约 **80.31 秒**（复用视频/运动索引、包含模型冷加载，不含最后导出）。最高进程 RSS **3.69GiB**，不等同于全部 GPU/统一内存占用。
* 另外从官方 TUM 压缩包按需提取 3 个原始小片段（总 18 秒、2fps），使用发布者 OpenLABEL 核对描述中的对象类别，推理不读取标签。该小样本只能做事实核对，**不提供事故 precision/recall 或安全性能结论**。
* 14 项离线测试通过（缺少本地模型时跳过一项 processor 测试），并通过真实浏览器验证。结果见 [实验报告](reports/EXPERIMENTS.md) 与 [浏览器检查](reports/browser-check.json)。

样例来源：comma.ai [comma2k19](https://huggingface.co/datasets/commaai/comma2k19)（MIT）、[TUM Traffic Accid3nD](https://accident-dataset.github.io/)（CC BY-NC-SA 4.0）。数据许可不随本项目 MIT 代码改变。视频默认不随源码发布；下载脚本、固定版本和来源清单可复现输入。[数据说明](docs/DATASETS.md) / [真实事件 JSON](examples/real-event.json)。

## CLI、评估与导出

```bash
risksifter --workspace workspace import /absolute/path/to/videos
risksifter --workspace workspace mine all --annotator smolvlm --max-candidates 4
risksifter --workspace workspace list
risksifter --workspace workspace export                 # 仅最新分析中的已确认事件
risksifter --workspace workspace export --include-pending # 显式包含自动伪标签
python -m pytest -q
```

API 文档位于 `/docs`。`GET /api/events` 默认返回最新分析；`?latest_only=false` 可查询历史。工作台是单用户本地应用，服务仅监听 loopback，不提供公网部署或多用户权限体系。

本轮实验复现：

```bash
python scripts/run_experiment.py --device auto \
  --output data/results/my-run.json data/public/comma2k19-driving.mp4

# 取得官方小型原始帧/标签集；只按需读取公开分卷，不下载整个 9GB 包
python scripts/fetch_tum_subset.py
python scripts/run_experiment.py --workspace data/raw-eval-workspace \
  --max-candidates 1 --output data/results/raw-subset-run.json \
  data/public/tum-raw/r09_s02.mp4 data/public/tum-raw/r09_s03.mp4 data/public/tum-raw/r09_s04.mp4
python scripts/evaluate_object_claims.py
```

事件时间定位评估器接受**独立标注清单**，提供同预算 motion/uniform 对比、tIoU 0.1/0.3/0.5、事件 precision/recall、匹配 IoU、候选回看时长。不能用自动描述作为真值：

```bash
python scripts/evaluate.py /path/to/independent-manifest.json --output data/results/events.json
```

清单格式如下，路径相对清单所在目录；同一行程的 `group_id` 不得跨 split，未标注的视频不要填写空事件冒充负例：

```json
{"videos":[{"path":"labeled-video.mp4","group_id":"trip-001","split":"test",
"label_source":"publisher temporal annotation / independent human review",
"sha256":"REPLACE_WITH_INPUT_HASH","events":[{"start":12.0,"end":17.5}]}]}
```

## 导出约定

```text
workspace/exports/v1-<id>/
  manifest.json          # 来源、许可、运行版本、文件 SHA256
  annotations.jsonl      # 秒单位 [start,end)，审核状态，原始模型输出与修订历史
  clips/<event>.mp4      # 静音 H.264，可独立播放
  evidence/<event>/*.jpg # 模型实际查看的证据帧
  README.txt
```

默认只导出已确认事件；被拒绝事件不导出。人工修改边界后，原始模型证据仍保留原时间，并注明是否在新边界内。输出原片段来自 15fps 代理，时间误差约一帧；不含原始音轨，原视频仍在本地保存。

## 目录与下一步

`risksifter/` 为应用；`scripts/` 为下载与评估；`tests/` 为测试；`examples/` 为一份真实输出；`reports/` 只保留最终实验依据。工程原则写在 [AGENTS.md](AGENTS.md)。新实验输出写入忽略的 `data/results/`；权重、视频、工作区、日志和构建产物不进 Git。实验命令只写结果，不自动导出数据集。

下一阶段按价值排序：

1. 取得有许可的 DoTA/Nexar 独立事件标注子集，完成覆盖正常/事故/险情的固定测试集和人工核验。
2. 对比专用异常 proposal 模型与当前运动/均匀基线，改善低运动风险漏检及普通交通误检。
3. 加入可校验的结构化 VLM 输出与拒判机制，处理时间关系幻觉，提升中文查询表达与检索覆盖。
4. 视觉近重复去重、跨运行事件合并、长视频断点恢复与硬件预算/批量吞吐优化。
5. 在实际 8GB Mac 和 4090 上测量端到端资源、速度、质量；再决定量化/MLX/向量检索路线。

设计依据见 [技术决策](docs/TECHNICAL_DECISIONS.md)。上述任务均未包装成已实现功能。
