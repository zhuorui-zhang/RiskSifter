# 公开数据与许可

## 本轮实际输入

### comma2k19 官方驾驶样例

来源：[comma.ai 官方数据仓库](https://huggingface.co/datasets/commaai/comma2k19)、[原项目](https://github.com/commaai/comma2k19)、[MIT 许可](https://github.com/commaai/comma2k19/blob/master/LICENSE)。作者：Harald Schafer、Eder Santana、Andrew Haden、Riccardo Biasini；论文 *A Commute in Data: The comma2k19 Dataset* (2018), arXiv:1812.05752。

固定 revision `4bff77c7254c654c28d4c2726186b4e825adccee`，文件 `compression_challenge/b0c9d2329ad1606b|2018-07-27--06-03-57/10/video.hevc`。1200 帧、按数据集 20fps 封装为 60 秒 MP4。保留原 HEVC 哈希与封装说明；无内容合成。夜间高速驾驶，有相邻车道车辆。该样例没有事故时间真值，不能据此报告正式事故召回率。

下载：`python scripts/download_sample.py`；网络环境需要时显式设置 `HF_ENDPOINT=https://hf-mirror.com`。应用推理本身不会访问此镜像。

### TUM Traffic Accid3nD 官方展示片段

来源：[官方项目](https://accident-dataset.github.io/)，许可 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)，署名 Zimmer et al., *Towards Vision Zero: The TUM Traffic Accid3nD Dataset* (2025)。

`r09_s02_scaled.mp4` 为 119 秒路侧摄像头演示，有彩色 3D 框/文本覆盖，属于分布外、带展示标注的输入，**不是原始测试集**。保留来源、哈希与许可；它适合测试工作流和发现失败模式，不能将叠加文字当独立模型证据或用作正式评测。

媒体均下载到 gitignored 的 `data/public`，不因工作台代码使用 MIT 就改变媒体许可。导出派生 clip 保留源许可；TUM 派生媒体仍需遵守非商业及相同方式共享。默认不将公开样例、车牌、人脸或用户视频上传任何服务，也不提供隐私识别功能。Phase 1 未自动匿名化，分享真实媒体前由使用者处理隐私与授权。

### 后续取得的 TUM 原始小样本与独立标注

2026-10-15 已通过官方公开 Nextcloud 共享文件的 Range 读取和 ZIP CRC 校验，从 `train1.zip` / `train2.zip` 提取三个固定片段的原始 JPEG 与 OpenLABEL。`scripts/fetch_tum_subset.py` 可复现：`r09_s02/s03/s04`、north_50mm 相机、源时间 57–62.5s、2fps，每段 12 帧、6 秒。没有叠加检测框；这是稀疏 sample-and-hold 重建，不是原始 25fps 连续视频。

标签只用于模型提及的对象类别核对，推理过程不读取标签。第三个序列的标签跨到第二分卷；缺失时评估标为未评分，不计为负例。原始数据路径、时间戳、图像哈希、标签来源分卷在 `data/public/tum-raw/subset-manifest.json`。这不是正式事故时间定位基准，不声称三个片段统计独立，也没有将发布者的 train 分卷称为官方 test。

## 后续独立评测首选

1. **DoTA**：有异常起止帧与类型，最适合事件 tIoU/precision/recall。使用官方 train/val 分组，源长视频/行程不可跨集合；下载原始帧并按官方 10fps 重建。当前尚未取得所需视频包，不能报告该基准结果。
2. **Nexar**：[官方仓库](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction) 有正常/碰撞/险情和 event/alert 时间。2026-10-14 访问要求账号、提供联系信息并接受 Nexar Open Data License；本轮没有代替用户填写资料、接受条件或转向镜像绕过权限。事件时间与区间标签不同，评测需明确转换规则。
3. **TUM 原始帧与 OpenLABEL**：适合路侧事故，但不能将多相机同步视角随机分割。官方演示文件的时间轴不能直接套原始标签。

评估器已支持通用 JSON 事件清单及同预算 motion/uniform 对比。清单必须说明标签来源、视频哈希、group_id、split；测试结果缺失时返回空值，不将未知记作 0 或 100%。
