# 事件与导出格式 v1.0

时间统一使用回放代理起点的秒数，区间为 `[start,end)`。源帧率可能不同，代理为 15fps；源视频 SHA256 与代理 SHA256 均保留。

| 字段 | 含义 |
|---|---|
| `id`, `video_id`, `run_id` | 稳定事件/视频/运行标识。同内容同配置可续跑，重新分析可产生独立 run。 |
| `start`, `end` | 当前审核后的区间；必须在视频时长内。 |
| `proposal` | 原始候选区间、运动峰值、非概率分数和选取原因。 |
| `description`, `query`, `tags` | 当前可编辑标注。搜索索引使用这些字段。 |
| `annotation` | 不可变的原始模型/降级输出，包括原文、prompt、模型版本、预处理与实际查看证据哈希。 |
| `evidence` | 帧时间、相对导出文件路径和 SHA256。人工改边界不替换原模型证据。 |
| `review_status` | pending / approved / rejected；任何自动输出初始均为 pending。 |
| `label_kind` | 自动伪标签，或人工确认后 human_reviewed。不能仅依据模型文字把记录升级为真值。 |
| `revision`, `history` | 乐观锁版本与人工变更前后值、时间、actor。 |
| `confidence`, `risk_level`, `ttc_seconds`, `distance_meters` | 当前没有可靠校准/几何依据，保持 null；confidence 位于 annotation 内。 |
| `clip` | 导出中的可播放静音片段；校验值统一在 manifest.files。 |

`manifest.json` 包含源信息、许可、运行配置、所有文件（manifest 本身除外）的 SHA256。`SOURCE_LICENSES.txt` 保留源媒体的许可条款/链接与改动说明。ZIP 版本名不覆写旧导出。原始视频不随 ZIP 复制。

模型文件缺失或不可读时 `annotation_error` 非空、`annotation.model.backend=none`、`viewed_evidence=[]`，提取的图片只供人工查看，不能声称模型已经看过。`generation_truncated=true` 表示达到生成长度预算，需要补充审核。

未预期的推理/程序错误标记运行失败并向调用者抛出，不转换成无模型标注。
