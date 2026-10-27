# TWM only：验证集恢复与实验 B 第一行结果

完成时间：2026-10-20 14:31:49 UTC。结果已填写至 experiment-design.md 的 B 部分消融表第一行。

| Variant | W-MPJPE ↓ | Onset MAE ↓ |
| --- | --- | --- |
| TWM only | 37.38 mm | 69.47 ms |

使用10段冻结HOT3D测试视频，共38,582帧；按原有右手有效且连续不少于16帧的范围，评估34,504帧。Onset MAE是36个成功一对一匹配事件的平均绝对起始时刻误差；共91个参考事件，预测186个，TP=36、FP=150、FN=55，precision=19.35%、recall=39.56%、F1=0.25993。没有把漏检算成零时序误差。

## 实际设置

- 权重：training_v2/checkpoints/latest.pt；HandLoop v2 motion_pretrain，第5000步。SHA256：26078e9b5d587f38130d51316dad90b598d4a426dbc312f8db2360eb0d9a0ab2。
- 前端：既有HaWoR权重，未重训HaWoR。SHA256：4d1cc43853c190d6f2c10d9b6295c73109f0faf9ef41ac817a2b31d94b4823f2。
- 当前candidate=raw；历史只保存过去原始视觉状态；每个当前状态仅预测一次；不调用视觉校正器、不把恢复结果写入历史。batch=8的推理与逐帧路径经非零残差模型等价测试和独立审查。
- 实际输入框是GT投影框，oracle_assisted=true，与旧基线的框来源范围一致。该结果属于GT框辅助的离线评估，不能表述为完全独立检测的在线测试结果。
- HaWoR保留16帧双向上下文；TWM数值历史因果，视频前端整体不是严格在线。
- Pinch参考为GT几何导出的事件代理，不是真实软件日志或人工事件标注；使用已冻结预测阈值0.275、参考阈值0.2、至少3帧、0.2秒匹配容差，没有在测试集调阈值。

## 恢复与核验

原始10段Aria数据仍在hot3d/dataset，从start_frame=20重建图像、视频、MANO标注、相机参考和时间戳。恢复后参考关节与旧eval_log一致（坐标最大差约1.2e-7米）。生成10段掩码、重算43个1000帧分段的masked DROID-SLAM/Metric3D尺度；43个轨迹长度、有限性及正尺度检查全部通过。SLAM为本次重算，因此与论文已有HaWoR数字不是严格同一次相机轨迹/前端执行的配对对照。

本行W-MPJPE 37.38mm和Onset MAE 69.47ms均未优于文中既有HaWoR 34.60mm/61.24ms。由于执行链路重算，这只是已有数字的直接对照，不能据此把差异全部归因于TWM；严格增益判断需要使用本次同一前端/SLAM输入再计算HaWoR对照。未执行或填写帧内反馈和完整双反馈两行。

## 产物索引

- 全量运行：twm_only/runs/restore_full_evaluation_20261020/
- 精确结果：twm_only/runs/restore_full_evaluation_20261020/table_row.json
- 空间指标：该目录 predictions_and_wmpjpe/evaluation_summary_10videos.json
- 事件指标与匹配：该目录 onset/summary.json 及各视频 events.json
- 每视频预测/trace/来源哈希：该目录 predictions_and_wmpjpe/<sequence>/
- 三GPU分片、命令、日志与返回码：该目录 inference_workers/
- 恢复全流程：restore_validation/README.md、runs/full_evaluation_20261020/status.json
- 参考核对：restore_validation/restored_reference_check.json
- 最终SLAM核对：restore_validation/final_slam_check.json
- 训练：training_v2/README.md、train.log、launch.json、training_status.json
- 实验文档修改记录：records/experiment_design_update.json（仅替换TWM only占位行，保留原换行格式）
