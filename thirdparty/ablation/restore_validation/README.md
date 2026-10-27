# HOT3D 10视频恢复与 TWM-only 评估记录

目标：恢复 datasets/hot3d_valset_export，使用 training_v2/checkpoints/latest.pt 计算 experiment-design.md B部分消融第一行。

原始来源：hot3d/dataset 中冻结 protocol_frozen.json 指定的10个Aria序列，recording.vrs、MANO轨迹、相机和物体信息均在本地。导出 start_frame=20，与仓库既有流程一致。

环境修复：

- hot3d/dataset_api.py 将 QuestDataProvider 导入移到 Quest 分支，避免 Aria 导出要求未使用的 pyvrs/pyvrs2。
- hot3d/mano_v1_2/models/MANO_LEFT.pkl、MANO_RIGHT.pkl 原为空文件，分别从本地 _DATA/data_left/mano_left 与 _DATA/data/mano 恢复；没有下载模型。
- run_legacy.py 在独立进程内补齐旧 chumpy 所需 NumPy 标量别名及 inspect.getargspec，不修改安装包。原始失败栈保留在 export.log、export_retry*.log。
- 为减少导出等待，将本任务自己启动的顺序导出进程 PID 3031403 在其首视频尚未写出时停止，再用 export_parallel.py 分配互斥序列，4个CPU导出进程；每序列日志 export_<sequence>.log，全部成功才写 val.json。没有终止其他用户任务。

导出入口：`.env/hawor/bin/python -u thirdparty/ablation/restore_validation/export_parallel.py`。

后续阶段和实际执行状态见 after_export.py、各阶段日志及结构化状态文件。恢复数据不得用训练集替代，正式指标不得引用随机权重或原HaWoR指标。

补充：导出过程中先按冻结协议写出完整val.json以支持已完成视频的pilot；全量完成判据仍是export_parallel.py最终成功和所有10段逐项帧数检查，不能只看val.json存在。

执行分组：pilot_first_video_retry1使用GPU1，prepare_group_gpu3使用GPU3且仅处理另3段，目录互斥。首次pilot因runner未显式添加仓库根到sys.path而失败，已修复并保留失败status。原eval_log预测通过只读symlink复用，不覆盖原eval_log.txt。

可选fast_mask.py验证了2帧rasterizer-only输出与旧alpha掩码逐像素一致（0差异），单轮约1.11x，收益有限；本次未启用该替代路径，继续用原渲染流程。

全量导出：10段共38,582帧。check_reference.py使用当前MANO几何重算GT joints，与旧eval_log各视频gt_world_joints_m比较，全部一致（最大坐标差约1.2e-7m）。详细结果 restored_reference_check.json。

调度加速：在GPU2/3已完成首视频mask后提前启动其SLAM，与后续mask重叠；run_legacy.py对SLAM按序列加排他文件锁，并将np.savez写入同目录临时文件后原子替换，防止后续分组重复写或读到半个NPZ。计算流程、分段和精度不变。GPU2原有小型旧任务保留运行，本任务新增进程未结束它。

首视频pilot：P0001_a68492d5全链路完成，3223评估帧，W-MPJPE=43.5913mm；pinch参考事件1、匹配0，Onset MAE=null。pipeline正确记录complete_diagnostic_no_formal_row，不把子集或缺失Onset当作正式行。实际日志在 twm_only/runs/restore_pilot_first_video_retry1；后续自动继续剩余视频准备。

调度修正：GPU3原分组的SLAM进程3074518实际在文件锁包装更新前已启动。发现额外overlap_gpu3进程3075916与其重复计算首区间后，已对本任务新增的3075916发送TERM；其日志尚无estimated scale，确认未写出SLAM结果。原GPU3分组继续作为该组唯一写者。GPU2 overlap开始时其分组仍在mask，后续会从新wrapper加载锁，保持互斥。此修正不删除或覆盖已完成轨迹。

最终多GPU推理调度：run_pipeline.py的可选--inference_gpus按视频互斥分组，各worker独立CUDA_VISIBLE_DEVICES和日志；全部成功后才进入统一评估。静态10/10覆盖、无重复、CLI检查通过；gpt-6-sol/high只读审查未发现P1/P2。为给最终full命令增加该flag，仅停止并重启了本任务的finish_all supervisor（3055830→3079168）；已有准备子进程3076274继续运行，未重启。

## 最终状态：完成

2026-10-20 14:31:49 UTC全量恢复与评估完成，workflow_status.json=complete。10段视频、38,582帧、10段mask、43个SLAM区间；所有43个轨迹长度、有限性和正尺度检查通过。全量评估34,504帧，W-MPJPE 37.375274658203125mm；36个匹配事件的Onset MAE 69.47025088888746ms。结果与限制见 ../RESULTS.md，精确产物位于 ../twm_only/runs/restore_full_evaluation_20261020/。
