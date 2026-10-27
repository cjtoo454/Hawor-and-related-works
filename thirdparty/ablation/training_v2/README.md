# HandLoop v2 TWM 数值预训练

2026-10-20 启动。只训练 v2 TrajectoryRestorer，不重训 HaWoR；使用原始 256 维、6 层、30 帧历史、4 模式/60 步未来架构。此阶段产生 motion_pretrain 权重，可被 ../twm_only/run_pipeline.py 加载；训练结束不等于消融评估已完成。

数据：从 thirdparty/try/data/raw_trajectories 转换实际手腕位置及旋转。33 段 train、20 段 calibration，无缺失源。使用 datasets/hot3d_train100_export/training_split.json 的既有划分，并从冻结 pinch 协议 test_sequences 检查 train/calibration/test 受试者互斥。数据及来源哈希位于 data/motion_manifest.json，转换命令及输出见 prepare.log 与 prepare_motion.py。没有以训练数据替代消融测试集。

运行：物理 GPU 1（CUDA_VISIBLE_DEVICES=1，因此进程内 cuda:0）；batch=8、5000 步、seed=7、每50步验证和保存；模型配置与学习率沿用原配置，快照 config.yaml。

- launch.json：实际启动参数、PID、GPU、配置哈希。
- training_status.json：running/completed/failed 生命周期；如进程被外部强杀，应同时检查 PID 与日志，不能只依赖此文件。
- train.log：训练和验证 stdout/stderr。
- progress.jsonl：第1步与每10步的loss、梯度范数、显存峰值和时间。
- checkpoints/latest.pt：每50步和最终一步原子替换保存模型、优化器、随机状态及数据游标。只保留最新保存点，不声称是最优权重。

查看进度：

```bash
tail -f thirdparty/ablation/training_v2/train.log
```

训练调用现有 train_handloop.py，train_logged.py 只增加记录，不修改优化和损失。校准沿用现有训练器的最多8个固定窗口，只是训练期间快速检查，不是完整校准集的正式评估。

最终训练状态：completed，2026-10-20 12:37:44 UTC完成5000步，最终校准loss=0.8728112578392029；latest.pt已用于全10视频TWM-only评估，结果见 ../RESULTS.md。
