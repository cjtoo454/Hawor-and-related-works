# B第二行：+ Intra-frame Feedback

本次只新增帧内反馈，使用第一行的相同基础HaWoR和5000步TWM权重，固定基础模型。需要先训练尚未训练的HandLoopVisual模块；不将随机初始化模块的输出作为实验结果。

训练：保持既有受试者划分，train=P0002/P0003/P0009的33段视频，calibration=P0015的20段。prepared_train100的183,631个图像路径已检查全部存在；当前有176,397个有效参考帧、148,728个有效视觉帧。GT只进入监督损失，历史只使用原始视觉状态。使用独立校准选择visual权重，不根据10段测试结果选择checkpoint或阈值。

推理：两轮候选→同帧投影→读取IAM图像证据→视觉候选→TWM恢复。第二轮必须使用更新后的候选投影；跨帧提交仍为原始视觉状态。无corrected-state反馈，历史uncertainty标记unknown，不保存恢复uncertainty作为后续输入。

评估：沿用第一行的相同10视频、同RGB/GT投影框、同43个SLAM区间和冻结pinch协议。核对重新提取的raw state、轨迹文件/框来源及评估帧ID与第一行一致。生成W-MPJPE和Onset MAE后，才填写experiment-design.md第二行。第一行及其产物保留。

冻结约定见 records/protocol.json；代码/数据/训练及评估将全部保存本目录。runtime由gpt-6-luna/max实现，视觉独训与数据适配由gpt-6-sol/xhigh实现，主代理负责集成与真实运行。
