# TWM only 执行记录

日期：2026-10-20 UTC。目标：experiment-design.md B 部分第二表第一行，W-MPJPE（mm）与 Onset MAE（ms）。

## 实验定义

使用过去原始视觉状态和当前原始视觉状态。当前 candidate=raw；仅调用一次 v2 TWM；不调用视觉校正器；提交历史的是 raw，而不是 restored。历史和当前使用同一锚点转换，输出恢复到原始世界坐标。传感器/手部真值只用于离线评估。

## 当前资源检查

- `git status --short`：存在用户已有修改；本任务新增代码与记录仅写 thirdparty/ablation。
- `rg --files -g AGENTS.md`：工作目录内未发现额外 AGENTS.md；遵循用户提供的全局规则。
- 枚举 thirdparty/try/runs 内 .pt 文件，均为旧版 full_dualloop/prior/cv_prior；README 明确旧版权重不能初始化 v2。
- 使用 environment_audit.py 静态检查 checkpoint zip 的 pickle 字符串操作码，不执行反序列化；未发现 handloop_v2 或 motion_pretrain/joint_visual 标记。详细路径见 environment_audit.json。
- 原验证集 `datasets/hot3d_valset_export` 不存在。现有数据目录为 hot3d_train100 与 hot3d_train100_export，不能当作原 10 视频测试集。
- eval_log/*/evaluation_details.npz 仅包含 21 关节 XYZ 与已有指标，无法唯一恢复完整 MANO 旋转；不足以作为 v2 TWM 输入。
- 冻结 pinch 协议保存在 diagnostics/pinch_hawor_vs_ema_10videos_20261018_final/protocol_frozen.json；预测阈值 0.275、参考阈值 0.2、连续 3 帧、匹配容差 0.2s。避免在测试集重新选阈值。
- nvidia-smi：4 张 A10，另有用户任务运行；未停止或修改任何已有任务。

## 工具与审批记录

默认 sandbox 的只读 shell 报 bwrap RTM_NEWADDR 权限错误；后续使用经自动审批的 escalated 命令。
自动审批拒绝了 torch.load(weights_only=False) 批量读取旧文件，原因是通用 pickle 可能执行任意代码。已转为 weights_only=True（旧文件因自定义对象未通过安全加载）和不执行对象的静态操作码审计；没有绕过该拒绝。

## 正式结果状态

未运行正式 v2 TWM 测试集实验，未填写 experiment-design.md 中的数值。需要兼容的已训练 v2 TWM 权重，以及原验证集的图像/观测、相机、时间戳与离线评估真值，或经过验证的等价完整状态缓存。随机模型、旧版模型或训练集替代测试集得到的数字不能填入该行。

实现验证与运行命令见同目录测试日志和 ../README.md。测试中的小模型和合成数据只检查流程正确性，不构成论文实验结果。

## 主代理集成检查

执行命令（缺失权重参数为明确的占位路径，不是实验权重）：

```bash
.env/hawor/bin/python thirdparty/ablation/twm_only/run_pipeline.py \
  --checkpoint thirdparty/ablation/missing_handloop_v2.pt \
  --run_dir thirdparty/ablation/twm_only/runs/resource_check_20261020 \
  --device cpu
```

结果：退出码 2，符合缺资源预检行为。检查确认 HaWoR 和 MANO 资源存在，v2 checkpoint、验证集和 val.json 缺失。完整结构化结果：`../twm_only/runs/resource_check_20261020/blocked_status.json`；完整 stdout/stderr：`pipeline_resource_check.log`。未启动正式推理、未生成表格行。

最终代码补充输入/CUDA预检后，主代理使用相同命令、独立目录 `twm_only/runs/resource_check_final_20261020` 再检查一次，退出码仍为 2，缺资源记录正常。日志见 `pipeline_resource_check_final.log`。未重复实现者成功的小样本测试。

实现者报告最终验证：`.env/hawor/bin/python -m unittest thirdparty/ablation/twm_only/test_twm_only.py`，8 项测试全部通过，包含3个入口的 --help 检查。测试摘要见 tests.md。独立审查结论见 review.md。

## 最终完成（2026-10-20）

此前的资源缺失已解决：v2 TWM完成5000步motion预训练；从本地原始Aria数据恢复冻结10段验证集，共38,582帧，10段mask与43个SLAM区间完成。全量推理、W-MPJPE和Onset评估于14:31:49 UTC完成。评估34,504帧，W-MPJPE=37.375274658203125mm；36个匹配事件的Onset MAE=69.47025088888746ms，参考事件91个。B部分第一行已填写37.38mm/69.47ms；后两行保持未执行。

完整结果及解释见 ../RESULTS.md；精确机器结果见 ../twm_only/runs/restore_full_evaluation_20261020/table_row.json。原environment_audit.json及blocked_status是恢复前的历史记录，不代表当前资源状态。
