# 最终独立审计

审查者：gpt-6-sol / high，只读；未重复GPU运行或既有测试。结论：通过，无阻止填写B表第一行的P1/P2。

- 10视频worker互斥，退出码全0；与冻结protocol、空间与事件来源集合一致。
- 38,582/38,582帧已处理；34,504右手评估帧。预测valid 34,665帧，评估连续chunk筛选后为34,504帧。
- 每视频均为同一checkpoint SHA 26078e9b5d587f38130d51316dad90b598d4a426dbc312f8db2360eb0d9a0ab2，motion_pretrain/global_step=5000；frontend SHA一致；GT框/oracle标记均true。
- 实际prediction/details/metadata文件SHA与manifest/onset sources一致。
- 拼接空间NPZ复算W-MPJPE=37.375274658203125mm。
- 36个匹配事件复算Onset MAE=69.47025088888746ms。
- 以上复算与summary/table_row精确一致。

边界：GT框辅助、GT几何代理事件；SLAM重算不等于旧基线完全同输入；未执行帧内反馈或完整双反馈两行。详见../RESULTS.md。
