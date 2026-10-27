# TWM 批处理核对

实现：gpt-6-luna / max；只读独立审查：gpt-6-sol / high。

新增 test_batch_twm.py 两项真实 tiny TWM 非零 delta head 测试通过，对照原逐帧恢复路径，覆盖缺失/部分缺失、时间大间隔、重复时间戳、segment、handedness、零历史。命令：`.env/hawor/bin/python thirdparty/ablation/twm_only/test_batch_twm.py`。

独立审查未发现P1/P2：history索引严格早于当前帧、只读取raw_world；latest_reset与逐帧before掩码一致；左侧padding由availability/mask屏蔽，TWM间隔和Mamba跳过padding；anchor仅选择有效过去状态，输出仍为[N,99]。保留原逐帧函数，CLI默认batch8。

审查同时确认dataset_api.py仅将Quest专用依赖延迟到Quest分支，无明显Aria/Quest语义变化。

本记录不代表真实10视频指标已完成；实际结果见后续pipeline日志。
