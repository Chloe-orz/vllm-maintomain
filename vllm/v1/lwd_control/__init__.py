"""Lwd prefill-only 控制面内核(仅 vllm 仓):控制面调度与控制面通信,数据面
经既有接缝对接。配置见 vllm/config/lwd.py(additional_config 段 + env 覆盖),
引擎类选择内联在 vllm/v1/engine/core.py。PD_mix 混排是唯一调度形态。

诊断(开关/旁路/观测/打点)收口在 lwd_control_diag 子包,功能类不散读
env、不直接 import lwd_debug。
"""
