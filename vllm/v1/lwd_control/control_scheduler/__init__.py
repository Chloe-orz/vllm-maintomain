# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""控制面调度/执行层:边/云调度器与配置支撑。

依赖方向:本包单向依赖 control_communication;继承例外 1 个
(lwd_edge_scheduler 继承本包 LwdBaseScheduler,即 AsyncScheduler
子类;云侧唯一形态 LwdCloudScheduler 亦然);内核不继承
EngineCore,引擎触达收在两侧引擎子类(lwd_edge_engine /
lwd_cloud_engine)。
"""
