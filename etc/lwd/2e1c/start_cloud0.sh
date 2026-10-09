#!/usr/bin/env bash
# Configuration-only template; edge/cloud execution is not migrated yet.
# No cloud script was supplied in the Desktop example. These are placeholders.
set -euo pipefail

export VLLM_HOST_IP=76.76.26.232
export GLOO_SOCKET_IFNAME=REPLACE_WITH_CLOUD_NETWORK_INTERFACE
export TP_SOCKET_IFNAME=REPLACE_WITH_CLOUD_NETWORK_INTERFACE
export HCCL_SOCKET_IFNAME=REPLACE_WITH_CLOUD_NETWORK_INTERFACE
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export VLLM_LOGGING_LEVEL=INFO

# LWD control endpoints are recorded in the topology YAML.
unset MASTER_ADDR MASTER_PORT
unset VLLM_ASCEND_LWD_PRE_OUT_HOST VLLM_ASCEND_LWD_PRE_OUT_PORT
unset VLLM_ASCEND_LWD_POST_OUT_BIND VLLM_ASCEND_LWD_POST_OUT_PORT

# Model/batch/graph values below illustrate the interface, not a tuned cloud run.
exec vllm serve /path/to/Qwen3.6-27B \
  --headless \
  --served-model-name qwen3.6 \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 --data-parallel-size 1 \
  --trust-remote-code \
  --max-model-len 262144 \
  --max-num-seqs 230 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.95 \
  --enable-prefix-caching \
  --async-scheduling \
  --compilation-config '{"cudagraph_mode":"NONE"}' \
  --additional-config '{
    "enable_cpu_binding": true,
    "weight_nz_mode": 1,
    "lwd_config": {
      "path": "/home/a00954194/2e1c/topology.yaml",
      "role": "cloud",
      "instance_id": 0
    },
    "lwd_coordination": {
      "enabled": true,
      "listen_host": "0.0.0.0",
      "listen_port": 9133,
      "instance_id": "cloud-0"
    }
  }'
