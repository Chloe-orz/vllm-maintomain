#!/usr/bin/env bash
# Configuration-only template; edge/cloud execution is not migrated yet.
set -euo pipefail

export VLLM_HOST_IP=76.76.26.17
export GLOO_SOCKET_IFNAME=enp189s0f0
export TP_SOCKET_IFNAME=enp189s0f0
export HCCL_SOCKET_IFNAME=enp189s0f0
export ASCEND_RT_VISIBLE_DEVICES=7
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

exec vllm serve /home/extra/Qwen3.6-27B/ \
  --host 0.0.0.0 --port 8095 \
  --served-model-name qwen3.6 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 --data-parallel-size 1 \
  --trust-remote-code \
  --max-model-len 262144 \
  --max-num-seqs 230 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.95 \
  --no-enable-prefix-caching \
  --async-scheduling \
  --compilation-config '{"cudagraph_mode":"NONE"}' \
  --additional-config '{
    "enable_cpu_binding": true,
    "weight_nz_mode": 1,
    "lwd_config": {
      "path": "/home/a00954194/2e1c/topology.yaml",
      "role": "edge",
      "instance_id": 1
    },
    "lwd_coordination": {
      "enabled": true,
      "control_url": "http://76.76.26.232:9133/v1/chat/completions",
      "tenant_key_file": "/home/a00954194/edge-cloud-tenant-key",
      "consumer_id": "enterprise-a-edge1",
      "connect_timeout": 5.0
    }
  }'
