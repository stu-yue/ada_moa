# adamoa


1. 安装环境根据`requirements.txt`；

2. 部署模型，以下为三个模型的示例：
```shell
CUDA_VISIBLE_DEVICES=0 \
nohup vllm serve /root/autodl-fs/Qwen3-4B \
  --port 8001 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.40 \
  --max-model-len 4096 \
  > qwen4b.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 \
nohup vllm serve /root/autodl-fs/SmolLM3-3B \
  --port 8002 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.40 \
  --max-model-len 4096 \
  > smollm3b.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 \
nohup vllm serve /root/autodl-fs/Qwen3-8B \
  --port 8003 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.75 \
  --max-model-len 8192 \
  > qwen8b.log 2>&1 &
```

3. 根据部署的端口，修改`run_all_test.sh`中，`VLLM_MODEL_ENDPOINTS`配置信息：
```shell
export VLLM_MODEL_ENDPOINTS='{
  "Qwen/Qwen3-4B": "http://localhost:8001/v1",
  "HuggingFaceTB/SmolLM3-3B": "http://localhost:8002/v1",
  "Qwen/Qwen3-8B": "http://localhost:8003/v1"
}'
```

4. 运行`bash run_all_test.sh`，生成MoA输出结果；

5. 运行`bash eval_ceb_test.sh`，对模型输出结果进行评分（这里需要设置外部API作为评分者，主要修改`GRADER_OPENAI_BASE_URL`、`GRADER_OPENAI_API_KEY`和`GRADER_MODEL`）；


