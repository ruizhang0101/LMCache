# Tutorial: Running DRRIP Cache Policy Example

## Prerequisites

- GPU machine with at least 20GB HBM and at least 10GB CPU RAM
- CUDA and nvidia driver installed

## Environment Setup

Follow the tutorials for building LMCache from Source.

```bash
git clone https://github.com/zerofishnoodles/LMCache.git
cd LMCache
git checkout feat/drrip


uv venv --python 3.12
source .venv/bin/activate

# Need to install these packages manually to avoid build isolation
uv pip install -r requirements/build.txt

uv pip install vllm==0.11.0

# no build isolation requires torch to already be installed
# with your desired version
uv pip install -e . --no-build-isolation
```

## Running the Example Experiment

1. Navigate to the cache policy examples directory:

```bash
cd examples/cache_policy
```

2. Configuration Explanation

The `example.yaml` configuration file sets up the DRRIP cache policy with the parameters described above. This configuration enables the DRRIP policy for managing cache eviction, which dynamically adapts between SRRIP (Set Dueling Re-Reference Interval Prediction) and BRRIP (Bimodal Re-Reference Interval Prediction) based on workload characteristics.


- `drrip_max_rrpv` (default: 3)
Maximum Re-Reference Prediction Value. This determines the maximum age a cache entry can reach before being evicted. Higher values allow entries to stay longer before eviction.

- `drrip_brrip_short_insert_prob` (default: 32)
Probability denominator for BRRIP short insert. This controls how often BRRIP inserts entries with a shorter RRPV (1/N chance). Lower values mean more frequent short inserts.

- `drrip_psel_max` (default: 1023)
Maximum value for the policy selector counter. This controls how quickly DRRIP can switch between SRRIP and BRRIP policies. Higher values make the policy more stable but slower to adapt.
- `drrip_leader_set_mask` (default: 0x1F)
Bit mask for leader set selection. This determines which cache entries are used as "leaders" for policy selection. The default 0x1F (31 in decimal) means approximately 1/32 of entries are leaders.

Example Configuration:

```yaml
cache_policy: "DRRIP"
drrip_max_rrpv: 3
drrip_brrip_short_insert_prob: 32
drrip_psel_max: 1023
drrip_leader_set_mask: 0x1F
```

3. Run the vLLM server with LMCache:

```bash
PROMETHEUS_MULTIPROC_DIR=/tmp/lmcache_prometheus \
LMCACHE_LOG_LEVEL=DEBUG \
PYTHONHASHSEED=123 \
VLLM_USE_V1=1 \
LMCACHE_USE_EXPERIMENTAL=True \
LMCACHE_CONFIG_FILE=example.yaml \
vllm serve Qwen/Qwen3-8B \
    --port 30080 \
    --gpu-memory-utilization 0.8 \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

**Command Explanation:**

- `PROMETHEUS_MULTIPROC_DIR`: Directory for Prometheus multiprocess metrics
- `LMCACHE_LOG_LEVEL=DEBUG`: Enable debug logging for LMCache
- `PYTHONHASHSEED=123`: Set random seed for reproducibility
- `VLLM_USE_V1=1`: Use vLLM v1 API
- `LMCACHE_USE_EXPERIMENTAL=True`: Enable experimental LMCache features
- `LMCACHE_CONFIG_FILE=example.yaml`: Path to the DRRIP configuration file
- `--gpu-memory-utilization 0.8`: Use 80% of available GPU memory
- `--kv-transfer-config`: Configure LMCache connector for KV cache management

4. Run example benchmark

```bash
cd benchmarks/long_doc_qa
python3 long_doc_qa.py --num-documents 20  --document-length 4000   --model Qwen/Qwen3-8B       --max-inflight-requests 1 --port 30080 --repeat-count 1 --sleep-time-after-warmup 10
```

## Collecting Metrics

After the server is running, you can collect metrics using:

```bash
curl -s http://localhost:30080/metrics | grep -E 'vllm:time_to_first_token_seconds_sum|vllm:time_to_first_token_seconds_count|lmcache:num_hit_tokens_total|lmcache:num_prompt_tokens_total'
```

## Calculating Performance Metrics

**Time to First Token (TTFT):**

```
time_to_first_token = vllm:time_to_first_token_seconds_sum / vllm:time_to_first_token_seconds_count
```

**Hit Rate:**

```bash
hit_rate = lmcache:num_hit_tokens_total / lmcache:num_prompt_tokens_total
```

**Example Output**

```bash
$ curl -s http://localhost:30080/metrics | grep -E 'vllm:time_to_first_token_seconds_sum|vllm:time_to_first_token_seconds_count|lmcache:num_hit_tokens_total|lmcache:num_prompt_tokens_total'

vllm:time_to_first_token_seconds_sum{engine="0",model_name="Qwen/Qwen3-8B"} 2.971604108810425
vllm:time_to_first_token_seconds_count{engine="0",model_name="Qwen/Qwen3-8B"} 45.0
# HELP lmcache:num_hit_tokens_total Total number of tokens hit in lmcache
# TYPE lmcache:num_hit_tokens_total counter
lmcache:num_hit_tokens_total{model_name="Qwen/Qwen3-8B",worker_id="0"} 1529.0
# HELP lmcache:num_prompt_tokens_total Number of prompt tokens in lmcache
# TYPE lmcache:num_prompt_tokens_total counter
lmcache:num_prompt_tokens_total{model_name="Qwen/Qwen3-8B",worker_id="0"} 36089.0
```

**Calculated Metrics:**

- `ttft = 2.971604108810425 / 45.0 = 0.0660 seconds`
- `hit_rate = 1529.0 / 36089.0 = 0.0424 (4.24%)`
