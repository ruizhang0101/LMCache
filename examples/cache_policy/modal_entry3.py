# SPDX-License-Identifier: Apache-2.0
# This script simplifies building and running the LMCache DRRIP cache policy examples.
# Build with love by @Sergio Valderrama <sevalder@ucsc.edu>
# Standard
from typing import Any, Dict, List
import os
import subprocess
import sys
import time

# Third Party
from tabulate import tabulate  # type: ignore
import modal
import requests

# --- 1. Define the Environment ---
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install(
        "git",
        "curl",
        "wget",
        "ninja-build",
        "pkg-config",
        "python3-dev",
        "clang",
        "llvm",
    )
    .pip_install(
        "ninja",
        "packaging",
        "wheel",
        "setuptools",
        "vllm==0.11.0",
        "numpy",
        "pandas",
        "aiohttp",
        "tabulate",
    )
    .run_commands(
        "git clone https://github.com/zerofishnoodles/LMCache.git /root/LMCache",
        "cd /root/LMCache && git checkout feat/drrip",
    )
    .run_commands("cd /root/LMCache && pip install -r requirements/build.txt")
    .env({"TORCH_CUDA_ARCH_LIST": "8.0 8.6 8.9 9.0"})
    .run_commands(
        "cd /root/LMCache && pip install -e . --no-build-isolation", gpu="any"
    )
)

app = modal.App("lmcache-a10g-stress-test", image=image)


# --- 2. Config Generator ---
def get_config(policy: str) -> str:
    # CRITICAL CHANGE: Restrict CPU cache to 5GB.
    # We generate ~18GB total. GPU takes ~9GB.
    # ~9GB spills to CPU. 9GB > 5GB capacity -> Forced Evictions!
    base_config = f"""
cache_policy: "{policy}"
max_local_cpu_size: 5
"""
    if policy == "DRRIP":
        base_config += """
drrip_max_rrpv: 3
drrip_brrip_short_insert_prob: 32
drrip_psel_max: 1023
drrip_leader_set_mask: 0x1F
"""
    return base_config


# --- 3. Benchmark Script ---
BENCHMARK_SCRIPT = """
import asyncio
import aiohttp
import time
import random
import numpy as np
import argparse
import sys

sys.stdout.reconfigure(line_buffering=True)

class RoundRobinGenerator:
    def __init__(self, num_prefixes, prefix_len, vocab_size=32000):
        self.prefix_len = prefix_len
        self.vocab_size = vocab_size
        self.num_prefixes = num_prefixes
        print(
            f"⚡ Generating {num_prefixes} unique prefixes (Len: {prefix_len})...",
            flush=True,
        )
        self.prefixes = [
            np.random.randint(10, vocab_size, size=prefix_len).tolist()
            for _ in range(num_prefixes)
        ]
        self.counter = 0

    def sample_request(self):
        # Round Robin ensures we loop back to check if old items were evicted
        idx = self.counter % self.num_prefixes
        self.counter += 1
        prefix = self.prefixes[idx]
        suffix = np.random.randint(10, self.vocab_size, size=10).tolist()
        return prefix + suffix

async def run_benchmark(args):
    generator = RoundRobinGenerator(
        num_prefixes=args.num_prefixes,
        prefix_len=args.prefix_len
    )

    print(
        f"🔄 Starting Loop: {args.num_requests} reqs over {args.num_prefixes} prefixes",
        flush=True,
    )

    async with aiohttp.ClientSession() as session:
        completed = 0

        async def send_request(req_id):
            nonlocal completed
            token_ids = generator.sample_request()
            payload = {
                "model": args.model,
                "prompt": token_ids,
                "max_tokens": args.output_len,
                "temperature": 0.0,
                "stream": False,
            }
            try:
                async with session.post(
                    "http://localhost:30080/v1/completions", json=payload
                ) as resp:
                    await resp.json()
            except Exception:
                pass
            finally:
                completed += 1
                if completed % 50 == 0:
                    print(
                        f"   Progress: {completed}/{args.num_requests}",
                        flush=True,
                    )

        sem = asyncio.Semaphore(args.concurrency)

        async def bound_send(i):
            async with sem:
                await send_request(i)

        await asyncio.gather(*(bound_send(i) for i in range(args.num_requests)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 5 loops through the 80 prefixes
    parser.add_argument("--num-requests", type=int, default=400)
    parser.add_argument("--concurrency", type=int, default=16)

    # 80 prefixes * 0.22GB/prefix = 17.6GB Total
    parser.add_argument("--num-prefixes", type=int, default=80)
    parser.add_argument("--prefix-len", type=int, default=4096)
    parser.add_argument("--output-len", type=int, default=16)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    args = parser.parse_args()
    asyncio.run(run_benchmark(args))
"""


# --- 4. Experiment Logic ---
@app.function(gpu="A10G", timeout=3600)
def run_experiment(policy_name: str) -> Dict[str, Any]:
    os.environ["PYTHONUNBUFFERED"] = "1"

    config_path = "/root/LMCache/examples/cache_policy/example.yaml"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        f.write(get_config(policy_name))
    with open("/root/prefix_benchmark.py", "w") as f:
        f.write(BENCHMARK_SCRIPT)

    env = os.environ.copy()
    env.update(
        {
            "PROMETHEUS_MULTIPROC_DIR": "/tmp/lmcache_prometheus",
            "LMCACHE_LOG_LEVEL": "INFO",
            "PYTHONHASHSEED": "123",
            "VLLM_USE_V1": "1",
            "LMCACHE_USE_EXPERIMENTAL": "True",
            "LMCACHE_CONFIG_FILE": config_path,
        }
    )
    if os.path.exists("/tmp/lmcache_prometheus"):
        # Standard
        import shutil

        shutil.rmtree("/tmp/lmcache_prometheus")
    os.makedirs("/tmp/lmcache_prometheus", exist_ok=True)

    print(
        f"[{policy_name}] 🚀 Starting vLLM on A10G (5GB LMCache Limit)...",
        flush=True,
    )
    server_cmd = [
        "vllm",
        "serve",
        "Qwen/Qwen2.5-7B-Instruct",
        "--port",
        "30080",
        "--gpu-memory-utilization",
        "0.9",
        "--kv-transfer-config",
        '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}',
        "--max-model-len",
        "8192",
        "--enable-prefix-caching",
    ]

    server_process = subprocess.Popen(
        server_cmd,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
        bufsize=1,
    )

    server_ready = False
    for i in range(120):
        try:
            response = requests.get("http://localhost:30080/health")
            if response.status_code == 200:
                print(
                    f"[{policy_name}] ✅ Server Ready!",
                    flush=True,
                )
                server_ready = True
                break
        except Exception:
            if i % 10 == 0:
                print(
                    f"[{policy_name}] ... waiting ({i * 5}s)",
                    flush=True,
                )
            time.sleep(5)

    if not server_ready:
        server_process.kill()
        return {"policy": policy_name, "error": "Server start failed"}

    print(f"[{policy_name}] 📊 Running Benchmark...", flush=True)
    try:
        subprocess.run(
            [
                "python3",
                "/root/prefix_benchmark.py",
                "--num-requests",
                "400",
                "--num-prefixes",
                "80",
                "--prefix-len",
                "4096",
                "--concurrency",
                "16",
            ],
            check=True,
        )
    except Exception:
        server_process.terminate()
        return {"policy": policy_name, "error": "Benchmark failed"}

    print(f"[{policy_name}] 💤 Waiting 10s for metrics...", flush=True)
    time.sleep(10)

    results: Dict[str, Any] = {"policy": policy_name}
    try:
        metrics_text = requests.get("http://localhost:30080/metrics").text

        ttft_sum: float = 0.0
        ttft_count: float = 0.0
        hits: float = 0.0
        total: float = 0.0
        evictions: float = 0.0

        for line in metrics_text.splitlines():
            if line.startswith("#"):
                continue

            if "vllm:time_to_first_token_seconds_sum" in line:
                ttft_sum = float(line.split()[-1])
            if "vllm:time_to_first_token_seconds_count" in line:
                ttft_count = float(line.split()[-1])
            if "lmcache:num_hit_tokens" in line:
                hits += float(line.split()[-1])
            if "lmcache:num_prompt_tokens" in line:
                total += float(line.split()[-1])
            if "lmcache:local_cpu_evict_count" in line:
                evictions += float(line.split()[-1])
            elif "evict" in line and "count" in line:
                evictions += float(line.split()[-1])

        results["ttft_avg"] = (ttft_sum / ttft_count) if ttft_count > 0 else 0.0
        results["hit_rate"] = (hits / total) if total > 0 else 0.0
        results["evictions"] = int(evictions)

        print(
            f"[{policy_name}]",
            f"Hits: {int(hits)}",
            f"Total: {int(total)}",
            f"Evictions: {int(evictions)}",
            flush=True,
        )

    except Exception as e:
        results["error"] = str(e)

    server_process.terminate()
    return results


@app.local_entrypoint()
def main() -> None:
    print("🚀 Launching Stress Test on A10G...", flush=True)
    policies: List[str] = ["DRRIP", "LRU", "FIFO"]
    results: List[Dict[str, Any]] = list(run_experiment.map(policies))

    print("\n\n" + "=" * 60)
    print("🏆 FINAL COMPARISON RESULTS 🏆")
    print("=" * 60)

    table_data: List[List[Any]] = []
    for r in results:
        if "error" in r:
            table_data.append(
                [r.get("policy", "Unknown"), "ERROR", r["error"], "ERROR"]
            )
        else:
            table_data.append(
                [
                    str(r["policy"]),
                    f"{float(r.get('ttft_avg', 0)):.4f} s",
                    f"{float(r.get('hit_rate', 0)) * 100:.2f} %",
                    int(r.get("evictions", 0)),
                ]
            )

    print(
        tabulate(
            table_data,
            headers=["Policy", "Avg TTFT", "Hit Rate", "Evict Count"],
            tablefmt="grid",
        )
    )
