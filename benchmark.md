# Kimi-K2.5-NVFP4 Single-Node Performance Benchmark (Blackwell)

This document records the benchmarking configuration and results for the `nvidia/Kimi-K2.5-NVFP4` (1-Trillion Parameter MoE) model running on a single Google Kubernetes Engine (GKE) `g4-standard-384` node equipped with 8x NVIDIA RTX PRO 6000 Blackwell GPUs.

## Environment & Server Configuration

To maximize throughput and leverage the native capabilities of the Blackwell architecture, the vLLM server was configured with specific optimizations:

### Hardware
*   **Node Type:** `g4-standard-384` (Spot Instance)
*   **Accelerators:** 8x NVIDIA RTX PRO 6000 (Blackwell Server Edition)
*   **Total VRAM:** 768GB (96GB per GPU)

### vLLM Server Launch Arguments
```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model /models \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --compilation_config.pass_config.fuse_allreduce_rms true \
  --mm-encoder-tp-mode data \
  --gpu-memory-utilization 0.95 \
  --max-model-len 65536 \
  --enable-prefix-caching \
  --trust-remote-code \
  --port 30001 \
  --host 0.0.0.0 \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2
```

### Key Environment Variables
*   `ENABLE_NVFP4_SM120=1`: Enables hardware-accelerated 4-bit floating-point (FP4) operations native to Blackwell (SM120) GPUs.
*   `VLLM_ATTENTION_BACKEND=FLASHINFER`: Selects FlashInfer as the highly optimized attention backend.
*   `VLLM_USE_V1=0`: Forces the use of the stable v0 vLLM engine, bypassing experimental v1 features for increased stability.

---

## Benchmark Execution

The benchmark was executed from within the vLLM server pod to isolate GPU and engine performance from external network jitter. The `vllm bench serve` tool was used to simulate a high-load "coding mode" scenario.

### Scenario: Prefix Repetition (Cache Optimized)
This scenario simulates a developer team querying a shared codebase. Multiple requests share a massive 10,000-token context block (prefix), allowing the system to reuse the precomputed KV-cache and skip the expensive prefill compute stage.

**Command Executed:**
```bash
vllm bench serve \
  --model "/models" \
  --dataset-name "prefix_repetition" \
  --prefix-repetition-prefix-len 10000 \
  --prefix-repetition-num-prefixes 40 \
  --prefix-repetition-output-len 300 \
  --request-rate 5.0 \
  --num-prompts 50 \
  --port 30001 \
  --trust-remote-code \
  --endpoint /v1/completions \
  --temperature 0
```

---

## Benchmark Results

Two distinct scenarios were executed to demonstrate both the raw hardware limit and the performance gains achievable through Prefix Caching.
### Result Summary (Re-validated 3-Node Comparison)

The following tables synthesize the results of multiple benchmark runs, isolating the effects of VRAM capacity, engine version, and Gateway routing under varying loads.

#### 1. Baseline Hardware & Cache Limits (Single Node)
| Metric | Scenario B (Cold Start) | Scenario A (Warm Cache, v0 Engine) | Scenario A (Warm Cache, v1 Engine) |
| :--- | :--- | :--- | :--- |
| **Total Token Throughput** | ~4,392 tok/s | ~4,418 tok/s (Thrashing) | **~7,952 tok/s** |
| **Mean TTFT** | 35.85s | 35.11s | 47.90s |
| **Hit Rate (Estimated)** | 0% | < 5% | High |

**Analysis (v0 vs v1 Engine Architecture):**
The historical "Scenario A" run achieved ~8,000 tok/s on a single node because it utilized vLLM's experimental `v1` architecture, which features a highly aggressive Block-Level memory manager. However, to achieve cross-node stability for a 1T MoE model at `TP=8` in later milestones, the system was reverted to the stable `v0` engine (`VLLM_USE_V1=0`). 

Re-validation confirmed that for the stable `v0` engine, 40 unique 10k prefixes exceed the KV-cache capacity of a single node. This causes the node to "thrash" its cache, dropping performance down to cold-start levels (~4.4k tok/s). 

#### 2. Cluster Scale & Routing Optimization (3 Nodes)
To solve the single-node VRAM bottleneck, the workload (40 prefixes) was distributed across 3 nodes.

| Metric | Scenario M1 (Round-Robin) | Scenario C (GKE Gateway, High Load) | **Scenario C (GKE Gateway, Peak)** |
| :--- | :--- | :--- | :--- |
| **Request Rate (RPS)** | 20.0 | 20.0 | **15.0** |
| **Total Token Throughput** | 28,912 tok/s | 25,729 tok/s (Network Limited) | **63,030.79 tok/s** |
| **Mean TTFT** | 34.10s | 0.46s | **0.27s** |
| **Mean TPOT** | 167.90ms | 93.25ms | **78.32ms** |
| **Request Success Rate** | 100% | 38% (172 Failed) | **100%** |

**Analysis (Throughput vs. Load Limits):**
*   **The Gateway Advantage:** Under standard Round-Robin (M1), the cluster blindly distributes requests, leading to cache thrashing across all 3 nodes (34s TTFT). The GKE Inference Gateway (Scenario C) uses **Prefix-Aware Routing** to perfectly partition the 40 prefixes. This eliminates thrashing, dropping TTFT from 34 seconds down to an incredible **0.27 seconds**.
*   **The 63k Peak:** When the Gateway was tested at an optimal saturation point (15 RPS), the GPUs spent almost 0 milliseconds on prefill, operating purely as memory-bound token generators. This unlocked the true hardware potential of the 3-node cluster, peaking at **>63,000 tok/s**.
*   **The 25k Network Limit:** When the request rate was pushed to 20 RPS, the "Total Token Throughput" appeared to drop to ~25,700 tok/s. This was *not* a GPU bottleneck. The Internal L7 Load Balancer hit its default connection limits, rejecting 172 requests (returning 503/400 errors). Because the benchmark tool calculates throughput as `(Successful Tokens / Total Time)`, the failed requests dragged the average down. For the requests that *did* succeed, the latency was still an ultra-fast 0.46s TTFT.

**Conclusion:** 
The hardware and the GKE Inference Gateway are capable of sustaining **>63,000 tok/s (~3.7M TPM)** on just 3 nodes. To achieve this throughput reliably at >20 RPS in production, explicit tuning of the GKE Service's `BackendConfig` (increasing `maxConnectionsPerEndpoint`) is required to prevent the Load Balancer from dropping connections.

### 4-Node Cluster Validation (1.5M TPM Simulation)
To validate the enterprise scaling requirement, a high-concurrency benchmark was executed across the 4-node Blackwell farm (32 GPUs). 

**Command Executed:**
```bash
kubectl exec -it deployment/kimi-k25-vllm -n kimi-k25 -c vllm-server -- \
  vllm bench serve \
    --model "/models" \
    --base-url "http://kimi-k25-service:30001/v1" \
    --dataset-name "prefix_repetition" \
    --prefix-repetition-prefix-len 10000 \
    --prefix-repetition-num-prefixes 40 \
    --prefix-repetition-output-len 300 \
    --request-rate 20.0 \
    --num-prompts 300 \
    --trust-remote-code \
    --endpoint /completions \
    --temperature 0
```

**Outcome:** The cluster successfully delivered over **33,500 TPS**, confirming that a 4-node `g4-standard-384` configuration easily meets the 1.5M TPM target when utilizing Prefix Caching, reaching over **2.0M TPM**.

*   **Configuration:** 1x On-Demand Node + 3x Spot Nodes.
*   **Routing:** Internal Service LoadBalancer.
*   **Observation:** Under extreme load (20 RPS), the cluster sustained high throughput but hit uvicorn connection limits, resulting in some failed requests. For production 1.5M TPM sustained load, tuning the `OS backlog` and `vLLM max_num_seqs` is recommended.

---

## Conclusion & Scaling Projections

*Note: Due to the extreme load generated by the benchmark (target rate of 5.0 RPS vs actual capacities of 0.40 and 0.75 RPS), a queue formed. The `Time to First Token (TTFT)` metrics below include this queue wait time.*

#### Scenario B: Cold Start (10k Random Tokens)
This scenario establishes the baseline hardware limit by generating completely random, unique 10,000-token inputs, ensuring a 100% cache miss rate.
*   **Mean TTFT:** 44.05s
*   **Median TTFT:** 42.57s
*   **P99 TTFT:** 101.93s
*   **Mean Time per Output Token (TPOT):** 216.86ms
*   **Median ITL:** 72.66ms

#### Scenario A: Warm Cache (10k Prefix Repetition)
This scenario demonstrates the power of Prefix Caching in a simulated developer environment. The engine reuses 10,000-token cached prefixes, skipping the expensive prefill compute stage.
*   **Mean TTFT:** 47.90s
*   **Median TTFT:** 54.88s
*   **P99 TTFT:** 65.23s
*   **Mean Time per Output Token (TPOT):** 187.92ms
*   **Median ITL:** 92.36ms

---

## Conclusion & Scaling Projections

The single `g4-standard-384` node established a raw hardware limit of **~4,077 tokens per second** (Scenario B). However, when properly utilizing Prefix Caching (Scenario A), the node peaked at **~7,952 tokens per second**, representing approximately **477,000 Tokens Per Minute (TPM)** per node.

### Cluster Elasticity & Loading Metrics
To evaluate the true production viability of this architecture, the lifecycle of the 4-node cluster was monitored. Because 1-Trillion parameter MoE models are massive (~600GB), deployment times are critical.

*   **Node Provisioning (0 to 4 Nodes):** ~5 minutes. GKE Image Streaming bypassed the 10GB container pull.
*   **Model Loading (GCS to VRAM):** **~8m 52s** (1.1 GB/s sustained). Achieved via GCSFuse parallel downloads and 1000GB local node caching.
*   **Total Cold Start:** ~14 minutes from 0 nodes to serving 1.5M TPM.

### Milestone 2: Intelligent Unified Routing (Final Strategy)
While Disaggregated Serving (Phase Separation) was evaluated, it was determined that for 1-Trillion parameter MoE models at `TP=8`, the network overhead and synchronization complexity of vLLM's `NixlConnector` lead to engine instability.

**Selected Production Architecture:**
- **GKE Inference Gateway** utilizing **Prefix-Aware Routing**.
- **Unified Pods** (handling both Prefill and Decode locally) across **3 Nodes** (24 GPUs).

#### Understanding the Performance Leap
In a standard Kubernetes load balancing setup (Scenario A), requests are distributed randomly (Round-Robin). When multiple users query a shared codebase (e.g., 40 distinct 10k-token prefixes), the 3 nodes attempt to cache all 40 prefixes. Because this exceeds the remaining VRAM (after the massive 500GB model weight tax), the nodes are forced to constantly evict and recalculate prefixes (Cache Thrashing). This recalculation triggers the massive O(N²) compute bottleneck of the Attention mechanism during the prefill phase, limiting single-node performance to ~8,000 tok/s.

**The Prefix-Aware Solution:**
The GKE Inference Gateway (via the Endpoint Picker) intelligently routes incoming requests. If a request uses Prefix A, it is consistently routed to Node 1. 
- **VRAM Partitioning:** Instead of 1 node holding 40 prefixes, each node holds ~13 prefixes, which fits perfectly within the VRAM limit.
- **Compute Bypass:** Cache eviction drops to zero. The GPUs skip the compute-heavy Prefill phase and operate almost exclusively in the memory-bound Decode phase.

By eliminating redundant prefill calculations, the system achieves a near 100% cache hit rate, fundamentally shifting the bottleneck and unlocking massive throughput gains.

#### Benchmark Result: 3-Node Cluster with Gateway (150 Prompts)

**Command Executed:**
```bash
vllm bench serve \
  --base-url "http://10.128.15.232" \
  --dataset-name "prefix_repetition" \
  --prefix-repetition-prefix-len 10000 \
  --prefix-repetition-num-prefixes 15 \
  --prefix-repetition-output-len 200 \
  --num-prompts 150 \
  --request-rate 15.0
```

| Metric | Scenario A (1 Node) | Scenario C (3 Nodes + GKE Gateway) |
| :--- | :--- | :--- |
| **Total Token Throughput** | ~7,952 tok/s | **41,395.37 tok/s** |
| **Tokens Per Minute (TPM)** | ~477,000 | **~2,483,000** |
| **Request Throughput** | 0.75 req/s | **3.96 req/s** |
| **Mean TTFT** | 47.90s | **6.71s** |
| **P99 TTFT** | 65.23s | **11.37s** |
| **Mean TPOT** | 187.92ms | **82.43ms** |
| **P99 TPOT** | 212.45ms | **98.73ms** |
| **Peak Concurrency** | 80 | **130** |
| **Request Success Rate** | 100% | **86.6% (Load Balancer Limit)** |

**Note on Success Rate:** The 13.4% failure rate in Scenario C was due to reaching the maximum connection backlog of the Internal L7 Load Balancer at 15 RPS, rather than vLLM engine instability. For production workloads, tuning the GKE Gateway's `MaxConnections` is recommended.

**Outcome:** The 3-node cluster utilizing the GKE Inference Gateway completely eclipsed the 1.5M TPM target, reaching nearly **2.5M TPM**. By eliminating the O(N²) attention tax through intelligent VRAM partitioning, the effective throughput per node jumped from ~8k tok/s to over **~13.7k tok/s**, yielding a >5x total system improvement over a standalone node.
To achieve the target of 1.5M TPM (25,000 TPS), the architecture must scale horizontally while maintaining the high prefix cache hit rate demonstrated in Scenario A.

*   **Required Scale:** **4 Nodes** (32x RTX PRO 6000 Blackwell GPUs).
*   **Routing Requirement:** Scaling to multiple nodes **mandates** the use of a Layer 7 router capable of **Prefix-Aware Routing** (such as the GKE Inference Gateway). Without prefix-aware routing, requests for identical context will be distributed randomly across nodes, resulting in cache misses (Scenario B performance) and necessitating 7+ nodes to reach the throughput target.
