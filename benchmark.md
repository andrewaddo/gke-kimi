# Kimi-K2.5-NVFP4 Performance Benchmark (Blackwell)

This document records the benchmarking configuration and results for the `nvidia/Kimi-K2.5-NVFP4` (1-Trillion Parameter MoE) model on GKE `g4-standard-384` nodes (8x NVIDIA RTX PRO 6000 Blackwell GPUs).

## Environment & Server Configuration

### Hardware
*   **Node Type:** `g4-standard-384`
*   **Accelerators:** 8x NVIDIA RTX PRO 6000 (Blackwell)
*   **Total VRAM:** 768GB (96GB per GPU)

### vLLM Server Launch Arguments
```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model /models \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --mm-encoder-tp-mode data \
  --gpu-memory-utilization 0.95 \
  --max-model-len 65536 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --trust-remote-code
```

---

## Result Summary (Comparative Analysis)

The following tables synthesize multiple benchmark runs, isolating the effects of engine version, load, and routing.

### 1. Single Node Baseline (Cache & Engine Limits)
| Metric | Scenario B (Cold Start) | Scenario A (Warm, v0 Engine) | Scenario A (Warm, v1 Engine) |
| :--- | :--- | :--- | :--- |
| **Total Throughput** | ~4,392 tok/s | ~4,418 tok/s (Thrashing) | **~7,952 tok/s** |
| **Mean TTFT** | 35.85s | 35.11s | 47.90s |
| **Engine Version** | Stable (v0) | Stable (v0) | Experimental (v1) |

**Analysis (v0 vs v1 Architecture):**
The historical ~8k tok/s run utilized vLLM's experimental `v1` engine, which has an aggressive Block-Level memory manager. The stable `v0` engine (`VLLM_USE_V1=0`) is used for production stability but experiences "Cache Thrashing" on a single node when unique prefixes exceed VRAM.

### 2. Cluster Scale & Gateway Optimization (3 Nodes)
| Metric | Scenario M1 (Round-Robin) | Scenario C (Gateway, Peak) |
| :--- | :--- | :--- |
| **Total Throughput** | 28,912 tok/s | **63,030 tok/s** |
| **Mean TTFT** | 34.10s | **0.27s** |
| **Mean TPOT** | 167.90ms | **78.32ms** |
| **Hit Rate** | ~15% | **~98%** |

---

## Technical Deep Dive: The Gateway Advantage

### The Problem: Cache Thrashing
Without a gateway, nodes use Round-Robin routing. All nodes attempt to cache all incoming prefixes. For a 1T model, the 500GB weight tax leaves only ~200GB for KV-cache. When unique prefixes exceed this limit, nodes constantly evict and recalculate the $O(N^2)$ attention matrices, dropping performance to cold-start levels.

### The Solution: Prefix-Aware Routing
The **GKE Inference Gateway** partitions the prefix workspace. By ensuring specific prefixes always hit the same node:
1.  **VRAM Partitioning:** Each node holds a stable subset of prefixes that fits in memory.
2.  **Compute Bypass:** Nodes skip the compute-heavy Prefill phase and operate in the memory-bound Decode phase.
3.  **Result:** TTFT drops from **34s** to **0.27s**, and throughput doubles from **28k** to **63k** tok/s.

---

## Conclusion & Scaling Projections

*   **Target (1.5M TPM):** Achieved and exceeded with just 3 nodes (~3.7M TPM peak).
*   **Required Infrastructure:** Horizontal scaling **mandates** a Layer 7 router like the GKE Inference Gateway to maintain prefix affinity.
*   **Production Note:** To sustain >20 RPS, tune the GKE `BackendConfig` to increase `maxConnectionsPerEndpoint`.
