# Serving Frameworks for Kimi K2.5 on GKE (Blackwell Edition)

This document captures the architectural decisions regarding the selection of an LLM serving framework for the Kimi K2.5 (1T MoE) model on Google Kubernetes Engine (GKE), utilizing the **NVIDIA Blackwell (G4)** architecture for a **prefix-heavy, coding-centric workload**.

## Executive Summary
For deploying a massive 1-Trillion parameter MoE model, the **GCP G4 series (RTX PRO 6000 Blackwell)** provides a generational leap. With **96GB of VRAM per GPU**, a single 8-GPU node provides **768GB of aggregate VRAM**, allowing the entire 1T model (quantized via NVFP4) to reside on a **single node**. This eliminates inter-node networking bottlenecks and allows for ultra-fast **Tensor Parallelism (TP=8)**.

The strategy relies on **Blackwell-Native NVFP4** and **Prefix Caching** to maximize throughput.

## Framework Comparisons

### 1. vLLM (Selected for Blackwell-Native NVFP4)
*   **Status:** Primary "Well-Lit" Path on GKE.
*   **Blackwell Optimizations:**
    *   **Native NVFP4 Support:** vLLM leverages the 5th Gen Tensor Cores to execute 4-bit Floating Point (FP4) math natively. This provides up to **2x the throughput** of standard INT4/FP8 while maintaining higher accuracy for the 1T MoE model.
    *   **Single-Node Tensor Parallelism (TP=8):** Because the entire model fits in 768GB VRAM, we use TP=8 within the node. This utilizes the high-bandwidth internal interconnect rather than slower multi-node PCIe/Ethernet hops.
    *   **Automatic Prefix Caching (APC):** Essential for coding. Caches KV states of large codebases. Subsequent requests skip the "Prefill" phase, resulting in near-instant TTFT.
    *   **Deep llm-d Integration:** vLLM provides the most mature metrics for GKE's `llm-d` to perform AI-aware routing and autoscaling.

### 2. SGLang (High-Performance Alternative)
*   **Status:** Fully supported on GKE; excellent for complex prefix branching.
*   **Key Strengths:**
    *   **RadixAttention:** A superior intra-node KV cache management structure. It is even more aggressive than vLLM's APC at deduplicating overlapping prefixes in VRAM.
    *   **Structured Output:** Optimized for strict JSON/grammar-constrained decoding, common in agentic coding tasks.
*   **Trade-offs:** While SGLang supports Blackwell, the integration with `llm-d` for cluster-wide cache-state awareness is still maturing compared to the vLLM path.

### 3. TensorRT-LLM
*   **Status:** Ultimate theoretical performance, but rejected for Kubernetes flexibility.
*   **Trade-offs:** Requires complex AOT compilation tied to static hardware topologies. vLLM's JIT approach is preferred for the dynamic scaling requirements of GKE.

---

## GKE-Native Optimizations & Compatibility

### GKE Inference Gateway (Prefix-Aware Routing)
*   **Mechanism:** Hashes prompt prefixes (e.g., the first 2048 tokens of a codebase) and routes identical hashes to the same Blackwell node.
*   **Synergy:** Ensures that the **96GB of VRAM** per card acts as a massive "hot cache" for specific codebases, maximizing the hit rate of vLLM's APC or SGLang's RadixAttention.

### llm-d (LLM-Distributed Daemon)
*   **Function:** Orchestrates routing based on real-time model state (e.g., KV cache fullness).
*   **Blackwell Context:** On G4 nodes, `llm-d` monitors the 768GB VRAM pool to ensure requests are distributed to nodes with available "Reasoning/Thinking" capacity.

### Disaggregated Serving (KV Cache Streaming) vs. RadixAttention

1.  **RadixAttention (Local Memory Manager):**
    *   Manages the 768GB VRAM of a **single Blackwell node** to ensure no duplicate prefixes are stored.
2.  **llm-d Disaggregated Serving (Network Streaming):**
    *   Streams KV caches between "Prefill" nodes and "Decode" nodes.
    *   **Relevance to Blackwell:** Because a single Blackwell node can handle both prefill and decode for a 1T model efficiently, disaggregated serving is less critical here than it was on the older 48GB Ada architecture. Local computation is often faster than streaming gigabytes of cache over the VPC network.

## Conclusion & Recommendation

For the Kimi K2.5 coding workload on **NVIDIA Blackwell (G4)** hardware:
1.  **Primary Strategy:** Use **vLLM** with **TP=8** and **NVFP4** quantization. This leverages the 5th Gen Tensor Cores for maximum throughput on a single node.
2.  **Routing:** Enable **GKE Inference Gateway (Prefix-Aware Routing)** to maintain cache affinity for large codebases.
3.  **Scaling:** Use `llm-d` with HPA to scale horizontally based on the high-fidelity metrics exported by vLLM.
