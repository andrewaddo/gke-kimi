# Optimizing Kimi K2.5 Inference on GKE with G4 (Blackwell)

## Overview
This document outlines the strategy for deploying and optimizing the inference of the Kimi K2.5 (1-trillion parameter MoE) model on Google Kubernetes Engine (GKE) utilizing **NVIDIA G4 machine types**.

## Hardware Architecture: NVIDIA Blackwell (G4)
Contrary to previous workstation-class RTX cards, the **GCP G4 series** features the **NVIDIA RTX PRO 6000 Blackwell Server Edition**. This hardware represents a generational leap in inference efficiency.

### Key Hardware Advantages:
*   **Architecture:** 5th Generation Tensor Cores (Blackwell).
*   **VRAM:** **96GB GDDR7** per GPU.
*   **Native NVFP4:** Native hardware acceleration for the 4-bit Floating Point (FP4) format.
*   **Aggregate Memory:** A single 8-GPU node provides **768GB of VRAM**, sufficient to host the entire 1T Kimi K2.5 model (quantized) without inter-node communication.

## Selected Optimization Strategy: "Blackwell-Native" Architecture

For Kimi K2.5, we leverage the native low-precision capabilities of Blackwell to maximize throughput and minimize latency by deploying the official NVIDIA-optimized weights.

### 1. Model Selection: `nvidia/Kimi-K2.5-NVFP4`
*   **Strategy:** Utilize the pre-quantized **`nvidia/Kimi-K2.5-NVFP4`** model directly from Hugging Face.
*   **Benefit:** This model is officially optimized by NVIDIA using the TensorRT Model Optimizer specifically for the Blackwell architecture. NVFP4 uses a two-level micro-scaling strategy that captures the dynamic range of Kimi's 1-trillion parameters far more accurately than INT4, preserving more of the model's reasoning and coding capability while minimizing the footprint.
*   **Throughput:** Native FP4 execution on 5th Gen Tensor Cores provides up to 2x the throughput of FP8/INT4.

### 2. Deployment Topology: Single-Node (TP=8)
*   **Strategy:** Deploy the entire model on a single `g4-standard-384` instance (8x 96GB GPUs).
*   **Why:** With 768GB of aggregate VRAM, we no longer need Pipeline Parallelism (PP) across nodes or complex orchestration like Ray. We use **Tensor Parallelism (TP=8)** within the node, leveraging the ultra-high bandwidth Titanium offload engine for intra-node communication. This eliminates the multi-node PCIe latency bottleneck.

### 3. GKE-Native Optimizations (The Core Advantage)
*   **GKE Inference Gateway (Prefix-Aware Routing):** This is the critical component for coding workloads.
    *   *Mechanism:* The Gateway hashes the prefix of incoming prompts (e.g., a large injected codebase). It routes requests with identical prefixes to the exact same GPU pod that previously processed it.
    *   *Impact:* Leverages vLLM's Automatic Prefix Caching. The model skips the expensive "Prefill" phase entirely for reused context, resulting in near-instant Time-To-First-Token (TTFT) and drastically reduced compute load. The **96GB VRAM** per card acts as a massive cache drive for code context.
*   **Model-Aware Autoscaling (llm-d / HPA):** Scale based on inference metrics (Request Concurrency or Duty Cycle) exported by vLLM, rather than raw GPU utilization (which is an inaccurate metric for LLMs).
*   **GCSFuse for Fast Model Loading:** Mount the 600GB model weights directly from Google Cloud Storage via the GCSFuse sidecar. This avoids copying hundreds of gigabytes of weights to the local pod disk, reducing pod startup time from 30+ minutes to seconds, and enabling rapid elastic scaling.
*   **GKE Image Streaming:** Enabled at the cluster level to dynamically pull container images (like the multi-gigabyte vLLM image) from Artifact Registry. The pod boots immediately, and container image data is streamed on-demand, further drastically reducing node provisioning and pod boot latency.
*   **Automated Weight Preparation (GKE Job):** Rather than manually copying weights, a dedicated Kubernetes Job (`model-download-job.yaml`) natively utilizes the GCSFuse mount to download the models from HuggingFace directly into the secure, centralized GCS bucket, creating a fully automated deployment pipeline.

## Scaling Strategy: Achieving 1.5M Tokens Per Minute (TPM)

To support a high-demand enterprise environment requiring 1.5 million TPM (25,000 TPS), the cluster must scale horizontally while maintaining cache affinity.

### 1. Hardware Requirements
*   **Total Throughput Target:** 25,000 TPS.
*   **Optimized Configuration (Prefix Hits):** 3 nodes (24x RTX PRO 6000 Blackwell).
*   **Baseline Configuration (Cold Starts):** 7 nodes (56x RTX PRO 6000 Blackwell).
*   **Recommended Starting Point:** **4 nodes** (32 GPUs) with Prefix-Aware Routing enabled.

### 2. Automated Scaling (HPA)
Use the GKE Horizontal Pod Autoscaler with custom vLLM metrics.
*   **Metric:** `vllm_avg_generation_throughput_10s` or `vllm_num_requests_running`.
*   **Threshold:** Trigger scale-up when a node reaches 80% of its tested Scenario A capacity (~6,800 TPS).

### 3. Capacity Tiering
*   **Primary Tier (Spot G4):** Use Spot instances for the bulk of the 25k TPS capacity to reduce costs by 60-90%.
*   **Safety Tier (On-Demand G4):** Maintain a minimum of 1 node on-demand to ensure baseline availability for critical tasks during GCE preemption events.

---

## Technical Correction (Historical Archive)
*   **Previous Assumption:** It was previously stated that G4 used Ada Lovelace architecture and rejected NVFP4. We originally planned a complex multi-node INT4 deployment.
*   **Correction:** Official GCP documentation confirms G4 uses **Blackwell Server Edition**. Consequently, NVFP4 is **fully supported**. Shifting to the `nvidia/Kimi-K2.5-NVFP4` model allows us to consolidate the entire 1T parameter MoE onto a single, highly efficient 8-GPU node.

## Appendix: Implementation Details
*   **Quantization:** Use `vLLM` which natively supports NVFP4 on Blackwell hardware.
*   **KV Cache:** Use `fp8` for KV cache to maximize context window length (up to 256k tokens) on the 96GB cards.
