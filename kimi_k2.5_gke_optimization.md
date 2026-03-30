# Optimizing Kimi K2.5 Inference on GKE with RTX 6000 Ada

## Overview
This document outlines the strategy for deploying and optimizing the inference of the Kimi K2.5 (1-trillion parameter MoE) model on Google Kubernetes Engine (GKE) utilizing NVIDIA RTX 6000 Ada Generation (48GB) GPUs.

## The Challenge
Kimi K2.5 is a massive Mixture-of-Experts (MoE) model. Even with INT4 quantization, the model requires approximately **600GB of memory**.
The NVIDIA RTX 6000 Ada has **48GB of VRAM** and relies on PCIe for inter-GPU communication (lacking high-bandwidth NVLink found in data center GPUs like H100). This presents a significant bottleneck for standard Tensor Parallelism across many cards.

## Selected Optimization Strategy: "The Coding Cache" Architecture`

For a coding-centric workload (where large context prefixes like codebases and documentation are frequently reused), we select a hybrid approach maximizing VRAM caching and GKE's intelligent routing.

### 1. Hardware Configuration
*   **GPU:** NVIDIA RTX 6000 Ada Generation (48GB VRAM).
*   **Scale:** 14x to 16x RTX 6000 Ada GPUs per Engine Group (distributed across nodes as needed by GKE availability, prioritizing compact placement).
*   **Quantization:** **INT4 (AWQ/GPTQ)**. This shrinks the model footprint to ~550GB-600GB, allowing it to reside entirely within the aggregated VRAM of the 14-16 GPUs.

### 2. Software & Framework Optimizations (vLLM)
*   **Parallelism Strategy:** **Pipeline Parallelism (PP)** combined with **Expert Parallelism (EP)**.
    *   *Why:* RTX 6000s lack NVLink. High Tensor Parallelism (TP) over PCIe will cripple performance. PP passes activations sequentially between GPUs, drastically reducing PCIe congestion. EP ensures specific MoE experts reside on specific GPUs.
*   **KV Cache:** `--kv-cache-dtype fp8`. The Ada architecture has native 4th-gen Tensor Cores optimized for FP8. This halves the memory required for conversation history, massively increasing concurrent batch size capability.
*   **Speculative Decoding:** Deploy an EAGLE-3 (or similar 1B-3B parameter) draft model alongside K2.5 to increase tokens-per-second (TPS) by 2x-3x.

### 3. GKE-Native Optimizations (The Core Advantage)
*   **GKE Inference Gateway (Prefix-Aware Routing):** This is the critical component for coding workloads.
    *   *Mechanism:* The Gateway hashes the prefix of incoming prompts (e.g., a large injected codebase). It routes requests with identical prefixes to the exact same GPU pod that previously processed it.
    *   *Impact:* Leverages vLLM's Automatic Prefix Caching. The model skips the expensive "Prefill" phase entirely for reused context, resulting in near-instant Time-To-First-Token (TTFT) and drastically reduced compute load. The 48GB VRAM per card acts as a massive cache drive for code context.
*   **Model-Aware Autoscaling (llm-d / HPA):** Scale based on inference metrics (Request Concurrency or Duty Cycle) exported by vLLM, rather than raw GPU utilization (which is an inaccurate metric for LLMs).
*   **GCSFuse for Fast Boot:** Mount the 600GB model weights directly from Google Cloud Storage via the GCSFuse sidecar. This reduces pod startup time from 30+ minutes to seconds, enabling elastic scaling.

---

## Alternative Options Considered (And Why They Were Rejected)

### Alternative 1: NVIDIA Blackwell NVFP4 Quantization
*   **Description:** Utilizing the new 4-bit Floating Point (FP4) format designed for extreme throughput.
*   **Why Rejected:** **Hardware Incompatibility.** NVFP4 requires 5th Generation Tensor Cores found *only* in the Blackwell (B100/B200) architecture. The RTX 6000 Ada (Ada Lovelace architecture) has 4th Gen Tensor Cores and physically cannot process FP4 math natively.

### Alternative 2: Extreme Quantization (ExLlamaV2 at 2.5bpw)
*   **Description:** Quantizing the model further to 2.5 bits-per-weight to fit the entire model into 8x RTX 6000s (a single server node).
*   **Why Rejected:** **Quality Degradation.** While it reduces hardware requirements significantly and offers high throughput, coding tasks require high precision and complex reasoning. Compressing a 1T model to 2.5bpw risks unacceptable degradation in coding accuracy and logical coherence compared to INT4.

### Alternative 3: CPU-GPU Offloading (KTransformers)
*   **Description:** Using 2x-4x RTX 6000s and offloading the MoE "expert" layers to massive, high-speed System RAM (DDR5) via PCIe.
*   **Why Rejected:** **Throughput Limits.** While highly cost-effective for a single user or hobbyist setup, the constant swapping of expert weights across the PCIe bus limits throughput to ~10-14 tokens/sec. This is insufficient for a multi-user enterprise environment where high concurrency is required.

## Conclusion
By combining the 48GB VRAM capacity of the RTX 6000 Ada with GKE Inference Gateway's Prefix-Aware Routing and INT4 quantization, we create an environment perfectly tuned for coding tasks. This strategy minimizes the PCIe bottleneck inherent to RTX workstation cards and maximizes throughput by ensuring large, repetitive code contexts are cached and routed intelligently.

## Hardware Validation Appendix

The strategies and rejections outlined in this document have been validated against official NVIDIA hardware specifications and current Google Cloud Platform (GCP) capabilities.

### 1. GCP Availability Validation
*   **Validated Fact:** The `nvidia-rtx-pro-6000` (NVIDIA RTX 6000 Ada Generation) is available as a native accelerator type on Google Cloud Compute Engine in multiple regions (e.g., `us-central1`, `europe-west1`, `us-west1`).
*   **GKE Support:** These accelerators can be attached to GKE Standard node pools running versions 1.34+ and 1.35+, which fully support the `InferencePool` and `HTTPRoute` resources required for the Inference Gateway.

### 2. NVFP4 and Architecture Validation
*   **Validated Fact:** The RTX 6000 Ada is built on the **Ada Lovelace** architecture and utilizes **4th Generation Tensor Cores**.
*   **NVFP4 Hardware Dependency:** **NVFP4** (NVIDIA 4-bit Floating Point) is a specialized format introduced with the **Blackwell** architecture. Native hardware acceleration for NVFP4 strictly requires **5th Generation Tensor Cores** (found in B100, B200, and Blackwell Ultra).
*   **Impact on Strategy:** While custom software kernels (e.g., AdaLLM) can emulate NVFP4 on Ada hardware to save VRAM, this emulation bypasses the hardware Tensor Cores. This results in a net *loss* of throughput compared to native INT4 or FP8 execution. Therefore, NVFP4 was correctly rejected for high-throughput enterprise inference on the RTX 6000 Ada platform.
