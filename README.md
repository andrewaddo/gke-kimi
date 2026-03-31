# Kimi K2.5 Blackwell (NVFP4) GKE Deployment

This project provides the definitive configuration for deploying and benchmarking the **Kimi K2.5 (1T MoE)** model on Google Kubernetes Engine (GKE) using **NVIDIA G4 (RTX PRO 6000 Blackwell)** GPUs.

## Prerequisites
- Google Cloud Project with Billing Enabled.
- Quotas for `nvidia-rtx-pro-6000` (at least 8 GPUs in `us-central1`).
- `gcloud` and `kubectl` CLIs installed.

## Hardware: NVIDIA Blackwell (G4)
The GCP G4 machine series utilizes **NVIDIA RTX PRO 6000 Blackwell Server Edition** GPUs.
- **VRAM:** 96GB GDDR7 per GPU.
- **Aggregate VRAM:** 768GB (Single 8-GPU node).
- **Quantization:** Native **NVFP4** support via 5th Gen Tensor Cores for extreme throughput.

---

## Setup Instructions

### 1. Infrastructure Provisioning
Run the automated cluster setup script.
```bash
./scripts/setup_cluster.sh
```
**What this script does:**
1.  **Creates a GKE Cluster (`kimi-k25-cluster`):** Enables `GcsFuseCsiDriver` (for fast storage access) and `Image Streaming` (to boot the massive vLLM container instantly without waiting for a full 10GB+ image pull).
2.  **Provisions a Multi-Zone Node Pool:** Attempts to create a single `g4-standard-384` node (8x GPUs, 384 CPUs, 1.4TB RAM) across `us-central1-a,c,f`. This multi-zone request mitigates GCE Spot/On-Demand stockouts for high-demand Blackwell hardware. It also provisions a **1000GB boot disk** to provide ample room for GCSFuse caching.
3.  **Configures Workload Identity:** Creates a Google Service Account (`kimi-gcsfuse-sa`) with `roles/storage.objectAdmin` and binds it to a Kubernetes Service Account (`kimi-ksa`). This allows pods to securely access the GCS bucket without managing JSON keys.

### 2. Model Weight Preparation
A Kubernetes Job is provided to automatically download the official **`nvidia/Kimi-K2.5-NVFP4`** weights (~600GB) directly into a newly created GCS bucket using GCSFuse.
```bash
kubectl apply -f deploy/manifests/model-download-job.yaml
```
**Why this approach?**
- **Direct-to-Bucket:** We use a lightweight container running `huggingface-cli` mapped to the GCSFuse mount (`/models`). This streams the 600GB download directly into Cloud Storage, bypassing the node's local disk capacity limits entirely.
- **Parallelism:** We use the `hf_transfer` Rust-based library to maximize Hugging Face download bandwidth (often exceeding 150MB/s).

Monitor the download progress:
```bash
kubectl logs -f job/download-kimi-model -n kimi-k25 -c download-kimi
```
*Wait for this Job to reach `Completed` status before proceeding.*

### 3. Deploy vLLM
Once the weights are securely stored in the GCS bucket, deploy the Blackwell-optimized vLLM server:
```bash
kubectl apply -f deploy/manifests/vllm-deployment.yaml
kubectl apply -f deploy/manifests/vllm-service.yaml
```

---

## Configuration Reference: The vLLM Deployment (`vllm-deployment.yaml`)

This deployment is heavily tuned to extract maximum tokens-per-second (TPS) from the Blackwell architecture for a 1-Trillion parameter MoE.

### A. Environment Variables
*   `ENABLE_NVFP4_SM120="1"`: **Crucial.** Enables hardware-accelerated 4-bit floating-point (FP4) math natively on Blackwell (SM120) Tensor Cores.
*   `VLLM_ATTENTION_BACKEND="FLASHINFER"`: Forces vLLM to use FlashInfer, which provides highly optimized attention kernels specifically tuned for modern NVIDIA architectures.

### B. vLLM Server Arguments (`api_server.py`)
*   `--model /models/Kimi-K2.5-NVFP4`: The path to the NVIDIA-optimized NVFP4 weights.
*   `--tensor-parallel-size 8`: Distributes the model tensors across all 8 GPUs on the single node. We **do not** use Pipeline Parallelism (PP) here because the 768GB aggregate VRAM is sufficient to hold the entire model on one node, avoiding slow PCIe inter-node latency.
*   `--enable-expert-parallel`: **Crucial for MoE.** Instead of slicing every tensor across all GPUs (Tensor Parallelism), EP places entire "experts" on specific GPUs. This drastically reduces the inter-GPU communication overhead over the PCIe bus.
*   `--compilation_config.pass_config.fuse_allreduce_rms true`: A performance optimization that fuses the AllReduce communication step with the subsequent RMSNorm operation into a single GPU kernel, significantly reducing memory bandwidth pressure.
*   `--mm-encoder-tp-mode data`: Runs the small (~400M) vision encoder in Data Parallel mode rather than slicing it via Tensor Parallelism, preventing unnecessary PCIe communication bottlenecks for multimodal inputs.
*   `--enable-prefix-caching`: **The key to 1.5M TPM.** Enables KV-cache reuse. If multiple requests share the same system prompt or large codebase prefix, vLLM skips the expensive "prefill" compute phase for those tokens.
*   `--kv-cache-dtype fp8`: Stores the conversation history (KV cache) in 8-bit floating-point format instead of 16-bit. This halves the memory required for context, allowing the 96GB cards to handle massive concurrent batch sizes and massive contexts (up to 65,536 tokens).
*   `--max-model-len 65536`: Sets the maximum context window to 64k tokens.
*   `--gpu-memory-utilization 0.95`: Instructs vLLM to allocate 95% of the 96GB VRAM on each GPU immediately for the model weights and the KV cache.

### C. GCSFuse Optimizations (Volume Mounts)
To load the 600GB model quickly, we apply advanced GCSFuse volume attributes:
*   `implicit-dirs`: Allows GCSFuse to traverse the bucket structure without requiring explicit directory objects.
*   `sequential-read-size-mb=200`: Tells GCSFuse that vLLM will be reading the `.safetensors` files sequentially, instructing it to pre-fetch large 200MB chunks into memory, massively accelerating pod startup times.
*   `file-cache:max-size-mb:-1` & `file-cache:enable-parallel-downloads:true`: Leverages the node's 1000GB boot disk to cache the model files locally using parallel streaming.

---

## Scaling Strategy: Achieving 1.5M Tokens Per Minute (TPM)

To support a high-demand enterprise environment requiring 1.5 million TPM (25,000 TPS), the cluster must scale horizontally while maintaining cache affinity.

### 1. The Scaling Math
Based on G4 Blackwell benchmarks:
*   **Cold Start (Cache Miss):** ~3,880 TPS per 8-GPU node. Requires **7 nodes (56 GPUs)** to hit 1.5M TPM.
*   **Cache Optimized (Prefix Hits):** ~8,500 TPS per 8-GPU node. Requires **3 nodes (24 GPUs)** to hit 1.5M TPM.

### 2. The Solution: GKE Inference Gateway
To scale efficiently to 3 nodes, you **must** utilize the GKE Inference Gateway with Prefix-Aware Routing.
*   **Without Gateway:** A standard Kubernetes Service round-robins requests. "Developer A" and "Developer B" querying the same codebase might hit different nodes, resulting in a 0% cache hit rate (requiring 7 nodes).
*   **With Gateway:** The Gateway hashes the incoming prompt prefix and routes identical prefixes (e.g., the same codebase context) to the exact same GPU node. This guarantees a near-100% prefix cache hit rate, bypassing the expensive prefill phase and unlocking the 8,500 TPS per-node performance tier.

### 3. Automated Scaling (HPA)
Use the GKE Horizontal Pod Autoscaler configured with custom vLLM metrics.
*   **Metric:** Scale based on `vllm_num_requests_running` or `vllm_gpu_cache_usage_perc` rather than raw CPU/GPU utilization (which are inaccurate for LLMs).
*   **Threshold:** Trigger scale-up when a node reaches 80% of its tested Cache-Optimized capacity.

---

## Performance Benchmarking (vLLM Bench)

To validate the Prefix Caching performance, use the provided vLLM benchmarking tool. Two passes of the benchmark should be executed to demonstrate the impact of caching.

### Port-forward to the Service:
```bash
kubectl port-forward svc/kimi-k25-service -n kimi-k25 30001:30001 &
```

### Run 1 (Cold Cache): Establishing the Baseline
This first pass establishes a baseline hardware limit, as the KV-cache is empty and the engine must compute the full 10,000-token prefill for every unique prefix.

```bash
# Clear cache first to ensure a clean cold start
curl -X POST http://localhost:30001/reset_prefix_cache

# Run the benchmark from inside the pod to bypass network jitter
kubectl exec -it deployment/kimi-k25-vllm -n kimi-k25 -c vllm-server -- \
  vllm bench serve \
    --model "/models/Kimi-K2.5-NVFP4" \
    --dataset-name "prefix_repetition" \
    --prefix-repetition-prefix-len 10000 \
    --prefix-repetition-num-prefixes 40 \
    --prefix-repetition-output-len 300 \
    --request-rate 5.0 \
    --num-prompts 100 \
    --port 30001 \
    --trust-remote-code \
    --endpoint /v1/completions
```

### Run 2 (Warm Cache): Cache Optimized
This second pass demonstrates the power of Prefix Caching in a simulated developer team environment. The engine reuses the cached 10,000-token prefixes from Run 1, dramatically increasing overall throughput.

```bash
# DO NOT clear the cache. Run the benchmark again immediately.
kubectl exec -it deployment/kimi-k25-vllm -n kimi-k25 -c vllm-server -- \
  vllm bench serve \
    --model "/models/Kimi-K2.5-NVFP4" \
    --dataset-name "prefix_repetition" \
    --prefix-repetition-prefix-len 10000 \
    --prefix-repetition-num-prefixes 40 \
    --prefix-repetition-output-len 300 \
    --request-rate 5.0 \
    --num-prompts 100 \
    --port 30001 \
    --trust-remote-code \
    --endpoint /v1/completions
```

### Run Scenario B: Cold Start (Random Inputs)
This alternative scenario completely invalidates caching by generating 10,000 completely random, unique tokens per request.
```bash
# Clear cache first
curl -X POST http://localhost:30001/reset_prefix_cache

# Run benchmark with random inputs
kubectl exec -it deployment/kimi-k25-vllm -n kimi-k25 -c vllm-server -- \
  vllm bench serve \
    --model "/models/Kimi-K2.5-NVFP4" \
    --dataset-name "random" \
    --random-input-len 10000 \
    --random-output-len 300 \
    --request-rate 5.0 \
    --num-prompts 100 \
    --port 30001 \
    --trust-remote-code \
    --endpoint /v1/completions
```
