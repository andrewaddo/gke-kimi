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

### 3. Deploy vLLM (Choose Scenario)

We provide two deployment configurations depending on your testing needs:

#### Option A: Single Node (Baseline & Scenario Testing)
Deploys 1 replica of the vLLM server to establish baseline hardware performance.
```bash
kubectl apply -f deploy/manifests/vllm-deployment-single.yaml
kubectl apply -f deploy/manifests/vllm-service.yaml
```

#### Option B: 4-Node Farm (1.5M TPM Simulation)
Deploys 4 replicas across the cluster (combining On-Demand and Spot instances) using `podAntiAffinity` to ensure each 1T model replica gets its own dedicated 8-GPU node.
```bash
# Make sure your node pools are scaled up (e.g., Spot pool max-nodes: 4)
kubectl apply -f deploy/manifests/vllm-deployment-farm.yaml
kubectl apply -f deploy/manifests/vllm-service.yaml
```

#### Switching Between Scenarios
If you have one scenario running and want to switch to the other, delete the active deployment first to free up the GPUs, then apply the new one:
```bash
kubectl delete deployment kimi-k25-vllm -n kimi-k25
kubectl apply -f deploy/manifests/vllm-deployment-<single|farm>.yaml
```

---

## Configuration Reference: The vLLM Deployments

This deployment is heavily tuned to extract maximum tokens-per-second (TPS) from the Blackwell architecture for a 1-Trillion parameter MoE.

### A. Environment Variables
*   `ENABLE_NVFP4_SM120="1"`: **Crucial.** Enables hardware-accelerated 4-bit floating-point (FP4) math natively on Blackwell (SM120) Tensor Cores.
*   `VLLM_ATTENTION_BACKEND="FLASHINFER"`: Forces vLLM to use FlashInfer, which provides highly optimized attention kernels specifically tuned for modern NVIDIA architectures.

### B. vLLM Server Arguments (`api_server.py`)
*   `--model /models`: The path to the NVIDIA-optimized NVFP4 weights.
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
*   `metadata-cache:ttl-secs:-1`: Indefinitely caches model file metadata, reducing latency for file system lookups.
*   `file-cache:max-size-mb:-1`: Uses the node's local disk as a full cache for model weights, bypassing the network for subsequent pod restarts.
*   `file-cache:enable-parallel-downloads:true`: **Crucial.** Enables GCSFuse to stream model shards via multiple parallel threads, reducing 1T model load times from hours to minutes.
*   `file-cache:cache-file-for-range-read:true`: Optimizes the multi-head read pattern of the vLLM loader.

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

## Milestone 2: Intelligent Unified Serving with GKE Inference Gateway

To achieve massive scale (>2.0M TPM) and support high concurrency, a single node is insufficient. However, simply scaling standard Kubernetes pods (Round-Robin) leads to "Cache Thrashing" as all nodes attempt to process identical 10k-token prefills.

The solution is the **GKE Inference Gateway** utilizing **Prefix-Aware Routing**. By intelligently routing requests based on their context, the Gateway ensures that 10k-token prefixes are perfectly partitioned across the cluster's VRAM. This allows the GPUs to skip the expensive $O(N^2)$ prefill compute stage entirely and operate at maximum memory-bound throughput (achieving >5x the single-node performance).

### 1. Prerequisites (Infrastructure & CRDs)
Ensure your cluster has the necessary Load Balancing and Autoscaling addons enabled, and that the required Gateway API Inference Extension CRDs are installed:

```bash
# 1. Enable Required GKE Addons
gcloud container clusters update kimi-k25-cluster \
  --zone us-central1-b \
  --update-addons=HttpLoadBalancing=ENABLED,HorizontalPodAutoscaling=ENABLED

# 2. Install Gateway API Inference Extension CRDs (GKE 1.34+)
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/v1.0.0/config/crd/bases/inference.networking.x-k8s.io_inferenceobjectives.yaml

# 3. Create Firewall Rule for Internal L7 Load Balancer (Proxy-Only Subnet)
# Replace with your actual proxy-only subnet range (e.g., 10.129.0.0/23) and node tag
gcloud compute firewall-rules create allow-gke-l7-rilb-inference \
    --action=ALLOW \
    --direction=INGRESS \
    --rules=tcp:30001 \
    --source-ranges=10.129.0.0/23,10.127.0.0/26 \
    --target-tags=gke-kimi-k25-cluster-5699cb9b-node \
    --network=default
```

### 2. Deploying the Unified Architecture
Apply the specialized manifests that configure the vLLM farm, the Endpoint Picker (EPP), and the Gateway routing rules.

#### Understanding the Gateway Configuration
The core of the intelligent routing lies in the **Endpoint Picker (EPP)** configuration (`kimi-epp-config` ConfigMap in `vllm-unified-gateway.yaml`). It uses a weighted scoring system to determine the best pod for each incoming request:
*   **`prefix-cache-scorer` (Weight: 10):** The Gateway hashes the incoming prompt and asks the vLLM pods if they have those tokens in their KV-Cache. It heavily biases routing toward the pod with the highest cache hit rate, completely skipping the compute-heavy Prefill phase.
*   **`queue-scorer` (Weight: 1):** Acts as a tie-breaker or fallback. If no pod has a cache hit (or if multiple do), it routes to the pod with the shortest active request queue to prevent latency spikes.

```bash
# 1. Deploy the vLLM Farm (3 Replicas)
kubectl apply -f deploy/manifests/vllm-deployment-farm.yaml

# 2. Deploy the GKE Inference Gateway, EPP, and InferencePool
# This configures the EPP with the `prefix-cache-scorer` (Weight: 10)
kubectl apply -f deploy/manifests/vllm-unified-gateway.yaml

# 3. Apply the Health Check Policy (Fixes 503 Backend Errors)
# Tells the Load Balancer to check the /health endpoint
kubectl apply -f deploy/manifests/vllm-health-check-policy.yaml

# 4. Deploy the Metric-Driven Autoscaler (HPA)
kubectl apply -f deploy/manifests/vllm-unified-hpa.yaml
```

### 3. Verification & Access
The Gateway will take a few minutes to provision the Internal L7 Load Balancer and assign an IP address.

```bash
# 1. Wait for the Gateway to become 'Programmed' and copy the IP ADDRESS
kubectl get gateway kimi-gateway-unified -n kimi-k25

# 2. Monitor the Endpoint Picker (EPP) logs to ensure it discovers the vLLM pods
kubectl logs -l app=kimi-unified-epp -n kimi-k25 --tail=50
```

Once the Gateway has an IP (e.g., `10.128.15.232`), you can access the OpenAI-compatible endpoint from within the cluster VPC:
`http://10.128.15.232/v1/chat/completions`

### 4. Running a High-Concurrency Benchmark
Because the Gateway assigns an *Internal* Load Balancer IP address, you cannot run the benchmark directly from your local macOS/Windows terminal (you will get a timeout or a `bash: vllm: command not found` error). 

To execute the benchmark and simulate real cluster traffic, you must run the `vllm bench` command **from inside one of the vLLM pods** in your cluster, targeting the Gateway's internal IP.

```bash
# 1. Get the name of a running vLLM pod
POD_NAME=$(kubectl get pods -n kimi-k25 -l app=kimi-k25-vllm -o jsonpath='{.items[0].metadata.name}')

# 2. Execute the benchmark from inside the pod, pointing to your Gateway IP
# IMPORTANT: Replace "10.128.15.232" with your actual Gateway IP from Step 3!
kubectl exec -it pod/$POD_NAME -n kimi-k25 -c vllm-server -- \
  vllm bench serve \
    --model "/models" \
    --served-model-name "/models" \
    --base-url "http://10.128.15.232" \
    --dataset-name "prefix_repetition" \
    --prefix-repetition-prefix-len 10000 \
    --prefix-repetition-num-prefixes 15 \
    --prefix-repetition-output-len 200 \
    --num-prompts 150 \
    --request-rate 15.0 \
    --trust-remote-code \
    --endpoint /v1/completions \
    --temperature 0
```


To test the model manually from your local machine using the OpenAI-compatible API:

### 1. Establish a Secure Tunnel
```bash
kubectl port-forward svc/kimi-k25-service -n kimi-k25 30001:30001
```

### 2. Send a Test Request (New Terminal)
```bash
curl -X POST http://localhost:30001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models",
    "messages": [
      {"role": "user", "content": "Write a Python function to compute the Fibonacci sequence."}
    ]
  }'
```

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
    --model "/models" \
    --dataset-name "prefix_repetition" \
    --prefix-repetition-prefix-len 10000 \
    --prefix-repetition-num-prefixes 40 \
    --prefix-repetition-output-len 300 \
    --request-rate 5.0 \
    --num-prompts 100 \
    --port 30001 \
    --trust-remote-code \
    --endpoint /v1/completions \
    --temperature 0

### Notes on Benchmarking Warnings
*   **"Using a slow tokenizer"**: This warning is **expected** for Kimi-K2.5. The model uses a custom `tiktoken` implementation which is Python-based. Our testing confirms this does **not** bottleneck throughput on Blackwell hardware. Do **not** attempt to use `--tokenizer-mode fast` or a different model's tokenizer, as this will cause a vocabulary mismatch and significantly degrade performance.
*   **"temperature==0 (greedy)"**: We explicitly add `--temperature 0` to our commands to ensure deterministic results and suppress this warning.
```

### Run 2 (Warm Cache): Cache Optimized
This second pass demonstrates the power of Prefix Caching in a simulated developer team environment. The engine reuses the cached 10,000-token prefixes from Run 1, dramatically increasing overall throughput.

```bash
# DO NOT clear the cache. Run the benchmark again immediately.
kubectl exec -it deployment/kimi-k25-vllm -n kimi-k25 -c vllm-server -- \
  vllm bench serve \
    --model "/models" \
    --dataset-name "prefix_repetition" \
    --prefix-repetition-prefix-len 10000 \
    --prefix-repetition-num-prefixes 40 \
    --prefix-repetition-output-len 300 \
    --request-rate 5.0 \
    --num-prompts 100 \
    --port 30001 \
    --trust-remote-code \
    --endpoint /v1/completions \
    --temperature 0

### Notes on Benchmarking Warnings
*   **"Using a slow tokenizer"**: This warning is **expected** for Kimi-K2.5. The model uses a custom `tiktoken` implementation which is Python-based. Our testing confirms this does **not** bottleneck throughput on Blackwell hardware. Do **not** attempt to use `--tokenizer-mode fast` or a different model's tokenizer, as this will cause a vocabulary mismatch and significantly degrade performance.
*   **"temperature==0 (greedy)"**: We explicitly add `--temperature 0` to our commands to ensure deterministic results and suppress this warning.
```

### Run Scenario B: Cold Start (Random Inputs)
This alternative scenario completely invalidates caching by generating 10,000 completely random, unique tokens per request.
```bash
# Clear cache first
curl -X POST http://localhost:30001/reset_prefix_cache

# Run benchmark with random inputs
kubectl exec -it deployment/kimi-k25-vllm -n kimi-k25 -c vllm-server -- \
  vllm bench serve \
    --model "/models" \
    --dataset-name "random" \
    --random-input-len 10000 \
    --random-output-len 300 \
    --request-rate 5.0 \
    --num-prompts 100 \
    --port 30001 \
    --trust-remote-code \
    --endpoint /v1/completions \
    --temperature 0

### Notes on Benchmarking Warnings
*   **"Using a slow tokenizer"**: This warning is **expected** for Kimi-K2.5. The model uses a custom `tiktoken` implementation which is Python-based. Our testing confirms this does **not** bottleneck throughput on Blackwell hardware. Do **not** attempt to use `--tokenizer-mode fast` or a different model's tokenizer, as this will cause a vocabulary mismatch and significantly degrade performance.
*   **"temperature==0 (greedy)"**: We explicitly add `--temperature 0` to our commands to ensure deterministic results and suppress this warning.
```

### 4-Node Cluster Benchmark (1.5M TPM Simulation)
To benchmark the entire 4-node cluster (32 GPUs total), target the service endpoint so traffic is distributed across all replicas. Note the increased request rate and prompt count to saturate the cluster.

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

### Notes on Benchmarking Warnings
*   **"Using a slow tokenizer"**: This warning is **expected** for Kimi-K2.5. The model uses a custom `tiktoken` implementation which is Python-based. Our testing confirms this does **not** bottleneck throughput on Blackwell hardware. Do **not** attempt to use `--tokenizer-mode fast` or a different model's tokenizer, as this will cause a vocabulary mismatch and significantly degrade performance.
*   **"temperature==0 (greedy)"**: We explicitly add `--temperature 0` to our commands to ensure deterministic results and suppress this warning.
```


---

## Cost Management & Cleanup

Blackwell 8-GPU nodes are expensive resources. Follow these steps to manage your costs during idle periods and tear down the infrastructure completely.

### 1. Scale Down to 0 (Suspend State)
To keep your GCS bucket, Artifact Registry, and cluster settings intact but stop paying for the expensive GPUs, run:

```bash
# Scale On-Demand pool to 0
gcloud container clusters resize kimi-k25-cluster --node-pool rtx-6000-pool --num-nodes 0 --zone us-central1-b --quiet

# Scale Deployment to 0 so the Autoscaler can terminate Spot nodes
kubectl scale deployment kimi-k25-vllm -n kimi-k25 --replicas=0
```
*Note: Any Spot nodes (`rtx-6000-spot-pool`) with Autoscaling enabled will automatically spin down to 0 after ~10 minutes of having no pods scheduled to them.*

### 2. Complete Teardown (Delete Everything)
To completely destroy the cluster, the node pools, and all associated storage:

```bash
PROJECT_ID=$(gcloud config get-value project)

# 1. Delete the GKE Cluster
gcloud container clusters delete kimi-k25-cluster --zone us-central1-b --quiet

# 2. Delete the GCS Bucket containing the 600GB NVFP4 Model
gcloud storage rm --recursive gs://kimi-k25-weights-bucket-$PROJECT_ID

# 3. Delete the Artifact Registry repository
gcloud artifacts repositories delete kimi-repo --location us-central1 --quiet

# 4. Remove the Service Account
gcloud iam service-accounts delete kimi-gcsfuse-sa@$PROJECT_ID.iam.gserviceaccount.com --quiet
```

---

## Future Considerations: Scaling for High Concurrency (500+ Users)

While the current architecture (3 nodes + GKE Inference Gateway) successfully achieves ~41,400 tok/s and handles ~130 concurrent requests with sub-100ms TPOT, scaling to support 500+ concurrent users requires specific tuning across the "Triple Constraint" of LLM serving: **Memory (VRAM)**, **Compute (Attention)**, and **Connection Stability**.

### 1. The Trade-offs of High Concurrency

When pushing a 1-Trillion parameter MoE model to high concurrency on a finite set of GPUs, you encounter the following trade-offs:

*   **VRAM Capacity vs. Context Length:** To fit more concurrent user "slots" (KV-cache) in the remaining VRAM (after the ~500GB weight tax), you must either reduce the `max_model_len` or use more aggressive quantization (e.g., INT4 KV cache), which can impact long-context reasoning accuracy.
*   **Throughput vs. Latency (TPOT):** As more users share the same 8 GPUs, the memory bandwidth is saturated. The Time-Per-Output-Token (TPOT) will slow down for all active users to accommodate the sheer volume of concurrent generations.
*   **Prefill Interference:** When a new user submits a massive 10k-token prompt, it can "freeze" the generation of the 100 users already chatting while the GPU computes the prefill attention matrix.

### 2. Required Technical Tuning for 500+ Concurrency

To safely scale the current architecture, the following configuration changes are required at both the engine and infrastructure layers:

#### A. vLLM Engine Tuning (The "VRAM Capacity" Layer)
Update the `kimi-k25-vllm` deployment arguments to optimize for concurrent slots over raw single-stream speed:

```bash
# Recommended vLLM Flags for High Concurrency
python3 -m vllm.entrypoints.openai.api_server \
  ... (existing flags) ... \
  --gpu-memory-utilization 0.98 \     # Maximize VRAM allocation for KV-cache
  --max-num-seqs 512 \                # Increase the hard limit on concurrent requests (default is often 256)
  --enable-chunked-prefill \          # Crucial: Breaks massive prefills into smaller chunks
  --max-num-batched-tokens 8192 \     # Size of prefill chunks to prevent "generation stutters"
  --kv-cache-dtype fp8                # Keep FP8 (or test INT4 if supported) to double capacity
```

#### B. GKE Gateway & Infrastructure Tuning (The "Pipe" Layer)
Our benchmarks showed a 13.4% failure rate at 15 RPS not because the GPUs crashed, but because the **Internal L7 Load Balancer** hit its default connection limits.

*   **BackendConfig Connection Pooling:** You must apply a `BackendConfig` CRD to the GKE Service to explicitly increase `maxConnectionsPerEndpoint` and tune the `timeoutSec`. Without this, the Gateway will drop requests (returning 503 Service Unavailable) even if the GPUs are idle.
*   **OS-Level TCP Tuning:** Ensure the underlying node OS and container network namespaces are tuned to handle sudden bursts of concurrent handshakes (e.g., increasing `net.core.somaxconn`).

#### C. Horizontal Scaling (The "Throughput" Layer)
Even with Prefix-Aware Routing partitioning the cache, each 8-GPU node has a hard mathematical limit on how many KV-cache blocks it can hold before it must evict and trigger the O(N²) prefill penalty. 

To maintain a responsive TPOT (< 100ms) for 500+ concurrent users with 10k-token contexts, the cluster must be scaled horizontally to **5-7 nodes** (40-56 GPUs). The GKE Inference Gateway will seamlessly distribute the cache state across these new nodes.
