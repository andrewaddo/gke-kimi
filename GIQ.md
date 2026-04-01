# GKE Inference Quickstart (GIQ) Commands

GKE Inference Quickstart (GIQ) provides optimized Kubernetes configurations for AI/ML inference workloads.

## 1. Prerequisites & Setup
Ensure your `gcloud` version is up to date and the required API is enabled.

```bash
# Update gcloud
gcloud components update

# Enable the Inference Quickstart API
gcloud services enable gkerecommender.googleapis.com

# Set the billing quota project (Required for the API)
gcloud config set billing/quota_project $(gcloud config get-value project)
```

## 2. Discovery
Find supported models, model servers, and hardware accelerators.

```bash
# List all supported models
gcloud container ai profiles models list

# Check if Kimi K2.5 is supported
gcloud container ai profiles models list --filter="moonshotai/Kimi-K2.5"

# List supported model servers for Kimi K2.5
gcloud container ai profiles model-servers list --model="moonshotai/Kimi-K2.5"

# List supported accelerators for Kimi K2.5 (Requires alpha component)
gcloud alpha container ai profiles accelerators list --model="moonshotai/Kimi-K2.5"
```

## 3. Generate Optimized Manifests
Generate a tailored Kubernetes manifest with optimized resource requests and tuning.

```bash
# Generate manifest for Kimi K2.5 on RTX 6000 Ada
gcloud container ai profiles manifests create \
    --model="moonshotai/Kimi-K2.5" \
    --model-server="vllm" \
    --accelerator-type="nvidia-rtx-pro-6000" \
    --use-case="Code Generation" \
    --output-path="kimi-k25-giq.yaml"
```

### Key Flags for `manifests create`:
- `--model`: The model name (e.g., `moonshotai/Kimi-K2.5`).
- `--model-server`: The serving framework (e.g., `vllm`).
- `--accelerator-type`: The GPU/TPU type (e.g., `nvidia-rtx-pro-6000`).
- `--model-bucket-uri`: (Optional) GCS bucket path for model weights.
- `--use-case`: Optimize for specific workloads (e.g., `Code Generation`, `Chatbot`).
- `--target-ttft-milliseconds`: (Optional) Target Time-To-First-Token for HPA.

## 4. Deployment
Apply the generated manifest to your GKE cluster.

```bash
kubectl apply -f kimi-k25-giq.yaml
```
