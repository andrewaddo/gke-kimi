#!/bin/bash
set -e
export USE_GKE_GCLOUD_AUTH_PLUGIN=True

PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1"
ZONE="us-central1-b"
CLUSTER_NAME="kimi-k25-cluster"
NAMESPACE="kimi-k25"

# 0. Create Artifact Registry
echo "Creating Artifact Registry: kimi-repo..."
gcloud artifacts repositories create kimi-repo \
    --repository-format=docker \
    --location=$REGION \
    --description="Repository for Kimi K2.5 model images" || true

# 0.5 Create GCS Bucket for Model Weights
echo "Creating GCS Bucket for Model Weights: kimi-k25-weights-bucket-$PROJECT_ID..."
gcloud storage buckets create gs://kimi-k25-weights-bucket-$PROJECT_ID --location=$REGION --project=$PROJECT_ID || true

# 1. Create GKE Cluster
echo "Checking if GKE Cluster: $CLUSTER_NAME exists..."
if ! gcloud container clusters describe $CLUSTER_NAME --zone $ZONE > /dev/null 2>&1; then
  echo "Creating GKE Cluster: $CLUSTER_NAME..."
  gcloud container clusters create $CLUSTER_NAME \
      --project=$PROJECT_ID \
      --zone=$ZONE \
      --release-channel="rapid" \
      --gateway-api=standard \
      --workload-pool="$PROJECT_ID.svc.id.goog" \
      --addons GcsFuseCsiDriver \
      --enable-image-streaming \
      --enable-managed-prometheus \
      --num-nodes=1 \
      --machine-type="n2-standard-8"
else
  echo "Cluster $CLUSTER_NAME already exists."
  gcloud container clusters get-credentials $CLUSTER_NAME --zone $ZONE
fi

# 2. Create Node Pool with RTX PRO 6000 Blackwell (On-Demand, Multi-Zone to avoid Stockouts)
echo "Checking if Node Pool: rtx-6000-pool exists..."
if ! gcloud container node-pools describe rtx-6000-pool --cluster $CLUSTER_NAME --zone $ZONE > /dev/null 2>&1; then
  echo "Creating Node Pool with RTX PRO 6000 Blackwell GPUs..."
  gcloud container node-pools create rtx-6000-pool \
      --project=$PROJECT_ID \
      --cluster=$CLUSTER_NAME \
      --zone=$ZONE \
      --node-locations=us-central1-b \
      --accelerator=type=nvidia-rtx-pro-6000,count=8 \
      --machine-type=g4-standard-384 \
      --num-nodes=0 \
      --enable-autoscaling \
      --min-nodes=0 \
      --max-nodes=4 \
      --disk-size=1000GB \
      --workload-metadata=GKE_METADATA
else
  echo "Node pool rtx-6000-pool already exists."
fi

# 2.5 Create Spot Node Pool
echo "Checking if Node Pool: rtx-6000-spot-pool exists..."
if ! gcloud container node-pools describe rtx-6000-spot-pool --cluster $CLUSTER_NAME --zone $ZONE > /dev/null 2>&1; then
  echo "Creating Spot Node Pool with RTX PRO 6000 Blackwell GPUs..."
  gcloud container node-pools create rtx-6000-spot-pool \
      --project=$PROJECT_ID \
      --cluster=$CLUSTER_NAME \
      --zone=$ZONE \
      --node-locations=us-central1-b \
      --accelerator=type=nvidia-rtx-pro-6000,count=8 \
      --machine-type=g4-standard-384 \
      --num-nodes=0 \
      --spot \
      --enable-autoscaling \
      --min-nodes=0 \
      --max-nodes=4 \
      --disk-size=1000GB \
      --workload-metadata=GKE_METADATA
else
  echo "Node pool rtx-6000-spot-pool already exists."
fi

# 3. Create Namespace
kubectl create namespace $NAMESPACE || true

# 4. Set up Workload Identity for GCSFuse & Artifact Registry
echo "Setting up Workload Identity..."
GSA_NAME="kimi-gcsfuse-sa"
KSA_NAME="kimi-ksa"

gcloud iam service-accounts create $GSA_NAME --project=$PROJECT_ID || true

# Grant Storage Admin for GCSFuse
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$GSA_NAME@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"

# Grant Artifact Registry Writer for the Image Copy Job
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$GSA_NAME@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/artifactregistry.writer"

kubectl create serviceaccount $KSA_NAME --namespace $NAMESPACE || true

gcloud iam service-accounts add-iam-policy-binding $GSA_NAME@$PROJECT_ID.iam.gserviceaccount.com \
    --role="roles/iam.workloadIdentityUser" \
    --member="serviceAccount:$PROJECT_ID.svc.id.goog[$NAMESPACE/$KSA_NAME]"

kubectl annotate serviceaccount $KSA_NAME \
    --namespace $NAMESPACE \
    iam.gke.io/gcp-service-account=$GSA_NAME@$PROJECT_ID.iam.gserviceaccount.com

# 5. Template Bucket and Image in YAMLs
echo "Templating bucket and image name into manifests..."
VLLM_IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/kimi-repo/vllm-openai:latest"
BUCKET_NAME="kimi-k25-weights-bucket-$PROJECT_ID"

# Use python for safer replacement of strings with slashes
python3 -c "
import sys
content = open('deploy/manifests/vllm-deployment.yaml').read()
content = content.replace('image: vllm/vllm-openai:latest', f'image: {sys.argv[1]}')
content = content.replace('bucketName: kimi-k25-weights-bucket', f'bucketName: {sys.argv[2]}')
open('deploy/manifests/vllm-deployment.yaml', 'w').write(content)

content = open('deploy/manifests/model-download-job.yaml').read()
content = content.replace('bucketName: REPLACE_WITH_BUCKET_NAME', f'bucketName: {sys.argv[2]}')
open('deploy/manifests/model-download-job.yaml', 'w').write(content)

content = open('deploy/manifests/image-copy-job.yaml').read()
content = content.replace('REPLACE_WITH_AR_IMAGE', f'{sys.argv[1]}')
open('deploy/manifests/image-copy-job.yaml', 'w').write(content)
" "$VLLM_IMAGE" "$BUCKET_NAME"

echo "Setup Complete!"
