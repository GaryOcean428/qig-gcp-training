#!/usr/bin/env bash
# build_push.sh — Build and push Docker image to GCR for QIG-Native training
# Usage: ./scripts/build_push.sh [TAG]

set -euo pipefail

PROJECT_ID="agent-one-ffec8"
REGION="us-central1"
REPO="qig-training"
IMAGE="qig-native-train"
TAG="${1:-latest}"

FULL_IMAGE="us-central1-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:${TAG}"

echo "==> Authenticating with GCP..."
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet

echo "==> Building image: ${FULL_IMAGE}"
docker build \
  --platform linux/amd64 \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  -t "${FULL_IMAGE}" \
  .

echo "==> Pushing image: ${FULL_IMAGE}"
docker push "${FULL_IMAGE}"

echo "==> Image pushed successfully: ${FULL_IMAGE}"
echo "Use this URI in submit_vertex.py:"
echo "  ${FULL_IMAGE}"
