#!/usr/bin/env bash
# setup_gcs.sh — Verify and initialise GCS bucket structure for QIG training
# Note: Bucket qig-training-data already created via GCP console.
# This script uploads configs and verifies folder structure.

set -euo pipefail

BUCKET="gs://qig-training-data"
PROJECT_ID="agent-one-ffec8"

echo "==> Verifying bucket structure..."
FOLDERS=(
  "raw/"
  "coordized/"
  "checkpoints/"
  "coordizer_artifacts/"
  "configs/"
  "qktj/phase0_identity/"
  "qktj/phase1_coupling/"
  "qktj/phase2_integration/"
  "qktj/phase3_temporal/"
)

for folder in "${FOLDERS[@]}"; do
  # Create placeholder to ensure folder exists
  echo "" | gsutil cp - "${BUCKET}/${folder}.keep" 2>/dev/null || true
  echo "  OK: ${BUCKET}/${folder}"
done

echo "==> Uploading configs to GCS..."
gsutil cp configs/*.yaml "${BUCKET}/configs/"

echo "==> Bucket structure:"
gsutil ls -r "${BUCKET}/" | head -50

echo ""
echo "==> Setup complete."
echo "    Bucket: ${BUCKET}"
echo "    To upload QKTJ data:"
echo "      gsutil -m cp data/*.qktj.jsonl ${BUCKET}/qktj/phase0_identity/"
