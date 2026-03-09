"""submit_vertex.py — Submit QIG-Native training job to Vertex AI.

Usage:
    python scripts/submit_vertex.py --config configs/100m_phase0.yaml \
        --image us-central1-docker.pkg.dev/agent-one-ffec8/qig-training/qig-native-train:latest

Requires:
    google-cloud-aiplatform>=1.38.0
    gcloud auth application-default login
"""

import argparse
import yaml
from datetime import datetime
from pathlib import Path

from google.cloud import aiplatform


PROJECT_ID = "agent-one-ffec8"
REGION = "us-central1"
STAGING_BUCKET = "gs://qig-training-data"


def submit_job(
    config_path: str,
    image_uri: str,
    job_name: str = None,
    use_spot: bool = False,
) -> None:
    """Submit a Vertex AI Custom Training job.

    Args:
        config_path: path to YAML training config
        image_uri: Docker image URI in Artifact Registry
        job_name: optional job display name
        use_spot: use spot (preemptible) instances to reduce cost
    """
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=STAGING_BUCKET)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    display_name = job_name or f"qig-train-{Path(config_path).stem}-{timestamp}"

    machine_type = cfg.get("vertex_machine_type", "a2-highgpu-2g")
    accel_type = cfg.get("vertex_accelerator_type", "NVIDIA_TESLA_A100")
    accel_count = cfg.get("vertex_accelerator_count", 2)
    replica_count = cfg.get("vertex_replica_count", 1)

    # Worker pool spec
    worker_pool_spec = {
        "machine_spec": {
            "machine_type": machine_type,
            "accelerator_type": accel_type,
            "accelerator_count": accel_count,
        },
        "replica_count": replica_count,
        "container_spec": {
            "image_uri": image_uri,
            "args": [
                "python", "-m", "training.train",
                "--config", f"/gcs/qig-training-data/configs/{Path(config_path).name}",
                "--rank", "0",
                "--world_size", str(replica_count * accel_count),
            ],
        },
    }

    if use_spot:
        worker_pool_spec["machine_spec"]["scheduling"] = {
            "provisioning_model": "SPOT"
        }

    job = aiplatform.CustomJob(
        display_name=display_name,
        worker_pool_specs=[worker_pool_spec],
        base_output_dir=cfg.get("output_dir", f"{STAGING_BUCKET}/checkpoints/{display_name}"),
    )

    print(f"Submitting job: {display_name}")
    print(f"  Image: {image_uri}")
    print(f"  Machine: {machine_type} x{accel_count} {accel_type}")
    print(f"  Spot: {use_spot}")
    print(f"  Config: {config_path}")

    job.submit(
        service_account=None,  # uses default Compute SA
        network=None,
        timeout=86400 * 3,  # 3 days max
        restart_job_on_worker_restart=use_spot,
    )

    print(f"Job submitted. Resource name: {job.resource_name}")
    print(f"Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")


def main():
    parser = argparse.ArgumentParser(description="Submit QIG training to Vertex AI")
    parser.add_argument("--config", required=True, help="Path to YAML training config")
    parser.add_argument("--image", required=True, help="Docker image URI")
    parser.add_argument("--name", default=None, help="Job display name")
    parser.add_argument("--spot", action="store_true", help="Use spot instances (60-70% cheaper)")
    args = parser.parse_args()

    submit_job(
        config_path=args.config,
        image_uri=args.image,
        job_name=args.name,
        use_spot=args.spot,
    )


if __name__ == "__main__":
    main()
