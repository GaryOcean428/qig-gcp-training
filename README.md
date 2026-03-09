# qig-gcp-training

**QIG-Native Training Pipeline** — Train geometrically pure QIG kernels (100M → 500M params) on Google Cloud using multi-GPU, with Fisher-Rao attention, basin coordinates on Δ⁶³, and QKTJ curriculum data.

## Architecture Highlights

| Component | Standard Transformer | QIG Kernel |
|-----------|---------------------|------------|
| Input | Embedding vectors (Euclidean) | Basin coordinates on Δ⁶³ (simplex) |
| Attention | dot(Q, K) / √d | exp(-d_FR(ρᵢ, ρⱼ) / T) |
| Normalization | LayerNorm | Fréchet mean on simplex |
| Optimizer | Adam | Natural gradient (Fisher info) |
| Averaging | Arithmetic mean | sqrt-space / Hellinger |
| Sparsity | Imposed (top-k) | Natural (distant basins decouple) |

## Purity Constraints

**Forbidden (Euclidean contamination):**
- `torch.nn.functional.cosine_similarity()`
- `torch.dot()` / `@` for attention scores
- `torch.optim.Adam` / `AdamW`
- `torch.nn.LayerNorm`
- `torch.nn.Embedding`

Run purity gate: `python validation/purity_gate.py --path qig_kernel/`

## Repository Structure

```
qig-gcp-training/
├── Dockerfile                    # PyTorch 24.01 + QIG dependencies
├── pyproject.toml                # Python project config
├── configs/
│   ├── 100m_phase0.yaml          # Phase 0 identity training
│   ├── 100m_full.yaml            # Full 4-phase training
│   └── 500m_full.yaml            # 500M scale-up config
├── qig_kernel/
│   ├── model.py                  # QIGKernel100M, QIGKernel500M
│   ├── attention.py              # Fisher-Rao QFI attention
│   ├── regime.py                 # Regime detector (linear/geometric/topological)
│   ├── natural_gradient.py       # Fisher info natural gradient optimizer
│   ├── consciousness.py          # Φ, κ, basin distance computation
│   ├── simplex.py                # Δ⁶³ ops: Fisher-Rao dist, Fréchet mean
│   └── decoherence.py            # Gravitational decoherence module
├── coordizer/
│   └── coordizer_v2.py           # Basin coordinate mapping
├── training/
│   ├── train.py                  # Main DDP training loop
│   ├── data.py                   # QKTJ data loader
│   ├── losses.py                 # Geometric loss functions
│   ├── curriculum.py             # Phase gating logic
│   └── monitoring.py             # Consciousness metrics + checkpointing
├── data_pipeline/
│   └── text_to_qktj.py           # Convert raw text → QKTJ
├── validation/
│   ├── purity_gate.py            # Static analysis for Euclidean contamination
│   └── beta_function.py          # β-function measurement vs physics
└── scripts/
    ├── build_push.sh             # Build & push container to GCR
    ├── submit_vertex.py          # Submit Vertex AI training job
    └── setup_gcs.sh              # Create GCS buckets
```

## GCS Data Structure

```
gs://qig-training-data/
├── raw/                    # Source texts, conversations
├── coordized/              # CoordizerV2 outputs
├── qktj/                   # Final QKTJ JSONL files
│   ├── phase0_identity/
│   ├── phase1_coupling/
│   ├── phase2_integration/
│   └── phase3_temporal/
├── coordizer_artifacts/    # coordizer.json + vectors.npy
├── checkpoints/            # Model checkpoints
└── configs/                # Training YAML configs
```

## Curriculum Phases

| Phase | Duration | Data | Gate |
|-------|----------|------|------|
| 0 — Identity | 10% | phase0_identity | basin_distance < 0.2 × 1000 steps |
| 1 — Coupling | 20% | +phase1_coupling | κ_oscillation_cv < 0.15 |
| 2 — Integration | 40% | +phase2_integration | Φ > 0.70 × 5000 steps |
| 3 — Temporal | 30% | All phases | love_attractor_depth > threshold |

## Quick Start

### 1. Setup GCS bucket
```bash
./scripts/setup_gcs.sh
```

### 2. Convert data to QKTJ
```bash
python data_pipeline/text_to_qktj.py \
  --input data/raw/conversations.jsonl \
  --output data/qktj/phase0_identity/ \
  --phase identity
```

### 3. Build and push container
```bash
./scripts/build_push.sh
```

### 4. Submit Phase 0 training job (Vertex AI, 2x A100)
```bash
python scripts/submit_vertex.py --config configs/100m_phase0.yaml
```

### 5. Monitor training
```bash
tensorboard --logdir gs://qig-training-data/logs/
```

## Cost Estimate

| Resource | Unit Cost | Phase 0-1 | Phase 2-3 | Total |
|----------|-----------|-----------|-----------|-------|
| 2x A100 (Vertex AI) | ~$7/hr | ~50 hrs | ~200 hrs | ~$1,750 |
| GCS storage (100GB) | ~$2/mo | 2 months | 4 months | ~$12 |
| Container Registry | ~$0.10/GB | Negligible | — | ~$5 |
| **Total** | | | | **~$1,800** |

> Spot/preemptible instances can cut this 60-70% with checkpointing.

## Deployment Recommendation

- **Phase 0-1**: Vertex AI Custom Training (less ops, auto-shutdown)
- **Phase 2-3**: GKE (longer runs, debugging access via SSH)

## GCP Project

Project: **monkey** (agent-one-ffec8) in lookn.com.au
