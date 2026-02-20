#!/bin/bash
# ─────────────────────────────────────────────────────────────
# T-JEPA — GCP L4 Run Script
# Usage: bash run.sh
# ─────────────────────────────────────────────────────────────

set -e   # Exit immediately on any error

# ── 1. Install dependencies ───────────────────────────────────
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install datasets scikit-learn scipy   # eval extras

# ── 2. Smoke test (60 seconds) ───────────────────────────────
echo "Running smoke test..."
python main.py --config configs/debug.yaml

# ── 3. Full training run ──────────────────────────────────────
echo "Starting full training..."
python main.py --config configs/base.yaml

# ── 4. Evaluate after training ───────────────────────────────
echo "Running evaluation..."
python eval/compare.py --ckpt checkpoints/step_050000.pt --quick

# ── 5. Full eval with ablations (optional) ───────────────────
# python eval/compare.py --ckpt checkpoints/step_050000.pt \
#                        --ablation-dir checkpoints/ablations/

echo "Done."