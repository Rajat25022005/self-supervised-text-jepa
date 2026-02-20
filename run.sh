# Install the 3 extra deps (all lightweight)
pip install datasets scikit-learn scipy

# Basic: T-JEPA vs DistilBERT only
python eval/compare.py --ckpt checkpoints/step_050000.pt

# Smoke test first (runs in ~2 minutes)
python eval/compare.py --ckpt checkpoints/step_050000.pt --quick

# With ablation checkpoints
python eval/compare.py --ckpt checkpoints/step_050000.pt \
                       --ablation-dir checkpoints/ablations/
```

For ablations to work, you train each config separately and save checkpoints in this structure:
```
checkpoints/ablations/
    mask_10pct/best.pt
    mask_25pct/best.pt
    ema_fixed/best.pt
    span_1/best.pt
    ...
```

---

## What the printed table looks like
```
══════════════════════════════════════════════════════════════════════════
  T-JEPA vs DistilBERT (MLM Baseline) — Performance Comparison

  ── MLM Baseline ─────────────────────────────────────────────────────
  DistilBERT (baseline) │ Pretrained MLM   │ 0.6821 │ 0.5912 │ 0.0909 │ 0.7340 │ 0.51 │ 0.49 │ 0.31

  ── T-JEPA (main run) ────────────────────────────────────────────────
  T-JEPA (main)         │ Teacher @ step_0 │ 0.8103 │ 0.4201 │ 0.3902★│ 0.8120★│ 0.68★│ 0.61★│ 0.44★

  ── Ablations: Masking Ratio ─────────────────────────────────────────
  JEPA (mask=10%)       │ Mask ratio 0.10  │ 0.7612 │ 0.4890 │ 0.2722 │ 0.7810 │ 0.59 │ 0.54 │ 0.38
  JEPA (mask=25%)       │ Mask ratio 0.25  │ 0.7944 │ 0.4412 │ 0.3532 │ 0.8050 │ 0.65 │ 0.59 │ 0.42
  ...
  ★ = best in column