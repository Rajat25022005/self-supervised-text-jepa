import argparse
import yaml
import torch

from src.datasets.wikitext import build_wikitext_loader
from src.helper import build_models, build_optimizer, load_checkpoint
from src.train import train_loop
from src.utils.distributed import is_main_process
from src.utils.logging import init_wandb


def parse_args():
    parser = argparse.ArgumentParser(description="T-JEPA Pretraining")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML config file (default: configs/base.yaml)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint .pt file to resume training from",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable WandB experiment tracking",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="WandB run name (auto-generated if not set)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if is_main_process():
        print(f"\n{'='*60}")
        print(f"  T-JEPA Pretraining")
        print(f"  Config   : {args.config}")
        print(f"  Encoder  : {cfg['model']['encoder_name']}")
        print(f"  Latent D : {cfg['model']['proj_dim']}")
        print(f"  Max Steps: {cfg['training']['max_steps']:,}")
        print(f"  Batch    : {cfg['training']['batch_size']}")
        print(f"  Precision: {cfg['training']['precision']}")
        print(f"{'='*60}\n")

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_main_process():
        if device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
        else:
            print("WARNING: No GPU found, running on CPU (very slow)")

    # ── WandB ─────────────────────────────────────────────────────────────────
    if args.wandb and is_main_process():
        init_wandb(cfg, run_name=args.run_name)

    # ── Data ──────────────────────────────────────────────────────────────────
    if is_main_process():
        print("Loading dataset...")
    loader = build_wikitext_loader(cfg)
    if is_main_process():
        print(f"Dataset loaded: {len(loader.dataset):,} samples\n")

    # ── Models ────────────────────────────────────────────────────────────────
    student, teacher, predictor = build_models(cfg, device)

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    optimizer, scheduler = build_optimizer(cfg, student, predictor)

    # ── Resume from checkpoint (optional) ─────────────────────────────────────
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(
            args.resume, student, teacher, predictor, optimizer, scheduler
        )
        # Move models back to device after loading (load_checkpoint uses CPU map)
        student = student.to(device)
        teacher = teacher.to(device)
        predictor = predictor.to(device)

    # ── Train ─────────────────────────────────────────────────────────────────
    if is_main_process():
        print("Starting training...\n")

    train_loop(
        cfg=cfg,
        loader=loader,
        student=student,
        teacher=teacher,
        predictor=predictor,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        start_step=start_step,
    )


if __name__ == "__main__":
    main()