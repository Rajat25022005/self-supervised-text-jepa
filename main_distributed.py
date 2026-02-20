"""
Distributed training entry point for T-JEPA.

Usage (single L4):
    python main.py  # Use this — distributed is overkill for 1 GPU

Usage (multiple GPUs on one machine, e.g. A3 instance with 8xH100):
    torchrun --nproc_per_node=8 main_distributed.py --config configs/base.yaml

Usage (multi-node on GCP with SLURM):
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=$RANK \
             --master_addr=$MASTER_ADDR --master_port=29500 \
             main_distributed.py --config configs/base.yaml

Notes for GCP:
    - For a single L4 instance, just use main.py
    - DDP becomes useful if you upgrade to a multi-GPU instance (A100/H100)
    - Batch size in config is per-GPU; effective batch = batch_size * world_size
    - NCCL backend is used automatically (fastest for NVLink/InfiniBand)
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from src.datasets.wikitext import build_wikitext_loader
from src.helper import build_models, build_optimizer, load_checkpoint
from src.train import train_loop
from src.utils.distributed import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_rank,
    get_world_size,
)
from src.utils.logging import init_wandb


def parse_args():
    parser = argparse.ArgumentParser(description="T-JEPA Distributed Pretraining")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize distributed process group (no-op on single GPU)
    setup_distributed()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # On multi-GPU, each process uses its assigned GPU
    local_rank = int(__import__("os").environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if is_main_process():
        world_size = get_world_size()
        effective_batch = cfg["training"]["batch_size"] * world_size
        print(f"Distributed training: {world_size} GPUs")
        print(f"Per-GPU batch: {cfg['training']['batch_size']} | "
              f"Effective batch: {effective_batch}")

        if args.wandb:
            init_wandb(cfg, run_name=args.run_name)

    # Build models
    student, teacher, predictor = build_models(cfg, device)

    # Wrap student and predictor in DDP — teacher is never wrapped (frozen)
    if get_world_size() > 1:
        student = DDP(student, device_ids=[local_rank])
        predictor = DDP(predictor, device_ids=[local_rank])

    # Access underlying module for EMA (DDP wraps the module)
    student_module = student.module if isinstance(student, DDP) else student
    predictor_module = predictor.module if isinstance(predictor, DDP) else predictor

    optimizer, scheduler = build_optimizer(cfg, student_module, predictor_module)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(
            args.resume, student_module, teacher, predictor_module,
            optimizer, scheduler
        )

    # Build dataloader — DistributedSampler ensures each GPU sees different data
    loader = build_wikitext_loader(cfg)

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

    cleanup_distributed()


if __name__ == "__main__":
    main()