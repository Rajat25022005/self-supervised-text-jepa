import os
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.models.encoder import TextEncoder
from src.models.predictor import Predictor


def build_models(cfg: dict, device: torch.device):
    """
    Instantiate and initialize student encoder, teacher encoder, and predictor.

    Initialization:
        - Student and Teacher start with identical weights (teacher loaded from student)
        - Teacher parameters are frozen — no gradient flows through it
        - Teacher is updated only via EMA in the training loop

    Args:
        cfg    : config dict
        device : torch.device (cuda for L4)

    Returns:
        student   : TextEncoder (trained by backprop)
        teacher   : TextEncoder (updated by EMA only)
        predictor : Predictor (trained by backprop with student)
    """
    student = TextEncoder(
        model_name=cfg["model"]["encoder_name"],
        proj_dim=cfg["model"]["proj_dim"],
    )

    teacher = TextEncoder(
        model_name=cfg["model"]["encoder_name"],
        proj_dim=cfg["model"]["proj_dim"],
    )

    predictor = Predictor(
        proj_dim=cfg["model"]["proj_dim"],
        hidden_dim=cfg["model"]["predictor_hidden_dim"],
        max_len=cfg["data"]["max_length"],
    )

    # Teacher starts with exact same weights as student
    teacher.load_state_dict(student.state_dict())

    # Freeze teacher — EMA update handles its weights, not backprop
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Move everything to GPU
    student = student.to(device)
    teacher = teacher.to(device)
    predictor = predictor.to(device)

    # Count and display parameter counts
    student_params = sum(p.numel() for p in student.parameters()) / 1e6
    pred_params = sum(p.numel() for p in predictor.parameters()) / 1e6
    print(f"Student encoder: {student_params:.1f}M params")
    print(f"Predictor:       {pred_params:.2f}M params")
    print(f"Teacher encoder: {student_params:.1f}M params (frozen, EMA-updated)")

    return student, teacher, predictor


def build_optimizer(cfg: dict, student: torch.nn.Module, predictor: torch.nn.Module):
    """
    Build AdamW optimizer + linear warmup cosine decay LR scheduler.

    Only student and predictor parameters are optimized — teacher is frozen.

    Warmup + cosine decay is standard in SSL pretraining:
        - Warmup: LR rises linearly from 0 to cfg.lr over warmup_steps
          (prevents instability when weights are random early on)
        - Cosine decay: LR falls smoothly from cfg.lr to near 0 by max_steps
          (better final convergence than constant LR)

    Args:
        cfg       : config dict
        student   : student TextEncoder
        predictor : Predictor MLP

    Returns:
        optimizer : AdamW
        scheduler : LambdaLR with warmup + cosine decay
    """
    # Combine parameters from both trainable modules
    params = list(student.parameters()) + list(predictor.parameters())

    optimizer = AdamW(
        params,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        betas=(0.9, 0.95),   # Slightly higher beta2 than default — better for transformers
    )

    warmup_steps = cfg["training"]["warmup_steps"]
    max_steps = cfg["training"]["max_steps"]

    def lr_lambda(step: int) -> float:
        """Returns LR multiplier at given step."""
        if step < warmup_steps:
            # Linear warmup: 0 → 1 over warmup_steps
            return float(step) / float(max(1, warmup_steps))
        else:
            # Cosine decay: 1 → ~0 from warmup_steps to max_steps
            import math
            progress = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def save_checkpoint(
    step: int,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    predictor: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    cfg: dict,
):
    """Save full training state to disk for resuming."""
    ckpt_dir = cfg["training"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"step_{step:06d}.pt")

    torch.save({
        "step": step,
        "student_state_dict": student.state_dict(),
        "teacher_state_dict": teacher.state_dict(),
        "predictor_state_dict": predictor.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "cfg": cfg,
    }, path)

    print(f"Checkpoint saved → {path}")
    return path


def load_checkpoint(path: str, student, teacher, predictor, optimizer, scheduler):
    """Load training state from a checkpoint file for resuming."""
    ckpt = torch.load(path, map_location="cpu")

    student.load_state_dict(ckpt["student_state_dict"])
    teacher.load_state_dict(ckpt["teacher_state_dict"])
    predictor.load_state_dict(ckpt["predictor_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    print(f"Resumed from step {ckpt['step']} ← {path}")
    return ckpt["step"]