import torch
import torch.nn.functional as F

from src.models.ema import update_ema, get_ema_momentum
from src.helper import save_checkpoint
from src.utils.logging import MetricLogger
from src.utils.metrics import embedding_variance


def train_loop(
    cfg: dict,
    loader,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    predictor: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    start_step: int = 0,
):
    """
    Main T-JEPA training loop.

    Each step:
        1. Student reads masked tokens  → context latents z_s (B, T, D)
        2. Teacher reads clean tokens   → target latents z_t (B, T, D)  [no grad]
        3. Predictor takes z_s, predicts z_t at masked positions         [N, D]
        4. Loss = MSE(predictor_output, teacher_target) at masked positions only
        5. Backprop through student + predictor, optimizer step
        6. EMA update of teacher from student
        7. Log metrics, save checkpoints

    BF16 mixed precision:
        The L4's Ada tensor cores natively accelerate BF16 matmuls.
        We use torch.autocast("cuda", dtype=torch.bfloat16) to cast
        forward pass ops automatically. Backward pass remains in FP32
        via GradScaler (though BF16 doesn't need a scaler like FP16 does —
        BF16 has the same dynamic range as FP32).

    Args:
        cfg        : config dict
        loader     : DataLoader yielding dicts from TextTransform
        student    : context encoder (trained)
        teacher    : target encoder (EMA-updated)
        predictor  : MLP predictor (trained with student)
        optimizer  : AdamW
        scheduler  : LR scheduler with warmup
        device     : cuda device
        start_step : step to resume from (0 for fresh training)
    """
    student.train()
    predictor.train()
    teacher.eval()

    logger = MetricLogger()

    max_steps = cfg["training"]["max_steps"]
    log_every = cfg["training"]["log_every"]
    save_every = cfg["training"]["save_every"]
    grad_clip = cfg["training"]["grad_clip"]
    m_start = cfg["training"]["ema_momentum_start"]
    m_end = cfg["training"]["ema_momentum_end"]
    use_bf16 = cfg["training"]["precision"] == "bf16"

    step = start_step

    # Infinite loop over dataset — we step-count, not epoch-count
    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break

            # Skip None batches (from empty collate)
            if batch is None:
                continue

            # Move batch tensors to GPU
            clean_ids = batch["clean_ids"].to(device)
            masked_ids = batch["masked_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            mask_positions = batch["mask_positions"].to(device)     # (B, T) bool

            # Skip batch if masking produced zero masked tokens (rare edge case)
            if mask_positions.sum() == 0:
                continue

            # ── Forward pass (BF16 autocast for L4 tensor cores) ─────────────
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):

                # Student: encode masked context
                # z_s shape: (B, T, proj_dim)
                z_s = student(masked_ids, attention_mask)

                # Teacher: encode clean text — no gradient, EMA-updated only
                with torch.no_grad():
                    z_t = teacher(clean_ids, attention_mask)  # (B, T, proj_dim)

                # Predictor: predict teacher latents at masked positions
                # pred shape: (N, proj_dim) where N = total masked tokens
                pred = predictor(z_s, mask_positions)

                # Target: teacher latents at those same masked positions
                # (N, proj_dim) — extract matching positions from teacher output
                target = z_t[mask_positions]  # (N, proj_dim)

                # JEPA loss: MSE in latent space, masked positions only
                # This is the core objective — predict the semantic content
                # of masked spans without ever decoding to token vocabulary
                loss = F.mse_loss(pred, target.detach())

            # ── Backward pass ─────────────────────────────────────────────────
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping — prevents exploding gradients in transformers
            torch.nn.utils.clip_grad_norm_(
                list(student.parameters()) + list(predictor.parameters()),
                max_norm=grad_clip,
            )

            optimizer.step()
            scheduler.step()

            # ── EMA update — teacher tracks student with cosine momentum ──────
            momentum = get_ema_momentum(step, max_steps, m_start, m_end)
            update_ema(student, teacher, momentum)

            # ── Metrics & Logging ─────────────────────────────────────────────
            current_lr = scheduler.get_last_lr()[0]
            logger.update(loss=loss.item(), lr=current_lr, ema_m=momentum)

            if step % log_every == 0:
                # Compute embedding variance to detect representation collapse
                with torch.no_grad():
                    var = embedding_variance(z_t[mask_positions])

                logger.log(step, extra={"emb_var": f"{var:.4f}"})

            # ── Checkpoint ───────────────────────────────────────────────────
            if step > 0 and step % save_every == 0:
                save_checkpoint(step, student, teacher, predictor,
                                optimizer, scheduler, cfg)

            step += 1

    # Final checkpoint
    save_checkpoint(step, student, teacher, predictor, optimizer, scheduler, cfg)
    print(f"\nTraining complete at step {step}.")