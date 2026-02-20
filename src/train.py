import torch
import torch.nn.functional as F
from src.models.ema import update_ema

def train_loop(cfg, loader, student, teacher, predictor, optimizer, device):
    student.train()
    predictor.train()

    step = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        zs = student(batch["masked_ids"], batch["attention_mask"])
        with torch.no_grad():
            zt = teacher(batch["clean_ids"], batch["attention_mask"])

        pred = predictor(zs)
        loss = F.mse_loss(pred, zt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        update_ema(student, teacher, cfg["training"]["ema_momentum"])

        if step % 100 == 0:
            print(f"Step {step} | Loss {loss.item():.4f}")

        step += 1
        if step >= cfg["training"]["max_steps"]:
            break