import torch
from torch.optim import AdamW

from src.models.encoder import TextEncoder
from src.models.predictor import Predictor

def build_models(cfg, device):
    student = TextEncoder(cfg["model"]["encoder_name"], cfg["model"]["proj_dim"])
    teacher = TextEncoder(cfg["model"]["encoder_name"], cfg["model"]["proj_dim"])
    predictor = Predictor(cfg["model"]["proj_dim"])

    teacher.load_state_dict(student.state_dict())
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    return student.to(device), teacher.to(device), predictor.to(device)

def build_optimizer(cfg, student, predictor):
    return AdamW(
        list(student.parameters()) + list(predictor.parameters()),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )