import yaml
import torch

from src.datasets.wikitext import build_wikitext_loader
from src.helper import build_models, build_optimizer
from src.train import train_loop

def main():
    cfg = yaml.safe_load(open("configs/base.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = build_wikitext_loader(cfg)
    student, teacher, predictor = build_models(cfg, device)
    optimizer = build_optimizer(cfg, student, predictor)

    train_loop(cfg, loader, student, teacher, predictor, optimizer, device)

if __name__ == "__main__":
    main()