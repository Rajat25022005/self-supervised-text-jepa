from datasets import load_dataset
from torch.utils.data import DataLoader
from src.datasets.text_dataset import TextDataset
from src.transforms import TextTransform

def build_wikitext_loader(cfg):
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    dataset = TextDataset(raw)
    transform = TextTransform(cfg)

    def collate(batch):
        batch = [x for x in batch if len(x) > 20]
        return transform(batch)

    return DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate,
        drop_last=True,
    )