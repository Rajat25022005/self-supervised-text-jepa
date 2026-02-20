from datasets import load_dataset
from torch.utils.data import DataLoader

from src.datasets.text_dataset import TextDataset
from src.transforms import TextTransform


def build_wikitext_loader(cfg: dict, split: str = "train") -> DataLoader:
    """
    Build a DataLoader over the WikiText dataset.

    Data pipeline:
        WikiText (HF) → TextDataset → collate_fn (TextTransform) → DataLoader

    The collate function handles:
      - Filtering out empty/too-short strings (WikiText has many section headers)
      - Tokenization (via HuggingFace tokenizer)
      - Span masking (creates masked_ids and mask_positions)

    Args:
        cfg   : config dict loaded from YAML
        split : "train", "validation", or "test"

    Returns:
        DataLoader that yields dicts with keys:
            - "clean_ids"      : (B, T) original token ids
            - "masked_ids"     : (B, T) token ids with spans replaced by [MASK]
            - "attention_mask" : (B, T) 1 for real tokens, 0 for padding
            - "mask_positions" : (B, T) bool, True where [MASK] was applied
    """
    # Load dataset from HuggingFace hub (cached after first download)
    raw = load_dataset(
        cfg["data"]["dataset"],
        cfg["data"]["dataset_config"],
        split=split,
    )

    dataset = TextDataset(raw)
    transform = TextTransform(cfg)

    def collate_fn(batch: list) -> dict:
        # Filter empty lines and WikiText section headers (= Title =)
        # Require at least 20 characters to have meaningful spans to mask
        texts = [x for x in batch if isinstance(x, str) and len(x.strip()) > 20]

        if len(texts) == 0:
            # Edge case: entire batch was empty (rare but possible)
            # Return None and skip in training loop
            return None

        return transform(texts)

    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=(split == "train"),        # Only shuffle training data
        collate_fn=collate_fn,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,                   # Faster CPU→GPU transfer on L4
        drop_last=True,                    # Keep batch sizes uniform
        persistent_workers=True,           # Keep workers alive between epochs (faster)
    )

    return loader