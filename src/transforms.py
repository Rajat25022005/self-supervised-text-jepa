import torch
from transformers import AutoTokenizer

from src.masks.span_mask import span_mask


class TextTransform:
    """
    Transforms a list of raw text strings into a training-ready batch dict.

    This is the glue between raw text and model inputs. It runs inside the
    DataLoader's collate function (on CPU, in worker processes), so it needs
    to be fast. HuggingFace's fast tokenizer (Rust-backed) handles this well.

    Steps:
        1. Tokenize batch of strings → input_ids, attention_mask
        2. Apply span masking → masked_ids, mask_positions
        3. Return a dict with both clean and masked versions

    The reason we keep clean_ids alongside masked_ids:
        - Student encoder reads masked_ids (sees [MASK] tokens)
        - Teacher encoder reads clean_ids (sees the original tokens)
        - This is the fundamental JEPA setup: student learns from context,
          teacher provides clean semantic targets
    """

    def __init__(self, cfg: dict):
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg["model"]["encoder_name"],
            use_fast=True,      # Rust tokenizer — much faster than Python fallback
        )
        self.max_length = cfg["data"]["max_length"]
        self.mask_cfg = cfg["mask"]

    def __call__(self, texts: list) -> dict:
        """
        Args:
            texts : list of raw text strings (one per sample in the batch)

        Returns:
            dict with keys:
                clean_ids      : (B, T) LongTensor — original token ids
                masked_ids     : (B, T) LongTensor — ids with [MASK] spans
                attention_mask : (B, T) LongTensor — 1 for real, 0 for padding
                mask_positions : (B, T) BoolTensor — True where masking applied
        """
        # Tokenize: pad to max_length, truncate long sequences
        tokens = self.tokenizer(
            texts,
            padding="max_length",   # Pad all to same length for batch uniformity
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",    # Return PyTorch tensors directly
        )

        clean_ids = tokens["input_ids"]           # (B, T)
        attention_mask = tokens["attention_mask"] # (B, T)

        # Apply span masking to produce student input
        # masked_ids  : (B, T) with [MASK] tokens inserted
        # mask_positions: (B, T) bool marking exactly where masks are
        masked_ids, mask_positions = span_mask(
            input_ids=clean_ids,
            mask_token_id=self.tokenizer.mask_token_id,
            ratio=self.mask_cfg["ratio"],
            min_span=self.mask_cfg["min_span"],
            max_span=self.mask_cfg["max_span"],
            num_spans=self.mask_cfg["num_spans"],
        )

        # Sanity check: ensure mask_positions only marks real tokens (not padding)
        # Padding positions should never be counted as masked
        mask_positions = mask_positions & attention_mask.bool()

        return {
            "clean_ids": clean_ids,               # Teacher reads this
            "masked_ids": masked_ids,             # Student reads this
            "attention_mask": attention_mask,     # Both use this
            "mask_positions": mask_positions,     # Loss computed only here
        }