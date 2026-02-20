from transformers import AutoTokenizer
from src.masks.span_mask import span_mask

class TextTransform:
    def __init__(self, cfg):
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])
        self.max_length = cfg["data"]["max_length"]
        self.mask_cfg = cfg["mask"]

    def __call__(self, texts):
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        masked_ids = span_mask(
            tokens["input_ids"],
            self.tokenizer.mask_token_id,
            self.mask_cfg["ratio"],
            self.mask_cfg["min_span"],
            self.mask_cfg["max_span"],
        )

        return {
            "clean_ids": tokens["input_ids"],
            "masked_ids": masked_ids,
            "attention_mask": tokens["attention_mask"],
        }