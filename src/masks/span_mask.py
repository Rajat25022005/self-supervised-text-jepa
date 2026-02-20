import torch


def span_mask(input_ids, mask_token_id, ratio, min_span, max_span, num_spans=4):
    """
    Apply contiguous span masking to a batch of token sequences.

    This is the core data augmentation step for T-JEPA. For each sentence,
    we randomly select `num_spans` non-overlapping contiguous regions and
    replace them with [MASK] tokens. The total tokens masked targets `ratio`
    of the sequence length.

    Args:
        input_ids   : (B, T) tensor of token ids — WILL BE MODIFIED (clone first)
        mask_token_id: int — the tokenizer's [MASK] token id
        ratio       : float — target fraction of tokens to mask (e.g. 0.15)
        min_span    : int — minimum span length
        max_span    : int — maximum span length
        num_spans   : int — number of spans to mask per sequence

    Returns:
        masked_ids     : (B, T) — masked version of input_ids
        mask_positions : (B, T) — bool tensor, True where tokens were masked
    """
    ids = input_ids.clone()
    B, T = ids.shape
    mask_positions = torch.zeros(B, T, dtype=torch.bool)

    for i in range(B):
        # Track which positions are already masked to avoid overlap
        occupied = torch.zeros(T, dtype=torch.bool)

        # Special tokens: [CLS] is position 0, [SEP] is last real token.
        # Never mask these — they anchor the sequence structure.
        # Position 0 = [CLS], we protect first and last token.
        occupied[0] = True
        occupied[T - 1] = True

        masked_count = 0
        target_count = int(T * ratio)

        for _ in range(num_spans):
            if masked_count >= target_count:
                break

            # Try to place a span — retry a few times if it overlaps
            placed = False
            for _attempt in range(15):
                length = torch.randint(min_span, max_span + 1, (1,)).item()
                # Start at least 1 from boundary, end at least 1 from boundary
                if T - length - 1 <= 1:
                    break
                start = torch.randint(1, T - length - 1, (1,)).item()
                end = start + length

                # Check if this region is free
                if not occupied[start:end].any():
                    ids[i, start:end] = mask_token_id
                    mask_positions[i, start:end] = True
                    occupied[start:end] = True
                    masked_count += length
                    placed = True
                    break

    return ids, mask_positions