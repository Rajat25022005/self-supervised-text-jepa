import torch

def span_mask(input_ids, mask_token_id, ratio, min_span, max_span):
    ids = input_ids.clone()
    B, T = ids.shape

    for i in range(B):
        if torch.rand(1) < ratio:
            start = torch.randint(0, T - max_span, (1,))
            length = torch.randint(min_span, max_span + 1, (1,))
            ids[i, start:start + length] = mask_token_id

    return ids