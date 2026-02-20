import torch
import torch.nn.functional as F

def cosine_sim(a, b):
    return F.cosine_similarity(a, b, dim=-1)