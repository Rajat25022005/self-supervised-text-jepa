import torch

@torch.no_grad()
def update_ema(student, teacher, momentum):
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(momentum).add_((1.0 - momentum) * ps.data)