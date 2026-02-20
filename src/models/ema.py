import math
import torch


@torch.no_grad()
def update_ema(student: torch.nn.Module, teacher: torch.nn.Module, momentum: float):
    """
    Exponential Moving Average update of teacher weights from student weights.

    Formula:  θ_teacher = momentum * θ_teacher + (1 - momentum) * θ_student

    This is the core of what prevents representation collapse in JEPA/BYOL-style
    methods. Instead of computing gradients through the teacher (which creates a
    trivial shortcut), the teacher is a slowly-decaying memory of all past student
    states. This gives the student stable, meaningful targets to predict.

    Args:
        student  : the context encoder being trained by backprop
        teacher  : the target encoder updated only by EMA
        momentum : float in [0, 1]. Higher = slower teacher update.
                   Typical: starts at 0.990, ends at 0.9999
    """
    for param_s, param_t in zip(student.parameters(), teacher.parameters()):
        param_t.data.mul_(momentum).add_((1.0 - momentum) * param_s.data)


def get_ema_momentum(step: int, max_steps: int, m_start: float, m_end: float) -> float:
    """
    Cosine schedule for EMA momentum — increases from m_start to m_end.

    Why schedule momentum?
    ─────────────────────
    Early in training, the student produces poor representations. Using a high
    momentum (0.9999) at this stage means the teacher barely updates from random
    weights, giving the student useless targets.

    By starting at a lower momentum (0.990), the teacher updates more aggressively
    early on, tracking the rapidly-improving student. As training stabilizes, we
    increase momentum so the teacher becomes a smoother, more stable average.

    This follows the cosine schedule used in I-JEPA and BYOL:
        m(t) = m_end - (m_end - m_start) * (cos(π * t/T) + 1) / 2

    Args:
        step      : current training step
        max_steps : total training steps
        m_start   : momentum at step 0    (e.g. 0.990)
        m_end     : momentum at max_steps (e.g. 0.9999)

    Returns:
        current momentum value (float)
    """
    progress = step / max_steps
    cosine_decay = (1.0 + math.cos(math.pi * progress)) / 2.0
    momentum = m_end - (m_end - m_start) * cosine_decay
    return momentum