from collections import deque
import time

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class MetricLogger:
    """
    Tracks training metrics with exponential moving averages and logs them.

    Why moving averages?
        Loss at a single step is very noisy. A moving average over the last
        100 steps gives a much cleaner picture of whether training is improving.

    Supports:
        - Console printing (always on)
        - WandB dashboard logging (if wandb is installed and initialized)

    Usage:
        logger = MetricLogger()
        logger.update(loss=0.42, lr=1e-4, ema_m=0.996)
        logger.log(step=100, extra={"emb_var": "0.021"})
    """

    def __init__(self, window: int = 100, use_wandb: bool = False):
        """
        Args:
            window    : rolling window size for smoothed metrics
            use_wandb : whether to push metrics to WandB
        """
        self.window = window
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.metrics = {}          # {name: deque of recent values}
        self.start_time = time.time()

    def update(self, **kwargs):
        """Record new metric values. Call once per training step."""
        for name, value in kwargs.items():
            if name not in self.metrics:
                self.metrics[name] = deque(maxlen=self.window)
            self.metrics[name].append(float(value))

    def _smooth(self, name: str) -> float:
        """Return smoothed (mean over window) value for a metric."""
        vals = self.metrics.get(name, [0.0])
        return sum(vals) / len(vals)

    def log(self, step: int, extra: dict = None):
        """
        Print metrics to console and optionally push to WandB.

        Args:
            step  : current training step
            extra : dict of additional string-formatted metrics to print
                    (e.g. {"emb_var": "0.0212"})
        """
        elapsed = time.time() - self.start_time
        steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0

        # Build log dict for WandB
        log_dict = {name: self._smooth(name) for name in self.metrics}
        log_dict["steps_per_sec"] = steps_per_sec

        # Console output
        smooth_loss = self._smooth("loss")
        smooth_lr = self._smooth("lr")
        smooth_ema = self._smooth("ema_m")

        msg = (
            f"Step {step:6d} | "
            f"Loss {smooth_loss:.4f} | "
            f"LR {smooth_lr:.2e} | "
            f"EMA-m {smooth_ema:.4f} | "
            f"{steps_per_sec:.1f} steps/s"
        )

        if extra:
            for k, v in extra.items():
                msg += f" | {k} {v}"

        print(msg)

        # WandB push
        if self.use_wandb:
            wandb.log(log_dict, step=step)


def init_wandb(cfg: dict, run_name: str = None):
    """
    Initialize a WandB run for experiment tracking.

    Call this at the start of main() if you want dashboard logging.

    Args:
        cfg      : full config dict (logged as WandB config)
        run_name : optional run name (auto-generated if None)
    """
    if not WANDB_AVAILABLE:
        print("WandB not installed. Run: pip install wandb")
        return False

    wandb.init(
        project="t-jepa",
        name=run_name,
        config=cfg,
    )
    print(f"WandB run initialized: {wandb.run.url}")
    return True