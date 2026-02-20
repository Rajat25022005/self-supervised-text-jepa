import os
import torch
import torch.distributed as dist


def is_dist_available() -> bool:
    """Check if a distributed process group has been initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """
    Return the rank of the current process.
    - Single GPU / no DDP: returns 0
    - Multi-GPU DDP: returns 0, 1, 2, ... (GPU index)
    """
    if is_dist_available():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """
    Return total number of processes.
    - Single GPU: returns 1
    - Multi-GPU: returns number of GPUs
    """
    if is_dist_available():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """
    Returns True only on rank 0 (the main process).

    Use this to guard operations that should only happen once:
        - Printing logs
        - Saving checkpoints
        - WandB logging
        - Downloading datasets
    """
    return get_rank() == 0


def setup_distributed():
    """
    Initialize the distributed process group for multi-GPU training.

    Call this at the start of main() when using torchrun.
    Uses NCCL backend (fastest for GPU-to-GPU communication on L4).

    On a single L4, this is a no-op — DDP is not initialized.
    On multiple GPUs: run with `torchrun --nproc_per_node=N main_distributed.py`
    """
    if "RANK" not in os.environ:
        # Not launched with torchrun — single GPU mode
        print("Single GPU mode (no distributed training)")
        return

    rank       = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",    # NCCL is fastest for GPU<>GPU comms
        rank=rank,
        world_size=world_size,
    )

    if is_main_process():
        print(f"Distributed training initialized: {world_size} GPUs")


def cleanup_distributed():
    """Destroy the process group at end of training."""
    if is_dist_available():
        dist.destroy_process_group()