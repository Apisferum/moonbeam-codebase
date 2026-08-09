import math
import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset, Sampler
import logging

logger = logging.getLogger(__name__)


class MultiTaskDataset(ConcatDataset):
    def __init__(self, commu_ds, emopia_ds, slakh_ds):
        super().__init__([commu_ds, emopia_ds, slakh_ds])

        # Calculate boundaries for task routing
        c_len = len(commu_ds)
        e_len = len(emopia_ds)
        s_len = len(slakh_ds)

        # Store as (start, end) ranges -> O(1) membership check instead of
        # O(n) `idx in list` scans on every __getitem__ call.
        self.task_bounds = {
            0: (0, c_len),                              # CoMMU
            1: (c_len, c_len + e_len),                  # EMOPIA
            2: (c_len + e_len, c_len + e_len + s_len),  # SLakh
        }

        # Keep task_indices around too (as range objects, not materialized
        # lists) since HomogeneousTaskSampler iterates over the actual
        # index values, not just bounds.
        self.task_indices = {
            task_id: range(start, end)
            for task_id, (start, end) in self.task_bounds.items()
        }

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        
        # Unpack all 3 boundaries
        start0, end0 = self.task_bounds[0] # CoMMU
        start1, end1 = self.task_bounds[1] # EMOPIA
        start2, end2 = self.task_bounds[2] # SLakh
        
        # Explicitly stamp the task_id for ALL 3 datasets
        if start0 <= idx < end0:
            item["task_id"] = 0
        elif start1 <= idx < end1:
            item["task_id"] = 1
        elif start2 <= idx < end2:
            item["task_id"] = 2  # 🚀 BULLETPROOF: Guarantee SLakh gets task_id 2
            
        return item


class DDPSyncedTaskSampler(Sampler):
    """Ensures all GPUs process the SAME task at the SAME step to prevent DDP deadlocks.
    
    NOTE: This is a BATCH sampler (yields lists of indices), so it must be passed
    to DataLoader as `batch_sampler=`, never as `sampler=`.
    """
    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, shuffle=True, seed=42):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0

        self.dataset = dataset
        self.batch_size = batch_size
        self.macro_batch_size = batch_size * num_replicas
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        self.total_macro_batches = 0
        for task_id, indices in self.dataset.task_indices.items():
            n = len(indices) // self.macro_batch_size
            if n == 0:
                logger.warning(
                    f"[DDPSyncedTaskSampler] Task '{task_id}' has {len(indices)} samples, "
                    f"which is fewer than macro_batch_size={self.macro_batch_size} "
                    f"(batch_size={self.batch_size} x num_replicas={self.num_replicas}). "
                    f"This task will produce ZERO batches and be excluded from training entirely."
                )
            self.total_macro_batches += n

        if self.total_macro_batches == 0:
            raise ValueError(
                "DDPSyncedTaskSampler produced 0 total macro-batches. "
                "Check that macro_batch_size <= smallest task's sample count."
            )

    def set_epoch(self, epoch):
        """Must be called at the top of each epoch so shuffling actually changes
        across epochs, and so all ranks re-derive the same seed deterministically."""
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        all_macro_batches = []
        for task_id, indices in self.dataset.task_indices.items():
            indices = list(indices)
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=g).tolist()
                indices = [indices[i] for i in perm]

            for i in range(0, len(indices) - self.macro_batch_size + 1, self.macro_batch_size):
                macro_batch = indices[i : i + self.macro_batch_size]
                all_macro_batches.append(macro_batch)

        if self.shuffle:
            perm = torch.randperm(len(all_macro_batches), generator=g).tolist()
            all_macro_batches = [all_macro_batches[i] for i in perm]

        for macro_batch in all_macro_batches:
            start_idx = self.rank * self.batch_size
            end_idx = start_idx + self.batch_size
            yield macro_batch[start_idx:end_idx]

    def __len__(self):
        return self.total_macro_batches