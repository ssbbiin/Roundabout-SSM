from pathlib import Path
import sys
import time
import torch
from torch.utils.data import DataLoader

ROOT = Path.home() / "roundabout_ssm"
sys.path.insert(0, str(ROOT))

from dataset.torch_dataset import RoundaboutTorchDataset


BATCH_SIZE = 128
NUM_WORKERS = 2
NUM_BATCHES = 100


print("=" * 80)
print("OPTIMIZED DATALOADER BENCHMARK")
print("=" * 80)

ds = RoundaboutTorchDataset(
    split="train",
    history_frames=50,
    max_neighbors=8,
    neighbor_radius=30.0,
    max_cached_recordings=32,
)

print("Dataset size:", len(ds))

loader = DataLoader(
    ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)

it = iter(loader)

# ------------------------------------------------------------
# Warm-up
# Worker 시작 + recording cache 로딩
# ------------------------------------------------------------

print("\nWarming up ...")

warm_start = time.time()

for _ in range(20):
    batch = next(it)

warm_elapsed = time.time() - warm_start

print(
    f"Warm-up 20 batches: "
    f"{warm_elapsed:.2f} sec"
)

# ------------------------------------------------------------
# Actual benchmark
# SAME workers, SAME cache
# ------------------------------------------------------------

print("\nBenchmarking ...")

count = 0
start = time.time()

for i in range(NUM_BATCHES):

    batch = next(it)

    count += batch["agents"].shape[0]

elapsed = time.time() - start

print()
print("=" * 80)
print("RESULT")
print("=" * 80)

print(f"Batches       : {NUM_BATCHES}")
print(f"Samples       : {count}")
print(f"Elapsed       : {elapsed:.2f} sec")
print(f"Samples/sec   : {count / elapsed:.1f}")

epoch_seconds = len(ds) / (count / elapsed)

print(
    f"Estimated pure data time / epoch: "
    f"{epoch_seconds:.1f} sec "
    f"({epoch_seconds/60:.2f} min)"
)

print()
print("Batch shapes:")
print("agents       :", tuple(batch["agents"].shape))
print("agent_mask   :", tuple(batch["agent_mask"].shape))
print("entry_line   :", tuple(batch["entry_line"].shape))

print("=" * 80)
