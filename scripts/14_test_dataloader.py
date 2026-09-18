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


print("=" * 80)
print("PYTORCH DATALOADER TEST")
print("=" * 80)

ds = RoundaboutTorchDataset(
    split="train",
    history_frames=50,
    max_neighbors=8,
    neighbor_radius=30.0,
)

print("Dataset size:", len(ds))

loader = DataLoader(
    ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)

print()
print("Loading first batch ...")

t0 = time.time()

batch = next(iter(loader))

elapsed = time.time() - t0

print(f"First batch time: {elapsed:.3f} sec")

print()
print("=== SHAPES ===")

for key, value in batch.items():
    print(
        f"{key:18s}: "
        f"{tuple(value.shape)} "
        f"{value.dtype}"
    )


print()
print("=== BASIC CHECK ===")

print(
    "GO count  :",
    int((batch["decision"] == 1).sum())
)

print(
    "WAIT count:",
    int((batch["decision"] == 0).sum())
)

print(
    "TTE range :",
    float(batch["time_to_entry"].min()),
    "~",
    float(batch["time_to_entry"].max())
)

print(
    "Agents/sample:",
    batch["agent_mask"].sum(dim=1)[:10]
)


# ------------------------------------------------------------
# Throughput benchmark
# ------------------------------------------------------------

print()
print("=== DATALOADER BENCHMARK ===")

benchmark_loader = DataLoader(
    ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)

NUM_BATCHES = 100

count = 0
start = time.time()

for i, b in enumerate(benchmark_loader):

    count += b["agents"].shape[0]

    if i + 1 >= NUM_BATCHES:
        break

elapsed = time.time() - start

print(
    f"Batches : {NUM_BATCHES}"
)

print(
    f"Samples : {count}"
)

print(
    f"Time    : {elapsed:.2f} sec"
)

print(
    f"Speed   : {count / elapsed:.1f} samples/sec"
)

print()
print("=" * 80)
