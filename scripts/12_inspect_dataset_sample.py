from pathlib import Path
import sys
import numpy as np

ROOT = Path.home() / "roundabout_ssm"
sys.path.insert(0, str(ROOT))

from dataset.roundabout_dataset import RoundaboutDatasetCore


ds = RoundaboutDatasetCore(
    split="train",
    history_frames=50,
    max_neighbors=8,
    neighbor_radius=30.0,
)

print("=" * 80)
print("ROUNDABOUT DATASET SAMPLE TEST")
print("=" * 80)

print("Dataset length:", len(ds))

indices = [
    0,
    len(ds) // 3,
    len(ds) // 2,
    len(ds) - 1,
]

for index in indices:

    sample = ds[index]

    agents = sample["agents"]
    mask = sample["agent_mask"]
    entry_line = sample["entry_line"]

    print()
    print("-" * 80)
    print("INDEX:", index)

    print("Metadata:")
    print(sample["metadata"])

    print()
    print("agents shape     :", agents.shape)
    print("agent_mask shape :", mask.shape)
    print("entry_line shape :", entry_line.shape)

    print()
    print("agent mask:")
    print(mask)

    print(
        "number of agents:",
        int(mask.sum())
    )

    print()
    print("Ego history presence:")
    print(
        int(
            agents[0, :, 8].sum()
        ),
        "/ 50"
    )

    print()
    print("Ego current feature:")
    print(
        np.round(
            agents[0, -1],
            4
        )
    )

    print()
    print("Entry line in Ego coordinates:")
    print(
        np.round(
            entry_line,
            3
        )
    )

    print()
    print("Target:")
    print(sample["target"])

print()
print("=" * 80)
print("EXPECTED")
print("=" * 80)
print("agents shape should be     : (9, 50, 9)")
print("agent_mask shape should be : (9,)")
print("Ego presence should be     : 50 / 50")
print()
print("Ego current rel_x ~ 0")
print("Ego current rel_y ~ 0")
print("Ego current heading:")
print("sin ~ 0")
print("cos ~ 1")
print("=" * 80)
