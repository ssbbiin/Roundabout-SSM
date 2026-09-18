from pathlib import Path
import json
import random

ROOT = Path.home() / "roundabout_ssm"
OUT = ROOT / "configs/recording_split.json"

recordings = list(range(2, 24))

random.seed(42)
random.shuffle(recordings)

train = sorted(recordings[:16])
val = sorted(recordings[16:19])
test = sorted(recordings[19:])

split = {
    "seed": 42,
    "split_unit": "recording",
    "train": train,
    "val": val,
    "test": test
}

with open(OUT, "w") as f:
    json.dump(split, f, indent=2)

print("=" * 70)
print("RECORDING SPLIT")
print("=" * 70)
print("Train:", train)
print("Val  :", val)
print("Test :", test)
print()
print(f"Saved: {OUT}")
