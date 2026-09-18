from pathlib import Path
from collections import Counter
import pandas as pd

DATA_DIR = Path.home() / "roundabout_ssm/data/raw/data"

recording_files = sorted(DATA_DIR.glob("*_recordingMeta.csv"))

print("=" * 100)
print("rounD DATASET INSPECTION")
print("=" * 100)

print(f"\nNumber of recordings: {len(recording_files)}")

recording_rows = []
all_classes = Counter()

for rec_file in recording_files:

    rec_id = rec_file.stem.split("_")[0]

    rec_df = pd.read_csv(rec_file)
    meta_df = pd.read_csv(DATA_DIR / f"{rec_id}_tracksMeta.csv")

    r = rec_df.iloc[0]

    class_counts = meta_df["class"].value_counts().to_dict()

    for cls, count in class_counts.items():
        all_classes[cls] += count

    recording_rows.append({
        "recordingId": int(r["recordingId"]),
        "locationId": int(r["locationId"]),
        "frameRate": r["frameRate"],
        "duration_sec": r["duration"],
        "numTracks": int(r["numTracks"]),
        "numVehicles": int(r["numVehicles"]),
        "numVRUs": int(r["numVRUs"]),
        "speedLimit": r["speedLimit"],
        "orthoPxToMeter": r["orthoPxToMeter"],
    })


summary = pd.DataFrame(recording_rows)

print("\n=== RECORDING SUMMARY ===")
print(summary.to_string(index=False))

print("\n=== LOCATION DISTRIBUTION ===")
print(
    summary.groupby("locationId")
    .agg(
        recordings=("recordingId", "count"),
        total_tracks=("numTracks", "sum"),
        vehicles=("numVehicles", "sum"),
        VRUs=("numVRUs", "sum"),
        total_duration_sec=("duration_sec", "sum")
    )
)

print("\n=== FRAME RATE ===")
print(summary["frameRate"].value_counts().sort_index())

print("\n=== TRACK CLASSES ===")
for cls, count in sorted(all_classes.items()):
    print(f"{cls:20s}: {count}")

print("\n=== TOTAL ===")
print("Tracks   :", summary["numTracks"].sum())
print("Vehicles :", summary["numVehicles"].sum())
print("VRUs     :", summary["numVRUs"].sum())
print("Duration :", round(summary["duration_sec"].sum() / 60, 2), "minutes")

print("\n=== RECORDINGS BY LOCATION ===")
for loc, group in summary.groupby("locationId"):
    ids = group["recordingId"].tolist()
    print(f"Location {loc}: {ids}")

print("\n" + "=" * 100)
