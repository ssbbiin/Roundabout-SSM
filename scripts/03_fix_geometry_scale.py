from pathlib import Path
import json

ROOT = Path.home() / "roundabout_ssm"
GEOM = ROOT / "configs/location0_geometry.json"

SCALE_DOWN_FACTOR = 10

with open(GEOM, "r") as f:
    data = json.load(f)

scale = float(data["orthoPxToMeter"])

print("=" * 80)
print("FIX LOCATION 0 GEOMETRY SCALE")
print("=" * 80)

for entry in data["entries"]:
    p1 = entry["p1_pixel"]
    p2 = entry["p2_pixel"]

    entry["p1_world"] = [
        float(p1[0] * SCALE_DOWN_FACTOR * scale),
        float(-p1[1] * SCALE_DOWN_FACTOR * scale)
    ]

    entry["p2_world"] = [
        float(p2[0] * SCALE_DOWN_FACTOR * scale),
        float(-p2[1] * SCALE_DOWN_FACTOR * scale)
    ]

data["scaleDownFactor"] = SCALE_DOWN_FACTOR

with open(GEOM, "w") as f:
    json.dump(data, f, indent=2)

print("\nCorrected coordinates:")

for entry in data["entries"]:
    print(
        f"Entry {entry['id']}: "
        f"{entry['p1_world']} -> {entry['p2_world']}"
    )

print(f"\nSaved: {GEOM}")
print("=" * 80)
