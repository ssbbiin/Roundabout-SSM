from pathlib import Path
import pandas as pd
import json

ROOT = Path.home() / "roundabout_ssm"
DATA = ROOT / "data/raw/data"
GEOM = ROOT / "configs/location0_geometry.json"

with open(GEOM) as f:
    geom = json.load(f)

print("=" * 100)
print("LOCATION 0 COORDINATE VALIDATION")
print("=" * 100)

rows = []

for rec in range(2, 24):
    rid = f"{rec:02d}"

    meta = pd.read_csv(DATA / f"{rid}_recordingMeta.csv").iloc[0]
    tracks = pd.read_csv(
        DATA / f"{rid}_tracks.csv",
        usecols=["xCenter", "yCenter"]
    )

    rows.append({
        "recording": rid,
        "locationId": int(meta["locationId"]),
        "scale": float(meta["orthoPxToMeter"]),
        "xUtmOrigin": float(meta["xUtmOrigin"]),
        "yUtmOrigin": float(meta["yUtmOrigin"]),
        "xMin": tracks["xCenter"].min(),
        "xMax": tracks["xCenter"].max(),
        "yMin": tracks["yCenter"].min(),
        "yMax": tracks["yCenter"].max(),
    })

df = pd.DataFrame(rows)

print("\n=== RECORDINGS ===")
print(df.to_string(index=False))

print("\n=== UNIQUE VALUES ===")
print("locationId:")
print(df["locationId"].unique())

print("\northoPxToMeter:")
print(df["scale"].unique())

print("\nxUtmOrigin:")
print(df["xUtmOrigin"].unique())

print("\nyUtmOrigin:")
print(df["yUtmOrigin"].unique())

print("\n=== ENTRY LINES (WORLD COORDINATES) ===")
for e in geom["entries"]:
    print(
        f"Entry {e['id']}: "
        f"{e['p1_world']} -> {e['p2_world']}"
    )

print("\n=== CHECK ===")

same_location = df["locationId"].nunique() == 1
same_scale = df["scale"].nunique() == 1
same_x_origin = df["xUtmOrigin"].nunique() == 1
same_y_origin = df["yUtmOrigin"].nunique() == 1

print("Same location :", same_location)
print("Same scale    :", same_scale)
print("Same X origin :", same_x_origin)
print("Same Y origin :", same_y_origin)

if same_location and same_scale and same_x_origin and same_y_origin:
    print("\nRESULT: geometry can be shared across recordings 02~23.")
else:
    print("\nRESULT: coordinate systems differ. Do NOT share geometry yet.")

print("=" * 100)
