from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = Path.home() / "roundabout_ssm"
DATA_DIR = ROOT / "data/raw/data"
OUT_DIR = ROOT / "configs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RECORDING = "02"

bg_path = DATA_DIR / f"{RECORDING}_background.png"
meta_path = DATA_DIR / f"{RECORDING}_recordingMeta.csv"

meta = pd.read_csv(meta_path).iloc[0]
scale = float(meta["orthoPxToMeter"])
location_id = int(meta["locationId"])

img = mpimg.imread(bg_path)

fig, ax = plt.subplots(figsize=(14, 9))
ax.imshow(img)
ax.set_title(
    "Location 0 geometry\n"
    "Click 2 points across each ENTRY LINE.\n"
    "Order: Entry 0 -> Entry 1 -> Entry 2 -> Entry 3 (clockwise)"
)
ax.axis("off")

print()
print("=" * 70)
print("ENTRY LINE DEFINITION")
print("=" * 70)
print("회전교차로 진입선 4개를 시계방향으로 지정합니다.")
print()
print("각 진입선마다 도로를 가로지르는 점 2개를 클릭하세요.")
print("총 8번 클릭:")
print(" Entry 0: 2 points")
print(" Entry 1: 2 points")
print(" Entry 2: 2 points")
print(" Entry 3: 2 points")
print()
print("잘못 클릭했으면 창을 닫고 다시 실행하면 됩니다.")
print("=" * 70)

points = plt.ginput(8, timeout=0)

if len(points) != 8:
    print(f"ERROR: expected 8 points, got {len(points)}")
    raise SystemExit(1)

entries = []

for i in range(4):
    p1_px = points[i * 2]
    p2_px = points[i * 2 + 1]

    # rounD visualization conversion:
    # x_px = x_meter / scale
    # y_px = -y_meter / scale
    p1_world = [
        float(p1_px[0] * scale),
        float(-p1_px[1] * scale)
    ]

    p2_world = [
        float(p2_px[0] * scale),
        float(-p2_px[1] * scale)
    ]

    entries.append({
        "id": i,
        "p1_pixel": [float(p1_px[0]), float(p1_px[1])],
        "p2_pixel": [float(p2_px[0]), float(p2_px[1])],
        "p1_world": p1_world,
        "p2_world": p2_world
    })

    ax.plot(
        [p1_px[0], p2_px[0]],
        [p1_px[1], p2_px[1]],
        linewidth=3
    )

    mx = (p1_px[0] + p2_px[0]) / 2
    my = (p1_px[1] + p2_px[1]) / 2

    ax.text(
        mx, my,
        f"ENTRY {i}",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8)
    )

fig.canvas.draw()

output = {
    "locationId": location_id,
    "referenceRecording": int(RECORDING),
    "orthoPxToMeter": scale,
    "entries": entries
}

json_path = OUT_DIR / "location0_geometry.json"

with open(json_path, "w") as f:
    json.dump(output, f, indent=2)

preview_path = ROOT / "outputs/location0_entry_lines.png"
preview_path.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(preview_path, dpi=200, bbox_inches="tight")

print()
print("Saved:")
print(json_path)
print(preview_path)

print()
for e in entries:
    print(
        f"Entry {e['id']}: "
        f"{e['p1_world']} -> {e['p2_world']}"
    )

plt.show()
