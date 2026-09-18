from pathlib import Path
import pandas as pd

ROOT = Path.home() / "roundabout_ssm"

DATA = ROOT / "data/raw/data"
SPLITS = ROOT / "data/splits"

HORIZONS = [1.0, 2.0, 3.0, 4.0]


print("=" * 100)
print("FUTURE TRAJECTORY HORIZON ANALYSIS")
print("=" * 100)


for split in ["train", "val", "test"]:

    samples = pd.read_csv(
        SPLITS / f"{split}_samples.csv"
    )

    rows = []

    for rec_id, d in samples.groupby(
        "recordingId"
    ):
        rec_id = int(rec_id)
        rid = f"{rec_id:02d}"

        meta = pd.read_csv(
            DATA / f"{rid}_tracksMeta.csv"
        )

        final_map = dict(
            zip(
                meta["trackId"].astype(int),
                meta["finalFrame"].astype(int)
            )
        )

        dd = d.copy()

        dd["finalFrame"] = (
            dd["trackId"]
            .astype(int)
            .map(final_map)
        )

        rows.append(dd)

    df = pd.concat(
        rows,
        ignore_index=True
    )

    print()
    print("=" * 100)
    print(split.upper())
    print("=" * 100)

    print(
        f"Total samples: {len(df):,}"
    )

    print()

    for horizon in HORIZONS:

        future_frames = (
            horizon
            * df["frameRate"]
        ).round().astype(int)

        future_end = (
            df["currentFrame"]
            + future_frames
        )

        available = (
            future_end
            <= df["finalFrame"]
        )

        # Does this horizon actually include
        # the entry-line crossing?
        contains_entry = (
            df["timeToEntry"]
            <= horizon
        )

        both = (
            available
            & contains_entry
        )

        print(
            f"{horizon:.1f}s future | "
            f"available="
            f"{available.sum():6d} "
            f"({available.mean()*100:6.2f}%) | "
            f"contains entry="
            f"{contains_entry.sum():6d} "
            f"({contains_entry.mean()*100:6.2f}%) | "
            f"available+entry="
            f"{both.sum():6d} "
            f"({both.mean()*100:6.2f}%)"
        )


    print()
    print("=== FUTURE AVAILABILITY BY TTE ===")

    temp = []

    for tte, g in df.groupby(
        "timeToEntry"
    ):

        row = {
            "TTE": tte,
            "N": len(g),
        }

        for horizon in HORIZONS:

            frames = (
                horizon
                * g["frameRate"]
            ).round().astype(int)

            available = (
                g["currentFrame"]
                + frames
                <= g["finalFrame"]
            )

            row[
                f"{horizon:.0f}s_available_%"
            ] = (
                available.mean()
                * 100
            )

        temp.append(row)

    table = pd.DataFrame(temp)

    print(
        table.to_string(
            index=False,
            float_format=lambda x: f"{x:6.2f}"
        )
    )


print()
print("=" * 100)
