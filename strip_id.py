import numpy as np
import pandas as pd


def compute_strip_id_from_csv_row(
    x, y, z,
    NVANE=40,
    NStripPerRow=512,
    SensorOriginX=(90.96, 190.23),
    SensorOriginZ=(0.5, 99.77),
    SensorSize=98.77,
    PosFirstStrip=0.840,
    StripPitch=0.190,
    DoubleRow=True
):
    twopi = 2*np.pi

    if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(z):
        return -1

    # ---------------------------------
    # 1️⃣ Vane ID (manual definition)
    # ---------------------------------
    phi = np.arctan2(y, x)
    if phi < 0:
        phi += twopi

    vaneNum = int((1.0 - phi/twopi) * NVANE) % NVANE

    # ---------------------------------
    # 2️⃣ Rotate into vane-local frame
    # ---------------------------------
    rot = vaneNum / NVANE * twopi
    c, s = np.cos(rot), np.sin(rot)

    xr = c*x + s*y
    zr = z

    if xr < 0:
        return -1

    # ---------------------------------
    # 3️⃣ Determine sensor block
    # ---------------------------------

    # Radial block (0=inner, 1=outer)
    radial_block = 0 if xr < SensorOriginX[1] else 1

    # Vertical block (0=near center, 1=far)
    vertical_block = 0 if abs(zr) < SensorOriginZ[1] else 1

    # Top / Bottom
    isTop = zr > 0

    # ---------------------------------
    # 4️⃣ Determine R or Z plane
    # ---------------------------------
    

    isRstrip = xr < (SensorOriginX[0] + SensorSize)

    # ---------------------------------
    # 5️⃣ Correct sensorID mapping (Figure 6)
    # ---------------------------------

    if isRstrip:
        # R sensors
        base = 0 if isTop else 8
    else:
        # Z sensors
        base = 4 if isTop else 12

    # block index inside that group
    block_index = vertical_block*2 + radial_block

    sensorID = base + block_index

    # ---------------------------------
    # 6️⃣ Position inside sensor
    # ---------------------------------
    xs = xr - SensorOriginX[radial_block]
    ys = abs(zr) - SensorOriginZ[vertical_block]

    if not isRstrip:
        xs, ys = ys, xs

    # ---------------------------------
    # 7️⃣ Strip index
    # ---------------------------------
    idxStrip = int((xs - PosFirstStrip) / StripPitch + 0.5)

    if DoubleRow:
        idxRow = 1 if ys > SensorSize * 0.5 else 0
        idxStrip += idxRow * NStripPerRow

    maxStrips = (2 if DoubleRow else 1) * NStripPerRow
    if idxStrip < 0 or idxStrip >= maxStrips:
        return -1

    # ---------------------------------
    # 8️⃣ Pack strip ID (32-bit format)
    # ---------------------------------
    stripID = (
        idxStrip
        + (sensorID << 16)
        + (vaneNum << 24)
    )

    return stripID


# ---------------------------------
# Apply to CSV  + Minimal StripDigi
# ---------------------------------
if __name__ == "__main__":

    TIME_DIV = 5.0        # ns (from Table 12)
    E_THRESHOLD = 0.03    # MeV ≈ 0.3 MIP (approximate)

    hits = pd.read_csv("/home/christy/helix_generator/data_hits/detector_hits_33k.csv")

    # -------------------------------------------------
    # 1️⃣ Compute strip ID
    # -------------------------------------------------
    hits["strip_id"] = hits.apply(
        lambda r: compute_strip_id_from_csv_row(r.x, r.y, r.z),
        axis=1
    )

    # Remove invalid strips
    hits = hits[hits["strip_id"] >= 0]

    # -------------------------------------------------
    # 2️⃣ Apply energy threshold
    # -------------------------------------------------
    hits = hits[hits["cum_energy_MeV"] > E_THRESHOLD]

    # -------------------------------------------------
    # 3️⃣ Digitize time (5 ns bins)
    # -------------------------------------------------
    hits["time_bin"] = (hits["time_ns"] // TIME_DIV).astype(int)

    # -------------------------------------------------
    # 4️⃣ Stack hits in same strip + same time bin
    # -------------------------------------------------
    digitized = (
        hits
        .groupby(["event", "strip_id", "time_bin"])
        .agg({
            "cum_energy_MeV": "sum"
        })
        .reset_index()
    )

    # -------------------------------------------------
    # 5 Optional: Add ToT (100 ns at 1 MIP)
    # -------------------------------------------------
    # MIP_ENERGY = 0.12  # MeV approx in silicon
    # digitized["ToT_ns"] = 100.0 * (digitized["cum_energy_MeV"] / MIP_ENERGY)
    # digitized["ToT_bin"] = (digitized["ToT_ns"] // TIME_DIV).astype(int)

    # Save digitized strip hits
    digitized.to_csv("/home/christy/helix_generator/strip_ids/digitized_strip_hits_33k.csv", index=False)

    print("Saved digitized_strip_hits_33k.csv")