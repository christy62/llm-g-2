import numpy as np
import pandas as pd


# -------------------------------------------------------
# Strip ID computation (unchanged physics)
# -------------------------------------------------------
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

    # -------------------------
    # 1️⃣ Vane ID
    # -------------------------
    phi = np.arctan2(y, x)
    if phi < 0:
        phi += twopi

    vaneNum = int((1.0 - phi/twopi) * NVANE) % NVANE

    # -------------------------
    # 2️⃣ Rotate to vane frame
    # -------------------------
    rot = vaneNum / NVANE * twopi
    c, s = np.cos(rot), np.sin(rot)

    xr = c*x + s*y
    zr = z

    if xr < 0:
        return -1

    # -------------------------
    # 3️⃣ Sensor block
    # -------------------------
    radial_block = 0 if xr < SensorOriginX[1] else 1
    vertical_block = 0 if abs(zr) < SensorOriginZ[1] else 1
    isTop = zr > 0

    # -------------------------
    # 4️⃣ R/Z plane
    # -------------------------
    isRstrip = xr < (SensorOriginX[0] + SensorSize)

    # -------------------------
    # 5️⃣ Sensor ID
    # -------------------------
    if isRstrip:
        base = 0 if isTop else 8
    else:
        base = 4 if isTop else 12

    block_index = vertical_block*2 + radial_block
    sensorID = base + block_index

    # -------------------------
    # 6️⃣ Local sensor coords
    # -------------------------
    xs = xr - SensorOriginX[radial_block]
    ys = abs(zr) - SensorOriginZ[vertical_block]

    if not isRstrip:
        xs, ys = ys, xs

    # -------------------------
    # 7️⃣ Strip index
    # -------------------------
    idxStrip = int((xs - PosFirstStrip) / StripPitch + 0.5)

    if DoubleRow:
        idxRow = 1 if ys > SensorSize * 0.5 else 0
        idxStrip += idxRow * NStripPerRow

    maxStrips = 2 * NStripPerRow

    if idxStrip < 0 or idxStrip >= maxStrips:
        return -1

    # -------------------------
    # 8️⃣ Pack stripID
    # -------------------------
    stripID = (
        idxStrip
        + (sensorID << 16)
        + (vaneNum << 24)
    )

    return stripID


# -------------------------------------------------------
# Decode strip ID → (vane, sensor, strip)
# -------------------------------------------------------
def decode_strip_id(stripID):

    vane = (stripID >> 24) & 0xFF
    sensor = (stripID >> 16) & 0xFF
    strip = stripID & 0xFFFF

    return vane, sensor, strip


# -------------------------------------------------------
# Convert stripID → LLM token
# -------------------------------------------------------
def strip_token(stripID):

    vane, sensor, strip = decode_strip_id(stripID)

    return f"{vane}_{sensor}_{strip}"



# -------------------------------------------------------
# Main processing
# -------------------------------------------------------
if __name__ == "__main__":

    TIME_DIV = 5.0
    E_THRESHOLD = 0.03

    hits = pd.read_csv(
        "/home/christy/helix_generator/data_hits/detector_hits_33k.csv"
    )

    # -------------------------------------------------
    # 1️⃣ Compute strip IDs
    # -------------------------------------------------
    hits["strip_id"] = hits.apply(
        lambda r: compute_strip_id_from_csv_row(r.x, r.y, r.z),
        axis=1
    )

    hits = hits[hits["strip_id"] >= 0]

    # -------------------------------------------------
    # 2️⃣ Energy threshold
    # -------------------------------------------------
    hits = hits[hits["cum_energy_MeV"] > E_THRESHOLD]

    # -------------------------------------------------
    # 3️⃣ Digitize time
    # -------------------------------------------------
    hits["time_bin"] = (hits["time_ns"] // TIME_DIV).astype(int)

    # -------------------------------------------------
    # 4️⃣ Stack hits (same strip + same time)
    # -------------------------------------------------
    digitized = (
        hits
        .groupby(["event", "strip_id", "time_bin"])
        .agg({"cum_energy_MeV": "sum"})
        .reset_index()
    )

    # convert strip id → token
    digitized["strip_token"] = digitized["strip_id"].apply(strip_token)

    # sort hits inside each event by time
    digitized = digitized.sort_values(["event", "time_bin"])

    # group tokens per event (time ordered)
    event_tokens = (
        digitized
        .groupby("event")["strip_token"]
        .apply(lambda x: list(dict.fromkeys(x)))  # remove duplicates but keep order
        .reset_index()
    )

    # convert list → string
    event_tokens["input"] = event_tokens["strip_token"].apply(lambda x: " ".join(x))

    event_tokens = event_tokens.drop(columns=["strip_token"])

    # -------------------------------------------------
    # Remove events with < 5 hits
    # -------------------------------------------------
    event_tokens = event_tokens[
        event_tokens["input"].str.split().str.len() >= 5
    ]

    # -------------------------------------------------
    # 7️⃣ Save LLM dataset
    # -------------------------------------------------
    event_tokens.to_csv(
        "/home/christy/helix_generator/strip_ids/vane_sensor_strip_33k.csv",
        index=False
    )

    print("Saved vane_sensor_strip_33k.csv")