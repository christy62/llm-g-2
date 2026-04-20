import json
import time
import pandas as pd


def format_gen_event(event_info, axis):
    """Format target string"""

    vtx = [
        float(event_info['decay_x']),
        float(event_info['decay_y']),
        float(event_info['decay_z'])
    ]

    mom = [
        float(event_info['decay_px']),
        float(event_info['decay_py']),
        float(event_info['decay_pz'])
    ]

    vtx_str = f"[{vtx[0]:.1f}, {vtx[1]:.1f}, {vtx[2]:.1f}]"
    mom_str = f"[{mom[0]:.1f}, {mom[1]:.1f}, {mom[2]:.1f}]"

    return f"gen_event(vtx={vtx_str}, mom={mom_str}, axis='{axis}')"


def simple_conversion_single_csv(
    fileName="/home/christy/helix_generator/json_files/g4_test_3k",
    MAX_HITS=None
):

    # ✅ Load single CSV
    df = pd.read_csv("/home/christy/helix_generator/data_hits/g4_dataset_test_3k.csv")

    training_data = []

    # ✅ Group by event
    for event_id, event_hits in df.groupby("event"):

        # sort hits if you have time column (optional)
        # event_hits = event_hits.sort_values("time")

        if len(event_hits) <= 5:
            continue

        if MAX_HITS is not None:
            event_hits = event_hits.iloc[:MAX_HITS]

        # ✅ decay info (same for all rows in event)
        event_info = event_hits.iloc[0]

        # ------------------------
        # Build axis inputs
        # ------------------------
        x_hits = "[" + ",".join(
            f"{row.hit_x:.1f}" for row in event_hits.itertuples()
        ) + "]"

        y_hits = "[" + ",".join(
            f"{row.hit_y:.1f}" for row in event_hits.itertuples()
        ) + "]"

        z_hits = "[" + ",".join(
            f"{row.hit_z:.1f}" for row in event_hits.itertuples()
        ) + "]"

        # ------------------------
        # Create samples
        # ------------------------
        training_data.append({
            "input": x_hits,
            "target": format_gen_event(event_info, "x")
        })

        training_data.append({
            "input": y_hits,
            "target": format_gen_event(event_info, "y")
        })

        training_data.append({
            "input": z_hits,
            "target": format_gen_event(event_info, "z")
        })

    # ------------------------
    # Save JSON
    # ------------------------
    with open(f"{fileName}.json", "w") as f:
        json.dump(training_data, f, indent=2)

    print(f"{len(training_data)} samples written to {fileName}.json")
    print("Example input:", training_data[0]["input"][:120] + "...")
    print("Example target:", training_data[0]["target"])

    return training_data


if __name__ == "__main__":

    start_time = time.time()

    data = simple_conversion_single_csv(MAX_HITS=None)

    end_time = time.time()

    print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")
    print(f"Total samples generated: {len(data)}")