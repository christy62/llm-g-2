import json
import time
import pandas as pd


def format_gen_event(event_info, axis):
    """Format the target string in gen_event(...) format with axis."""

    vtx = [
        float(event_info['x_decay']),
        float(event_info['y_decay']),
        float(event_info['z_decay'])
    ]

    mom = [
        float(event_info['px_decay']),
        float(event_info['py_decay']),
        float(event_info['pz_decay'])
    ]

    vtx_str = f"[{vtx[0]:.1f}, {vtx[1]:.1f}, {vtx[2]:.1f}]"
    mom_str = f"[{mom[0]:.1f}, {mom[1]:.1f}, {mom[2]:.1f}]"

    return f"gen_event(vtx={vtx_str}, mom={mom_str}, axis='{axis}')"


def simple_conversion(
    fileName="/home/christy/helix_generator/json_files/testing_50k",
    MAX_HITS=None
):

    events_df = pd.read_csv(
        "/home/christy/helix_generator/decay_events/positron_decay_events_random_50k.csv"
    )

    hits_df = pd.read_csv(
        "/home/christy/helix_generator/data_hits/detector_hits_random_50k.csv"
    )

    training_data = []

    for event_id in events_df['event'].unique():

        event_info = events_df[events_df['event'] == event_id].iloc[0]

        # sort hits in time order
        event_hits = (
            hits_df[hits_df['event'] == event_id]
            .sort_values("time_ns")
        )

        if len(event_hits) <= 5:
            continue

        if MAX_HITS is not None:
            event_hits = event_hits.iloc[:MAX_HITS]

        # axis-specific hit lists
        x_hits = "[" + ",".join(
            f"{row.x:.1f}" for row in event_hits.itertuples()
        ) + "]"

        y_hits = "[" + ",".join(
            f"{row.y:.1f}" for row in event_hits.itertuples()
        ) + "]"

        z_hits = "[" + ",".join(
            f"{row.z:.1f}" for row in event_hits.itertuples()
        ) + "]"

        # create 3 training samples per event
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

    # write JSON
    with open(f"{fileName}.json", "w") as f:
        json.dump(training_data, f, indent=2)

    print(f"{len(training_data)} samples written to {fileName}.json")
    print("Example input:", training_data[0]["input"][:120] + "...")
    print("Example target:", training_data[0]["target"])

    return training_data


if __name__ == "__main__":

    start_time = time.time()

    data = simple_conversion(MAX_HITS=None)

    end_time = time.time()

    print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")
    print(f"Total samples generated: {len(data)}")