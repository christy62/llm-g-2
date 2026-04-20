import json
import time
import pandas as pd

def format_gen_event(event_info):
    """Format the target string in gen_event(...) format."""

    vtx = [
        float(event_info['x_decay']) ,
        float(event_info['y_decay']) ,
        float(event_info['z_decay']) 
    ]

    mom = [
        float(event_info['px_decay']),
        float(event_info['py_decay']),
        float(event_info['pz_decay'])
    ]

    vtx_str = f"[{vtx[0]:.2f}, {vtx[1]:.2f}, {vtx[2]:.2f}]"
    mom_str = f"[{mom[0]:.2f}, {mom[1]:.2f}, {mom[2]:.2f}]"

    return f"gen_event(vtx={vtx_str}, mom={mom_str})"


def simple_conversion(fileName="/home/christy/helix_generator/json_files/new_dataset_mcs_1", MAX_HITS=None):

    events_df = pd.read_csv("/home/christy/helix_generator/decay_events/positron_decay_events_50k.csv")
    hits_df = pd.read_csv("/home/christy/helix_generator/data_hits/detector_hits_mcs_50k.csv")

    training_data = []

    for event_id in events_df['event'].unique():

        event_info = events_df[events_df['event'] == event_id].iloc[0]

        event_hits = (
            hits_df[hits_df['event'] == event_id]
            .sort_values("time_bin")
        )

        if len(event_hits) <= 5:  # Skip events with 5 or fewer hits
            continue

        if MAX_HITS is not None:
            event_hits = event_hits.iloc[:MAX_HITS]

        # Build time-ordered input string
        input_str = "[" + ",".join(
            f"{row.strip_id}"
            for row in event_hits.itertuples()
        ) + "]"

        target_str = format_gen_event(event_info)

        training_data.append({
            "input": input_str,
            "target": target_str
        })

    with open(f"{fileName}.json", "w") as f:
        json.dump(training_data, f, indent=2)

    print(f"{len(training_data)} events written to {fileName}.json")
    print("Example input:", training_data[0]["input"][:120] + "...")
    print("Example target:", training_data[0]["target"])

    return training_data


if __name__ == "__main__":

    RMIN, RMAX = 70.0, 290.0
    ZMIN, ZMAX = -200.0, 200.0

    start_time = time.time()
    data = simple_conversion(MAX_HITS=None)  
    end_time = time.time()

    print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")
    print(f"Total events processed: {len(data)}")