import json
import time
import pandas as pd


def format_gen_event(event_info):
    """Format the target string in gen_event(...) format."""

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

    return f"gen_event(v={vtx_str}, p={mom_str})"


def helix_to_vertex_conversion(
    fileName="/home/christy/helix_generator/json_files/helix_to_vertex_dataset"
):

    # true decay info
    events_df = pd.read_csv(
        "/home/christy/helix_generator/decay_events/positron_decay_events_100k.csv"
    )

    # helix parameters you generated
    helix_df = pd.read_csv(
        "/home/christy/helix_generator/data_hits/belle_params_detector_accepted_100k.csv"
    )

    training_data = []

    for row in helix_df.itertuples():

        event_id = int(row.event)

        event_info = events_df[events_df['event'] == event_id].iloc[0]

        # helix parameters as input
        helix_input = [
            round(row.d0, 1),
            round(row.z0, 1),
            round(row.phi0, 2),
            round(row.omega, 4),
            round(row.tan_lambda, 2)
        ]

        training_data.append({
            "input": helix_input,
            "target": format_gen_event(event_info)
        })

    # write JSON
    with open(f"{fileName}.json", "w") as f:
        json.dump(training_data, f, indent=2)

    print(f"{len(training_data)} samples written to {fileName}.json")
    print("Example input:", training_data[0]["input"])
    print("Example target:", training_data[0]["target"])

    return training_data


if __name__ == "__main__":

    start_time = time.time()

    data = helix_to_vertex_conversion()

    end_time = time.time()

    print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")
    print(f"Total samples generated: {len(data)}")