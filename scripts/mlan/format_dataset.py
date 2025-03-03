import json
import multiprocessing
import copy
import argparse

from typing import Dict, List, Any

END_FORMAT: Dict[str, List[Dict[str, List[Any]]]] = {
    "messages": [],
    "images": [],
}

# TODO: encode vision flan SFT dataset as an example


def process_entry(entry):
    """Process a single dataset entry."""
    formatted_entry = copy.deepcopy(END_FORMAT)

    for message in entry["conversations"]:
        new_message = {
            "content": message["value"],
            "role": "user" if message["from"] == "human" else "assistant",
        }
        formatted_entry["messages"].append(new_message)

    formatted_entry["images"] = entry["image"]
    return formatted_entry


def format_dataset(dataset_path, output_path, num_workers=4):
    """Load dataset, format it using multiprocessing, and return formatted dataset."""
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    # Use multiprocessing to process dataset entries in parallel
    with multiprocessing.Pool(processes=num_workers) as pool:
        formatted_dataset = pool.map(process_entry, dataset)

    return formatted_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    formatted_dataset = format_dataset(args.dataset_path, args.output_path)

    with open(args.output_path, "w") as f:
        json.dump(formatted_dataset, f)
