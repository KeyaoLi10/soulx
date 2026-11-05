import os
import json
import argparse
from tqdm import tqdm


def calculate_first_packet(data_dir):
    json_files = []
    # Walk through the directory to find all infer_time_info.json files
    for root, dirs, files in os.walk(data_dir):
        if "infer_time_info.json" in files:
            json_files.append(os.path.join(root, "infer_time_info.json"))

    latency_list = []

    for file_path in tqdm(json_files, desc="Processing"):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            if data and isinstance(data, list):
                # Get the tts_latency from the last element
                last_item = data[-1]
                if "tts_latency" in last_item:
                    latency_list.append(last_item["tts_latency"])
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if not latency_list:
        print("No valid tts_latency data found.")
        return

    average_latency = sum(latency_list) / len(latency_list)

    print("-" * 50)
    print(f"Total files processed: {len(latency_list)}")
    print(f"Average LLM + TTS Latency (First Packet): {average_latency:.4f}s")
    print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir", type=str, required=True, help="Root directory to scan"
    )
    args = parser.parse_args()

    calculate_first_packet(args.root_dir)
