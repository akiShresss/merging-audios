import json
import pandas as pd
import os
import numpy as np
import soundfile as sf
from datasets import load_dataset
from huggingface_hub import login
import glob

# Define the path to the JSON file and the directory to save CSV files
json_path = "/home/bumblebee/wiseyak_backup/aakriti/aakriti/datasets.json"
csv_dir = "/home/bumblebee/wiseyak_backup/aakriti/aakriti/filtered_datasets/"  # Directory to save filtered CSV files
merged_csv_path = os.path.join(csv_dir, "merged_filtered_dataset.csv")  # Path for the merged CSV file

# Ensure the directory exists
os.makedirs(csv_dir, exist_ok=True)

# Function to log in to Hugging Face
def login_to_huggingface():
    token = "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"  # Replace with your Hugging Face token
    login(token=token)
    print("Successfully logged in to Hugging Face")

# Function to load datasets from JSON file
def load_datasets_from_json(json_file):
    with open(json_file, "r") as file:
        data = json.load(file)
    all_datasets = {}
    for dataset_name, dataset_info in data["datasets"].items():
        dataset_path = dataset_info[0]
        lang = dataset_info[1] if len(dataset_info) > 1 else None
        all_datasets[dataset_name] = (dataset_path, lang)
    return all_datasets

# Function to load all datasets, add language info, and save individual filtered CSV files
def load_all_datasets(datasets):
    loaded_datasets = {}
    for dataset_name, (dataset_path, lang) in datasets.items():
        try:
            # Load the dataset
            dataset = load_dataset(dataset_path, lang, split="train", cache_dir=csv_dir) if lang else load_dataset(dataset_path, split="train", cache_dir=csv_dir)
            loaded_datasets[dataset_name] = dataset
            
            # Convert the dataset to DataFrame and print the head to understand the structure
            df = dataset.to_pandas()
            print(f"Dataset structure for {dataset_name}:{df.head()}")
        
            # Extract only the required columns: index_no, sentence, audio_id, duration
            data = []
            for idx, example in enumerate(dataset):
                if 'sentence' in example and 'audio' in example:
                    transcript = example['sentence']
                    utterance_id = example['client_id'] if 'client_id' in example else f"{dataset_name}_{idx}"
                    duration = len(example['audio']['array']) / example['audio']['sampling_rate'] if 'array' in example['audio'] else None
                    index_no = idx
                    data.append([index_no, transcript, utterance_id, duration, lang, dataset_name])  # Add language info
            
            # Convert to DataFrame
            df_filtered = pd.DataFrame(data, columns=["index_no", "transcript", "utterance_id", "duration", "language","dataset_name"])
            output_path = f"{csv_dir}{dataset_name}_filtered_data.csv"
            
            # Save required columns to the CSV file
            df_filtered.to_csv(output_path, index=False, encoding="utf-8")
            print(f"Filtered data CSV created for {dataset_name} at {output_path}")

        except Exception as e:
            print(f"Error loading {dataset_name}: {str(e)}")
    return loaded_datasets

# Function to merge all filtered CSV files
def merge_filtered_csv_files(csv_directory, output_path):
    all_files = glob.glob(os.path.join(csv_directory, "*_filtered_data.csv"))
    combined_df = pd.concat((pd.read_csv(file) for file in all_files), ignore_index=True)
    combined_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"All filtered CSV files have been merged into {output_path}")

# Main execution
if __name__ == "__main__":
    # Log in to Hugging Face
    login_to_huggingface()
    
    # Load datasets from JSON and process each
    datasets = load_datasets_from_json(json_path)
    loaded_datasets = load_all_datasets(datasets)
    
    # Merge all filtered CSV files into a single CSV
    merge_filtered_csv_files(csv_dir, merged_csv_path)

save_path = "/home/bumblebee/wiseyak_backup/aakriti/aakriti/common_mixed_clips"
os.makedirs(save_path, exist_ok=True)


df = pd.read_csv(merged_csv_path)

mixed_clips_metadata_path = "/home/bumblebee/wiseyak_backup/aakriti/aakriti/common_mixed_clips_metadata.csv"
if not os.path.exists(mixed_clips_metadata_path):
    mixed_clips_df = pd.DataFrame(columns=['languages_used', 'index_numbers', 'mixed_clip_filename', 'transcripts', 'utterance_ids', 'total_duration'])
    mixed_clips_df.to_csv(mixed_clips_metadata_path, index=False)

def save_audio_file(audio_array, sample_rate, filename):
    sf.write(filename, audio_array, sample_rate)
    
clip_counter = 0
while len(df) > 0:
    mixed_audio = np.array([])  # Empty array to store mixed audio data
    total_duration = 0
    sample_rate = None
    mixed_languages, mixed_index_numbers, mixed_transcripts, mixed_utterance_ids = [], [], [], []

    rows_to_remove = []  # List to store indexes of rows that have been used
    while total_duration < 18 and len(df) > 0:
        row = df.sample(1).iloc[0]  # Randomly select one row
        row_index = row.name
        language = row['language']
        dataset_name = row['dataset_name']
        index = int(row['index_no'])
        audio_duration = float(row['duration'])

        try:
            if dataset_name in loaded_datasets:
                dataset = loaded_datasets[dataset_name]
                example = dataset[int(index)]
                audio_data = example['audio']
                audio_array = audio_data['array']
                sample_rate = audio_data['sampling_rate']

                if total_duration + audio_duration <= 25:
                    mixed_audio = np.concatenate((mixed_audio, audio_array))
                    total_duration += audio_duration

                    mixed_languages.append(language)
                    mixed_index_numbers.append(index)
                    mixed_transcripts.append(row['transcript'])
                    mixed_utterance_ids.append(str(row['utterance_id']))

                    rows_to_remove.append(row_index)

        except IndexError:
            print(f"Index {index} out of range for {dataset_name} dataset.")
            continue

    if 18 <= total_duration <= 25:
        clip_counter += 1
        filename = os.path.join(save_path, f"mixed_clip_{clip_counter}.wav")
        save_audio_file(mixed_audio, sample_rate, filename)
        print(f"Saved mixed clip {clip_counter} with duration {total_duration:.2f}s to {filename}")

        # Prepare metadata for the mixed clip
        mixed_clip_metadata = {
            'languages_used': ', '.join(mixed_languages),
            'index_numbers': ', '.join(map(str, mixed_index_numbers)),
            'mixed_clip_filename': f"mixed_clip_{clip_counter}.wav",
            'transcripts': ' | '.join(mixed_transcripts),
            'utterance_ids': ', '.join(mixed_utterance_ids),
            'total_duration': total_duration
        }

        mixed_clips_df = pd.DataFrame([mixed_clip_metadata])
        mixed_clips_df.to_csv(mixed_clips_metadata_path, mode='a', header=False, index=False)

    # Remove used rows
    df = df.drop(rows_to_remove)