import os
import json
import numpy as np
import pickle
from multiprocessing import Pool

def process_and_dump(params):
    """
    Reads an .npy file, combines it with its label, and saves it as a .pkl file.
    """
    npy_path, label, dump_folder = params
    
    try:
        # Extract the base filename to use for the output .pkl file
        base_filename = os.path.basename(npy_path).replace('.npy', '.pkl')
        dump_path = os.path.join(dump_folder, base_filename)

        # Load the EEG data from the .npy file
        eeg_data = np.load(npy_path)
        
        # Create the dictionary to be saved
        data_to_dump = {
            "X": eeg_data,
            "y": label
        }

        # Dump the dictionary to a .pkl file
        with open(dump_path, 'wb') as f:
            pickle.dump(data_to_dump, f)
            
    except Exception as e:
        print(f"Error processing file {npy_path}: {e}")


if __name__ == "__main__":
    # Configuration
    datasets_to_process = [
        {
            "name": "HMC",
            "root": "./data/HMC",
            "output": "./data/HMC/labram"
        },
        {
            "name": "TUSZ",
            "root": "./data/TUSZ",
            "output": "./data/TUSZ/labram"
        }
    ]
    num_processes = 12 

    # Mapping: TUSZ 'dev' -> 'val', 'eval' -> 'test'
    split_mapping = {
        "train": "train",
        "dev": "val",
        "eval": "test"
    }

    all_parameters = []

    # Iterate over each dataset defined in the configuration
    for dataset_config in datasets_to_process:
        dataset_name = dataset_config["name"]
        dataset_root = dataset_config["root"]
        processed_root = dataset_config["output"]
        
        print(f"\n===== Preparing dataset: {dataset_name} =====")

        # Create the main 'processed' directory for the current dataset
        os.makedirs(processed_root, exist_ok=True)

        for split, processed_split in split_mapping.items():
            print(f"--- Preparing {split} set for processing ---")
            
            # Define paths for the current split
            data_folder = os.path.join(dataset_root, split)
            label_file = os.path.join(dataset_root, f"{split}.json")
            dump_folder = os.path.join(processed_root, processed_split)

            # Create output directory for the split (e.g., 'processed/train')
            os.makedirs(dump_folder, exist_ok=True)

            # Check if data folder and label file exist
            if not os.path.isdir(data_folder) or not os.path.isfile(label_file):
                print(f"Warning: Could not find data or labels for '{split}' split in '{dataset_name}'. Skipping.")
                print(f"Checked for folder: {data_folder}")
                print(f"Checked for file: {label_file}")
                continue

            # Load labels from the JSON file
            with open(label_file, 'r') as f:
                labels_list = json.load(f)
            
            # Create a dictionary for quick label lookup: {id: label}
            labels_dict = {item['id']: item['label'] for item in labels_list}

            # Prepare the list of parameters for the processing function
            for filename in os.listdir(data_folder):
                if filename.endswith(".npy"):
                    file_id = filename.replace('.npy', '')
                    if file_id in labels_dict:
                        npy_path = os.path.join(data_folder, filename)
                        label = labels_dict[file_id]
                        all_parameters.append((npy_path, label, dump_folder))
                    else:
                        print(f"Warning: No label found for file {filename}. Skipping.")

    if not all_parameters:
        print("\nNo files found to process across all datasets. Exiting.")
    else:
        print(f"\nFound {len(all_parameters)} total files to process across all datasets.")
        print("Starting data processing with multiprocessing...")

        # Use a process pool to parallelize the file processing
        with Pool(processes=num_processes) as pool:
            pool.map(process_and_dump, all_parameters)

        print("\nProcessing complete for all datasets.")