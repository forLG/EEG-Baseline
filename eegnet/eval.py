import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import datetime
from sklearn.metrics import confusion_matrix

from dataset import SimpleDataset
from model import EEGNet

# --- Configuration ---
EVAL_JSON_PATH = './data/HMC/eval.json'
EVAL_EEG_DIR = './data/HMC/eval'

# Model and data parameters (should match the training script)
MODEL_PATH = './output/hmc_eegnet_model.pth'
BATCH_SIZE = 32
NUM_CHANNELS = 4
WINDOW_LENGTH = 2000
NUM_CLASSES = 4

# Result save path
RESULTS_SAVE_PATH = './output/hmc_eegnet_evaluation.txt'

def evaluate_model():
    """
    Loads the trained model and evaluates it on the evaluation dataset.
    Calculates and prints accuracy and balanced accuracy, then saves the results to a text file.
    """
    # Check if the model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        return

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Evaluation Data ---
    print("Loading evaluation data...")
    eval_dataset = SimpleDataset(json_path=EVAL_JSON_PATH, eeg_folder=EVAL_EEG_DIR)

    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=4, 
        pin_memory=True
    )
    print(f"Evaluation data loaded. Found {len(eval_dataset)} samples.")

    # --- Initialize and Load Model ---
    print("Loading the trained model...")
    model = EEGNet(
        num_chans=NUM_CHANNELS,
        window_length=WINDOW_LENGTH,
        num_classes=NUM_CLASSES,
        p_dropout=0.5,
        avgpool_factor=4
    )
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # --- Evaluation Loop ---
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(eval_loader):
            inputs = inputs.to(device)
            labels = labels.to(device, dtype=torch.long)
            outputs = model(inputs)
            
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- Calculate Metrics ---
    total_samples = len(all_labels)
    accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / total_samples
    
    # Calculate balanced accuracy from confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=np.arange(NUM_CLASSES))
    per_class_recall = np.diag(cm) / (np.sum(cm, axis=1) + 1e-6)
    balanced_accuracy = np.mean(per_class_recall)
    
    # --- Save Result ---
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Format confusion matrix for printing
    cm_string = "Confusion Matrix:\n"
    cm_string += str(cm)

    # Format per-class recall string safely
    recall_strings = [f"  - Class {i}: {recall:.4f}" for i, recall in enumerate(per_class_recall)]
    per_class_recall_string = "\n".join(recall_strings)
    
    results_string = f"""
==================================================
          Model Evaluation Report
==================================================
Evaluation Timestamp: {timestamp}
Model Path: {MODEL_PATH}
Evaluation Dataset: {EVAL_JSON_PATH}
--------------------------------------------------
Total Samples: {total_samples}

{cm_string}
--------------------------------------------------
Metrics:
  - Accuracy:          {accuracy * 100:.2f}%
  - Balanced Accuracy: {balanced_accuracy * 100:.2f}%
==================================================
Per-Class Recall:
{per_class_recall_string}
==================================================
""".strip()

    print("\n" + results_string)

    try:
        output_dir = os.path.dirname(RESULTS_SAVE_PATH)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        with open(RESULTS_SAVE_PATH, 'w', encoding='utf-8') as f:
            f.write(results_string)
        print(f"\n✅ Results successfully saved to '{RESULTS_SAVE_PATH}'")
        
    except IOError as e:
        print(f"\n❌ Error saving results to file: {e}")


if __name__ == '__main__':
    evaluate_model()