import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import numpy as np
from sklearn.metrics import confusion_matrix

from dataset import SimpleDataset
from model import CNNTransformer

# --- Configuration ---
NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
RANDOM_SEED = 42

TRAIN_JSON_PATH = './data/HMC/train.json'
TRAIN_EEG_DIR = './data/HMC/train' 
DEV_JSON_PATH = './data/HMC/dev.json'
DEV_EEG_DIR = './data/HMC/dev'

NUM_CHANNELS = 4
WINDOW_LENGTH = 2000
FS = 200  # your dataset's sampling frequency (Hz)
L_SECONDS = WINDOW_LENGTH // FS # Window length in seconds
NUM_CLASSES = 4 # Aligned with the dataset, 2 for TUSZ, 4 for HMC

# New model save path
MODEL_SAVE_PATH = './output/hmc_cnn_transformer_model.pth'

# --- Data Loading ---
print("Loading data...")
train_dataset = SimpleDataset(json_path=TRAIN_JSON_PATH, eeg_folder=TRAIN_EEG_DIR)
dev_dataset = SimpleDataset(json_path=DEV_JSON_PATH, eeg_folder=DEV_EEG_DIR)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
print(f"Data loaded. Train set: {len(train_dataset)} samples, dev set: {len(dev_dataset)} samples.")

# --- Initialize Model, Loss, and Optimizer ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = CNNTransformer(
    n_channels=NUM_CHANNELS,
    L_seconds=L_SECONDS,
    fs=FS,
    n_classes=NUM_CLASSES
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Training Loop (Structure is identical, with one minor data shape adjustment) ---
best_val_b_accuracy = 0.0

for epoch in range(NUM_EPOCHS):
    # --- Train ---
    model.train()
    total_train_loss = 0
    for i, (inputs, labels) in enumerate(train_loader):
        # # Expects (batch, channels, length)
        # # Assuming loader provides (batch, 1, channels, length), we squeeze the channel dim
        # if inputs.dim() == 4 and inputs.shape[1] == NUM_CHANNELS:
        #      # This case handles if dataset gives (batch, channels, length, 1) or similar
        #      inputs = inputs.squeeze(-1)
        # elif inputs.dim() == 4 and inputs.shape[1] == 1:
        #      # This case handles if dataset gives (batch, 1, channels, length)
        #      inputs = inputs.squeeze(1)

        inputs = inputs.to(device)
        labels = labels.to(device, dtype=torch.long)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        if (i + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    avg_train_loss = total_train_loss / len(train_loader)

    # --- Eval ---
    model.eval()
    total_val_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dev_loader:
            # # Adjust input shape for evaluation as well
            # if inputs.dim() == 4 and inputs.shape[1] == NUM_CHANNELS:
            #     inputs = inputs.squeeze(-1)
            # elif inputs.dim() == 4 and inputs.shape[1] == 1:
            #     inputs = inputs.squeeze(1)

            inputs = inputs.to(device)
            labels = labels.to(device, dtype=torch.long)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(dev_loader)

    # --- Metrics and Saving ---
    cm = confusion_matrix(all_labels, all_preds, labels=np.arange(NUM_CLASSES))
    per_class_recall = np.diag(cm) / (np.sum(cm, axis=1) + 1e-6)
    balanced_accuracy = np.mean(per_class_recall) * 100
    
    print(f'--- Epoch {epoch+1} end ---')
    print(f'Average train loss: {avg_train_loss:.4f} | Average val loss: {avg_val_loss:.4f} | Val balanced accuracy: {balanced_accuracy:.2f}%')

    if balanced_accuracy > best_val_b_accuracy:
        best_val_b_accuracy = balanced_accuracy
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f'Model saved, best val balanced accuracy: {best_val_b_accuracy:.2f}%')
    
    print('-' * 50)

print(f'Best val balanced accuracy: {best_val_b_accuracy:.2f}%')
print(f'Best model saved at: {MODEL_SAVE_PATH}')