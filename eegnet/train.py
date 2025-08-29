from dataset import SimpleDataset
from model import EEGNet

import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import numpy as np
from sklearn.metrics import confusion_matrix


NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
RANDOM_SEED = 42

TRAIN_JSON_PATH = './data/TUSZ/train.json'
TRAIN_EEG_DIR = './data/TUSZ/train' 
DEV_JSON_PATH = './data/TUSZ/dev.json'
DEV_EEG_DIR = './data/TUSZ/dev'
MODEL_SAVE_PATH = './output/tusz_eegnet_model.pth'

NUM_CHANNELS = 19
WINDOW_LENGTH = 2000
NUM_CLASSES = 2 # Aligned with the dataset, 2 for TUSZ, 4 for HMC

# Load data
print("Loading data...")
train_dataset = SimpleDataset(json_path=TRAIN_JSON_PATH, eeg_folder=TRAIN_EEG_DIR)
train_size = len(train_dataset)

dev_dataset = SimpleDataset(json_path=DEV_JSON_PATH, eeg_folder=DEV_EEG_DIR)
dev_size = len(dev_dataset)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

print(f"Data loaded. Train set: {len(train_dataset)} samples, dev set: {len(dev_dataset)} samples.")

# Initialize model, loss function and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Use device: {device}")

model = EEGNet(
    num_chans=NUM_CHANNELS,
    window_length=WINDOW_LENGTH,
    num_classes=NUM_CLASSES,
    p_dropout=0.5,
    avgpool_factor=4
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
best_val_b_accuracy = 0.0

for epoch in range(NUM_EPOCHS):
    # --- Train ---
    model.train()
    total_train_loss = 0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device, dtype=torch.long)

        # frward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        if (i + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    avg_train_loss = total_train_loss / len(train_loader)

    # --- Eval---
    model.eval()
    total_val_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dev_loader:
            inputs = inputs.to(device)
            labels = labels.to(device, dtype=torch.long)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()

            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(dev_loader)

    # Calculate balanced accuracy from confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=np.arange(NUM_CLASSES))
    # Calculate recall for each class (per-class accuracy)
    per_class_recall = np.diag(cm) / (np.sum(cm, axis=1) + 1e-6)
    # Balanced accuracy is the average of recalls
    balanced_accuracy = np.mean(per_class_recall) * 100
    
    print(f'--- Epoch {epoch+1} end ---')
    print(f'Average train loss: {avg_train_loss:.4f} | Average val loss: {avg_val_loss:.4f} | Val balanced accuracy: {balanced_accuracy:.2f}%')

    # --- Save best model ---
    if balanced_accuracy > best_val_b_accuracy:
        best_val_b_accuracy = balanced_accuracy
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f'Model saved, best val balanced accuracy: {best_val_b_accuracy:.2f}%')
    
    print('-' * 50)

print(f'Best val balanced accuracy: {best_val_b_accuracy:.2f}%')
print(f'Best model saved at: {MODEL_SAVE_PATH}')