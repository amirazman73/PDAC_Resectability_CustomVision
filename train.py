import os
import numpy as np
import nibabel as nib
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import torchio as tio  # Import torchio
from torch import nn, optim
from linformer import Linformer  # Import Linformer
import torch.nn.functional as F
from Data.data_loader import load_labels, setup_dir
from Data.utils import find_max_bounding_box_edges
from Data.preprocess import MedicalImageDataset
from Models.setup import setup_model


raw_ct_dir = r'.../GitHub/PDAC_Resectability_CustomVision/Data/files'
seg_dir = r'.../GitHub/PDAC_Resectability_CustomVision/Data/files'
label_dir = r'.../GitHub/PDAC_Resectability_CustomVision/Data/files'


labels_df = load_labels(label_dir)
seg_dir = setup_dir(seg_dir, raw_ct_dir=raw_ct_dir, segmentations_dir=seg_dir)

train_data, test_data = train_test_split(labels_df, test_size=0.1, random_state=42)

max_height, max_width, max_depth = find_max_bounding_box_edges(seg_dir)

def train_model(model, train_loader, criterion, optimizer, device, num_epochs):
    model.train()
    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

            accuracy = correct_predictions / total_predictions
            print(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / (i + 1):.3f}, Accuracy: {accuracy:.3f}")

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)

        # Save every step
        torch.save(model.state_dict(), f"FinalUpdated_panc_model_{epoch + 1}.pth")

    print("Training Completed")
    return train_losses, train_accuracies

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    test_losses = []
    test_accuracies = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

    avg_loss = running_loss / len(test_loader)
    accuracy = correct_predictions / total_predictions
    test_losses.append(avg_loss)
    test_accuracies.append(accuracy)

    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")

    return test_losses, test_accuracies

def setup_data_loaders(train_data, test_data, max_height, max_width, max_depth, batch_size):
    train_dataset = MedicalImageDataset(train_data, raw_ct_dir, seg_dir, max_height, max_width, max_depth)
    test_dataset = MedicalImageDataset(test_data, raw_ct_dir, seg_dir, max_height, max_width, max_depth)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def main():
    max_height, max_width, max_depth = 333, 206, 176
    batch_size = 1
    num_epochs = 30
    
    train_loader, test_loader = setup_data_loaders(train_data, test_data, max_height, max_width, max_depth, batch_size)
    model = setup_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_losses, train_accuracies = train_model(model, train_loader, criterion, optimizer, device, num_epochs)
    test_losses, test_accuracies = evaluate_model(model, test_loader, criterion, device)


if __name__ == "__main__":
    main()