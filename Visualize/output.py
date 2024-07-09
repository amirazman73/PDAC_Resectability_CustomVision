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
from predict import predictions_df
from matplotlib.backends.backend_pdf import PdfPages

raw_ct_dir = r'.../GitHub/PDAC_Resectability_CustomVision/Data/files'
seg_dir = r'.../GitHub/PDAC_Resectability_CustomVision/Data/files'
label_dir = r'.../GitHub/PDAC_Resectability_CustomVision/Data/files'

labels_df = load_labels(label_dir)
seg_dir = setup_dir(seg_dir, raw_ct_dir=raw_ct_dir, segmentations_dir=seg_dir)

train_data, test_data = train_test_split(labels_df, test_size=0.1, random_state=42)

max_height, max_width, max_depth = find_max_bounding_box_edges(seg_dir)

labels_df = load_labels(label_dir)
seg_dir = setup_dir(seg_dir, raw_ct_dir=raw_ct_dir, segmentations_dir=seg_dir)

train_data, test_data = train_test_split(labels_df, test_size=0.1, random_state=42)

test_dataset = MedicalImageDataset(test_data, raw_ct_dir, seg_dir, max_height, max_width, max_depth)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = setup_model()
model_path = r'.../GitHub/PDAC_Resectability_CustomVision/Data/saved_states/Final_All_model.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model = model.to(device)

# Compute Gradient-based attribution
def compute_gradients(input, model, target_class):
    input = input.clone().detach().requires_grad_(True)
    model.eval()
    output = model(input)
    loss = F.cross_entropy(output, torch.tensor([target_class]).to(input.device))
    model.zero_grad()
    loss.backward()
    gradients = input.grad
    return gradients

# Determine thresholds based on percentiles
def determine_percentile_thresholds(gradients, percentiles):
    thresholds = np.percentile(gradients, percentiles)
    return thresholds

# Create multi-label map from gradient split
def create_multi_label_map(gradients, thresholds):
    label_map = np.zeros_like(gradients, dtype=np.int16)
    for i, threshold in enumerate(thresholds):
        label_map[gradients > threshold] = i + 1
    return label_map

def save_to_nifti(data, output_path, affine=np.eye(4), dtype=np.int16):
    data = data.astype(dtype)
    nifti_img = nib.Nifti1Image(data, affine)
    nib.save(nifti_img, output_path)

pdf_path = 'FINAL_PANC_MASK.pdf'
percentiles = [20, 40, 60, 80] 

class MultiChannelMasker:
    # Choose 3 images with the biggest tumor
    def __init__(self, dataloader, labels_to_mask=(8, 10), num_slices=3):
        self.dataloader = dataloader
        self.labels_to_mask = labels_to_mask
        self.num_slices = num_slices

    def apply_mask(self):
        masked_batches = []
        for images, labels in self.dataloader:
            for img, label in zip(images, labels):
                largest_slices = self.find_slices_with_largest_tumor(img[1])
                masked_img = self.mask_labels(img, largest_slices)
                masked_batches.append((masked_img, label))
        return masked_batches

    def find_slices_with_largest_tumor(self, segmentation, tumor_label=10):
        slice_areas = []
        for i in range(segmentation.shape[0]):
            slice_ = segmentation[i, :, :]
            tumor_area = torch.sum(slice_ == tumor_label).item()
            slice_areas.append((i, tumor_area))
        
        slice_areas.sort(key=lambda x: x[1], reverse=True)
        largest_tumor_slices = [slice_areas[i][0] for i in range(min(self.num_slices, len(slice_areas)))]
        print(f"Chosen slices: {largest_tumor_slices}")
        
        return largest_tumor_slices

    def mask_labels(self, images, slice_indices):
        masked_images = images.clone()
        segmentation_channel = masked_images[1]  

        for slice_idx in slice_indices:
            slice_segmentation = segmentation_channel[:, :, slice_idx]
            for label in self.labels_to_mask:
                slice_segmentation[slice_segmentation == label] = 0
            segmentation_channel[:, :, slice_idx] = slice_segmentation

        masked_images[1] = segmentation_channel
        return masked_images

masked_dataset = MultiChannelMasker(test_loader, num_slices=3).apply_mask()
masker = MultiChannelMasker(test_loader, num_slices=3)
masked_dataset = masker.apply_mask()

with PdfPages(pdf_path) as pdf_pages:
    for sample_idx, (original, masked) in enumerate(zip(test_dataset, masked_dataset)):
        if sample_idx > 1:
            break
        img, label = original
        masked_img, label = masked

        original_sample = img.unsqueeze(0).to(device)
        masked_sample = masked_img.unsqueeze(0).to(device)

        predicted_label = predictions_df.loc[sample_idx, 'PredictedLabel']
        target_class = label.item()

        gradients = compute_gradients(original_sample, model, target_class).abs().squeeze().cpu().numpy()

        thresholds = determine_percentile_thresholds(gradients, percentiles)
        print(thresholds)

        # Create multi-label map from gradients
        multi_label_map = create_multi_label_map(gradients, thresholds)

        save_to_nifti(multi_label_map, f'multi_label_map_{sample_idx}.nii', dtype=np.int16)