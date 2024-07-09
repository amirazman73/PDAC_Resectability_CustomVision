import numpy as np
import torch
import os
from Data.data_loader import load_nifti_image
from torch.utils.data import DataLoader, Dataset

def find_bounding_box(segmentation):
    coords = np.array(np.nonzero(segmentation))
    top_left = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1)
    return top_left, bottom_right

def pad_image_to_depth(image, max_depth):
    current_depth = image.shape[2]
    if current_depth < max_depth:
        pad_before = (max_depth - current_depth) // 2
        pad_after = max_depth - current_depth - pad_before
        return np.pad(image, ((0, 0), (0, 0), (pad_before, pad_after)), mode='constant', constant_values=0)
    return image

def crop_image(image, top_left, bottom_right, max_height, max_width, max_depth):
    return image[
        top_left[0]:min(top_left[0] + max_height, bottom_right[0] + 1),
        top_left[1]:min(top_left[1] + max_width, bottom_right[1] + 1),
        top_left[2]:min(top_left[2] + max_depth, bottom_right[2] + 1)
    ]

class MedicalImageDataset(Dataset):
    def __init__(self, labels_df, ct_scan_dirs, full_seg_dir, max_height, max_width, max_depth, transform=None, visualize=False):
        self.labels_df = labels_df
        self.ct_scan_dirs = ct_scan_dirs
        self.full_seg_dir = full_seg_dir
        self.max_height = max_height
        self.max_width = max_width
        self.max_depth = max_depth
        self.transform = transform
        self.visualize = visualize

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        patient_id = row['Patient_ID']
        label = row['Label']
        
        ct_scan_path, full_seg_path = None, None
        
        for category, dir_path in self.ct_scan_dirs.items():
            for filename in os.listdir(dir_path):
                if patient_id in filename:
                    ct_scan_path = os.path.join(dir_path, filename)
                    break
            if ct_scan_path:
                break

        for filename in os.listdir(self.full_seg_dir):
            if patient_id in filename:
                full_seg_path = os.path.join(self.full_seg_dir, filename)
                break

        ct_scan = load_nifti_image(ct_scan_path)
        full_seg = load_nifti_image(full_seg_path)

        top_left, bottom_right = find_bounding_box(full_seg)

        def crop_image(image, top_left, bottom_right, max_height, max_width, max_depth):
            cropped_image = image[top_left[0]:min(top_left[0] + max_height, bottom_right[0]+1),
                                  top_left[1]:min(top_left[1] + max_width, bottom_right[1]+1),
                                  top_left[2]:min(top_left[2] + max_depth, bottom_right[2]+1)]
            return cropped_image

        ct_scan = crop_image(ct_scan, top_left, bottom_right, self.max_height, self.max_width, self.max_depth)
        full_seg = crop_image(full_seg, top_left, bottom_right, self.max_height, self.max_width, self.max_depth)
        
        ct_scan = pad_image_to_depth(ct_scan, self.max_depth)
        full_seg = pad_image_to_depth(full_seg, self.max_depth)

        full_seg = np.where((full_seg == 8) | (full_seg == 10), full_seg, 0)
        
        combined = np.stack([ct_scan, full_seg], axis=0)
        combined = torch.tensor(combined, dtype=torch.float32).permute(0, 3, 1, 2)

        label = torch.tensor(label, dtype=torch.long)

        # Apply transformations
        if self.transform:
            combined = self.transform(combined)
        
        
        if self.visualize:
            transformed_ct_scan = combined[0].permute(1, 2, 0).numpy()
            transformed_full_seg = combined[1].permute(1, 2, 0).numpy()
            #visualize_individual_images(transformed_ct_scan, transformed_full_seg, label, specific_slice_idx = 79)
            #visualize_individual_images1(ct_scan, full_seg, start_slice_idx=None, num_slices=3, interval=5)
        
        return combined, label