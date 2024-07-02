import os
import numpy as np
import nibabel as nib
import pandas as pd

def load_nifti_image(file_path):
    img = nib.load(file_path)
    data = img.get_fdata().astype(np.float32)
    return data

def load_labels(labels_path, file_type='excel'):
    if file_type == 'excel':
        return pd.read_excel(labels_path, engine="openpyxl")
    else:
        raise ValueError("Unsupported file type, not an excel file.")

def setup_dir(base_dir, raw_ct_dir='raw_ct', segmentations_dir='segmentations'):
    raw_ct_dir = os.path.join(base_dir, raw_ct_dir)
    segmentation_dir = os.path.join(base_dir, segmentations_dir)
    
    return {
        'resectable': os.path.join(raw_ct_dir, 'resectable'),
        'borderline_resectable': os.path.join(raw_ct_dir, 'borderline_resectable'),
        'locally_advanced': os.path.join(raw_ct_dir, 'locally_advanced')
    }, segmentation_dir