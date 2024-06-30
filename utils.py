import os
import numpy as np
import nibabel as nib

def find_max_bounding_box_edges(segmentations):
    max_height = max_width = max_depth = 0

    for filename in os.listdir(segmentations):
        segmentation_path = os.path.join(segmentations, filename)
        segmentation = nib.load(segmentation_path).get_fdata()
        label_mask = (segmentation == 10)
        non_zero_indices = np.nonzero(label_mask)

        if non_zero_indices[0].size > 0:
            height = np.max(non_zero_indices[0]) - np.min(non_zero_indices[0]) + 1
            width = np.max(non_zero_indices[1]) - np.min(non_zero_indices[1]) + 1
            depth = np.max(non_zero_indices[2]) - np.min(non_zero_indices[2]) + 1
            max_height, max_width, max_depth = max(max_height, height), max(max_width, width), max(max_depth, depth)

    # print(f"Max Height: {max_height}, Max Width: {max_width}, Max Depth: {max_depth}")
    return max_height, max_width, max_depth