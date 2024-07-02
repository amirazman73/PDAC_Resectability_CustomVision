import numpy as np

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
