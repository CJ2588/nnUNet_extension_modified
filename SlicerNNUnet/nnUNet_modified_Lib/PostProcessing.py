import numpy as np
from scipy.ndimage import label

def Volume_Threshold(binary_mask, min_size:int = 50000):
    """
    Removes connected components smaller than `min_size` from a binary mask.

    Parameters:
        binary_mask (ndarray): Binary image (e.g., output of model > threshold).
        min_size (int): Minimum size (in voxels or pixels) to keep a component.

    Returns:
        cleaned_mask (ndarray): Binary mask with small components removed.
    """
    labeled_mask, num_features = label(binary_mask)
    cleaned_mask = np.zeros_like(binary_mask)

    for i in range(1, num_features + 1):
        component = (labeled_mask == i)
        if np.sum(component) >= min_size:
            cleaned_mask[component] = 1

    return cleaned_mask