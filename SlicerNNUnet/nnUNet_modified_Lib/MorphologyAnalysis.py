import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, label
from skimage.measure import regionprops
from skimage.morphology import skeletonize_3d
import slicer
import os

def compute_vessel_metrics(vessel_mask: np.ndarray, min_volume: int = 50000):
    """Compute morphological metrics for each vessel in a 3D binary mask."""
    # Remove small components
    labeled_mask, num = label(vessel_mask)
    props = regionprops(labeled_mask)

    results = []
    for region in props:
        if region.area < min_volume:
            continue

        vessel_separated = (labeled_mask == region.label)
        skeleton = skeletonize_3d(vessel_separated)
        distance_map = distance_transform_edt(vessel_separated)
        radii = distance_map[skeleton > 0]
        diameters = 2 * radii

        length = np.sum(skeleton)
        results.append({
            "VesselLabel": region.label,
            "Volume": region.area,
            "Length (voxels)": float(length),
            "Mean Diameter": float(np.mean(diameters)),
            "Min Diameter": float(np.min(diameters)),
            "Max Diameter": float(np.max(diameters)),
        })

    return pd.DataFrame(results)


def save_metrics_to_file(df: pd.DataFrame, base_name: str = "VesselMetrics"):
    """Save results to a CSV file in the Slicer temp directory and return the path."""
    output_dir = slicer.app.temporaryPath
    csv_path = os.path.join(output_dir, f"{base_name}.csv")
    df.to_csv(csv_path, index=False)
    return csv_path
