import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, label, convolve
from skimage.morphology import skeletonize  # unified 2D/3D skeletonization
import slicer
import os

# 26-neighborhood offsets (excluding center)
_NEIGHBOR_OFFSETS = [
    (dz, dy, dx)
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if not (dz == 0 and dy == 0 and dx == 0)
]


def _build_skeleton_graph(skeleton: np.ndarray):
    """
    Build adjacency for skeleton voxels and classify endpoints / branchpoints.

    Returns
    -------
    neighbors_dict : dict[(z,y,x)] -> [ (z,y,x), ... ]
    endpoints      : set[(z,y,x)]
    branchpoints   : set[(z,y,x)]
    """
    coords = np.argwhere(skeleton)
    skel_set = {tuple(c) for c in coords}

    neighbors_dict: dict[tuple, list] = {}
    endpoints: set[tuple] = set()
    branchpoints: set[tuple] = set()

    for z, y, x in skel_set:
        p = (z, y, x)
        nb = []
        for dz, dy, dx in _NEIGHBOR_OFFSETS:
            q = (z + dz, y + dy, x + dx)
            if q in skel_set:
                nb.append(q)
        neighbors_dict[p] = nb
        deg = len(nb)
        if deg == 1:
            endpoints.add(p)
        elif deg >= 3:
            branchpoints.add(p)

    return neighbors_dict, endpoints, branchpoints


def _extract_branches(neighbors_dict, endpoints, branchpoints):
    """
    Extract branches as paths from endpoints / branchpoints along degree-2 voxels.
    Each branch is a list of (z,y,x) voxels.
    """
    branches: list[list[tuple]] = []
    visited_edges: set[tuple] = set()

    def edge_key(a, b):
        # order-independent key for undirected edge
        return (a, b) if a <= b else (b, a)

    def trace(start, neighbor):
        path = [start]
        prev = start
        curr = neighbor

        while True:
            path.append(curr)
            visited_edges.add(edge_key(prev, curr))
            next_neighbors = [n for n in neighbors_dict.get(curr, []) if n != prev]
            if len(next_neighbors) != 1:
                # reached endpoint or branchpoint
                break
            prev, curr = curr, next_neighbors[0]
        return path

    start_nodes = list(endpoints) + list(branchpoints)
    for start in start_nodes:
        for nb in neighbors_dict.get(start, []):
            ekey = edge_key(start, nb)
            if ekey in visited_edges:
                continue
            branches.append(trace(start, nb))

    return branches


def _branch_length_um(branch, spacing_zyx: np.ndarray) -> float:
    """
    Compute physical length of a voxel path using spacing_zyx = (sz, sy, sx) in µm.
    """
    length = 0.0
    for p1, p2 in zip(branch[:-1], branch[1:]):
        v1 = np.array(p1, dtype=float)
        v2 = np.array(p2, dtype=float)
        diff = (v2 - v1) * spacing_zyx
        length += np.linalg.norm(diff)
    return float(length)


def _prune_branches_scale_aware(
    branches: list[list[tuple]],
    dist_um: np.ndarray,
    spacing: np.ndarray,
    branchpoints: set[tuple],
    pruning_scale: float,
) -> list[list[tuple]]:
    """
    Prune branches whose length is shorter than pruning_scale * radius at branchpoint.

    Parameters
    ----------
    branches : list of voxel paths (z,y,x)
    dist_um : distance transform (radii in µm)
    spacing : (sx, sy, sz) in µm
    branchpoints : set of branchpoint coordinates (z,y,x)
    pruning_scale : s; prune branch if L < s * B

    Returns
    -------
    kept_branches : list[list[tuple]]
    """
    if pruning_scale <= 0 or not branches:
        return branches

    spacing_zyx = np.array((spacing[2], spacing[1], spacing[0]), dtype=float)
    kept: list[list[tuple]] = []

    for branch in branches:
        if len(branch) < 2:
            continue

        L = _branch_length_um(branch, spacing_zyx)

        # choose radius at branchpoint end if available
        p_start = branch[0]
        p_end = branch[-1]

        if p_start in branchpoints:
            B = float(dist_um[p_start])
        elif p_end in branchpoints:
            B = float(dist_um[p_end])
        else:
            # fallback: max radius at endpoints
            B = float(max(dist_um[p_start], dist_um[p_end]))

        if B <= 0:
            # no meaningful radius, keep branch
            kept.append(branch)
            continue

        if L >= pruning_scale * B:
            kept.append(branch)
        # else: prune branch

    return kept


def _r2(x: float) -> float:
    """Round to 2 decimal places as float."""
    return float(f"{x:.2f}")

def _clean_mask(mask: np.ndarray, min_object_size_vox: int = 0) -> np.ndarray:
    """
    Optionally remove connected components smaller than `min_object_size_vox`.
    """
    mask = mask.astype(bool)
    if min_object_size_vox <= 0:
        return mask

    labeled, num = label(mask)
    cleaned = np.zeros_like(mask, dtype=bool)

    for comp_idx in range(1, num + 1):
        comp = (labeled == comp_idx)
        if comp.sum() >= min_object_size_vox:
            cleaned[comp] = True

    return cleaned


def compute_global_metrics(
    vessel_mask: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    min_object_size_vox: int = 0,
    pruning_scale: float = 1.5,
) -> pd.DataFrame:

    """
    Compute sample-level lymphatic vessel metrics for a 3D binary mask.

    Parameters
    ----------
    vessel_mask : np.ndarray
        3D binary array (1 = vessel).
    spacing : (sx, sy, sz)
        Voxel spacing in micrometers (µm).
    min_object_size_vox : int
        Remove connected components smaller than this size (in voxels)
        before skeletonization and metric computation.
    pruning_scale : float
        Scale-aware pruning factor s. Branches with length < s * radius_at_branchpoint are removed.


    Returns
    -------
    df : pandas.DataFrame
        One row with global metrics (lengths in µm, volumes in µm^3).
    """
    spacing = np.asarray(spacing, dtype=float)
    voxel_volume = float(spacing.prod())  # [µm^3]

    # 1) Cleanup: remove tiny components if requested
    mask_clean = _clean_mask(vessel_mask > 0, min_object_size_vox=min_object_size_vox)

    if mask_clean.sum() == 0:
        # Empty mask after cleanup
        roi_volume = float(mask_clean.size * voxel_volume)
        result = {
            "Vessel volume (%)": 0.0,
            "Total length (µm)": 0.0,
            "Mean diameter (µm)": 0.0,
            "Min diameter (µm)": 0.0,
            "Max diameter (µm)": 0.0,
            "Branching points": 0,
            "Vessel volume (µm³)": 0.0,
            "Total sample volume (µm³)": _r2(roi_volume),
        }
        return pd.DataFrame([result])

    # 2) Skeletonize
    skeleton = skeletonize(mask_clean)
    skeleton = skeleton.astype(bool)

    if not skeleton.any():
        # Degenerate case: no skeleton even though mask is non-empty
        vessel_voxels = int(mask_clean.sum())
        vessel_volume = float(vessel_voxels * voxel_volume)
        roi_volume = float(mask_clean.size * voxel_volume)
        volume_fraction = 100.0 * vessel_volume / roi_volume if roi_volume > 0 else 0.0

        result = {
            "Vessel volume (%)": _r2(volume_fraction),
            "Total length (µm)": 0.0,
            "Mean diameter (µm)": 0.0,
            "Min diameter (µm)": 0.0,
            "Max diameter (µm)": 0.0,
            "Branching points": 0,
            "Vessel volume (µm³)": _r2(vessel_volume),
            "Total sample volume (µm³)": _r2(roi_volume),
        }
        return pd.DataFrame([result])

    # 3) Distance transform with spacing (µm) → radii in µm
    sampling = spacing[::-1]  # (sz, sy, sx)
    dist = distance_transform_edt(mask_clean, sampling=sampling)

    # 4) Build skeleton graph and extract branches
    neighbors_dict, endpoints, branchpoints = _build_skeleton_graph(skeleton)
    branches = _extract_branches(neighbors_dict, endpoints, branchpoints)

    # 5) Scale-aware pruning of branches
    spacing_arr = np.asarray(spacing, dtype=float)
    pruned_branches = _prune_branches_scale_aware(
        branches,
        dist_um=dist,
        spacing=spacing_arr,
        branchpoints=branchpoints,
        pruning_scale=pruning_scale,
    )

    # 6) Rebuild pruned skeleton
    skeleton_pruned = np.zeros_like(skeleton, dtype=bool)
    for branch in pruned_branches:
        for z, y, x in branch:
            skeleton_pruned[z, y, x] = True

    # Fallback: if everything got pruned, revert to original skeleton
    if not skeleton_pruned.any():
        skeleton_use = skeleton
    else:
        skeleton_use = skeleton_pruned

    # 7) Radii / diameters at pruned skeleton voxels
    skel_indices = np.where(skeleton_use)
    if skel_indices[0].size == 0:
        mean_diam = min_diam = max_diam = 0.0
        total_length = 0.0
        total_branch_points = 0
    else:
        radii = dist[skel_indices]
        diameters = 2.0 * radii

        mean_diam = float(diameters.mean())
        min_diam = float(diameters.min())
        max_diam = float(diameters.max())

        # Approximate total length: #skeleton voxels * mean spacing
        n_skel_voxels = int(skeleton_use.sum())
        mean_step = float(np.asarray(spacing, dtype=float).mean())
        total_length = float(n_skel_voxels * mean_step)

        # Branch points on pruned skeleton: voxels with >=3 neighbors (26-neighborhood)
        kernel = np.ones((3, 3, 3), dtype=np.int32)
        neighbor_count = convolve(skeleton_use.astype(np.int32), kernel, mode="constant", cval=0)
        neighbors = neighbor_count - skeleton_use.astype(np.int32)
        branch_mask = (skeleton_use > 0) & (neighbors >= 3)
        total_branch_points = int(branch_mask.sum())

    # 8) Volumes (unchanged)
    vessel_voxels = int(mask_clean.sum())
    vessel_volume = float(vessel_voxels * voxel_volume)
    roi_volume = float(mask_clean.size * voxel_volume)
    volume_fraction = 100.0 * vessel_volume / roi_volume if roi_volume > 0 else 0.0

    result = {
        "Vessel volume (%)": _r2(volume_fraction),
        "Total length (µm)": _r2(total_length),
        "Mean diameter (µm)": _r2(mean_diam),
        "Min diameter (µm)": _r2(min_diam),
        "Max diameter (µm)": _r2(max_diam),
        "Branching points": int(total_branch_points),
        "Vessel volume (µm³)": _r2(vessel_volume),
        "Total sample volume (µm³)": _r2(roi_volume),
    }

    return pd.DataFrame([result])




def save_metrics_to_file(df: pd.DataFrame, base_name: str = "VesselMetrics", output_dir: str = slicer.app.temporaryPath):
    """Save results to a CSV file in the Slicer temp directory and return the path."""
    csv_path = os.path.join(output_dir, f"{base_name}.csv")
    df.to_csv(csv_path, index=False)
    return csv_path
