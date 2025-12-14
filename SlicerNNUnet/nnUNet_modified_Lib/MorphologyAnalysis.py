import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, label, convolve, binary_opening, binary_closing
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
    endpoints: set[tuple],
    branchpoints: set[tuple],
    pruning_scale: float,
    min_branch_length_um: float,
) -> list[list[tuple]]:
    """
    Prune only terminal branches (ending in endpoints), based on physical length
    and vessel radius at the branching point.

    Interior segments between two branchpoints are always kept to preserve
    the main vessel trunks.
    """
    if pruning_scale <= 0 and min_branch_length_um <= 0:
        return branches

    spacing_zyx = np.array((spacing[2], spacing[1], spacing[0]), dtype=float)
    kept: list[list[tuple]] = []

    for branch in branches:
        if len(branch) < 2:
            continue

        p_start = branch[0]
        p_end = branch[-1]

        # Is this branch terminal? (touching an endpoint)
        is_terminal = (p_start in endpoints) or (p_end in endpoints)

        # Interior branches (between branchpoints) are kept untouched
        if not is_terminal:
            kept.append(branch)
            continue

        # Physical length of this branch in µm
        L = _branch_length_um(branch, spacing_zyx)

        # 1) Absolute minimum length (e.g. 100 µm): drop if shorter
        if min_branch_length_um > 0 and L < min_branch_length_um:
            continue

        # 2) Scale-aware pruning: compare to vessel radius at branchpoint
        if pruning_scale > 0:
            if p_start in branchpoints:
                B = float(dist_um[p_start])
            elif p_end in branchpoints:
                B = float(dist_um[p_end])
            else:
                # fallback: use max radius at endpoints
                B = float(max(dist_um[p_start], dist_um[p_end]))

            if B > 0 and L < pruning_scale * B:
                # too short relative to local thickness → prune
                continue

        # If we made it here, keep this terminal branch
        kept.append(branch)

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


def _smooth_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Simple binary smoothing: a few rounds of opening+closing with a 3x3x3 structuring element.

    This reduces jagged edges and tiny spurs that cause fake branches, similar in spirit
    to VesselExpress' segmentation smoothing.
    """
    mask = mask.astype(bool)
    if not mask.any():
        return mask

    struct = np.ones((3, 3, 3), dtype=bool)
    smoothed = mask.copy()
    for _ in range(iterations):
        smoothed = binary_opening(smoothed, structure=struct)
        smoothed = binary_closing(smoothed, structure=struct)
    return smoothed


def _branch_mask_from_skeleton(skeleton_use: np.ndarray) -> np.ndarray:
    """
    From a skeleton, compute a branchpoint mask with exactly ONE voxel per
    branchpoint (cluster center), using a 6-neighborhood (axis-aligned).
    """
    if not skeleton_use.any():
        return np.zeros_like(skeleton_use, dtype=bool)

    skel_int = skeleton_use.astype(np.int32)

    # 6-neighborhood kernel (no center voxel)
    kernel = np.zeros((3, 3, 3), dtype=np.int32)
    kernel[1, 1, 0] = 1  # -x
    kernel[1, 1, 2] = 1  # +x
    kernel[1, 0, 1] = 1  # -y
    kernel[1, 2, 1] = 1  # +y
    kernel[0, 1, 1] = 1  # -z
    kernel[2, 1, 1] = 1  # +z

    neighbor_count = convolve(skel_int, kernel, mode="constant", cval=0)

    # Raw branch voxels: skeleton voxels that have >=3 axis-aligned neighbors
    raw_branch_mask = (skeleton_use > 0) & (neighbor_count >= 3)

    if not raw_branch_mask.any():
        return raw_branch_mask

    # Collapse each connected blob of branch voxels to a single voxel
    labeled, n_components = label(raw_branch_mask.astype(np.uint8))
    branch_mask = np.zeros_like(raw_branch_mask, dtype=bool)

    for comp_idx in range(1, n_components + 1):
        coords = np.argwhere(labeled == comp_idx)
        if coords.size == 0:
            continue
        # Use the rounded center-of-mass of the cluster
        center = np.round(coords.mean(axis=0)).astype(int)
        z, y, x = center
        branch_mask[z, y, x] = True

    return branch_mask


def compute_skeleton_and_branch_masks(
    vessel_mask: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    min_object_size_vox: int = 0,
    pruning_scale: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the pruned skeleton and branchpoint mask for a 3D binary vessel mask.

    Returns
    -------
    skeleton_use : np.ndarray (bool)
        Pruned skeleton mask (or original skeleton if everything was pruned).
    branch_mask   : np.ndarray (bool)
        Branchpoints computed from the PRUNED skeleton.
    """
    spacing_arr = np.asarray(spacing, dtype=float)

    # helper for consistent empty returns
    def _empty(shape):
        z = np.zeros(shape, dtype=bool)
        return z, z

    if vessel_mask.size == 0:
        return _empty(vessel_mask.shape)

    # 1) Cleanup
    mask_clean = _clean_mask(vessel_mask > 0, min_object_size_vox=min_object_size_vox)
    if mask_clean.sum() == 0:
        return _empty(mask_clean.shape)

    # 1b) Smoothing before skeletonization (NEW)
    mask_smooth = _smooth_mask(mask_clean, iterations=1)
    if not mask_smooth.any():
        # Fallback: if smoothing killed everything, use cleaned mask
        mask_smooth = mask_clean

    # 2) Skeletonize on smoothed mask
    skeleton = skeletonize(mask_smooth).astype(bool)
    if not skeleton.any():
        return _empty(mask_clean.shape)

    # 3) Distance transform for radii in µm (on same mask used for skeleton)
    sampling = spacing_arr[::-1]  # (sz, sy, sx)
    dist = distance_transform_edt(mask_smooth, sampling=sampling)

    # 4) Graph + branches on original skeleton
    neighbors0, endpoints0, branchpoints0 = _build_skeleton_graph(skeleton)
    
    #####DEBUG########################################
    print(f"[Morphology] (viz) Graph 1: initial branchpoints = {len(branchpoints0)}")
    #####END##########################################

    branches = _extract_branches(neighbors0, endpoints0, branchpoints0)

    # 5) Prune branches
    min_branch_length_um = 100.0
    effective_pruning_scale = max(pruning_scale, 2.0)

    pruned_branches = _prune_branches_scale_aware(
        branches=branches,
        dist_um=dist,
        spacing=spacing_arr,
        endpoints=endpoints0,
        branchpoints=branchpoints0,
        pruning_scale=effective_pruning_scale,
        min_branch_length_um=min_branch_length_um,
    )

    # 6) Rebuild pruned skeleton
    skeleton_pruned = np.zeros_like(skeleton, dtype=bool)
    for branch in pruned_branches:
        for p in branch:
            skeleton_pruned[p] = True

    # Fallback: if everything got pruned, revert to original skeleton
    skeleton_use = skeleton if not skeleton_pruned.any() else skeleton_pruned

    #####DEBUG#########################################
    _, _, branchpoints_after_graph = _build_skeleton_graph(skeleton_use)
    print(f"[Morphology] (viz) Graph 2: branchpoints after pruning = {len(branchpoints_after_graph)}")
    #####END###########################################


    # 7) Branchpoints on PRUNED skeleton: 26-neighborhood + cluster collapse
    kernel = np.ones((3, 3, 3), dtype=np.int32)
    neighbor_count = convolve(skeleton_use.astype(np.int32), kernel, mode="constant", cval=0)
    neighbors = neighbor_count - skeleton_use.astype(np.int32)
    raw_branch_mask = (skeleton_use > 0) & (neighbors >= 3)

    #####DEBUG#########################################
    print(f"[Morphology] (viz) After pruning: raw branch voxels = {int(raw_branch_mask.sum())}")
    #####END###########################################

    labeled, n_components = label(raw_branch_mask.astype(np.uint8))

    #####DEBUG#########################################
    print(f"[Morphology] (viz) After pruning: connected branchpoint components = {int(n_components)}")
    #####END###########################################


    branch_mask = np.zeros_like(raw_branch_mask, dtype=bool)

    for comp_idx in range(1, n_components + 1):
        coords = np.argwhere(labeled == comp_idx)
        if coords.size == 0:
            continue
        center = np.round(coords.mean(axis=0)).astype(int)
        z, y, x = center
        branch_mask[z, y, x] = True

    return skeleton_use, branch_mask


def compute_global_metrics(
    vessel_mask: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    min_object_size_vox: int = 0,
    pruning_scale: float = 1.5,
) -> pd.DataFrame:
    """
    Compute sample-level lymphatic vessel metrics for a 3D binary mask.
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

    # 1b) Smoothing (same as in visualization)
    mask_smooth = _smooth_mask(mask_clean, iterations=5)
    if not mask_smooth.any():
        mask_smooth = mask_clean

    # 2) Skeletonize
    skeleton = skeletonize(mask_smooth).astype(bool)

    if not skeleton.any():
        # Degenerate case: no skeleton even though mask is non-empty
        vessel_voxels = int(mask_clean.sum())  # volume from cleaned (unsmoothed) mask
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
    dist = distance_transform_edt(mask_smooth, sampling=sampling)

    # 4) Build skeleton graph and extract branches
    neighbors_dict, endpoints, branchpoints = _build_skeleton_graph(skeleton)

    #####DEBUG#########################################
    print(f"[Morphology] (metrics) Graph 1: initial branchpoints = {len(branchpoints)}")
    #####END###########################################

    branches = _extract_branches(neighbors_dict, endpoints, branchpoints)

    # 5) Scale-aware pruning of branches (same as visualization)
    spacing_arr = np.asarray(spacing, dtype=float)
    min_branch_length_um = 100.0
    effective_pruning_scale = max(pruning_scale, 2.0)

    pruned_branches = _prune_branches_scale_aware(
        branches=branches,
        dist_um=dist,
        spacing=spacing_arr,
        endpoints=endpoints,
        branchpoints=branchpoints,
        pruning_scale=effective_pruning_scale,
        min_branch_length_um=min_branch_length_um,
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

    _, _, branchpoints_after_graph = _build_skeleton_graph(skeleton_use)
    #####DEBUG#########################################
    print(f"[Morphology] (metrics) Graph 2: branchpoints after pruning = {len(branchpoints_after_graph)}")
    #####END###########################################


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

        # Branch points on PRUNED skeleton: 26-neighborhood + cluster collapse
        kernel = np.ones((3, 3, 3), dtype=np.int32)
        neighbor_count = convolve(skeleton_use.astype(np.int32), kernel, mode="constant", cval=0)
        neighbors = neighbor_count - skeleton_use.astype(np.int32)
        raw_branch_mask = (skeleton_use > 0) & (neighbors >= 3)

        #####DEBUG#########################################
        print(f"[Morphology] (metrics) After pruning: raw branch voxels = {int(raw_branch_mask.sum())}")
        #####END###########################################


        labeled, n_components = label(raw_branch_mask.astype(np.uint8))

        #####DEBUG#########################################
        print(f"[Morphology] (metrics) After pruning: connected branchpoint components = {int(n_components)}")
        #####END###########################################

        total_branch_points = int(n_components)

    # 8) Volumes (computed on cleaned mask, not smoothed, to preserve volume)
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
