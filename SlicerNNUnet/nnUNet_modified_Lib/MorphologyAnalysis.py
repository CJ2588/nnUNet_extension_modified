import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, label, convolve, binary_opening, binary_closing
from skimage.morphology import skeletonize  # unified 2D/3D skeletonization
import slicer
import os


from skimage.morphology import remove_small_objects, ball
import networkx as nx
from itertools import product


def _get_branching_points(mask, degree_threshold=2):
    """
    Build a 3D skeleton graph (networkx) and return branching points + graph.

    Parameters
    ----------
    mask : 3D array-like
        Binary vessel mask.
    degree_threshold : int
        Nodes with degree > degree_threshold are considered branchpoints.
        (With 2, that means degree >= 3.)

    Returns
    -------
    branch_nodes : list[(z,y,x)]
    G            : nx.Graph with nodes = (z,y,x)
    """
    # Find branching points in a 3D skeleton volume.
    skeleton = skeletonize(mask.astype(bool))
    skeleton = skeleton.astype(bool)

    # 26-neighborhood offsets (exclude center)
    neighbour_voxels = [
        offset for offset in product((-1, 0, 1), repeat=3)
        if offset != (0, 0, 0)
    ]

    G = nx.Graph()

    z_max, y_max, x_max = skeleton.shape

    # Loop over all voxels
    for z in range(z_max):
        for y in range(y_max):
            for x in range(x_max):
                if not skeleton[z, y, x]:
                    continue
                for dz, dy, dx in neighbour_voxels:
                    nz, ny, nx_ = z + dz, y + dy, x + dx
                    if 0 <= nz < z_max and 0 <= ny < y_max and 0 <= nx_ < x_max:
                        if skeleton[nz, ny, nx_]:
                            G.add_edge((z, y, x), (nz, ny, nx_))

    # Branching points = nodes with degree > degree_threshold
    branch_nodes = [n for n, d in G.degree() if d > degree_threshold]
    return branch_nodes, G


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
    G,
    spacing,
    dist_um,
    threshold_um: float = 0.0,
    pruning_scale: float = 0.0,
    diameter_scale: float = 0.0,
):
    """
    Prune short terminal branches in a skeleton graph G.

    Uses three criteria (all optional):
      1) threshold_um      : absolute minimum physical length (µm)
      2) diameter_scale    : L < diameter_scale * mean_diameter_along_branch
      3) pruning_scale     : L < pruning_scale  * local_radius_at_junction

    IMPORTANT: this version does a *single pass* on the original graph.
    Only branches that start at an original leaf (degree == 1) are pruned.
    Interior trunk segments (between branchpoints) are not touched,
    and no new leaves are created/used in later iterations.
    """

    # If all criteria are off, just return a copy
    if threshold_um <= 0 and pruning_scale <= 0 and diameter_scale <= 0:
        return G.copy()

    G = G.copy()  # work on a copy

    # Use degrees from the original graph to define leaves and path continuation
    original_degrees = dict(G.degree())
    leaves = [n for n, d in original_degrees.items() if d == 1]

    for leaf in leaves:
        if not G.has_node(leaf):
            continue  # might have been removed already by a previous path

        # ---- 1) Trace branch from leaf until ORIGINAL degree != 2 ----
        path = [leaf]
        current = leaf
        prev = None

        while True:
            # neighbors that are still in the graph and not where we came from
            neighbor_nodes = [n for n in G.neighbors(current) if n != prev]
            if not neighbor_nodes:
                break

            nxt = neighbor_nodes[0]
            # stop if this next node was not a simple degree-2 node in the original graph
            if original_degrees.get(nxt, 0) != 2:
                path.append(nxt)
                break

            path.append(nxt)
            prev, current = current, nxt

        if len(path) < 2:
            continue  # nothing to prune

        # Physical length of the branch (µm)
        L = _branch_length_um(path, spacing)

        # Decide whether to prune this path
        prune_this = False

        # ---- 2) Absolute length criterion ----
        if threshold_um > 0 and L < threshold_um:
            prune_this = True

        # ---- 3) Diameter-scale pruning (diaScale-like) ----
        if not prune_this and diameter_scale > 0:
            radii_branch = np.array([dist_um[p] for p in path], dtype=float)
            if radii_branch.size > 0:
                mean_diam_branch = 2.0 * radii_branch.mean()  # µm
                if mean_diam_branch > 0 and L < diameter_scale * mean_diam_branch:
                    prune_this = True

        # ---- 4) Pruning based on local radius at junction (pruning_scale) ----
        if not prune_this and pruning_scale > 0:
            junction = path[-1]  # last node where we stopped

            # use original degree to decide if junction was a real branchpoint
            deg_junc = original_degrees.get(junction, 0)

            if deg_junc >= 3:
                B = float(dist_um[junction])
            else:
                # fallback: use max radius at path endpoints
                B = float(max(dist_um[path[0]], dist_um[path[-1]]))

            if B > 0 and L < pruning_scale * B:
                prune_this = True

        # ---- 5) Apply pruning for this branch only ----
        if prune_this:
            G.remove_nodes_from(path)

    return G



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

    cleaned = remove_small_objects(mask, min_size=min_object_size_vox)
    return cleaned


def _smooth_mask(mask: np.ndarray, radius: int = 1,
                 do_opening: bool = True,
                 do_closing: bool = True) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return mask

    selem = ball(radius)

    smoothed = mask.copy()
    if do_opening:
        smoothed = binary_opening(smoothed, selem)
    if do_closing:
        smoothed = binary_closing(smoothed, selem)

    return smoothed



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
    spacing_xyz = np.asarray(spacing, dtype=float)       # (sx, sy, sz)
    spacing_zyx = spacing_xyz[::-1]                      # (sz, sy, sx) for distances

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

    # 1b) Smoothing before skeletonization
    mask_smooth = _smooth_mask(mask_clean, radius=5, do_opening=False, do_closing=True)
    if not mask_smooth.any():
        # Fallback: if smoothing killed everything, use cleaned mask
        mask_smooth = mask_clean

    # 2) Distance transform for radii in µm (on same mask used for skeleton)
    dist = distance_transform_edt(mask_smooth, sampling=spacing_zyx)

    # 3) Build graph from skeleton of smoothed mask
    branch_nodes0, G0 = _get_branching_points(mask_smooth, degree_threshold=2)
    print(f"[Morphology] (viz) Graph 1: initial branchpoints = {len(branch_nodes0)}")

    # 4) Prune short terminal branches in the graph
    min_branch_length_um = 100.0             # your absolute threshold (can be >0 if you want)
    effective_pruning_scale = max(pruning_scale, 0.0)
    diameter_scale = 2.0                   # like diaScale in VesselExpress

    G_pruned = _prune_branches_scale_aware(
        G=G0,
        spacing=spacing_zyx,
        dist_um=dist,
        threshold_um=min_branch_length_um,
        pruning_scale=effective_pruning_scale,
        diameter_scale=diameter_scale,
    )

    # 5) Rebuild pruned skeleton mask from pruned graph
    skeleton_use = np.zeros_like(mask_smooth, dtype=bool)
    for (z, y, x) in G_pruned.nodes:
        skeleton_use[z, y, x] = True

    # Debug: branchpoints after pruning (in graph sense)
    branch_nodes_after = [n for n, d in G_pruned.degree() if d >= 3]
    print(f"[Morphology] (viz) Graph 2: branchpoints after pruning = {len(branch_nodes_after)}")

    # 6) Branchpoints on PRUNED skeleton: 26-neighborhood + cluster collapse
    kernel = np.ones((3, 3, 3), dtype=np.int32)
    neighbor_count = convolve(skeleton_use.astype(np.int32), kernel, mode="constant", cval=0)
    neighbors = neighbor_count - skeleton_use.astype(np.int32)
    raw_branch_mask = (skeleton_use > 0) & (neighbors >= 3)

    print(f"[Morphology] (viz) After pruning: raw branch voxels = {int(raw_branch_mask.sum())}")

    labeled, n_components = label(raw_branch_mask.astype(np.uint8))
    print(f"[Morphology] (viz) After pruning: connected branchpoint components = {int(n_components)}")

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
    spacing_xyz = np.asarray(spacing, dtype=float)       # (sx, sy, sz)
    spacing_zyx = spacing_xyz[::-1]                      # (sz, sy, sx)
    voxel_volume = float(spacing_xyz.prod())             # [µm^3]

    # 1) Cleanup: remove tiny components if requested
    mask_clean = _clean_mask(vessel_mask > 0, min_object_size_vox=min_object_size_vox)

    if mask_clean.sum() == 0:
        roi_volume = float(mask_clean.size * voxel_volume)
        result = {
            "Vessel volume (%)": 0.0,
            "Total length (µm)": 0.0,
            "Mean diameter (µm)": 0.0,
            "Min diameter (µm)": 0.0,
            "Max diameter (µm)": 0.0,
            "Approx. diameter V/L (µm)": 0.0,
            "Branching points": 0,
            "Vessel volume (µm³)": 0.0,
            "Total sample volume (µm³)": _r2(roi_volume),
        }
        return pd.DataFrame([result])

    # 1b) Smoothing (same as in visualization)
    mask_smooth = _smooth_mask(mask_clean, radius=5, do_opening=False, do_closing=True)
    if not mask_smooth.any():
        mask_smooth = mask_clean

    # 2) Distance transform with spacing (µm) → radii in µm
    dist = distance_transform_edt(mask_smooth, sampling=spacing_zyx)

    # 3) Build graph from skeleton of smoothed mask
    branch_nodes0, G0 = _get_branching_points(mask_smooth, degree_threshold=2)
    print(f"[Morphology] (metrics) Graph 1: initial branchpoints = {len(branch_nodes0)}")

    # 4) Prune in graph domain
    min_branch_length_um = 100.0
    effective_pruning_scale = max(pruning_scale, 0.0)
    diameter_scale = 2.0

    G_pruned = _prune_branches_scale_aware(
        G=G0,
        spacing=spacing_zyx,
        dist_um=dist,
        threshold_um=min_branch_length_um,
        pruning_scale=effective_pruning_scale,
        diameter_scale=diameter_scale,
    )

    # 5) Rebuild pruned skeleton mask
    skeleton_use = np.zeros_like(mask_smooth, dtype=bool)
    for (z, y, x) in G_pruned.nodes:
        skeleton_use[z, y, x] = True

    branch_nodes_after = [n for n, d in G_pruned.degree() if d >= 3]
    print(f"[Morphology] (metrics) Graph 2: branchpoints after pruning = {len(branch_nodes_after)}")

    # 6) Radii / diameters at pruned skeleton voxels
    skel_indices = np.where(skeleton_use)
    if skel_indices[0].size == 0:
        mean_diam = min_diam = max_diam = 0.0
        total_length = 0.0
        total_branch_points = 0
        approx_diam_vl = 0.0
    else:
        radii = dist[skel_indices]
        diameters = 2.0 * radii

        mean_diam = float(diameters.mean())
        min_diam = float(diameters.min())
        max_diam = float(diameters.max())

        # Approximate total length: #skeleton voxels * mean spacing
        n_skel_voxels = int(skeleton_use.sum())
        mean_step = float(spacing_xyz.mean())
        total_length = float(n_skel_voxels * mean_step)

        # Branch points on PRUNED skeleton: 26-neighborhood + cluster collapse
        kernel = np.ones((3, 3, 3), dtype=np.int32)
        neighbor_count = convolve(skeleton_use.astype(np.int32), kernel, mode="constant", cval=0)
        neighbors = neighbor_count - skeleton_use.astype(np.int32)
        raw_branch_mask = (skeleton_use > 0) & (neighbors >= 3)

        print(f"[Morphology] (metrics) After pruning: raw branch voxels = {int(raw_branch_mask.sum())}")

        labeled, n_components = label(raw_branch_mask.astype(np.uint8))
        print(f"[Morphology] (metrics) After pruning: connected branchpoint components = {int(n_components)}")

        total_branch_points = int(n_components)

        # Approximate diameter from Volume / Length
        if total_length > 0.0:
            vessel_voxels_tmp = int(mask_clean.sum())
            vessel_volume_tmp = float(vessel_voxels_tmp * voxel_volume)
            mean_cs_area = vessel_volume_tmp / total_length   # µm²
            approx_diam_vl = 2.0 * np.sqrt(mean_cs_area / np.pi)
        else:
            approx_diam_vl = 0.0

    # 7) Volumes (computed on cleaned mask, not smoothed)
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
        "Approx. diameter V/L (µm)": _r2(approx_diam_vl),
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
