"""
Script to compare and visualize two overlapping meshes (PLY and STL).
Optimized with Scipy cKDTree and Multiprocessing for fast execution.
"""

import trimesh
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.spatial import cKDTree
from scipy.interpolate import CubicSpline
import re
import os

# --- MULTIPROCESSING WORKER ENGINE ---
def _parallel_closest_point_worker(args):
    """Worker uses a pre-built global KDTree for instantaneous lookups."""
    kdtree, stl_centroids, points_chunk = args
    
    # Query the tree for the nearest face centroid instantly
    _, closest_face_ids = kdtree.query(points_chunk, k=1)
    
    # Get the actual point coordinates of those closest centroids
    closest_points = stl_centroids[closest_face_ids]
    
    return closest_points, closest_face_ids

def get_closest_points_parallel(mesh_stl, points, num_workers=6, chunk_size=50000):
    """
    Builds a single KDTree and searches it instantly across multiple CPU cores.
    """
    print(f"    -> Building unified Spatial KD-Tree...")
    # Build a tree using the STL mesh's face centroids
    stl_centroids = mesh_stl.vertices[mesh_stl.faces].mean(axis=1)
    kdtree = cKDTree(stl_centroids)
    print(f"    -> Tree ready. Querying across {num_workers} CPU cores...")
    
    # Split target points into small chunks
    chunks = [points[i:i + chunk_size] for i in range(0, len(points), chunk_size)]
    worker_args = [(kdtree, stl_centroids, chunk) for chunk in chunks]
    
    closest_points_list = []
    face_indices_list = []
    
    with Pool(processes=num_workers) as pool:
        for closest, face_id in pool.map(_parallel_closest_point_worker, worker_args):
            closest_points_list.append(closest)
            face_indices_list.append(face_id)
            
    return np.vstack(closest_points_list), np.concatenate(face_indices_list)

def load_camera_trajectory(position_file_path, axis_transform=None):
    """
    Reads the camera position file and returns an (N, 3) array with coordinates.
    Expected format: "Frame XXXXX Position: X=..., Y=..., Z=..."
    
    axis_transform: Function to transform coordinates (e.g., to invert or swap axes)
                    If None, uses coordinates as they are.
    """
    positions = []
    
    try:
        with open(position_file_path, 'r') as f:
            for line in f:
                # Extract X, Y, Z using regex
                match = re.search(r'X=([-\d.]+),\s*Y=([-\d.]+),\s*Z=([-\d.]+)', line)
                if match:
                    x, y, z = float(match.group(1)), float(match.group(2)), float(match.group(3))
                    
                    # Apply transformation if provided
                    if axis_transform:
                        x, y, z = axis_transform(x, y, z)
                    
                    positions.append([x, y, z])
        
        if positions:
            num_frames = len(positions)
            print(f"  ✓ Trajectory loaded: {num_frames} frames")
            return np.array(positions, dtype=np.float64)
        else:
            print(f"  ✗ No positions found in the file")
            return None
            
    except FileNotFoundError:
        print(f"  ✗ File not found: {position_file_path}")
        return None

def interpolate_trajectory(positions, factor=10):
    """
    Interpolates the trajectory using cubic spline to increase the number of frames.
    
    positions: (N, 3) array with trajectory coordinates
    factor: multiplication factor (default 10 = 10x frames)
    
    Returns:
        Interpolated array of shape (N*factor - factor + 1, 3)
    """
    if positions is None or len(positions) < 2:
        return positions
    
    print(f"  Interpolating trajectory ({factor}x factor)...")
    
    num_original = len(positions)
    
    # Original frames
    t_original = np.arange(num_original)
    
    # Create cubic spline for each coordinate
    cs_x = CubicSpline(t_original, positions[:, 0])
    cs_y = CubicSpline(t_original, positions[:, 1])
    cs_z = CubicSpline(t_original, positions[:, 2])
    
    # Interpolated frames (increase density by factor)
    t_interpolated = np.linspace(0, num_original - 1, num_original * factor - (factor - 1))
    
    # Interpolate coordinates
    x_interp = cs_x(t_interpolated)
    y_interp = cs_y(t_interpolated)
    z_interp = cs_z(t_interpolated)
    
    positions_interpolated = np.column_stack([x_interp, y_interp, z_interp])
    
    num_interpolated = len(positions_interpolated)
    print(f"  ✓ Trajectory interpolated: {num_original} → {num_interpolated} frames")
    
    return positions_interpolated

def create_trajectory_line(positions, color=(255, 0, 0, 255)):
    """
    Creates a LineSegments mesh from the camera trajectory.
    color: RGBA tuple, default red
    """
    if positions is None or len(positions) < 2:
        return None
    
    # Create line segments between consecutive points
    segments = []
    for i in range(len(positions) - 1):
        segments.append([positions[i], positions[i + 1]])
    
    # Create the line mesh
    trajectory_mesh = trimesh.load_path(np.array(segments))
    
    # Color the trajectory
    if hasattr(trajectory_mesh, 'entities'):
        for entity in trajectory_mesh.entities:
            entity.color = color
    
    return trajectory_mesh

def create_reference_frame(origin=(0, 0, 0), size=5.0):
    """
    Creates an XYZ reference axis system.
    X -> Red, Y -> Green, Z -> Blue
    origin: point of origin (default global origin)
    size: axis length in units
    """
    origin = np.array(origin)
    
    # Define the three axes
    x_axis = np.array([origin, origin + np.array([size, 0, 0])])
    y_axis = np.array([origin, origin + np.array([0, size, 0])])
    z_axis = np.array([origin, origin + np.array([0, 0, size])])
    
    # Create three paths for the three axes
    x_line = trimesh.load_path(x_axis)
    y_line = trimesh.load_path(y_axis)
    z_line = trimesh.load_path(z_axis)
    
    # Color the axes
    if hasattr(x_line, 'entities'):
        for entity in x_line.entities:
            entity.color = [255, 0, 0, 255]  # Red
    
    if hasattr(y_line, 'entities'):
        for entity in y_line.entities:
            entity.color = [0, 255, 0, 255]  # Green
    
    if hasattr(z_line, 'entities'):
        for entity in z_line.entities:
            entity.color = [0, 0, 255, 255]  # Blue
    
    # Combine the three axes into a list
    return [x_line, y_line, z_line]

def save_colormap_legend(output_dir):
    from matplotlib.colors import Normalize

    fig, axes = plt.subplots(2, 1, figsize=(12, 3.5))

    # 2nd plot: radial distance
    norm_distance = Normalize(vmin=0.0, vmax=1.0)
    sm_distance = cm.ScalarMappable(norm=norm_distance, cmap=plt.colormaps["turbo"])
    sm_distance.set_array([])
    cbar1 = fig.colorbar(sm_distance, cax=axes[0], orientation="horizontal")
    cbar1.set_label("Normalized radial distance", fontsize=20)
    axes[0].set_title("Colormap used in 2nd plot", fontsize=18, fontweight="bold")
    axes[0].tick_params(labelsize=14)

    # 3rd plot: normal consistency
    norm_normal = Normalize(vmin=0.0, vmax=90.0)
    sm_normal = cm.ScalarMappable(norm=norm_normal, cmap=plt.colormaps["plasma"])
    sm_normal.set_array([])
    cbar2 = fig.colorbar(sm_normal, cax=axes[1], orientation="horizontal")
    cbar2.set_label("Normal difference [deg]", fontsize=20)
    axes[1].set_title("Colormap used in 3rd plot", fontsize=18, fontweight="bold")
    axes[1].tick_params(labelsize=14)

    fig.tight_layout()

    output_path = os.path.join(output_dir, "colormaps_distance_normal_consistency.png")
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"  ✓ Colormap saved: {output_path}")

    plt.close(fig)
    
def compute_rms_distance_per_frame(mesh_ply, trajectory_positions, distances_radial):
    """
    Computes the RMS of radial distances for each trajectory frame.
    """
    face_centroids = mesh_ply.vertices[mesh_ply.faces].mean(axis=1)

    kdtree_trajectory = cKDTree(trajectory_positions)
    _, closest_frame_indices = kdtree_trajectory.query(face_centroids, k=1)

    num_frames = len(trajectory_positions)
    rms_per_frame = np.zeros(num_frames)

    for frame_idx in range(num_frames):
        triangle_indices = np.where(closest_frame_indices == frame_idx)[0]

        if len(triangle_indices) == 0:
            rms_per_frame[frame_idx] = 0.0
            continue

        nearby_distances = distances_radial[triangle_indices]
        rms_per_frame[frame_idx] = np.sqrt(np.mean(nearby_distances ** 2))

    return rms_per_frame

def compute_hausdorff_distance_per_frame(mesh_ply, mesh_stl, trajectory_positions):
    """
    For each PLY mesh triangle, identifies the closest trajectory frame.
    Then, for each trajectory frame:
    1. Collects PLY triangles associated with that frame
    2. Computes Hausdorff distance between vertices of these triangles and the STL mesh
    3. Hausdorff distance is the maximum of these distances
    
    Returns:
        (num_frames,) array with Hausdorff distance for each frame
    """
    print("\n  Computing Hausdorff distance for each trajectory frame...")
    
    # Compute PLY triangle centroids
    face_centroids = mesh_ply.vertices[mesh_ply.faces].mean(axis=1)
    
    # For each PLY centroid, find the closest trajectory frame
    kdtree_trajectory = cKDTree(trajectory_positions)
    _, closest_frame_indices = kdtree_trajectory.query(face_centroids, k=1)
    
    # Build KDTree of STL centroids for fast distance calculation
    stl_centroids = mesh_stl.vertices[mesh_stl.faces].mean(axis=1)
    kdtree_stl = cKDTree(stl_centroids)
    
    # For each frame, group the closest PLY triangles
    num_frames = len(trajectory_positions)
    hausdorff_per_frame = np.zeros(num_frames)
    
    for frame_idx in range(num_frames):
        # Find all PLY triangles associated with this frame
        triangle_indices = np.where(closest_frame_indices == frame_idx)[0]
        
        if len(triangle_indices) == 0:
            # If no triangle is associated with this frame, use 0
            hausdorff_per_frame[frame_idx] = 0.0
            continue
        
        # Collect all vertices of these triangles (deduplicated)
        triangles = mesh_ply.faces[triangle_indices]  # (num_tri, 3)
        vertices_idx = np.unique(triangles.flatten())
        vertices = mesh_ply.vertices[vertices_idx]  # (num_vertices, 3)
        
        # For each vertex, compute the minimum distance to the STL mesh
        dists_to_stl, _ = kdtree_stl.query(vertices, k=1)
        
        # Hausdorff distance for this frame is the maximum
        hausdorff_per_frame[frame_idx] = np.max(dists_to_stl)
    
    print(f"  ✓ Computation completed")
    return hausdorff_per_frame

def compute_normal_consistency_per_frame(mesh_ply, positions_scaled, normal_difference_values):
    face_centroids = mesh_ply.vertices[mesh_ply.faces].mean(axis=1)

    kdtree_trajectory = cKDTree(positions_scaled)
    _, closest_frame_indices = kdtree_trajectory.query(face_centroids, k=1)

    num_frames = len(positions_scaled)
    normal_consistency_per_frame = np.zeros(num_frames)

    for frame_idx in range(num_frames):
        triangle_indices = np.where(closest_frame_indices == frame_idx)[0]

        if len(triangle_indices) == 0:
            normal_consistency_per_frame[frame_idx] = 0.0
            continue

        normal_consistency_per_frame[frame_idx] = np.mean(normal_difference_values[triangle_indices])

    return normal_consistency_per_frame

def plot_metric_vs_trajectory(values, ylabel, title, num_frames, output_path=None):
    """
    Plots a single metric vs trajectory progression.
    """
    frame_percentages = np.linspace(0, 100, num_frames)

    fig, ax = plt.subplots(figsize=(25, 10))
    ax.plot(frame_percentages, values, linewidth=2, color="#4C78A8", marker="o", markersize=4)

    ax.set_xlabel("Trajectory Progress (%)", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.set_title(title, fontsize=25, fontweight="bold")

    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.tick_params(axis="both", which="minor", labelsize=18)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        print(f"  ✓ Chart saved: {output_path}")

    plt.show()
    plt.close(fig)

def compute_radial_distance_per_frame(mesh_ply, trajectory_positions, distances_radial):
    """
    For each PLY mesh triangle, identifies the closest trajectory frame.
    Then, for each trajectory frame:
    1. Groups PLY triangles associated with that frame
    2. Computes the mean radial distance of these triangles from the STL mesh
    
    Returns:
        (num_frames,) array with mean radial distance for each frame
    """
    print("\n  Computing mean radial distance for each trajectory frame...")
    
    # Compute PLY face centroids
    face_centroids = mesh_ply.vertices[mesh_ply.faces].mean(axis=1)
    
    # For each PLY centroid, find the closest trajectory frame
    kdtree_trajectory = cKDTree(trajectory_positions)
    _, closest_frame_indices = kdtree_trajectory.query(face_centroids, k=1)
    
    # For each frame, group the closest PLY triangles
    num_frames = len(trajectory_positions)
    distances_per_frame = np.zeros(num_frames)
    
    for frame_idx in range(num_frames):
        # Find all PLY triangles associated with this frame
        triangle_indices = np.where(closest_frame_indices == frame_idx)[0]
        
        if len(triangle_indices) == 0:
            # If no triangle is associated with this frame, use 0
            distances_per_frame[frame_idx] = 0.0
            continue
        
        # Radial distances of these triangles
        nearby_distances = distances_radial[triangle_indices]
        
        # Mean of radial distances
        distances_per_frame[frame_idx] = np.mean(nearby_distances)
    
    print(f"  ✓ Computation completed")
    return distances_per_frame

def filter_mesh_small_regions(mesh, min_triangles=10):
    """
    Remove all connected regions with fewer than min_triangles triangles from a trimesh.
    Uses edge-based connectivity without external dependencies.
    
    Args:
        mesh: trimesh.Trimesh object to filter
        min_triangles: Minimum number of triangles per region (default: 10)
    
    Returns:
        Filtered trimesh.Trimesh with small regions removed, or None if no regions remain
    """
    if mesh is None or len(mesh.faces) == 0:
        return None
    
    if len(mesh.faces) < min_triangles:
        print(f"  ⚠️  Mesh has {len(mesh.faces)} triangles, fewer than minimum {min_triangles}")
        return None
    
    print(f"  Filtering connected regions (min {min_triangles} triangles)...")
    
    faces = np.asarray(mesh.faces)
    n_faces = len(faces)
    
    # Build face adjacency using edges
    # Two faces are adjacent if they share an edge (2 vertices)
    face_adjacency = [set() for _ in range(n_faces)]
    
    # Extract all edges and associate with faces
    edge_to_faces = {}
    for face_idx, face in enumerate(faces):
        v0, v1, v2 = face
        edges = [(min(v0, v1), max(v0, v1)), 
                 (min(v1, v2), max(v1, v2)), 
                 (min(v2, v0), max(v2, v0))]
        for edge in edges:
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(face_idx)
    
    # Build adjacency from shared edges
    for face_indices in edge_to_faces.values():
        for i in range(len(face_indices)):
            for j in range(i + 1, len(face_indices)):
                face_adjacency[face_indices[i]].add(face_indices[j])
                face_adjacency[face_indices[j]].add(face_indices[i])
    
    # Find connected components using DFS
    visited = np.zeros(n_faces, dtype=bool)
    components = []
    
    def dfs(start_face):
        stack = [start_face]
        component = []
        while stack:
            face = stack.pop()
            if visited[face]:
                continue
            visited[face] = True
            component.append(face)
            for neighbor in face_adjacency[face]:
                if not visited[neighbor]:
                    stack.append(neighbor)
        return component
    
    for face_idx in range(n_faces):
        if not visited[face_idx]:
            component = dfs(face_idx)
            components.append(component)
    
    # Filter components by size
    large_components = [c for c in components if len(c) >= min_triangles]
    
    if len(large_components) == 0:
        print(f"  ⚠️  No regions with ≥{min_triangles} triangles found")
        return None
    
    removed_count = len(components) - len(large_components)
    print(f"  Found {len(components)} regions → Keeping {len(large_components)}, removing {removed_count}")
    
    # Extract large component faces
    kept_face_indices = np.concatenate(large_components)
    kept_faces = faces[kept_face_indices]
    
    # Build submesh from kept faces
    used_vertices = np.unique(kept_faces.flatten())
    vertex_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
    
    new_vertices = mesh.vertices[used_vertices]
    new_faces = np.array([[vertex_mapping[v] for v in face] for face in kept_faces], dtype=np.int32)
    
    filtered_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
    
    print(f"  ✓ Filtered mesh: {len(filtered_mesh.faces)} faces, {len(filtered_mesh.vertices)} vertices")
    
    return filtered_mesh

# --- UNIFIED COMPUTATION & VISUALIZATION ---
def compute_and_visualize(ply_path, stl_path, trajectory_path=None, max_distance=1.2, axis_transform=None, output_dir=None):
    """
    Computes metrics and visualizes in a single pass (NO double calculations).
    trajectory_path: path to the camera position file (optional)
    axis_transform: function to transform trajectory coordinates (e.g., invert axes)
    output_dir: folder to save results (meshes, charts, metrics) - optional
    """
    # Create output folder if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"  ✓ Output folder created: {output_dir}")
        
    print("\n[1/5] Loading meshes...")
    mesh_ply = trimesh.load(ply_path)
    mesh_stl = trimesh.load(stl_path)
    
    # Filter small connected regions from PLY mesh (keep only regions with ≥20 triangles)
    print("  Cleaning PLY mesh: removing small connected regions...")
    mesh_ply = filter_mesh_small_regions(mesh_ply, min_triangles=20)
    if mesh_ply is None:
        print("✗ Error: PLY mesh has no regions with ≥20 triangles")
        return
    
    print(f"✓ PLY Mesh: {len(mesh_ply.faces)} faces, {len(mesh_ply.vertices)} vertices")
    print(f"✓ STL Mesh: {len(mesh_stl.faces)} faces, {len(mesh_stl.vertices)} vertices")
    
    # --- RADIAL DISTANCE ---
    print("\n[2/5] Computing metrics (Parallel cKDTree)...")
    print("  Computing radial distances...")
    face_centroids = mesh_ply.vertices[mesh_ply.faces].mean(axis=1)
    normals_missing = np.asarray(mesh_ply.face_normals, dtype=np.float64)

    closest_points, closest_face_indices_faces = get_closest_points_parallel(mesh_stl, face_centroids, num_workers=6)
    closest_points = np.asarray(closest_points, dtype=np.float64)
    closest_face_indices_faces = np.asarray(closest_face_indices_faces, dtype=int)

    vecs = closest_points - face_centroids
    radial_distances = np.sum(vecs * normals_missing, axis=1)
    distances = np.abs(radial_distances)
    rms_distance = np.sqrt(np.mean(distances ** 2))

    # --- NORMAL CONSISTENCY / NORMAL CORRELATION ---
    print("  Computing normal consistency and correlation...")
    vertices_missing = np.asarray(mesh_ply.vertices, dtype=np.float64)
    normals_missing_v = np.asarray(mesh_ply.vertex_normals, dtype=np.float64)

    closest_points_v, closest_face_indices_vertices = get_closest_points_parallel(mesh_stl, vertices_missing, num_workers=6)
    closest_points_v = np.asarray(closest_points_v, dtype=np.float64)
    closest_face_indices_vertices = np.asarray(closest_face_indices_vertices, dtype=int)

    faces_stl = mesh_stl.faces[closest_face_indices_vertices]
    face_vertices_all = mesh_stl.vertices[faces_stl]

    diffs = face_vertices_all - closest_points_v[:, np.newaxis, :]
    distances_all = np.linalg.norm(diffs, axis=2)
    closest_in_face = np.argmin(distances_all, axis=1)
    closest_vertex_indices = faces_stl[np.arange(len(faces_stl)), closest_in_face]

    # Angular consistency using closest GT vertex normal
    normals_gt_vertex = np.asarray(mesh_stl.vertex_normals, dtype=np.float64)
    normals_gt_corresponding = normals_gt_vertex[closest_vertex_indices]

    dot_products = np.sum(normals_missing_v * normals_gt_corresponding, axis=1)
    dot_products = np.abs(np.clip(dot_products, -1, 1))
    angles = np.arccos(np.clip(dot_products, -1, 1)) * 180 / np.pi

    # New metric: compare each PLY vertex normal to the GT face normal
    # of the same closest face used for the distance correspondence
    face_normals_gt = np.asarray(mesh_stl.face_normals, dtype=np.float64)
    face_normals_gt_corresponding = face_normals_gt[closest_face_indices_vertices]

    normal_norms_ply = np.linalg.norm(normals_missing_v, axis=1)
    normal_norms_gt = np.linalg.norm(face_normals_gt_corresponding, axis=1)
    safe_norms_ply = np.where(normal_norms_ply > 0, normal_norms_ply, 1.0)
    safe_norms_gt = np.where(normal_norms_gt > 0, normal_norms_gt, 1.0)

    normalized_missing = normals_missing_v / safe_norms_ply[:, np.newaxis]
    normalized_gt = face_normals_gt_corresponding / safe_norms_gt[:, np.newaxis]
    cosine_similarity = np.sum(normalized_missing * normalized_gt, axis=1)
    cosine_similarity = np.clip(cosine_similarity, -1, 1)
    normal_correlation = np.abs(cosine_similarity)
    
    # --- PRINT RESULTS ---
    print("\n[3/5] Results...")
    print("\n" + "="*70)
    print("FIDELITY METRICS - POISSON VALIDATION")
    print("="*70)
    
    print(f"\n--- RADIAL DISTANCE ---")
    print(f"Mean:      {np.mean(distances):.6f}")
    print(f"Median:    {np.median(distances):.6f}")
    print(f"Std Dev:   {np.std(distances):.6f}")
    print(f"RMS:       {rms_distance:.6f}")
    print(f"Min/Max:   {np.min(distances):.6f} / {np.max(distances):.6f}")

    print(f"\n--- NORMAL CONSISTENCY ---")
    print(f"Mean angle:        {np.mean(angles):.2f}°")
    print(f"Median angle:      {np.median(angles):.2f}°")
    print(f"Std Dev:           {np.std(angles):.2f}°")
    print(f"Min/Max:           {np.min(angles):.2f}° / {np.max(angles):.2f}°")

    print(f"\n--- NORMAL CORRELATION (vertex normals vs closest GT face normals) ---")
    print(f"Mean:      {np.mean(normal_correlation):.6f}")
    print(f"Median:    {np.median(normal_correlation):.6f}")
    print(f"Std Dev:   {np.std(normal_correlation):.6f}")
    print(f"Min/Max:   {np.min(normal_correlation):.6f} / {np.max(normal_correlation):.6f}")
    
    print("\n" + "="*70)
    
    # --- SAVE METRICS IN TEXT FILE ---
    if output_dir:
        metrics_file = os.path.join(output_dir, "validation_metrics.txt")
        with open(metrics_file, 'w') as f:
            f.write("FIDELITY METRICS - POISSON VALIDATION\n")
            f.write("=" * 70 + "\n")
            f.write(f"\n--- RADIAL DISTANCE ---\n")
            f.write(f"Mean:      {np.mean(distances):.6f}\n")
            f.write(f"Median:    {np.median(distances):.6f}\n")
            f.write(f"Std Dev:   {np.std(distances):.6f}\n")
            f.write(f"RMS:       {rms_distance:.6f}\n")
            f.write(f"Min/Max:   {np.min(distances):.6f} / {np.max(distances):.6f}\n")

            f.write(f"\n--- NORMAL CONSISTENCY ---\n")
            f.write(f"Mean angle:        {np.mean(angles):.2f}°\n")
            f.write(f"Median angle:      {np.median(angles):.2f}°\n")
            f.write(f"Std Dev:           {np.std(angles):.2f}°\n")
            f.write(f"Min/Max:           {np.min(angles):.2f}° / {np.max(angles):.2f}°\n")

            f.write(f"\n--- NORMAL CORRELATION (vertex normals vs closest GT face normals) ---\n")
            f.write(f"Mean:      {np.mean(normal_correlation):.6f}\n")
            f.write(f"Median:    {np.median(normal_correlation):.6f}\n")
            f.write(f"Std Dev:   {np.std(normal_correlation):.6f}\n")
            f.write(f"Min/Max:   {np.min(normal_correlation):.6f} / {np.max(normal_correlation):.6f}\n")
        print(f"  ✓ Metrics saved: {metrics_file}")
    
    # --- PREPARE COLORMAP FOR METRICS ---
    print(f"\n[4/5] Generating visualization...")

    def build_colored_mesh(mesh, values, cmap_name='viridis', vmin=None, vmax=None):
        values = np.asarray(values, dtype=np.float64)
        if vmin is None:
            vmin = np.nanmin(values)
        if vmax is None:
            vmax = np.nanmax(values)
        if vmax <= vmin:
            vmax = vmin + 1.0

        normalized = np.clip((values - vmin) / (vmax - vmin), 0, 1)
        cmap = plt.colormaps[cmap_name]
        colors_rgba = cmap(normalized)
        colors = (colors_rgba[:, :3] * 255).astype(np.uint8)
        colors = np.column_stack([colors, np.full(len(colors), 255, dtype=np.uint8)])

        faces_double = np.vstack([mesh.faces, mesh.faces[:, [2, 1, 0]]])
        colors_double = np.vstack([colors, colors])

        mesh_vis = trimesh.Trimesh(vertices=mesh.vertices, faces=faces_double, process=False)
        mesh_vis.visual.face_colors = colors_double
        return mesh_vis

    # 1. RED MISSING REGIONS MESH
    faces_double = np.vstack([mesh_ply.faces, mesh_ply.faces[:, [2, 1, 0]]])
    mesh_ply_vis_red = trimesh.Trimesh(vertices=mesh_ply.vertices, faces=faces_double, process=False)
    # RGBA Color: Bright red (255, 0, 0, 255) for all faces
    red_colors = np.full((len(faces_double), 4), [255, 0, 0, 255], dtype=np.uint8)
    mesh_ply_vis_red.visual.face_colors = red_colors

    # 2. RADIAL DISTANCE MESH
    distance_values = np.clip(distances / max_distance, 0, 1) ** 2.5
    mesh_ply_vis_distance = build_colored_mesh(mesh_ply, distance_values, cmap_name='turbo')
    if output_dir:
        save_colormap_legend(output_dir)

    # 3. NORMAL CONSISTENCY MESH
    face_normals_ply = np.asarray(mesh_ply.face_normals, dtype=np.float64)
    face_normals_gt = np.asarray(mesh_stl.face_normals, dtype=np.float64)
    face_normals_gt_corresponding = face_normals_gt[closest_face_indices_faces]

    face_dot_products = np.sum(face_normals_ply * face_normals_gt_corresponding, axis=1)
    face_dot_products = np.abs(np.clip(face_dot_products, -1, 1))
    normal_difference_values = np.degrees(np.arccos(face_dot_products))

    mesh_ply_vis = build_colored_mesh(
        mesh_ply,
        normal_difference_values,
        cmap_name='plasma',
        vmin=0.0,
        vmax=90.0,
    )

    mesh_stl_vis = mesh_stl.copy()
    mesh_stl_vis.visual.vertex_colors = [100, 100, 100, 50]
    
    # --- 3D MESH BOUNDING BOX ---
    mesh_for_bbox = mesh_ply_vis_distance
    bounds = mesh_for_bbox.bounds
    min_corner = bounds[0]
    max_corner = bounds[1]

    bbox_extents = max_corner - min_corner
    bbox_center = (min_corner + max_corner) / 2.0

    print("\n--- MESH BOUNDING BOX ---")
    print(f"Min corner: {min_corner}")
    print(f"Max corner: {max_corner}")
    print(f"Extents (Lx, Ly, Lz): {bbox_extents}")
    print(f"Center: {bbox_center}")
    print(f"Volume: {np.prod(bbox_extents):.6f}")

    bbox_mesh = trimesh.creation.box(
        extents=bbox_extents,
        transform=trimesh.transformations.translation_matrix(bbox_center)
    )
    bbox_mesh.visual.face_colors = [0, 0, 0, 0]
    
    if output_dir:
        mesh_red_output_path = os.path.join(output_dir, "mesh_validation_red_mr.ply")
        mesh_ply_vis_red.export(mesh_red_output_path)
        print(f"  ✓ Red PLY mesh saved: {mesh_red_output_path}")

        mesh_distance_output_path = os.path.join(output_dir, "mesh_validation_colored_distance.ply")
        mesh_ply_vis_distance.export(mesh_distance_output_path)
        print(f"  ✓ Distance-colored PLY mesh saved: {mesh_distance_output_path}")

        mesh_normal_diff_output_path = os.path.join(output_dir, "mesh_validation_colored_normal_diff.ply")
        mesh_ply_vis.export(mesh_normal_diff_output_path)
        print(f"  ✓ Normal difference-colored PLY mesh saved: {mesh_normal_diff_output_path}")
    
    # --- LOAD TRAJECTORY (if provided) ---
    trajectory_mesh = None
    num_frames = 0
    distances_per_frame = None
    hausdorff_per_frame = None
    rms_per_frame = None
    
    if trajectory_path and os.path.exists(trajectory_path):
        print(f"  Loading trajectory from file...")
        positions = load_camera_trajectory(trajectory_path, axis_transform=axis_transform)
        if positions is not None:
            positions = interpolate_trajectory(positions, factor=10)
            num_frames = len(positions)
            positions_scaled = positions * 10 
            trajectory_mesh = create_trajectory_line(positions_scaled, color=(255, 0, 0, 255))
            if trajectory_mesh:
                print(f"  ✓ Trajectory added - {num_frames} frames to display")
            
            print(f"\n[4/5] Analyzing distance vs trajectory...")
            distances_per_frame = compute_radial_distance_per_frame(mesh_ply, positions_scaled, distances)
            hausdorff_per_frame = compute_hausdorff_distance_per_frame(mesh_ply, mesh_stl, positions_scaled)
            print("\n--- HAUSDORFF DISTANCE ---")
            print(f"Mean:      {np.mean(hausdorff_per_frame):.6f}")
            print(f"Median:    {np.median(hausdorff_per_frame):.6f}")
            print(f"Std Dev:   {np.std(hausdorff_per_frame):.6f}")
            print(f"Min/Max:   {np.min(hausdorff_per_frame):.6f} / {np.max(hausdorff_per_frame):.6f}")

            rms_per_frame = compute_rms_distance_per_frame(mesh_ply, positions_scaled, distances)
            normal_consistency_per_frame = compute_normal_consistency_per_frame(
                mesh_ply, positions_scaled, normal_difference_values
            )
    elif trajectory_path:
        print(f"  ⚠️ Trajectory not found at specified path. Proceeding without trajectory.")

    # --- CREATE REFERENCE AXIS SYSTEM ---
    print(f"  Creating axis system...")
    axis_lines = create_reference_frame(origin=(0, 0, 0), size=5.0)
    
    print(f"\n✓ Visualization ready")
    print(f"[5/5] SUMMARY:")
    print(f"  • PLY Mesh: {len(mesh_ply.vertices)} vertices, {len(mesh_ply.faces)} faces")
    print(f"  • STL Mesh: {len(mesh_stl.vertices)} vertices, {len(mesh_stl.faces)} faces")
    if num_frames > 0:
        print(f"  • Trajectory: {num_frames} frames")
    print(f"\nStarting sequential visualization (3 3D windows)...")
    print("Close each window to proceed to the next one\n")
    
    # ---------------------------------------------------------
    # PLOT 1: RED Missing Regions on Ground Truth + Trajectory
    # ---------------------------------------------------------
    scene_objects_red = [mesh_stl_vis, mesh_ply_vis_red, bbox_mesh] + axis_lines
    if trajectory_mesh:
        scene_objects_red.append(trajectory_mesh)

    scene_red = trimesh.Scene(scene_objects_red)
    print("--> 1/3: Displaying Missing Regions (Red)...")
    scene_red.show()

    print("\nClose the first plot to see the distance map...\n")

    # ---------------------------------------------------------
    # PLOT 2: Radial Distance Colormap
    # ---------------------------------------------------------
    scene_objects_dist = [mesh_stl_vis, mesh_ply_vis_distance, bbox_mesh] + axis_lines
    if trajectory_mesh:
        scene_objects_dist.append(trajectory_mesh)

    scene_dist = trimesh.Scene(scene_objects_dist)
    print("--> 2/3: Displaying Radial Distance...")
    scene_dist.show()

    print("\nClose the second plot to see Normal Consistency...\n")

    # ---------------------------------------------------------
    # PLOT 3: Normal Consistency Colormap
    # ---------------------------------------------------------
    scene_objects_norm = [mesh_stl_vis, mesh_ply_vis, bbox_mesh] + axis_lines
    if trajectory_mesh:
        scene_objects_norm.append(trajectory_mesh)

    scene_norm = trimesh.Scene(scene_objects_norm)
    print("--> 3/3: Displaying Normal Consistency...")
    scene_norm.show()

    # Distance charts
    if distances_per_frame is not None and normal_consistency_per_frame is not None and hausdorff_per_frame is not None and rms_per_frame is not None and num_frames > 0:
        print(f"\n[6/6] Generating charts...")
        if output_dir:
            plot_metric_vs_trajectory(
                distances_per_frame,
                "Mean radial distance [mm]",
                "Mean radial distance",
                num_frames,
                os.path.join(output_dir, "mean_radial_distance.png")
            )

            plot_metric_vs_trajectory(
                rms_per_frame,
                "RMS [mm]",
                "RMS",
                num_frames,
                os.path.join(output_dir, "rms_distance.png")
            )

            plot_metric_vs_trajectory(
                hausdorff_per_frame,
                "Hausdorff distance [mm]",
                "Hausdorff distance",
                num_frames,
                os.path.join(output_dir, "hausdorff_distance.png")
            )

            plot_metric_vs_trajectory(
                normal_consistency_per_frame,
                "Normal difference [deg]",
                "Normal consistency",
                num_frames,
                os.path.join(output_dir, "normal_consistency.png")
            )
        print(f"  ✓ Charts saved and displayed")

# --- ENTRY POINT ---
if __name__ == "__main__":
    # ----------------------------------------------------------------------------------
    # PATH CONFIGURATION (To be filled before execution)
    # ----------------------------------------------------------------------------------
    
    ply_file = "PATH/TO/YOUR/MESH.ply"                  # E.g.: "data/MR_pymeshfix.ply"
    stl_file = "PATH/TO/YOUR/GROUND_TRUTH.stl"          # E.g.: "data/colon_model_v3_STL.stl"
    trajectory_file = "PATH/TO/YOUR/TRAJECTORY.txt"     # E.g.: "data/Camera_Position_Data.txt" (Optional)
    output_folder = "PATH/TO/YOUR/OUTPUT_DIRECTORY"     # E.g.: "data/output"
    
    # NOTE: This function inverts the Y axis. Modify or set to None if not needed for your data.
    axis_transform = lambda x, y, z: (x, -y, z)
    
    try:
        compute_and_visualize(
            ply_path=ply_file, 
            stl_path=stl_file, 
            trajectory_path=trajectory_file, 
            max_distance=1.2, 
            axis_transform=axis_transform, 
            output_dir=output_folder
        )
        
    except FileNotFoundError as e:
        print(f"Error: File not found - Ensure correct paths are provided. Details: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()