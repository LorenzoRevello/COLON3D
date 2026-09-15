"""
NUOVO_RECONSTRUCTION+POSE_REALTIME.py
=====================================
Pipeline Real-time integrata:
  1. DEPTH estimation frame-by-frame with SUMNet
  2. POSE estimation frame-by-frame with PoseNet 
  3. TSDF integration 
  4. Poisson mesh closing + missing regions analysis
  5. real-time and offline visualization
"""


import sys
import time
import traceback
from pathlib import Path

import cv2
import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering
from PIL import Image
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

import tkinter as tk
from tkinter import Listbox, Button, Label

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# ✅ IMPORT POSE ESTIMATION UTILITIES
sys.path.insert(0, str(PROJECT_ROOT / "bimodal_camera_pose"))
import test_MODULO as pose_utils
from inverse_warp import get_bins_quat
import models as pose_models

matplotlib.use("Agg")

# ══════════════════════════════════════════════════════════════════════
#  DEVICE SETUP
# ══════════════════════════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ══════════════════════════════════════════════════════════════════════
#  UTILITY TIMING
# ══════════════════════════════════════════════════════════════════════

def _fmt_seconds(seconds):
    seconds = float(seconds)
    mins = int(seconds // 60)
    secs = seconds % 60.0
    return f"{mins:02d}m {secs:05.2f}s"

def print_final_timing_report(avg_depth_s, avg_pose_s, reconstruction_s, closing_s, missing_analysis_s):
    total_s = float(reconstruction_s) + float(closing_s) + float(missing_analysis_s)
    print(f"\n{'=' * 68}")
    print("FINAL TIMING REPORT")
    print(f"{'=' * 68}")
    print(f"Avg Depth Estimation (per frame)   : {avg_depth_s * 1000:.2f} ms")
    print(f"Avg Pose Estimation (per frame)    : {avg_pose_s * 1000:.2f} ms")
    print(f"Reconstruction (Depth + Pose + TSDF) : {_fmt_seconds(reconstruction_s)}")
    print(f"Closing (Poisson)                  : {_fmt_seconds(closing_s)}")
    print(f"Missing regions extraction/analysis: {_fmt_seconds(missing_analysis_s)}")
    print(f"Total                              : {_fmt_seconds(total_s)}")
    print(f"{'=' * 68}")


# ══════════════════════════════════════════════════════════════════════
#  DEPTH ESTIMATION MODEL (SUMNet)
# ══════════════════════════════════════════════════════════════════════

class conv_bn(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_bn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
        )
    def forward(self, x):
        return self.conv(x)

class SUMNet(nn.Module):
    def __init__(self):
        super(SUMNet, self).__init__()
        self.encoder   = models.vgg11(weights='IMAGENET1K_V1').features
        self.conv1     = nn.Sequential(self.encoder[0], nn.BatchNorm2d(64))
        self.pool1     = nn.MaxPool2d(2, 2, return_indices=True)
        self.conv2     = nn.Sequential(self.encoder[3], nn.BatchNorm2d(128))
        self.pool2     = nn.MaxPool2d(2, 2, return_indices=True)
        self.conv3a    = nn.Sequential(self.encoder[6], nn.BatchNorm2d(256))
        self.conv3b    = nn.Sequential(self.encoder[8], nn.BatchNorm2d(256))
        self.pool3     = nn.MaxPool2d(2, 2, return_indices=True)
        self.conv4a    = nn.Sequential(self.encoder[11], nn.BatchNorm2d(512))
        self.conv4b    = nn.Sequential(self.encoder[13], nn.BatchNorm2d(512))
        self.pool4     = nn.MaxPool2d(2, 2, return_indices=True)
        self.conv5a    = nn.Sequential(self.encoder[16], nn.BatchNorm2d(512))
        self.conv5b    = nn.Sequential(self.encoder[18], nn.BatchNorm2d(512))
        self.pool5     = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool5   = nn.MaxUnpool2d(2, 2)
        self.donv5b    = conv_bn(1024, 512)
        self.donv5a    = conv_bn(512, 512)
        self.unpool4   = nn.MaxUnpool2d(2, 2)
        self.donv4b    = conv_bn(1024, 512)
        self.donv4a    = conv_bn(512, 256)
        self.unpool3   = nn.MaxUnpool2d(2, 2)
        self.donv3b    = conv_bn(512, 256)
        self.donv3a    = conv_bn(256, 128)
        self.unpool2   = nn.MaxUnpool2d(2, 2)
        self.donv2     = conv_bn(256, 64)
        self.unpool1   = nn.MaxUnpool2d(2, 2)
        self.donv1     = conv_bn(128, 32)
        self.output    = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        conv1          = F.relu(self.conv1(x), inplace=True)
        pool1, idxs1   = self.pool1(conv1)
        conv2          = F.relu(self.conv2(pool1), inplace=True)
        pool2, idxs2   = self.pool2(conv2)
        conv3a         = F.relu(self.conv3a(pool2), inplace=True)
        conv3b         = F.relu(self.conv3b(conv3a), inplace=True)
        pool3, idxs3   = self.pool3(conv3b)
        conv4a         = F.relu(self.conv4a(pool3), inplace=True)
        conv4b         = F.relu(self.conv4b(conv4a), inplace=True)
        pool4, idxs4   = self.pool4(conv4b)
        conv5a         = F.relu(self.conv5a(pool4), inplace=True)
        conv5b         = F.relu(self.conv5b(conv5a), inplace=True)
        pool5, idxs5   = self.pool5(conv5b)
        unpool5        = torch.cat([self.unpool5(pool5, idxs5), conv5b], 1)
        donv5b         = F.relu(self.donv5b(unpool5), inplace=True)
        donv5a         = F.relu(self.donv5a(donv5b), inplace=True)
        unpool4        = torch.cat([self.unpool4(donv5a, idxs4), conv4b], 1)
        donv4b         = F.relu(self.donv4b(unpool4), inplace=True)
        donv4a         = F.relu(self.donv4a(donv4b), inplace=True)
        unpool3        = torch.cat([self.unpool3(donv4a, idxs3), conv3b], 1)
        donv3b         = F.relu(self.donv3b(unpool3), inplace=True)
        donv3a         = F.relu(self.donv3a(donv3b))
        unpool2        = torch.cat([self.unpool2(donv3a, idxs2), conv2], 1)
        donv2          = F.relu(self.donv2(unpool2), inplace=True)
        unpool1        = torch.cat([self.unpool1(donv2, idxs1), conv1], 1)
        donv1          = F.relu(self.donv1(unpool1), inplace=True)
        output         = self.output(donv1)
        return torch.sigmoid(output)


# ══════════════════════════════════════════════════════════════════════
#  POSE ESTIMATION UTILITIES
# ══════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════
#  SIMCOL3D UTILITIES
# ══════════════════════════════════════════════════════════════════════

def load_intrinsics(path):
    """Load intrinsics from SimCol3D cam.txt - Scaled to 237x237"""
    with open(path, 'r') as f:
        vals = list(map(float, f.read().split()))
    K = np.array(vals).reshape(3, 3)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    width, height = 237, 237
    fx = fx * 0.5
    fy = fy * 0.5
    cx = cx * 0.5
    cy = cy * 0.5
    
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    return intrinsic

def trajectory_from_poses(poses_mat):
    """Extract 3D trajectory from pose matrices."""
    return np.array([p[:3, 3] for p in poses_mat], dtype=float)


# ══════════════════════════════════════════════════════════════════════
#  REAL-TIME VISUALIZATION
# ══════════════════════════════════════════════════════════════════════

class RealTimeTSDFViewer:
    """Dashboard for real-time TSDF + Pose monitoring."""

    def __init__(self, update_every=20, width=1280, height=720, point_size=1.0):
        self.update_every = max(1, int(update_every))
        self.frame_count = 0
        self.start_time = time.time()
        self.closed = False
        self.point_size = float(point_size)

        self.app = gui.Application.instance
        try:
            self.app.initialize()
        except Exception:
            pass

        self.window = self.app.create_window(
            "3D RECONSTRUCTION with REAL-TIME DEPTH & POSE ESTIMATION", 
            width, height
        )
        self.window.set_on_close(self._on_close)
        self.window.set_on_layout(self._on_layout)

        self.left_panel = gui.Vert(10, gui.Margins(8, 8, 8, 8))
        self.left_panel.background_color = gui.Color(0.4, 0.4, 0.4, 1.0)
        
        self.left_panel.add_child(gui.Label(""))


        depth_label = gui.Label("ESTIMATED DEPTH")
        depth_label.text_color = gui.Color(1, 1, 1)

        self.left_panel.add_child(depth_label)
        
        self.depth_widget = gui.ImageWidget(
            o3d.geometry.Image(np.zeros((240, 320, 3), dtype=np.uint8))
        )
        self.left_panel.add_child(self.depth_widget)
        
        separator1 = gui.Label("-" * 75)
        separator1.text_color = gui.Color(0.5, 0.5, 0.5, 1.0)
        self.left_panel.add_child(separator1)

        self.left_panel.add_child(gui.Label(""))


        pose_title = gui.Label("ESTIMATED POSE (Rototraslation Matrix)")
        pose_title.text_color = gui.Color(1, 1, 1)
        self.left_panel.add_child(pose_title)
        
        self.pose_label = gui.Label("[ - ]\n[ - ]\n[ - ]\n[ - ]")
        self.pose_label.text_color = gui.Color(1, 1, 1)

        self.left_panel.add_child(self.pose_label)
        
        separator2 = gui.Label("-" * 75)
        separator2.text_color = gui.Color(0.5, 0.5, 0.5, 1.0)
        self.left_panel.add_child(separator2)
        
        self.left_panel.add_child(gui.Label(""))

        self.progress_label = gui.Label("PROGRESS: 0/0 (0.0%)")
        self.progress_label.text_color = gui.Color(1, 1, 1)
        self.left_panel.add_child(self.progress_label)

        self.timer_label = gui.Label("TIMER: 00:00")
        self.timer_label.text_color = gui.Color(1, 1, 1)
        self.left_panel.add_child(self.timer_label)

        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([0.9, 0.9, 0.9, 1.0])

        self.material = rendering.MaterialRecord()
        self.material.shader = "defaultUnlit"
        self.material.point_size = self.point_size

        self.traj_material = rendering.MaterialRecord()
        self.traj_material.shader = "unlitLine"
        self.traj_material.line_width = 3.0

        self.cam_material = rendering.MaterialRecord()
        self.cam_material.shader = "defaultUnlit"

        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20.0, origin=[0, 0, 0])
        self.scene.scene.add_geometry("coord_frame", coord, rendering.MaterialRecord())
        
        self.window.add_child(self.left_panel)
        self.window.add_child(self.scene)

        self.geometry_added = False
        self._camera_initialized = False
        self.traj_points = []

        self._tick()

    def _on_close(self):
        self.closed = True
        return True

    def _on_layout(self, ctx):
        r = self.window.content_rect
        left_w = min(460, max(360, int(r.width * 0.28)))

        self.left_panel.frame = gui.Rect(r.x, r.y, left_w, r.height)
        self.scene.frame = gui.Rect(r.x + left_w, r.y, r.width - left_w, r.height)

    def _tick(self):
        if not self.closed:
            self.app.run_one_tick()

    def _to_o3d_image_depth_colormap(self, depth):
        if depth is None:
            return o3d.geometry.Image(np.zeros((240, 320, 3), dtype=np.uint8))
        depth = np.asarray(depth)
        if depth.size == 0:
            return o3d.geometry.Image(np.zeros((240, 320, 3), dtype=np.uint8))
        
        if depth.dtype == np.uint16 or np.max(depth) > 1.0:
            d = cv2.normalize(depth.astype(float), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            d = (depth * 255).astype(np.uint8)
            
        d = cv2.applyColorMap(d, cv2.COLORMAP_JET)
        d = cv2.cvtColor(d, cv2.COLOR_BGR2RGB)
        return o3d.geometry.Image(np.ascontiguousarray(d))

    def _update_pose_trajectory(self, pose):
        if pose is None:
            return

        self.traj_points.append(np.asarray(pose[:3, 3]).copy())

        cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=8.0)
        cam.transform(pose)
        if self.scene.scene.has_geometry("camera_pose"):
            self.scene.scene.remove_geometry("camera_pose")
        self.scene.scene.add_geometry("camera_pose", cam, self.cam_material)

        # ✅ Add SafeGuard for LineSet
        if len(self.traj_points) >= 2:
            try:
                pts = np.asarray(self.traj_points, dtype=np.float64)
                
                # Verify points are not degenerate
                if np.all(np.isfinite(pts)) and len(np.unique(pts, axis=0)) > 1:
                    lines = np.array([[i - 1, i] for i in range(1, len(pts))], dtype=np.int32)
                    colors = np.tile([[1.0, 0.1, 0.1]], (len(lines), 1))
                    ls = o3d.geometry.LineSet()
                    ls.points = o3d.utility.Vector3dVector(pts)
                    ls.lines = o3d.utility.Vector2iVector(lines)
                    ls.colors = o3d.utility.Vector3dVector(colors)
                    if self.scene.scene.has_geometry("camera_traj"):
                        self.scene.scene.remove_geometry("camera_traj")
                    self.scene.scene.add_geometry("camera_traj", ls, self.traj_material)
            except Exception as e:
                print(f"⚠️ Trajectory visualization error: {e}")

    def update(self, volume, frame_idx, total_frames, rgb_frame=None, depth_frame=None, pose=None):
        if self.closed:
            return

        self.frame_count += 1

        depth_small = None
        if depth_frame is not None:
            depth_small = cv2.resize(depth_frame, (320, 240), interpolation=cv2.INTER_NEAREST)
        self.depth_widget.update_image(self._to_o3d_image_depth_colormap(depth_small))

        if pose is not None:
            p = np.asarray(pose)
            self.pose_label.text = (
                f"[{p[0,0]:+.3f}  {p[0,1]:+.3f}  {p[0,2]:+.3f}  {p[0,3]:+8.2f}]\n"
                f"[{p[1,0]:+.3f}  {p[1,1]:+.3f}  {p[1,2]:+.3f}  {p[1,3]:+8.2f}]\n"
                f"[{p[2,0]:+.3f}  {p[2,1]:+.3f}  {p[2,2]:+.3f}  {p[2,3]:+8.2f}]\n"
                f"[{p[3,0]:+.3f}  {p[3,1]:+.3f}  {p[3,2]:+.3f}  {p[3,3]:+8.2f}]"
            )

        progress_pct = 100.0 * frame_idx / max(1, total_frames)
        self.progress_label.text = f"PROGRESS: {frame_idx}/{total_frames} ({progress_pct:.1f}%)"

        elapsed = int(time.time() - self.start_time)
        self.timer_label.text = f"TIMER: {elapsed // 60:02d}:{elapsed % 60:02d}"

        self._update_pose_trajectory(pose)

        should_update = (self.frame_count % self.update_every == 0) or (frame_idx == total_frames)
        if should_update:
            try:
                temp_pcd = volume.extract_point_cloud()
                
                # ✅ DOWNSAMPLING: riduce significativamente il lag
                if len(temp_pcd.points) > 100000:
                    temp_pcd = temp_pcd.voxel_down_sample(voxel_size=2.0)
                elif len(temp_pcd.points) > 50000:
                    temp_pcd = temp_pcd.voxel_down_sample(voxel_size=1.5)
                
                if len(temp_pcd.points) > 0:
                    if not temp_pcd.has_colors():
                        temp_pcd.paint_uniform_color([0.3, 0.3, 0.3])
                    if self.geometry_added and self.scene.scene.has_geometry("tsdf_pcd"):
                        self.scene.scene.remove_geometry("tsdf_pcd")
                    self.scene.scene.add_geometry("tsdf_pcd", temp_pcd, self.material)
                    self.geometry_added = True
                    if not self._camera_initialized:
                        bbox = temp_pcd.get_axis_aligned_bounding_box()
                        self.scene.setup_camera(60.0, bbox, bbox.get_center())
                        self._camera_initialized = True
            except Exception as e:
                print(f"⚠️ Viewer update error at frame {frame_idx}: {e}")

        self._tick()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if hasattr(self, 'window') and self.window is not None:
                self.window.close()
                self.window = None
        except Exception as e:
            print(f"Error closing viewer: {e}")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════
#  OFFLINE ANALYSIS (MESH CLOSING + MISSING REGIONS)
# ══════════════════════════════════════════════════════════════════════

def _region_centroid(verts):
    if verts is None or len(verts) == 0:
        return np.zeros(3, dtype=float)
    mean_pt = np.mean(verts, axis=0)
    idx = int(np.argmin(np.linalg.norm(verts - mean_pt, axis=1)))
    return verts[idx]

def _triangle_areas(vertices, triangles):
    if len(triangles) == 0:
        return np.array([], dtype=float)
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

def _build_submesh(vertices, triangles, vertex_colors=None):
    if len(triangles) == 0:
        return None
    used = np.unique(triangles.reshape(-1))
    remap = {old: new for new, old in enumerate(used)}
    new_vertices = vertices[used]
    new_triangles = np.array([[remap[v] for v in tri] for tri in triangles], dtype=np.int32)
    submesh = o3d.geometry.TriangleMesh()
    submesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    submesh.triangles = o3d.utility.Vector3iVector(new_triangles)
    
    # Preserve vertex colors if provided
    if vertex_colors is not None and len(vertex_colors) == len(vertices):
        new_colors = vertex_colors[used]
        submesh.vertex_colors = o3d.utility.Vector3dVector(new_colors)
    
    submesh.remove_duplicated_vertices()
    submesh.remove_duplicated_triangles()
    submesh.remove_degenerate_triangles()
    submesh.remove_unreferenced_vertices()
    if len(submesh.vertices) > 0 and len(submesh.triangles) > 0:
        submesh.compute_vertex_normals()
    return submesh

def remove_disconnected_components(mesh, keep_largest_only=True):
    """
    Remove disconnected mesh components, keeping only the largest.
    
    Args:
        mesh: o3d.geometry.TriangleMesh to process
        keep_largest_only: if True, keep only the largest connected component
    
    Returns:
        mesh_clean: filtered mesh with only the largest component
        stats: dict with component statistics
    """
    print("\n🔍 Analyzing mesh connectivity...")
    
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    vertex_colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
    
    if len(triangles) == 0:
        return mesh, {"components_found": 0, "largest_triangles": 0, "removed_triangles": 0}
    
    # Cluster connected triangles
    tri_clusters, cluster_n_triangles, cluster_areas = mesh.cluster_connected_triangles()
    tri_clusters = np.asarray(tri_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_areas = np.asarray(cluster_areas)
    
    num_components = len(cluster_n_triangles)
    print(f"  Found {num_components} connected component(s)")
    
    if num_components == 0:
        return mesh, {"components_found": 0, "largest_triangles": 0, "removed_triangles": 0}
    
    # Get statistics for each component
    component_info = []
    for cid in range(num_components):
        n_tri = cluster_n_triangles[cid]
        area = cluster_areas[cid]
        component_info.append({"id": cid, "n_triangles": n_tri, "area": area})
    
    # Sort by number of triangles (largest first)
    component_info.sort(key=lambda x: x["n_triangles"], reverse=True)
    
    # Print component info
    # for idx, info in enumerate(component_info):
    #     print(f"    Component {idx+1}: {info['n_triangles']} triangles, area={info['area']:.2f} mm²")
    
    if not keep_largest_only or num_components == 1:
        stats = {
            "components_found": num_components,
            "largest_triangles": component_info[0]["n_triangles"],
            "removed_triangles": sum(c["n_triangles"] for c in component_info[1:])
        }
        if num_components == 1:
            print(f"  ✅ Mesh has single component - no filtering needed")
            return mesh, stats
    
    # Keep only the largest component
    largest_cid = component_info[0]["id"]
    keep_tri_idx = np.where(tri_clusters == largest_cid)[0]
    kept_triangles = triangles[keep_tri_idx]
    
    # Build filtered mesh (with colors preserved)
    mesh_clean = _build_submesh(vertices, kept_triangles, vertex_colors)
    
    if mesh_clean is None:
        print(f"  ⚠️ Failed to build filtered mesh")
        return mesh, {"components_found": num_components, "largest_triangles": 0, "removed_triangles": 0}
    
    removed_triangles = len(triangles) - len(kept_triangles)
    removed_percentage = 100.0 * removed_triangles / len(triangles) if len(triangles) > 0 else 0
    
    print(f"  ✅ Kept largest component: {len(kept_triangles)}/{len(triangles)} triangles ({100-removed_percentage:.1f}%)")
    if removed_triangles > 0:
        print(f"  🗑️  Removed: {removed_triangles} triangles ({removed_percentage:.1f}%) from {num_components-1} artifact component(s)")
    
    stats = {
        "components_found": num_components,
        "largest_triangles": len(kept_triangles),
        "removed_triangles": removed_triangles,
        "removed_percentage": removed_percentage
    }
    
    return mesh_clean, stats

def visualize_missing_regions_menu(mesh_original, mesh_closed, missing_mesh, trajectory_xyz=None):
    if missing_mesh is None or len(missing_mesh.triangles) == 0:
        print("No missing regions mesh to visualize.")
        return

    print("\n=== MISSING REGIONS VISUALIZATION ===")
    print("1. Closed mesh only")
    print("2. Missing regions mesh only")
    print("3. Original TSDF mesh + colored regions + trajectory")
    vis_choice = input("Choose visualization option (1/2/3, other to skip): ").strip()

    if vis_choice == "1":
        closed_show = o3d.geometry.TriangleMesh(mesh_closed)
        closed_show.compute_vertex_normals()
        o3d.visualization.draw_geometries([closed_show], window_name="Closed Mesh", width=1280, height=720, mesh_show_back_face=True)
        return

    if vis_choice == "2":
        miss_show = o3d.geometry.TriangleMesh(missing_mesh)
        miss_show.paint_uniform_color([1.0, 0.0, 0.0])
        miss_show.compute_vertex_normals()
        o3d.visualization.draw_geometries([miss_show], window_name="Missing Regions Mesh", width=1280, height=720, mesh_show_back_face=True)
        return

    if vis_choice != "3":
        return

    # ✅ Scale trajectory from cm to mm to match mesh scale
    if trajectory_xyz is not None:
        trajectory_xyz = np.asarray(trajectory_xyz) * 10.0

    geoms = []
    original_show = o3d.geometry.TriangleMesh(mesh_original)
    original_show.paint_uniform_color([0.95, 0.95, 0.95])
    original_show.compute_vertex_normals()
    geoms.append(original_show)

    tri_clusters, cluster_n_triangles, cluster_area = missing_mesh.cluster_connected_triangles()
    tri_clusters = np.asarray(tri_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    tris_all = np.asarray(missing_mesh.triangles)
    verts_all = np.asarray(missing_mesh.vertices)

    if len(cluster_area) > 0:
        cmap = mcolors.LinearSegmentedColormap.from_list("yor", ["yellow", "orange", "red"])
        a_min, a_max = float(np.min(cluster_area)), float(np.max(cluster_area))

        for cid, (n_tri, area) in enumerate(zip(cluster_n_triangles, cluster_area)):
            tri_idx = np.where(tri_clusters == cid)[0]
            if len(tri_idx) == 0:
                continue

            tris = tris_all[tri_idx]
            region_mesh = _build_submesh(verts_all, tris)
            if region_mesh is None:
                continue

            t = (float(area) - a_min) / (a_max - a_min) if a_max > a_min else 0.5
            color = cmap(t)[:3]
            region_mesh.paint_uniform_color(color)
            region_mesh.compute_vertex_normals()
            geoms.append(region_mesh)

            # ✅ Add green sphere for region centroid
            rv = verts_all[np.unique(tris.reshape(-1))]
            centroid = _region_centroid(rv)
            c_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
            c_sphere.translate(centroid)
            c_sphere.paint_uniform_color([0.0, 1.0, 0.0])
            geoms.append(c_sphere)

            # ✅ DISABLED: Blue trajectory point sphere removed for better camera control

    if trajectory_xyz is not None and len(trajectory_xyz) >= 2:
        tr = np.asarray(trajectory_xyz)
        lines = [[i, i + 1] for i in range(len(tr) - 1)]
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(tr),
            lines=o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
        )
        ls.colors = o3d.utility.Vector3dVector(np.tile([[0.0, 0.0, 0.0]], (len(lines), 1)))
        geoms.append(ls)

    for geom in geoms:
        if isinstance(geom, o3d.geometry.TriangleMesh):
            geom.remove_degenerate_triangles()

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Mesh + Missing Regions + Trajectory", width=1280, height=720)
    for geom in geoms:
        vis.add_geometry(geom)
    
    render_option = vis.get_render_option()
    render_option.mesh_show_back_face = False 
    
    vis.run()
    vis.destroy_window()

    print("\n📋 Launching interactive region selection menu...")
    show_interactive_regions_gui(mesh_original, missing_mesh, trajectory_xyz)


def show_interactive_regions_gui(mesh_original, missing_mesh, trajectory_xyz=None):
    """Menu GUI interattivo per selezionare regioni singole da visualizzare."""
    
    tri_clusters, cluster_n_triangles, cluster_area = missing_mesh.cluster_connected_triangles()
    tri_clusters = np.asarray(tri_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    tris_all = np.asarray(missing_mesh.triangles)
    verts_all = np.asarray(missing_mesh.vertices)

    regions = []
    if len(cluster_area) > 0:
        cmap = mcolors.LinearSegmentedColormap.from_list("yor", ["yellow", "orange", "red"])
        a_min, a_max = float(np.min(cluster_area)), float(np.max(cluster_area))

        for cid in range(len(cluster_area)):
            tri_idx = np.where(tri_clusters == cid)[0]
            if len(tri_idx) == 0:
                continue
            
            t = (float(cluster_area[cid]) - a_min) / (a_max - a_min) if a_max > a_min else 0.5
            color = cmap(t)[:3]
            
            regions.append({
                "id": cid,
                "area": cluster_area[cid],
                "n_tri": cluster_n_triangles[cid],
                "color": color,
                "tri_idx": tri_idx,
                "centroid": None  # Calcolato dopo
            })
    
    regions.sort(key=lambda r: r["area"], reverse=True)
    regions = regions[:50]

    # centroid computation
    for region in regions:
        tri_idx = region["tri_idx"]
        tris = tris_all[tri_idx]
        rv = verts_all[np.unique(tris.reshape(-1))]
        centroid = _region_centroid(rv)
        region["centroid"] = centroid

        # trajectory % computation
        if trajectory_xyz is not None and len(trajectory_xyz) > 0:
            tr = np.asarray(trajectory_xyz)
            j = int(np.argmin(np.linalg.norm(tr - centroid, axis=1)))
            traj_pct = (j / max(1, len(tr) - 1)) * 100.0
            region["traj_percentage"] = traj_pct
        else:
            region["traj_percentage"] = 0.0

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Region Detail View", width=960, height=720)
    render_option = vis.get_render_option()
    render_option.mesh_show_back_face = False

    def update_region_visualization(idx):
        region = regions[idx]
        
        print(f"\n🔍 Visualizing Region {idx+1}:")
        print(f"   Area: {region['area']:.2f} mm²")
        print(f"   Triangles: {region['n_tri']}")
        
        view_control = vis.get_view_control()
        try:
            camera_params = view_control.convert_to_pinhole_camera_parameters()
            has_camera = True
        except Exception:
            has_camera = False
            camera_params = None
        
        vis.clear_geometries()
        
        mesh_show = o3d.geometry.TriangleMesh(mesh_original)
        mesh_show.paint_uniform_color([0.95, 0.95, 0.95])
        mesh_show.compute_vertex_normals()
        vis.add_geometry(mesh_show)
        
        tri_idx = region["tri_idx"]
        tris = tris_all[tri_idx]
        region_mesh = _build_submesh(verts_all, tris)
        if region_mesh is not None:
            region_mesh.paint_uniform_color(region["color"])
            region_mesh.compute_vertex_normals()
            vis.add_geometry(region_mesh)
            
            centroid = region["centroid"]
            c_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
            c_sphere.translate(centroid)
            c_sphere.paint_uniform_color([0.0, 1.0, 0.0])
            vis.add_geometry(c_sphere)

        
        if trajectory_xyz is not None and len(trajectory_xyz) >= 2:
            tr = np.asarray(trajectory_xyz)
            lines = [[i, i + 1] for i in range(len(tr) - 1)]
            ls = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(tr),
                lines=o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
            )
            ls.colors = o3d.utility.Vector3dVector(np.tile([[0.0, 0.0, 0.0]], (len(lines), 1)))
            vis.add_geometry(ls)
        

        if has_camera and camera_params is not None:
            try:
                view_control.convert_from_pinhole_camera_parameters(camera_params)
            except Exception:
                pass

    def on_visualize_region():
        selection = listbox.curselection()
        if not selection:
            print("⚠️ Please select a region first")
            return
        
        idx = selection[0]
        update_region_visualization(idx)

    def render_loop():
        try:
            if not vis.poll_events():
                # Finestra chiusa
                try:
                    root.destroy()
                except:
                    pass
                return
            
            vis.update_renderer()
        except Exception as e:
            print(f"Rendering error: {e}")
        
        root.after(16, render_loop)

    def on_close_gui():
        try:
            vis.destroy_window()
        except:
            pass
        root.destroy()

    root = tk.Tk()
    root.title("Missing Regions - Interactive Selection")
    root.geometry("550x750+960+0")
    
    Label(root, text="Top 50 Missing Regions", font=("Arial", 16, "bold")).pack(pady=10)
    Label(root, text="Select region → Press 'Visualize' → Rotate/Zoom with mouse\nSelect another region to update the view", 
          font=("Arial", 12)).pack(pady=5)
    
    listbox = Listbox(root, font=("Arial", 12), height=20)
    listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    for idx, r in enumerate(regions):
        listbox.insert(tk.END, f"  Region {idx+1:<2} │ Area: {r['area']:>10.2f} mm² │ Tri: {r['n_tri']:>6} │ Traj: {r['traj_percentage']:>5.1f}%")
    
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10, fill=tk.X, padx=10)
    
    Button(button_frame, text="Visualize Selected", command=on_visualize_region, 
           bg="#4CAF50", fg="white", font=("Arial", 13, "bold"), padx=15, pady=8).pack(side=tk.LEFT, padx=5)
    
    Button(button_frame, text="Close", command=on_close_gui,
           bg="#f44336", fg="white", font=("Arial", 13, "bold"), padx=15, pady=8).pack(side=tk.RIGHT, padx=5)
    
    
    mesh_show = o3d.geometry.TriangleMesh(mesh_original)
    mesh_show.paint_uniform_color([0.95, 0.95, 0.95])
    mesh_show.compute_vertex_normals()
    vis.add_geometry(mesh_show, reset_bounding_box=True)
    
    if regions:
        update_region_visualization(0)
        listbox.selection_set(0)
    
    root.after(16, render_loop)
    root.mainloop()

def remove_single_triangles(mesh):
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters, cluster_n_triangles = np.asarray(triangle_clusters), np.asarray(cluster_n_triangles)
    remove_mask = np.array([cluster_n_triangles[c] == 1 for c in triangle_clusters], dtype=bool)
    mesh.remove_triangles_by_mask(remove_mask)
    mesh.remove_unreferenced_vertices()
    return mesh

def extract_missing_regions(original_mesh, repaired_mesh, distance_threshold=1.0, min_triangles=5, min_area=1.0, density_threshold=10):
    print("Extracting missing regions...")
    original_points = original_mesh.sample_points_uniformly(number_of_points=600000)
    original_pts = np.asarray(original_points.points)

    repaired_vertices, repaired_faces = np.asarray(repaired_mesh.vertices), np.asarray(repaired_mesh.triangles)
    total_repaired_area = float(_triangle_areas(repaired_vertices, repaired_faces).sum())
    
    _empty_stats = {
        "missing_vertices_count": 0, "missing_triangles_count": 0, "missing_percentage": 0.0,
        "max_distance": 0.0, "avg_distance": 0.0, "min_distance": 0.0,
        "adaptive_threshold_used": distance_threshold, "boundary_triangles": 0,
        "filtered_regions_count": 0, "total_regions_found": 0, "removed_small_regions": 0,
        "missing_area_mm2": 0.0, "total_repaired_area_mm2": total_repaired_area,
    }

    if len(repaired_vertices) == 0 or len(repaired_faces) == 0:
        return None, _empty_stats

    edge_lengths = []
    for face in repaired_faces[: min(2000, len(repaired_faces))]:
        for k in range(3):
            v1, v2 = face[k], face[(k + 1) % 3]
            edge_lengths.append(np.linalg.norm(repaired_vertices[v1] - repaired_vertices[v2]))
    
    avg_edge = float(np.mean(edge_lengths)) if edge_lengths else distance_threshold
    median_edge = float(np.median(edge_lengths)) if edge_lengths else distance_threshold
    adaptive_threshold = min(distance_threshold, avg_edge * 5.0, median_edge * 8.0, 3.0)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(original_pts)
    distances, _ = nbrs.kneighbors(repaired_vertices)
    distances = distances.flatten()

    far_vertices = distances > adaptive_threshold
    far_indices = np.where(far_vertices)[0]
    refined_missing = np.zeros(len(repaired_vertices), dtype=bool)
    
    if len(far_indices) > 0:
        nbrs_radius = NearestNeighbors(algorithm="ball_tree").fit(original_pts)
        counts = nbrs_radius.radius_neighbors(
            repaired_vertices[far_indices], radius=adaptive_threshold * 2.0, return_distance=False
        )
        for local_i, orig_neighbours in enumerate(counts):
            if len(orig_neighbours) < density_threshold:
                refined_missing[far_indices[local_i]] = True
    
    missing_vertex_mask = refined_missing
    vertex_neighbors = {}
    for face in repaired_faces:
        for k in range(3):
            v = face[k]
            if v not in vertex_neighbors:
                vertex_neighbors[v] = set()
            vertex_neighbors[v].add(face[(k + 1) % 3])
            vertex_neighbors[v].add(face[(k + 2) % 3])

    expanded = missing_vertex_mask.copy()
    for i, is_missing in enumerate(missing_vertex_mask):
        if is_missing and i in vertex_neighbors:
            for nb in vertex_neighbors[i]:
                if nb < len(distances) and distances[nb] > adaptive_threshold * 0.8:
                    expanded[nb] = True
    missing_vertex_mask = expanded

    missing_vertex_indices = np.where(missing_vertex_mask)[0]
    vertex_mapping, new_vertices = {}, []
    for new_idx, old_idx in enumerate(missing_vertex_indices):
        vertex_mapping[old_idx] = new_idx
        new_vertices.append(repaired_vertices[old_idx])

    new_faces, boundary_faces = [], []
    for face in repaired_faces:
        in_missing = [v in vertex_mapping for v in face]
        count = sum(in_missing)
        if count == 3:
            new_faces.append([vertex_mapping[v] for v in face])
        elif count == 2:
            new_face = []
            for v in face:
                if v in vertex_mapping:
                    new_face.append(vertex_mapping[v])
                else:
                    if distances[v] > adaptive_threshold * 0.5:
                        vertex_mapping[v] = len(new_vertices)
                        new_vertices.append(repaired_vertices[v])
                        new_face.append(vertex_mapping[v])
                    else:
                        new_face = []
                        break
            if len(new_face) == 3:
                boundary_faces.append(new_face)
    new_faces.extend(boundary_faces)

    if len(new_vertices) == 0 or len(new_faces) == 0:
        return None, _empty_stats

    missing_mesh = o3d.geometry.TriangleMesh()
    missing_mesh.vertices = o3d.utility.Vector3dVector(np.array(new_vertices))
    missing_mesh.triangles = o3d.utility.Vector3iVector(np.array(new_faces))
    missing_mesh.remove_duplicated_vertices()
    missing_mesh.remove_degenerate_triangles()
    missing_mesh.remove_unreferenced_vertices()
    missing_mesh.compute_vertex_normals()

    tc, cluster_n_tri, cluster_area = missing_mesh.cluster_connected_triangles()
    tc, cluster_n_tri, cluster_area = np.asarray(tc), np.asarray(cluster_n_tri), np.asarray(cluster_area)

    large_clusters = [i for i, (n, area) in enumerate(zip(cluster_n_tri, cluster_area)) if n >= min_triangles and area >= min_area]

    if not large_clusters:
        return None, _empty_stats

    keep_tri_idx = np.concatenate([np.where(tc == cid)[0] for cid in large_clusters])
    kept_triangles = np.asarray(missing_mesh.triangles)[keep_tri_idx]
    filtered_missing = _build_submesh(np.asarray(missing_mesh.vertices), kept_triangles)
    
    if filtered_missing is None:
        return None, _empty_stats

    fv, ff = np.asarray(filtered_missing.vertices), np.asarray(filtered_missing.triangles)
    missing_area = float(_triangle_areas(fv, ff).sum())
    missing_percentage = (missing_area / total_repaired_area * 100.0) if total_repaired_area > 0 else 0.0

    stats = {
        "missing_vertices_count": len(fv), "missing_triangles_count": len(ff), "missing_percentage": missing_percentage,
        "max_distance": adaptive_threshold * 2.0, "avg_distance": adaptive_threshold * 1.5, "min_distance": adaptive_threshold,
        "adaptive_threshold_used": adaptive_threshold, "boundary_triangles": len(boundary_faces),
        "filtered_regions_count": len(large_clusters), "total_regions_found": len(cluster_n_tri),
        "removed_small_regions": len(cluster_n_tri) - len(large_clusters),
        "missing_area_mm2": missing_area, "total_repaired_area_mm2": total_repaired_area,
    }
    return filtered_missing, stats

def print_missing_regions_report(stats):
    print(f"\n{'=' * 64}")
    print("MISSING REGIONS — SIMCOL3D ANALYSIS")
    print(f"{'=' * 64}")
    if not stats:
        print("No missing regions found.")
        print(f"{'=' * 64}")
        return
    print(f"  Regions (significant)  : {int(stats['filtered_regions_count'])}")
    print(f"  Total regions found    : {int(stats['total_regions_found'])}")
    print(f"  Missing vertices       : {int(stats['missing_vertices_count'])}")
    print(f"  Missing triangles      : {int(stats['missing_triangles_count'])}")
    print(f"  Missing area           : {float(stats['missing_area_mm2']):.2f} mm²")
    
    # Show percentage of missing area vs closed mesh
    if 'missing_area_percentage' in stats:
        pct = float(stats['missing_area_percentage'])
        print(f"  Missing area %         : {pct:.2f}% of closed mesh surface")
    
    print(f"{'=' * 64}")

def poisson_closing_pipeline(mesh_original, trajectory_xyz, output_path, poisson_depth=5, seed_points=60000, missing_min_triangles=5, missing_min_area=1.0, missing_density_threshold=10):
    print(f"\n{'=' * 60}")
    print("🔧 POISSON MESH CLOSING")
    print(f"{'=' * 60}")

    mesh_original = remove_single_triangles(mesh_original)
    if not mesh_original.has_vertex_normals():
        mesh_original.compute_vertex_normals()

    closing_t0 = time.perf_counter()
    pcd_for_poisson = mesh_original.sample_points_uniformly(number_of_points=seed_points)
    mesh_poisson, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_for_poisson, depth=poisson_depth)
    mesh_poisson = mesh_poisson.crop(mesh_original.get_axis_aligned_bounding_box())
    mesh_poisson.compute_vertex_normals()
    closing_time_s = time.perf_counter() - closing_t0

    analysis_t0 = time.perf_counter()
    try:
        missing_mesh, stats = extract_missing_regions(mesh_original, mesh_poisson, min_triangles=missing_min_triangles, min_area=missing_min_area, density_threshold=missing_density_threshold)
    except Exception:
        traceback.print_exc()
        missing_mesh, stats = None, None

    # Calculate percentage of missing area vs total closed mesh surface
    if stats is not None and missing_mesh is not None and len(missing_mesh.triangles) > 0:
        poisson_vertices = np.asarray(mesh_poisson.vertices)
        poisson_triangles = np.asarray(mesh_poisson.triangles)
        poisson_area = float(_triangle_areas(poisson_vertices, poisson_triangles).sum())
        
        missing_area = float(stats.get('missing_area_mm2', 0.0))
        if poisson_area > 0:
            missing_percentage = (missing_area / poisson_area) * 100.0
            stats['missing_area_percentage'] = missing_percentage
            stats['poisson_area_mm2'] = poisson_area

    print_missing_regions_report(stats)
    missing_analysis_time_s = time.perf_counter() - analysis_t0

    if input("\nSave closed (Poisson) mesh? (y/n): ").strip().lower() in ["y", "yes"]:
        o3d.io.write_triangle_mesh(output_path, mesh_poisson)
        print(f"Saved: {output_path}")

    if missing_mesh is not None and input("Save missing regions mesh? (y/n): ").strip().lower() in ["y", "yes"]:
        mp = output_path.replace(".ply", "_missing_regions.ply")
        missing_save = o3d.geometry.TriangleMesh(missing_mesh)
        missing_save.paint_uniform_color([1.0, 0.0, 0.0])
        o3d.io.write_triangle_mesh(mp, missing_save)
        print(f"Saved: {mp}")

    return mesh_poisson, missing_mesh, stats, {"closing_time_s": closing_time_s, "missing_analysis_time_s": missing_analysis_time_s}


# ══════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS - POSE ESTIMATION 
# ══════════════════════════════════════════════════════════════════════

def _prepare_pose_tensor(rgb_image):
    """
    Preprocess RGB image to tensor matching SequenceFolder pipeline.
    Args:
        rgb_image: numpy array (H, W, 3), uint8, 0-255
    Returns:
        tensor: (1, 3, 256, 256), normalized with ImageNet stats
    """
    rgb_resized = cv2.resize(rgb_image, (256, 256), interpolation=cv2.INTER_AREA)
    pil_img = Image.fromarray(rgb_resized.astype(np.uint8))
    
    pose_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    tensor = pose_transform(pil_img).unsqueeze(0).to(device)
    return tensor


def accumulate_pose_mm(current_pose, trans_rel, rot_rel):
    """
    Accumulate relative pose into global pose (in mm scale, cm -> mm).
    Args:
        current_pose: (4, 4) current global pose
        trans_rel: (3,) relative translation in cm
        rot_rel: (3, 3) relative rotation matrix
    Returns:
        new_pose: (4, 4) updated global pose in mm
    """
    trans_rel_mm = trans_rel * 10.0
    P_rel = np.concatenate((rot_rel, trans_rel_mm.reshape((3, 1))), 1)
    P_rel = np.vstack((P_rel, np.array([0, 0, 0, 1])))
    return current_pose @ P_rel


# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "bimodal_camera_pose"))
SCENE = "S4"
BASE_PATH = SCRIPT_DIR / "DATA" / SCENE
INPUT_FRAMES = BASE_PATH / "Frames"
CAM_FILE = BASE_PATH / "cam.txt"
MODEL_DEPTH_PATH = PROJECT_ROOT / "Colonoscopy-Depth-Estimation-main" / "sumnet_model" / "checkpoint_50.pt"
MODEL_POSE_PATH = PROJECT_ROOT / "bimodal_camera_pose" / "trained_models" / "posenet_binned" / "posenet.tar"
VOXEL_LENGTH = 0.5
SDF_TRUNC = 3.0
DEPTH_SCALE = 327.0
POSE_STRIDE = 5
FRAME_STEP = 5


# ══════════════════════════════════════════════════════════════════════
#  MAIN FUNCTION - ✅ CORRECTED LOGIC
# ══════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*60}")
    print(" REAL-TIME DEPTH + POSE ESTIMATION + 3D RECONSTRUCTION")
    print(f"{'='*60}\n")
    
    # 1. LOAD DEPTH MODEL (SUMNet)
    print("Loading depth model...")
    depth_model = SUMNet().to(device)
    checkpoint = torch.load(str(MODEL_DEPTH_PATH), map_location=device, weights_only=True)
    depth_model.load_state_dict(checkpoint['model_state_dict'])
    depth_model.eval()
    print("✅ SUMNet loaded\n")

    # 2. LOAD POSE MODEL (PoseNet)
    print("Loading pose model...")
    pose_net = pose_models.PoseCorrNet(fs=256, pose_decoder='conv').to(device)
    weights = torch.load(str(MODEL_POSE_PATH), map_location=device)
    pose_net.load_state_dict(weights['state_dict'], strict=True)
    pose_net.eval()
    softmax = nn.Softmax(dim=1)
    bins = get_bins_quat(FRAME_STEP).to(device)
    print("✅ PoseNet loaded\n")

    # 3. IMAGE PREPROCESSING
    depth_transform = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize([0.7050, 0.4508, 0.2961], [0.2382, 0.2037, 0.1276])
    ])

    # 4. LOAD INTRINSICS
    intrinsic = load_intrinsics(str(CAM_FILE))
    
    # 5. LOAD FRAME FILES
    frame_paths = sorted(list(Path(INPUT_FRAMES).glob("FrameBuffer_*.png")))
    frame_paths = frame_paths[::-1]  # Reverse (last frame first)
    total_frames = len(frame_paths)

    print(f"📊 Total frames: {total_frames}\n")

    if total_frames == 0:
        print("⚠️ No frames found. Check paths.")
        return

    # 6. INITIALIZE TSDF
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=VOXEL_LENGTH,
        sdf_trunc=SDF_TRUNC,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    # 7. INITIALIZE POSE & FRAME BUFFER
    accumulated_pose_cm = np.eye(4)      # ✅ Trajectory in cm (test_NUOVO.py scale)
    accumulated_pose_mm = np.eye(4)      # ✅ TSDF integration in mm
    poses_estimated = [accumulated_pose_cm.copy()]
    frame_buffer = {}                     # ✅ Dictionary for 5-frame lookback

    # 8. SETUP VIEWER
    viewer = None
    try:
        viewer = RealTimeTSDFViewer(update_every=20, width=1280, height=720, point_size=1.0)
        print(f"🔄 Starting real-time processing for {total_frames} frames...\n")

        stage_t0 = time.perf_counter()
        depth_est_total_time = 0.0
        pose_est_total_time = 0.0
        pose_count = 0  # ✅ Count successful pose estimations

        # ════════════════════════════════════════════════════════════════
        # ✅ MAIN LOOP - CORRECTED LOGIC FROM test_NUOVO.py
        # ════════════════════════════════════════════════════════════════
        
        for i in range(total_frames):
            if i % 100 == 0:
                print(f"\nProcessing frame {i+1}/{total_frames}")
            
            # ════════════════════════════════════════════════════════════
            # A. LOAD RGB FRAME i
            # ════════════════════════════════════════════════════════════
            rgb_bgr = cv2.imread(str(frame_paths[i]))
            rgb_input = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            frame_buffer[i] = rgb_input.copy()

            # ════════════════════════════════════════════════════════════
            # B. DEPTH ESTIMATION FOR FRAME i
            # ════════════════════════════════════════════════════════════
            pil_img = Image.fromarray(rgb_input)
            input_tensor = depth_transform(pil_img).unsqueeze(0).to(device)
            
            t_depth_start = time.perf_counter()
            with torch.no_grad():
                pred = depth_model(input_tensor)
                pred_np = F.interpolate(pred, size=(237, 237), mode='bilinear').squeeze().cpu().numpy()
            depth_est_total_time += (time.perf_counter() - t_depth_start)
            
            pred_np = cv2.GaussianBlur(pred_np, (5, 5), 1)
            depth_u16 = (pred_np * 65535.0).astype(np.uint16)

            # ════════════════════════════════════════════════════════════
            # ✅ C. POSE ESTIMATION (CORRECT LOGIC: frame i-5 → frame i)
            # ════════════════════════════════════════════════════════════
            
            # Solo quando abbiamo sia il frame reference che current
            if i > 0 and i % POSE_STRIDE == 0:
                try:
                    t_pose_start = time.perf_counter()
                    
                    # ✅ Reference frame: i - POSE_STRIDE (VECCHIO/5 frame prima)
                    # ✅ Current frame: i (NUOVO)
                    rgb_reference = frame_buffer[i - POSE_STRIDE]
                    rgb_current = frame_buffer[i]
                    
                    # Prepare tensors (ImageNet normalized)
                    rgb_ref_tensor = _prepare_pose_tensor(rgb_reference)
                    rgb_cur_tensor = _prepare_pose_tensor(rgb_current)
                    
                    # ✅ UNIFIED CALL: Compute, extract, and accumulate in ONE call
                    accumulated_pose_cm, confidence, trans_rel, rot_rel = pose_utils.process_frame_and_accumulate_pose(
                        pose_net, rgb_cur_tensor, [None, rgb_ref_tensor],
                        accumulated_pose_cm, bins, softmax, binned=True
                    )
                    
                    if accumulated_pose_cm is not None:
                        if np.all(np.isfinite(trans_rel)) and np.all(np.isfinite(rot_rel)):
                            pose_est_total_time += (time.perf_counter() - t_pose_start)
                            pose_count += 1
                            
                            # ✅ Poses already accumulated in test_MODULO!
                            # Just update mm-scale for TSDF
                            accumulated_pose_mm = accumulate_pose_mm(accumulated_pose_mm, trans_rel, rot_rel)
                            
                            # ✅ Store only when we accumulate
                            poses_estimated.append(accumulated_pose_cm.copy())
                            
                            trans_norm = np.linalg.norm(trans_rel)
                        else:
                            print(f"  ⚠️ Pose has NaN/Inf values, maintaining previous pose")
                    else:
                        print(f"  ⚠️ Pose estimation returned None")
                        
                except Exception as e:
                    print(f"  ⚠️ Pose estimation error: {e}")
                    traceback.print_exc()
            
            # ════════════════════════════════════════════════════════════
            # D. PREPARE RGBD AND INTEGRATE INTO TSDF (REMOVED DUPLICATE APPEND!)
            # ════════════════════════════════════════════════════════════
            rgb_for_tsdf = cv2.resize(rgb_input, (237, 237), interpolation=cv2.INTER_AREA)
            rgb_o3d = o3d.geometry.Image(rgb_for_tsdf)
            depth_o3d = o3d.geometry.Image(depth_u16)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_o3d, depth_o3d,
                depth_scale=DEPTH_SCALE,
                depth_trunc=250.0,
                convert_rgb_to_intensity=False
            )
            
            # Integrate with mm-scale accumulated pose
            volume.integrate(rgbd, intrinsic, np.linalg.inv(accumulated_pose_mm))

            # ════════════════════════════════════════════════════════════
            # F. UPDATE VIEWER (use mm-scale pose for visualization)
            # ════════════════════════════════════════════════════════════
            pose_for_viewer = accumulated_pose_cm.copy()
            pose_for_viewer[:3, 3] *= 10.0  # Scale translation cm → mm for visualization

            viewer.update(
                volume, i + 1, total_frames,
                rgb_frame=rgb_input,
                depth_frame=depth_u16,
                pose=pose_for_viewer
            )

            # ════════════════════════════════════════════════════════════
            # G. CLEAN OLD FRAMES FROM BUFFER
            # ════════════════════════════════════════════════════════════

            if i - 2*POSE_STRIDE in frame_buffer:
                del frame_buffer[i - 2*POSE_STRIDE]
        
        print("\n✅ Frame processing completed. Closing viewer...")
        
        reconstruction_time_s = time.perf_counter() - stage_t0
        avg_depth_time = depth_est_total_time / total_frames if total_frames > 0 else 0
        avg_pose_time = pose_est_total_time / max(1, pose_count)

    finally:
        if viewer is not None:
            try:
                viewer.close()
                viewer = None
            except Exception as e:
                print(f"Error closing viewer: {e}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 9. EXTRACT & SAVE POSES (in cm scale)
    print("\n✅ Reconstruction completed. Saving estimated poses...")
    output_dir = Path(BASE_PATH) / "output_realtime"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pose_utils.save_pose_trajectory(np.array(poses_estimated), str(output_dir))

    # 10. MESH EXTRACTION
    print("\n Extracting mesh from TSDF volume...")
    mesh_t0 = time.perf_counter()
    mesh_original = volume.extract_triangle_mesh()
    mesh_original.compute_vertex_normals()
    mesh_time = time.perf_counter() - mesh_t0
    
    print(f"✅ Mesh extracted in {_fmt_seconds(mesh_time)}")
    print(f"📊 Vertices: {len(mesh_original.vertices):,}, Triangles: {len(mesh_original.triangles):,}")

    # 11. REMOVE DISCONNECTED COMPONENTS (artifacts/noise)
    print("\n🧹 Cleaning mesh from disconnected components...")
    mesh_clean, connectivity_stats = remove_disconnected_components(mesh_original, keep_largest_only=True)
    if mesh_clean is not None:
        print(f"✅ Mesh cleaned: {connectivity_stats['components_found']} component(s) found, kept {connectivity_stats['largest_triangles']} triangles")
        mesh_original = mesh_clean

    if input("\nVisualize cleaned reconstruction? (y/n): ").strip().lower() in ["y", "yes"]:
        o3d.visualization.draw_geometries([mesh_original], window_name="Reconstruction (Cleaned)", width=1280, height=720, mesh_show_back_face=True)

    # 12. SAVE MESH
    mesh_filtered_out = str(output_dir / "mesh_tsdf_realtime.ply")
    o3d.io.write_triangle_mesh(mesh_filtered_out, mesh_original)
    print(f"✅ Mesh saved: {mesh_filtered_out}")
    print(f"   Vertices: {len(mesh_original.vertices):,}, Triangles: {len(mesh_original.triangles):,}")

    # 13. POISSON CLOSING
    closing_time = 0.0
    missing_analysis_time = 0.0
    
    if input("\nProceed with Poisson closing? (y/n): ").strip().lower() in ["y", "yes"]:
        trajectory_xyz = trajectory_from_poses(poses_estimated)
        closed_out = str(output_dir / "mesh_closed_poisson_realtime.ply")
        
        closed_mesh, missing_mesh, stats, timing = poisson_closing_pipeline(
            mesh_original, trajectory_xyz, closed_out,
            poisson_depth=10, seed_points=100000,
            missing_min_triangles=5, missing_min_area=1.0, missing_density_threshold=10
        )
        
        closing_time = timing["closing_time_s"]
        missing_analysis_time = timing["missing_analysis_time_s"]

        if missing_mesh is not None and len(missing_mesh.triangles) > 0:
            visualize_missing_regions_menu(mesh_original, closed_mesh, missing_mesh, trajectory_xyz)
        else:
            print("\nNo missing regions detected.")

    # 14. FINAL REPORT
    print_final_timing_report(
        avg_depth_s=avg_depth_time,
        avg_pose_s=avg_pose_time,
        reconstruction_s=reconstruction_time_s,
        closing_s=closing_time,
        missing_analysis_s=missing_analysis_time
    )
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\n✅ Pipeline completed successfully.")
    exit(0)


if __name__ == "__main__":
    main()