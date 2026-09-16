import time
import traceback
from pathlib import Path

import cv2
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R

try:
    import OpenEXR
    import Imath
    OPENEXR_AVAILABLE = True
except ImportError:
    OPENEXR_AVAILABLE = False
    print("⚠️ OpenEXR not available. Install with: pip install OpenEXR")

matplotlib.use("Agg")


# === STAGE 1: TSDF RECONSTRUCTION ===

class RealTimeTSDFViewer:
    """Single-window dashboard for real-time TSDF visualization."""

    def __init__(self, update_every=10, width=1600, height=900, point_size=2.0):
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

        self.window = self.app.create_window("LIVE TSDF Dashboard", width, height)
        self.window.set_on_close(self._on_close)
        self.window.set_on_layout(self._on_layout)

        self.left_panel = gui.Vert(10, gui.Margins(8, 8, 8, 8))
        self.left_panel.add_child(gui.Label("Depth"))
        self.depth_widget = gui.ImageWidget(
            o3d.geometry.Image(np.zeros((240, 320, 3), dtype=np.uint8))
        )
        self.left_panel.add_child(self.depth_widget)

        self.left_panel.add_child(gui.Label("Camera Pose"))
        self.pose_label = gui.Label("[ - ]\n[ - ]\n[ - ]\n[ - ]")
        self.left_panel.add_child(self.pose_label)

        self.progress_label = gui.Label("Progress: 0/0 (0.0%)")
        self.left_panel.add_child(self.progress_label)

        self.timer_label = gui.Label("Timer: 00:00")

        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([1.0, 1.0, 1.0, 1.0])

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
        self.window.add_child(self.timer_label)

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
        timer_h = 34
        self.left_panel.frame = gui.Rect(r.x, r.y, left_w, r.height - timer_h)
        self.scene.frame = gui.Rect(r.x + left_w, r.y, r.width - left_w, r.height)
        self.timer_label.frame = gui.Rect(r.x + 10, r.get_bottom() - timer_h, left_w - 20, timer_h)

    def _tick(self):
        if not self.closed:
            self.app.run_one_tick()

    def _to_o3d_image_depth_colormap(self, depth):
        if depth is None:
            return o3d.geometry.Image(np.zeros((240, 320, 3), dtype=np.uint8))
        depth = np.asarray(depth)
        if depth.size == 0:
            return o3d.geometry.Image(np.zeros((240, 320, 3), dtype=np.uint8))
        d = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
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

        if len(self.traj_points) >= 2:
            pts = np.asarray(self.traj_points, dtype=np.float64)
            lines = np.array([[i - 1, i] for i in range(1, len(pts))], dtype=np.int32)
            colors = np.tile([[1.0, 0.1, 0.1]], (len(lines), 1))
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(pts)
            ls.lines = o3d.utility.Vector2iVector(lines)
            ls.colors = o3d.utility.Vector3dVector(colors)
            if self.scene.scene.has_geometry("camera_traj"):
                self.scene.scene.remove_geometry("camera_traj")
            self.scene.scene.add_geometry("camera_traj", ls, self.traj_material)

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
        self.progress_label.text = f"Progress: {frame_idx}/{total_frames} ({progress_pct:.1f}%)"

        elapsed = int(time.time() - self.start_time)
        self.timer_label.text = f"Timer: {elapsed // 60:02d}:{elapsed % 60:02d}"

        self._update_pose_trajectory(pose)

        should_update = (self.frame_count % self.update_every == 0) or (frame_idx == total_frames)
        if should_update:
            try:
                temp_pcd = volume.extract_point_cloud()
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
        self.closed = True
        try:
            self.window.close()
        except Exception:
            pass

    def __del__(self):
        self.close()


def _fmt_seconds(seconds):
    seconds = float(seconds)
    mins = int(seconds // 60)
    secs = seconds % 60.0
    return f"{mins:02d}m {secs:05.2f}s"


def print_final_timing_report(reconstruction_s, closing_s, missing_analysis_s):
    total_s = float(reconstruction_s) + float(closing_s) + float(missing_analysis_s)
    print("\n" + "=" * 50)
    print("FINAL TIMING REPORT")
    print("=" * 50)
    print(f"Reconstruction (TSDF)              : {_fmt_seconds(reconstruction_s)}")
    print(f"Closing (Poisson)                  : {_fmt_seconds(closing_s)}")
    print(f"Missing regions extraction/analysis: {_fmt_seconds(missing_analysis_s)}")
    print(f"Total                              : {_fmt_seconds(total_s)}")
    print("=" * 50)


# === DATASET CONFIG & UTILITIES ===

def load_intrinsics_txt(path):
    """Load camera intrinsics from a text file."""
    print(f"Loading intrinsics from: {path}")
    
    focal_length_mm = None
    sensor_size_x = None
    sensor_size_y = None
    principal_point_x = None
    principal_point_y = None
    image_width = None
    image_height = None
    
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if "Focal Length (mm)" in line:
                focal_length_mm = float(line.split(":")[-1].strip())
            
            if "Sensor Size (mm)" in line:
                parts = line.split(":")[-1]
                sensor_size_x = float(parts.split("X=")[1].split()[0])
                sensor_size_y = float(parts.split("Y=")[1].split()[0])
            
            if "Principal point" in line and "X=" in line:
                parts = line.split(":")[-1]
                principal_point_x = float(parts.split("X=")[1].split()[0])
                principal_point_y = float(parts.split("Y=")[1].split()[0])
            
            if "Resolution (pixel)" in line:
                res_str = line.split(":")[-1].strip()
                w, h = res_str.split("*")
                image_width = int(w.strip())
                image_height = int(h.strip())
    
    if None in [focal_length_mm, sensor_size_x, sensor_size_y, image_width, image_height]:
        raise ValueError(f"Failed to parse intrinsics from {path}")
    
    fx = focal_length_mm * image_width / sensor_size_x
    fy = focal_length_mm * image_height / sensor_size_y
    cx = principal_point_x if principal_point_x != 0 else image_width / 2.0
    cy = principal_point_y if principal_point_y != 0 else image_height / 2.0
    
    print(f"  Focal length (mm): {focal_length_mm}")
    print(f"  Sensor size: {sensor_size_x} x {sensor_size_y} mm")
    print(f"  Image resolution: {image_width} x {image_height} px")
    print(f"  fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        image_width, image_height, fx, fy, cx, cy
    )
    return intrinsic, (image_width, image_height)


def load_poses_from_pos_rot_files(position_file, rotation_file, use_quaternion=True):
    """Load camera poses from position and rotation text files."""
    print(f"Loading poses from:\n  Position: {position_file}\n  Rotation: {rotation_file}")
    
    locations = []
    rotations = []
    
    with open(position_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or "Position:" not in line:
                continue
            parts = line.split("Position:")[-1]
            x = float(parts.split("X=")[1].split(",")[0])
            y = float(parts.split("Y=")[1].split(",")[0])
            z = float(parts.split("Z=")[1].split()[0])
            locations.append([x, y, z])
    
    with open(rotation_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or "Rotation:" not in line:
                continue
            parts = line.split("Rotation:")[-1]
            
            if use_quaternion:
                x = float(parts.split("X=")[1].split(",")[0])
                y = float(parts.split("Y=")[1].split(",")[0])
                z = float(parts.split("Z=")[1].split(",")[0])
                w = float(parts.split("W=")[1].split()[0])
                rotations.append([x, y, z, w])
            else:
                x = float(parts.split("X=")[1].split(",")[0])
                y = float(parts.split("Y=")[1].split(",")[0])
                z = float(parts.split("Z=")[1].split()[0])
                rotations.append([x, y, z])
    
    if len(locations) != len(rotations):
        print(f"⚠️ Warning: position frames ({len(locations)}) != rotation frames ({len(rotations)})")
    
    locations = np.array(locations) * 10.0  # Convert cm to mm
    rotations = np.array(rotations)
    print(f"  Loaded {len(locations)} positions, {len(rotations)} rotations")
    
    if use_quaternion:
        r = R.from_quat(rotations).as_matrix()
    else:
        r = R.from_euler('xyz', rotations, degrees=True).as_matrix()
    
    TM = np.eye(4)
    TM[1, 1] = -1
    
    poses_mat = []
    for i in range(locations.shape[0]):
        Pi = np.concatenate((r[i], locations[i].reshape((3, 1))), 1)
        Pi = np.vstack((Pi, np.array([0, 0, 0, 1])))
        poses_mat.append(TM @ Pi @ TM)
    
    return poses_mat


def load_depth_exr(path, depth_scale=5):
    """Load depth data from an EXR file."""
    if not OPENEXR_AVAILABLE:
        raise RuntimeError("OpenEXR not available. Install with: pip install OpenEXR")
    
    exr_file = OpenEXR.InputFile(str(path))
    header = exr_file.header()
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    channel_str = exr_file.channel("R", pt)
    
    data_win = header['dataWindow']
    width = data_win.max.x - data_win.min.x + 1
    height = data_win.max.y - data_win.min.y + 1
    
    depth_array = np.frombuffer(channel_str, dtype=np.float32).reshape((height, width))
    return (depth_array * depth_scale).astype(np.float32)


def trajectory_from_poses(poses_mat):
    return np.array([p[:3, 3] for p in poses_mat], dtype=float)


# === OFFLINE VIEWERS ===

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


def _build_submesh(vertices, triangles):
    if len(triangles) == 0:
        return None
    used = np.unique(triangles.reshape(-1))
    remap = {old: new for new, old in enumerate(used)}
    new_vertices = vertices[used]
    new_triangles = np.array([[remap[v] for v in tri] for tri in triangles], dtype=np.int32)
    submesh = o3d.geometry.TriangleMesh()
    submesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    submesh.triangles = o3d.utility.Vector3iVector(new_triangles)
    submesh.remove_duplicated_vertices()
    submesh.remove_duplicated_triangles()
    submesh.remove_degenerate_triangles()
    submesh.remove_unreferenced_vertices()
    if len(submesh.vertices) > 0 and len(submesh.triangles) > 0:
        submesh.compute_vertex_normals()
    return submesh


def visualize_missing_regions_menu(mesh_original, mesh_closed, missing_mesh, trajectory_xyz=None):
    if missing_mesh is None or len(missing_mesh.triangles) == 0:
        print("No missing regions mesh to visualize.")
        return

    print("\n=== MISSING REGIONS VISUALIZATION ===")
    print("1. Closed mesh only")
    print("2. Missing regions mesh only")
    print("3. Original TSDF mesh + colored regions + centroids (green) + trajectory (black)")
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
    
    print(f"\n🔍 DEBUG: Found {len(cluster_n_triangles)} clusters in missing_mesh")

    if len(cluster_area) == 0:
        o3d.visualization.draw_geometries(geoms, window_name="Mesh + Missing Regions", width=1280, height=720, mesh_show_back_face=True)
        return

    cmap = mcolors.LinearSegmentedColormap.from_list("yor", ["yellow", "orange", "red"])
    a_min, a_max = float(np.min(cluster_area)), float(np.max(cluster_area))

    centroid_count = 0
    for cid, (n_tri, area) in enumerate(zip(cluster_n_triangles, cluster_area)):
        if n_tri < 10:
            continue
        tri_idx = np.where(tri_clusters == cid)[0]
        if len(tri_idx) == 0:
            continue

        tris = tris_all[tri_idx]
        region_mesh = _build_submesh(verts_all, tris)
        if region_mesh is None:
            continue

        t = (float(area) - a_min) / (a_max - a_min) if a_max > a_min else 0.5
        region_mesh.paint_uniform_color(cmap(t)[:3])
        region_mesh.compute_vertex_normals()
        geoms.append(region_mesh)

        rv = verts_all[np.unique(tris.reshape(-1))]
        centroid = _region_centroid(rv)
        c_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        c_sphere.translate(centroid)
        c_sphere.paint_uniform_color([0.0, 1.0, 0.0])
        c_sphere.compute_vertex_normals()
        geoms.append(c_sphere)
        centroid_count += 1

        if trajectory_xyz is not None and len(trajectory_xyz) > 0:
            tr = np.asarray(trajectory_xyz)
            j = int(np.argmin(np.linalg.norm(tr - centroid, axis=1)))
            tp = tr[j]
            t_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
            t_sphere.translate(tp)
            t_sphere.paint_uniform_color([0.0, 0.5, 1.0])
            t_sphere.compute_vertex_normals()
            geoms.append(t_sphere)

    if trajectory_xyz is not None and len(trajectory_xyz) >= 2:
        tr = np.asarray(trajectory_xyz)
        lines = [[i, i + 1] for i in range(len(tr) - 1)]
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(tr),
            lines=o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
        )
        ls.colors = o3d.utility.Vector3dVector(np.tile([[0.0, 0.0, 0.0]], (len(lines), 1)))
        geoms.append(ls)

    print(f"✅ Plotted {centroid_count} green centroids")
    
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


# === TSDF RECONSTRUCTION ===

def create_camera_frames(poses_mat, frame_size=10.0, frame_spacing=1):
    frames_geom = []
    for i in range(0, len(poses_mat), frame_spacing):
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
        frame.transform(poses_mat[i])
        frames_geom.append(frame)
    return frames_geom

def tsdf_reconstruction(
    frames_folder,
    depth_folder,
    position_file,
    rotation_file,
    intrinsic_file,
    use_quaternion=True,
    voxel_length=1,
    sdf_trunc=2.0,
    depth_scale=5, 
    realtime_visualization=True,
    vis_update_every=5,
    visualize_final=True,
):
    stage_t0 = time.perf_counter()

    intrinsic, (image_width, image_height) = load_intrinsics_txt(intrinsic_file)
    poses = load_poses_from_pos_rot_files(position_file, rotation_file, use_quaternion=use_quaternion)

    rgb_files = sorted([f for f in Path(frames_folder).glob('*.png')] + [f for f in Path(frames_folder).glob('*.jpg')])
    depth_files = sorted(Path(depth_folder).glob('*.exr'))

    if len(rgb_files) == 0:
        print("⚠️ No RGB frames found!")
        rgb_files = [None] * len(depth_files)
    
    use_color = len(rgb_files) > len(depth_files) // 2

    print(f"\n=== TSDF RECONSTRUCTION SETUP ===")
    print(f"Depth files: {len(depth_files)}")
    print(f"RGB files: {len([f for f in rgb_files if f is not None])}")
    print(f"Color mode: {'RGB' if use_color else 'Grayscale'}")
    
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=(
            o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            if use_color else o3d.pipelines.integration.TSDFVolumeColorType.NoColor
        ),
    )

    viewer = None
    if realtime_visualization:
        viewer = RealTimeTSDFViewer(update_every=vis_update_every, width=1280, height=720, point_size=2.0)

    total_frames = min(len(depth_files), len(poses))
    print(f"\nStarting TSDF integration — {total_frames} frames\n")

    for i in range(total_frames):
        try:
            depth = load_depth_exr(depth_files[i], depth_scale=depth_scale)
        except Exception as e:
            print(f"⚠️ Error loading depth: {e}")
            continue

        depth = cv2.resize(depth, (image_width, image_height), interpolation=cv2.INTER_NEAREST)

        rgb_display = None
        rgb_for_tsdf = np.zeros((image_height, image_width, 3), dtype=np.uint8)

        if i < len(rgb_files) and rgb_files[i] is not None:
            try:
                rgb_bgr = cv2.imread(str(rgb_files[i]), cv2.IMREAD_COLOR)
                if rgb_bgr is not None:
                    rgb_bgr = cv2.resize(rgb_bgr, (image_width, image_height), interpolation=cv2.INTER_AREA)
                    rgb_display = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
                    if use_color:
                        rgb_for_tsdf = rgb_display.copy()
            except Exception as e:
                pass

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_for_tsdf),
            o3d.geometry.Image(depth),
            depth_scale=1.0,
            depth_trunc=250.0,
            convert_rgb_to_intensity=False,
        )

        volume.integrate(rgbd, intrinsic, np.linalg.inv(poses[i]))
        print(f"\rFrame {i + 1}/{total_frames}", end="", flush=True)

        if viewer is not None:
            viewer.update(
                volume, i + 1, total_frames,
                rgb_frame=rgb_display,
                depth_frame=depth,
                pose=poses[i],
            )

    if viewer is not None:
        viewer.close()

    print("\n✅ TSDF INTEGRATION COMPLETED")

    geo, choice = None, None
    if visualize_final:
        print("\n=== FINAL GEOMETRY EXTRACTION ===")
        print("1. Mesh (triangulated surface)")
        print("2. Point cloud (dense points)")
        choice = input("Choose output type (1 or 2): ").strip()

        if choice == "2":
            geo = volume.extract_point_cloud()
            if not geo.has_colors():
                geo.paint_uniform_color([0.3, 0.3, 0.3])
            print(f"✅ Point cloud: {len(geo.points):,} points")
        else:
            geo = volume.extract_triangle_mesh()
            geo.compute_vertex_normals()
            print(f"✅ Mesh: {len(geo.vertices):,} vertices, {len(geo.triangles):,} triangles")
            import psutil, os
            print(f"🔍 RAM USAGE: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.1f} MB")

    reconstruction_time_s = time.perf_counter() - stage_t0
    return geo, choice, volume, poses, reconstruction_time_s


# === STAGE 2: POISSON MESH CLOSING & MISSING REGIONS ===

def remove_single_triangles(mesh):
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    remove_mask = np.array([cluster_n_triangles[c] == 1 for c in np.asarray(triangle_clusters)], dtype=bool)
    mesh.remove_triangles_by_mask(remove_mask)
    mesh.remove_unreferenced_vertices()
    return mesh


def extract_missing_regions(
    original_mesh,
    repaired_mesh,
    distance_threshold=1.0,
    min_triangles=10,
    min_area=1.0,
    density_threshold=10,
):
    print("Extracting missing regions...")
    original_points = original_mesh.sample_points_uniformly(number_of_points=80000)
    original_pts = np.asarray(original_points.points)
    repaired_vertices = np.asarray(repaired_mesh.vertices)
    repaired_faces = np.asarray(repaired_mesh.triangles)
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

    edge_lengths = [np.linalg.norm(repaired_vertices[face[k]] - repaired_vertices[face[(k + 1) % 3]]) 
                    for face in repaired_faces[:2000] for k in range(3)]
    avg_edge = float(np.mean(edge_lengths)) if edge_lengths else distance_threshold
    median_edge = float(np.median(edge_lengths)) if edge_lengths else distance_threshold
    adaptive_threshold = min(distance_threshold, avg_edge * 5.0, median_edge * 8.0, 3.0)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(original_pts)
    distances, _ = nbrs.kneighbors(repaired_vertices)
    distances = distances.flatten()

    far_indices = np.where(distances > adaptive_threshold)[0]
    refined_missing = np.zeros(len(repaired_vertices), dtype=bool)
    if len(far_indices) > 0:
        nbrs_radius = NearestNeighbors(algorithm="ball_tree").fit(original_pts)
        counts = nbrs_radius.radius_neighbors(repaired_vertices[far_indices], radius=adaptive_threshold * 2.0, return_distance=False)
        for local_i, orig_neighbours in enumerate(counts):
            if len(orig_neighbours) < density_threshold:
                refined_missing[far_indices[local_i]] = True

    missing_vertex_mask = refined_missing
    vertex_neighbors = {}
    for face in repaired_faces:
        for k in range(3):
            v = face[k]
            if v not in vertex_neighbors: vertex_neighbors[v] = set()
            vertex_neighbors[v].update([face[(k + 1) % 3], face[(k + 2) % 3]])

    expanded = missing_vertex_mask.copy()
    for i, is_missing in enumerate(missing_vertex_mask):
        if is_missing and i in vertex_neighbors:
            for nb in vertex_neighbors[i]:
                if nb < len(distances) and distances[nb] > adaptive_threshold * 0.8:
                    expanded[nb] = True

    missing_vertex_mask = expanded
    missing_vertex_indices = np.where(missing_vertex_mask)[0]
    vertex_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(missing_vertex_indices)}
    new_vertices = [repaired_vertices[idx] for idx in missing_vertex_indices]

    new_faces, boundary_faces = [], []
    for face in repaired_faces:
        count = sum(v in vertex_mapping for v in face)
        if count == 3:
            new_faces.append([vertex_mapping[v] for v in face])
        elif count == 2:
            new_face = []
            for v in face:
                if v in vertex_mapping:
                    new_face.append(vertex_mapping[v])
                elif distances[v] > adaptive_threshold * 0.5:
                    vertex_mapping[v] = len(new_vertices)
                    new_vertices.append(repaired_vertices[v])
                    new_face.append(vertex_mapping[v])
                else:
                    new_face = []
                    break
            if len(new_face) == 3: boundary_faces.append(new_face)
            
    new_faces.extend(boundary_faces)

    if not new_vertices or not new_faces:
        return None, _empty_stats

    missing_mesh = o3d.geometry.TriangleMesh()
    missing_mesh.vertices = o3d.utility.Vector3dVector(np.array(new_vertices))
    missing_mesh.triangles = o3d.utility.Vector3iVector(np.array(new_faces))
    missing_mesh.remove_duplicated_vertices()
    missing_mesh.remove_degenerate_triangles()
    missing_mesh.remove_unreferenced_vertices()
    missing_mesh.compute_vertex_normals()

    tc, cluster_n_tri, cluster_area = missing_mesh.cluster_connected_triangles()
    large_clusters = [i for i, (n, area) in enumerate(zip(cluster_n_tri, cluster_area)) if n >= min_triangles and area >= min_area]

    if not large_clusters:
        return None, _empty_stats

    kept_triangles = np.asarray(missing_mesh.triangles)[np.concatenate([np.where(np.asarray(tc) == cid)[0] for cid in large_clusters])]
    filtered_missing = _build_submesh(np.asarray(missing_mesh.vertices), kept_triangles)
    
    if filtered_missing is None:
        return None, _empty_stats

    missing_area = float(_triangle_areas(np.asarray(filtered_missing.vertices), np.asarray(filtered_missing.triangles)).sum())

    stats = {
        "missing_vertices_count": len(filtered_missing.vertices),
        "missing_triangles_count": len(filtered_missing.triangles),
        "missing_percentage": (missing_area / total_repaired_area * 100.0) if total_repaired_area > 0 else 0.0,
        "max_distance": adaptive_threshold * 2.0,
        "avg_distance": adaptive_threshold * 1.5,
        "min_distance": adaptive_threshold,
        "adaptive_threshold_used": adaptive_threshold,
        "boundary_triangles": len(boundary_faces),
        "filtered_regions_count": len(large_clusters),
        "total_regions_found": len(cluster_n_tri),
        "removed_small_regions": len(cluster_n_tri) - len(large_clusters),
        "missing_area_mm2": missing_area,
        "total_repaired_area_mm2": total_repaired_area,
    }
    return filtered_missing, stats


def load_trajectory_from_file(traj_file_path):
    print(f"\nLoading trajectory from: {traj_file_path}")
    if not Path(traj_file_path).exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_file_path}")
    
    trajectory = []
    with open(traj_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if "Position:" in line:
                try:
                    parts = line.split("Position:")[-1]
                    trajectory.append([float(parts.split("X=")[1].split(",")[0]),
                                       float(parts.split("Y=")[1].split(",")[0]),
                                       float(parts.split("Z=")[1].split()[0])])
                    continue
                except Exception: pass
            try:
                vals = [float(x.strip()) for x in line.split(',')] if ',' in line else [float(x) for x in line.split()]
                if len(vals) >= 3: trajectory.append(vals[:3])
            except ValueError:
                continue
    
    if not trajectory: raise ValueError("No trajectory points found")
    return np.array(trajectory, dtype=np.float64) * 10.0


def create_trajectory_lineset(poses_mat=None, trajectory_xyz=None, color=[0.0, 0.0, 0.0]):
    if trajectory_xyz is None:
        if poses_mat is None: return None
        trajectory_xyz = trajectory_from_poses(poses_mat)
    if trajectory_xyz is None or len(trajectory_xyz) < 2: return None
    
    tr = np.asarray(trajectory_xyz, dtype=np.float64)
    lines = np.array([[i, i + 1] for i in range(len(tr) - 1)], dtype=np.int32)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(tr)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(np.tile(color, (len(lines), 1)))
    return ls


def filter_small_connected_regions(mesh, min_triangles=10):
    if mesh is None or len(mesh.triangles) < min_triangles: return None
    tc, cluster_n_tri, _ = mesh.cluster_connected_triangles()
    large_clusters = np.where(np.asarray(cluster_n_tri) >= min_triangles)[0]
    if len(large_clusters) == 0: return None
    kept_triangles = np.asarray(mesh.triangles)[np.concatenate([np.where(np.asarray(tc) == cid)[0] for cid in large_clusters])]
    return _build_submesh(np.asarray(mesh.vertices), kept_triangles)


def print_missing_regions_report(stats):
    print("\n" + "=" * 50)
    print("MISSING REGIONS ANALYSIS")
    print("=" * 50)
    if not stats:
        print("No missing regions found.")
        return
    for k, v in stats.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
    print("=" * 50)


def load_reference_mesh_from_ply(ply_path):
    print(f"\nLoading reference mesh from: {ply_path}")
    if not Path(ply_path).exists():
        raise FileNotFoundError(f"PLY file not found: {ply_path}")
    mesh_ref = o3d.io.read_triangle_mesh(str(ply_path))
    if not mesh_ref.has_vertex_normals(): mesh_ref.compute_vertex_normals()
    return mesh_ref


def gt_mesh_missing_regions_pipeline(mesh_original, mesh_gt, trajectory_xyz, output_path, missing_min_triangles=5, missing_min_area=1.0, missing_density_threshold=10):
    print("\n=== GT MESH MISSING REGIONS EXTRACTION ===")
    mesh_original = remove_single_triangles(mesh_original)
    if not mesh_original.has_vertex_normals(): mesh_original.compute_vertex_normals()
    if not mesh_gt.has_vertex_normals(): mesh_gt.compute_vertex_normals()

    analysis_t0 = time.perf_counter()
    try:
        missing_mesh, stats = extract_missing_regions(mesh_original, mesh_gt, min_triangles=missing_min_triangles, min_area=missing_min_area, density_threshold=missing_density_threshold)
    except Exception:
        traceback.print_exc()
        missing_mesh, stats = None, None

    print_missing_regions_report(stats)
    missing_analysis_time_s = time.perf_counter() - analysis_t0

    if input("\nSave GT mesh? (y/n): ").strip().lower() in ["y", "yes"]:
        o3d.io.write_triangle_mesh(output_path, mesh_gt)
        print(f"Saved: {output_path}")

    if missing_mesh is not None and input("Save missing regions mesh? (y/n): ").strip().lower() in ["y", "yes"]:
        missing_filtered = filter_small_connected_regions(missing_mesh, min_triangles=10)
        if missing_filtered is not None:
            mp = str(Path(output_path).parent / "MR_pymeshfix.ply")
            missing_filtered.paint_uniform_color([1.0, 0.0, 0.0])
            o3d.io.write_triangle_mesh(mp, missing_filtered)
            print(f"✓ Saved (cleaned): {mp}")
            missing_mesh = missing_filtered
            
    return mesh_gt, missing_mesh, stats, {"closing_time_s": 0.0, "missing_analysis_time_s": missing_analysis_time_s}


# === MAIN EXECUTION ===

if __name__ == "__main__":
    # ===== CONFIGURE DATASET PATHS =====
    scene = "sim_7"
    base_path = Path("./dataset") / scene
    sigla = "SOC"

    output_dir = base_path / "output"
    reference_ply_path = str(output_dir / "pymeshfix.ply")
    output_dir.mkdir(parents=True, exist_ok=True)

    frames_folder = str(base_path / "Frames")
    depth_folder = str(base_path / "Depth")
    position_file = str(base_path / f"{sigla}_Camera Position Data.txt")
    rotation_file = str(base_path / f"{sigla}_Camera Quaternion Rotation Data.txt")
    intrinsic_file = str(base_path / f"{sigla}_Intrinsic Data.txt")

    print("Dataset Configuration:")
    print(f"  Base path: {base_path}")

    show_realtime = input("\nDo you want to see real-time reconstruction progress? (y/n): ").strip().lower() in ["y", "yes"]

    geo, choice, volume, poses, reconstruction_time_s = tsdf_reconstruction(
        frames_folder=frames_folder, depth_folder=depth_folder,
        position_file=position_file, rotation_file=rotation_file, intrinsic_file=intrinsic_file,
        use_quaternion=True, voxel_length=1, sdf_trunc=3.0, depth_scale=5, 
        realtime_visualization=show_realtime, vis_update_every=5, visualize_final=True,
    )

    if geo is None:
        print("⚠️ No geometry produced. Exiting.")
        raise SystemExit(0)

    trajectory_ls = create_trajectory_lineset(poses_mat=poses, color=[0.0, 0.0, 0.0])
    trajectory_xyz = trajectory_from_poses(poses)
    
    if input("\nLoad trajectory from file? (y/n): ").strip().lower() in ["y", "yes"]:
        try:
            trajectory_xyz = load_trajectory_from_file(input("Enter trajectory file path: ").strip())
            trajectory_ls = create_trajectory_lineset(trajectory_xyz=trajectory_xyz, color=[0.0, 0.0, 0.0])
            print("✅ Trajectory loaded from file.")
        except Exception as e:
            print(f"⚠️ Error loading trajectory file: {e}")
    
    camera_frames = create_camera_frames(poses, frame_size=5.0, frame_spacing=max(1, len(poses)//20))

    if input("\nVisualize TSDF reconstruction? (y/n): ").strip().lower() in ["y", "yes"]:
        geoms_to_viz = [geo] + ([trajectory_ls] if trajectory_ls else []) + camera_frames
        o3d.visualization.draw_geometries(geoms_to_viz, window_name="TSDF Reconstruction", width=1280, height=720, mesh_show_back_face=True)

    if input("\nSave TSDF reconstruction? (y/n): ").strip().lower() in ["y", "yes"]:
        tsdf_out = str(output_dir / ("reconstruction_pointcloud.ply" if choice == "2" else "reconstruction_mesh.ply"))
        (o3d.io.write_point_cloud if choice == "2" else o3d.io.write_triangle_mesh)(tsdf_out, geo)
        print(f"✅ Saved to: {tsdf_out}")
        if trajectory_ls:
            traj_out = str(output_dir / "trajectory.ply")
            o3d.io.write_line_set(traj_out, trajectory_ls)
            print(f"✅ Saved trajectory to: {traj_out}")

    if choice == "2" or input("\nProceed with missing region analysis? (y/n): ").strip().lower() not in ["y", "yes"]:
        print_final_timing_report(reconstruction_time_s, 0.0, 0.0)
        raise SystemExit(0)

    print("\n=== STAGE 2: MISSING REGIONS EXTRACTION ===")
    mesh_ref = load_reference_mesh_from_ply(reference_ply_path)
    closed_mesh, missing_mesh, _, timing = gt_mesh_missing_regions_pipeline(
        geo, mesh_ref, trajectory_xyz, str(output_dir / "mesh_reference_pymeshfix.ply")
    )

    if missing_mesh and len(missing_mesh.triangles) > 0:
        visualize_missing_regions_menu(geo, closed_mesh, missing_mesh, trajectory_xyz)

    if trajectory_xyz is not None and len(trajectory_xyz) > 0 and input("\nSave trajectory used for analysis? (y/n): ").strip().lower() in ["y", "yes"]:
        traj_out = str(output_dir / "trajectory_analysis.txt")
        with open(traj_out, 'w') as f:
            for i, pt in enumerate(trajectory_xyz):
                f.write(f"Frame {i:05d} Position: X={pt[0]:.6f}, Y={pt[1]:.6f}, Z={pt[2]:.6f}\n")
        print(f"✅ Saved trajectory to: {traj_out}")
    
    print_final_timing_report(reconstruction_time_s, timing["closing_time_s"], timing["missing_analysis_time_s"])
    print("\n✅ Full pipeline completed.")