# COLON3D

---
<br>

This project presents a near real-time pipeline that reconstructs the 3D surface of the colon directly from standard monocular endoscopic videos. 

By fusing deep learning-based depth and camera pose estimations into a TSDF volume, the system generates dense, metric 3D meshes of the observed mucosa. Crucially, it introduces a novel geometric approach based on Poisson Surface Reconstruction to explicitly locate, map, and quantify "missing regions" : anatomical areas obscured from the camera's line of sight. This framework lays the foundation for an objective, spatially-aware assessment of colonoscopy coverage.

<figure>
            <img width="1268" height="793" alt="Screenshot 2026-09-11 at 10 33 28" src="https://github.com/user-attachments/assets/9fc426dd-b2e3-4ece-b157-9cc01f3b4b2e" />
            <figcaption>
                        Full Pipeline : Video Processing - 3D reconstruction - Missing Regions identification and mapping.
            </figcaption>


</figure>

---
<br>

## Requirements

- Linux (Ubuntu 20.04 or newer recommended)
- Python 3.10 (Python 3.8--3.10 is supported)
- NVIDIA GPU plus a matching CUDA-enabled PyTorch build is strongly recommended. CPU execution is supported, but much slower.
- A graphical desktop session: the pipeline opens an interactive Open3D visualizer. For a remote machine, use VNC/remote desktop with OpenGL support.

On Ubuntu, install the system packages for virtual environments, Tk, and Open3D:

```bash
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3-tk libgl1 libglib2.0-0
```
---
<br>

## Installation

Clone the repository and enter it:

```bash
git clone <https://github.com/LorenzoRevello/COLON3D> COLON3D
cd COLON3D
```

If the large model files are stored through Git LFS, install Git LFS before cloning or run `git lfs pull` after cloning. Check that these files exist (they are about 286 MB and 291 MB):

```text
Colonoscopy-Depth-Estimation-main/sumnet_model/checkpoint_50.pt
bimodal_camera_pose/trained_models/posenet_binned/posenet.tar
```

Create and activate a virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install PyTorch first. Choose the CUDA command for the machine from the [official PyTorch installer](https://pytorch.org/get-started/locally/). For CPU-only execution:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Install the rest of the dependencies and verify the environment:

```bash
python -m pip install -r requirements.txt
python -c "import cv2, numpy, open3d, torch, torchvision; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Open3D:', open3d.__version__)"
```
---
<br>

## Prepare the input scene

Download **SyntheticColon I** from the [SimCol3D dataset page](https://rdr.ucl.ac.uk/articles/dataset/Simcol3D_-_3D_Reconstruction_during_Colonoscopy_Challenge_Dataset/24077763). For the default scene, `S4`, create this layout:

```text
DATA/
└── S4/
    ├── cam.txt
    └── Frames/
        ├── FrameBuffer_0000.png
        ├── FrameBuffer_0001.png
        └── ...
```

- `Frames/` must contain RGB files named `FrameBuffer_*.png`.
- `cam.txt` must contain the 3 × 3 SimCol3D camera-intrinsic matrix supplied with the dataset.
- To choose another scene, change `SCENE = "S4"` in [main.py](main.py).

For a dataset arranged as `Frames_S4` with a shared calibration file:

```bash
mkdir -p DATA/S4
cp /path/to/dataset/cam.txt DATA/S4/cam.txt
cp -a /path/to/dataset/Frames_S4 DATA/S4/Frames
test -f DATA/S4/cam.txt
find DATA/S4/Frames -maxdepth 1 -name 'FrameBuffer_*.png' | head
```
---
<br>

## Run the reconstruction

From the repository root, with the virtual environment active:

```bash
python main.py
```

The first run may download the ImageNet VGG11 weights required by SUMNet; keep an internet connection available until it completes. The application displays the live reconstruction and later asks whether to visualize the cleaned mesh and run Poisson closing. Answer each prompt with `y` or `n`.

Results are written to:

```text
DATA/<SCENE>/output_realtime/
├── mesh_tsdf_realtime.ply
├── SavedPosition.txt
├── SavedRotationQuaternion.txt
└── mesh_closed_poisson_realtime.ply  # when Poisson closing is selected
```

---
<br>

## Folder Architecture

```text
COLON3D
├── 📂 DATA/                          
│   └── 📂 S1/
│   │   ├── 📂 Frames/
│   │   └── 📄 cam.txt
│    ...
│    ...
│   └── 📂 S15/
│       ├── 📂 Frames/
│       └── 📄 cam.txt
│                    
├── 📂 bimodal_camera_pose/               
├── 📂 Colonoscopy-Depth-Estimation-main/
│
├── 📂 Poisson validation/
│    └── 📄Poisson_validation.py 
│    └── 📄MR_extraction_with_Trimesh.py 
│    └── 📄MR_extraction_without_Poisson.py
│   
├── 📄 main.py     
├── 📄 requirements.txt                         
└── 📄 README.md
            
```
---
<br>

## Troubleshooting

| Symptom | Resolution |
| --- | --- |
| Checkpoint `FileNotFoundError` | Ensure Git LFS completed and both model files listed above are present. |
| `No frames found. Check paths.` | Ensure `DATA/<SCENE>/Frames/` contains `FrameBuffer_*.png` and that `SCENE` matches the folder name. |
| Missing `cam.txt` | Copy the scene calibration file to `DATA/<SCENE>/cam.txt`. |
| `No module named 'tkinter'` | Install `python3-tk`, then recreate or reactivate the virtual environment. |
| `libGL.so.1` / Open3D import error | Install `libgl1 libglib2.0-0`. |
| Empty Open3D window on a server | Use a graphical desktop or VNC session with OpenGL; plain SSH cannot open the viewer. |
| CUDA is unavailable | Install the CUDA build of PyTorch selected from the official installer. CPU execution still works. |

## Project layout

```text
main.py                            # integrated reconstruction entry point
requirements.txt                   # Python dependencies
Colonoscopy-Depth-Estimation-main/ # SUMNet depth model and checkpoint
bimodal_camera_pose/               # PoseCorrNet model and checkpoint
DATA/                              # user-supplied SimCol3D scenes
```


---
<br>

## 3D reconstruction in Near-Real-Time

<video src="https://github.com/user-attachments/assets/7bafcb44-d8aa-451e-973d-7399ddb69700" autoplay loop muted playsinline></video>


---
<br>


## Missing Region Analysis
<figure>
            <img width="1030" height="730" alt="photo_5834758139167837872_w" src="https://github.com/user-attachments/assets/97d47a3c-06f3-436b-b03f-5cbd0a72182d" />
            <figcaption>The extracted Missing Regions can be individually visualized and studied directly on the reconstructed anatomy.</figcaption>
            
</figure>





