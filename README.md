# COLON3D

---

<br>

This project presents a near real-time pipeline that reconstructs the 3D surface of the colon directly from standard monocular endoscopic videos.

By fusing deep learning-based depth and camera-pose estimations into a TSDF volume, the system generates dense, metric 3D meshes of the observed mucosa. It also uses Poisson Surface Reconstruction to locate, map, and quantify *missing regions*: anatomical areas obscured from the camera’s line of sight.

<figure>
  <img width="1268" height="793" alt="COLON3D full pipeline" src="https://github.com/user-attachments/assets/9fc426dd-b2e3-4ece-b157-9cc01f3b4b2e" />
  <figcaption>Full pipeline: video processing, 3D reconstruction, and missing-region identification and mapping.</figcaption>
</figure>

---

<br>

## Requirements

- Linux (Ubuntu 20.04 or newer recommended)
- [Miniconda or Anaconda](https://docs.conda.io/en/latest/miniconda.html)
- Python 3.10, installed in a dedicated Conda environment
- NVIDIA GPU and a matching CUDA-enabled PyTorch build are strongly recommended. CPU execution is supported but substantially slower.
- A graphical desktop session with OpenGL support. For remote machines, use VNC or remote desktop; plain SSH cannot display the Open3D visualizer.

Do not install the project dependencies in Conda’s `base` environment.

---

<br>

## Installation

### 1. Install system dependencies and Git LFS

On Ubuntu, run:

```bash
sudo apt update
sudo apt install -y python3-tk libgl1 libglib2.0-0 git-lfs
git lfs install
```

### 2. Clone the repository and download the model weights

```bash
git clone https://github.com/LorenzoRevello/COLON3D.git COLON3D
cd COLON3D
git lfs pull
```

The two checkpoint files must be present and have sizes of roughly 274–278 MB:

```bash
ls -lh \
  Colonoscopy-Depth-Estimation-main/sumnet_model/checkpoint_50.pt \
  bimodal_camera_pose/trained_models/posenet_binned/posenet.tar
```

Expected files:

```text
Colonoscopy-Depth-Estimation-main/sumnet_model/checkpoint_50.pt
bimodal_camera_pose/trained_models/posenet_binned/posenet.tar
```

### 3. Create and activate the Conda environment

```bash
conda create -n colon3d python=3.10 -y
conda activate colon3d
conda install -y tk

python -m pip install --upgrade pip setuptools wheel
```

Every subsequent command must be run with the `colon3d` environment active:

```bash
conda activate colon3d
```

### 4. Install PyTorch

Install a PyTorch build compatible with your NVIDIA driver and CUDA setup using the [official PyTorch installer](https://pytorch.org/get-started/locally/).

For CPU-only execution, use:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 5. Install the remaining Python dependencies

```bash
python -m pip install -r requirements.txt
```

Verify the environment:

```bash
python -c "import cv2, numpy, open3d, torch, torchvision; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Open3D:', open3d.__version__)"
```

---

<br>

## Prepare the input scene

Download **SyntheticColon I** from the [SimCol3D dataset page](https://rdr.ucl.ac.uk/articles/dataset/Simcol3D_-_3D_Reconstruction_during_Colonoscopy_Challenge_Dataset/24077763).

For the default scene, `S4`, create the following layout:

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
- To process another scene, change `SCENE = "S4"` in [main.py](main.py).

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

From the repository root, activate the environment and run:

```bash
conda activate colon3d
python main.py
```

On first execution, SUMNet may download the ImageNet VGG11 weights. Keep an internet connection available until this download has completed.

The application opens a live Open3D visualization, then asks whether to visualize the cleaned mesh and run Poisson closing. Answer the prompts with `y` or `n`.

Results are written to:

```text
DATA/<SCENE>/output_realtime/
├── mesh_tsdf_realtime.ply
├── SavedPosition.txt
├── SavedRotationQuaternion.txt
├── mesh_closed_poisson_realtime.ply
└── mesh_closed_poisson_realtime_missing_regions.ply
```

The Poisson mesh and missing-regions mesh are created only when the respective options are selected.

---

<br>

## Folder architecture

```text
COLON3D/
├── DATA/
│   └── S4/
│       ├── Frames/
│       └── cam.txt
├── bimodal_camera_pose/
├── Colonoscopy-Depth-Estimation-main/
├── Poisson Validation/
│   ├── Poisson_validation.py
│   ├── MR_ectraction_Trimesh.py
│   └── MR_extraction_without_Poisson.py
├── main.py
├── requirements.txt
└── README.md
```

---

<br>

## Troubleshooting

| Symptom | Resolution |
| --- | --- |
| Checkpoint `FileNotFoundError` | Run `git lfs pull`, then verify the two checkpoint files with the `ls -lh` command above. |
| Checkpoint files are only a few bytes | Git LFS has not downloaded the real weights. Run `git lfs install` followed by `git lfs pull`. |
| `No frames found. Check paths.` | Ensure `DATA/<SCENE>/Frames/` contains `FrameBuffer_*.png` and that `SCENE` matches the folder name. |
| Missing `cam.txt` | Copy the scene calibration file to `DATA/<SCENE>/cam.txt`. |
| `No module named 'tkinter'` | Activate the environment and run `conda install -y tk`. |
| `libGL.so.1` or Open3D import error | Install the system libraries with `sudo apt install -y libgl1 libglib2.0-0`. |
| Empty Open3D window on a server | Use a graphical desktop or a VNC session with OpenGL support; plain SSH cannot open the viewer. |
| CUDA is unavailable | Install the CUDA-enabled PyTorch build chosen from the official PyTorch installer. CPU execution remains available. |
| Packages were installed in Conda `base` | Create and activate the dedicated environment with `conda create -n colon3d python=3.10 -y` and `conda activate colon3d`, then reinstall the project dependencies there. |

---

<br>

## 3D reconstruction in near real time

<video src="https://github.com/user-attachments/assets/7bafcb44-d8aa-451e-973d-7399ddb69700" autoplay loop muted playsinline></video>

---

<br>

## Missing Region Analysis

<figure>
  <img width="1030" height="730" alt="Missing-region visualization" src="https://github.com/user-attachments/assets/97d47a3c-06f3-436b-b03f-5cbd0a72182d" />
  <figcaption>The extracted missing regions can be individually visualized and studied directly on the reconstructed anatomy.</figcaption>
</figure>
