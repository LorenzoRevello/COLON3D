# COLON3D

---

This project presents a near real-time pipeline that reconstructs the 3D surface of the colon directly from standard monocular endoscopic videos. 

By fusing deep learning-based depth and camera pose estimations into a TSDF volume, the system generates dense, metric 3D meshes of the observed mucosa. Crucially, it introduces a novel geometric approach based on Poisson Surface Reconstruction to explicitly locate, map, and quantify "missing regions" : anatomical areas obscured from the camera's line of sight. This framework lays the foundation for an objective, spatially-aware assessment of colonoscopy coverage.

---

## Code and Data

Full Data and codes will be made available after acceptance

---

Main script : [main.py](main.py).

Unity dataset acquisition : [simulator](https://github.com/zsustc/colon_reconstruction_dataset)

Poisson Validation : [Poisson_validation.py](Poisson_validation.py)

---

### Folder Architecture

```text
COLON3D
├── 📂 DATA/                          
│   └── 📂 S/
│       ├── 📂 Frames/
│       └── 📄 cam.txt                   
├── 📂 bimodal_camera_pose/               
├── 📂 Colonoscopy-Depth-Estimation-main/ 
├── 📄 main.py     
├── 📄 requirements.txt                         
└── 📄 README.md
            
```

---

### 3D reconstruction in Near-Real-Time


https://github.com/user-attachments/assets/7bafcb44-d8aa-451e-973d-7399ddb69700


Starting from monocular endoscopy frames, the system estimates depth and camera pose, using this information to generate a 3D visualization of the observed surface.

---


### Missing Region Analysis

<img width="1030" height="730" alt="photo_5834758139167837872_w" src="https://github.com/user-attachments/assets/97d47a3c-06f3-436b-b03f-5cbd0a72182d" />

The extracted Missing Regions can be individually visualized and studied directly on the reconstructed anatomy.
