# COLON3D

---

This project presents a near real-time pipeline that reconstructs the 3D surface of the colon directly from standard monocular endoscopic videos. 

By fusing deep learning-based depth and camera pose estimations into a TSDF volume, the system generates dense, metric 3D meshes of the observed mucosa. Crucially, it introduces a novel geometric approach based on Poisson Surface Reconstruction to explicitly locate, map, and quantify "missing regions" : anatomical areas obscured from the camera's line of sight. This framework lays the foundation for an objective, spatially-aware assessment of colonoscopy coverage.

<figure>
            <img width="1268" height="793" alt="Screenshot 2026-09-11 at 10 33 28" src="https://github.com/user-attachments/assets/9fc426dd-b2e3-4ece-b157-9cc01f3b4b2e" />
            <figcaption>
                        Full Pipeline : Video Processing - 3D reconstruction - Missing Regions identification and mapping.
            </figcaption>


</figure>

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

### 3D reconstruction in Near-Real-Time

<figure>
https://github.com/user-attachments/assets/7bafcb44-d8aa-451e-973d-7399ddb69700


<figcaption>Starting from monocular endoscopy frames, the system estimates depth and camera pose, using this information to generate a 3D visualization of the observed surface.</figcaption>

</figure>
---


### Missing Region Analysis
<figure>
            <img width="1030" height="730" alt="photo_5834758139167837872_w" src="https://github.com/user-attachments/assets/97d47a3c-06f3-436b-b03f-5cbd0a72182d" />
            <figcaption>The extracted Missing Regions can be individually visualized and studied directly on the reconstructed anatomy.</figcaption>
            
</figure>





