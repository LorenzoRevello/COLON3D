# COLON3D
Data and codes will be made available after acceptance


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

### 3D reconstruction in Near-Real-Time

https://github.com/user-attachments/assets/7aa263ad-de19-4aac-bcfa-f058d9cf9d16

Starting from monocular endoscopy frames, the system estimates depth and camera pose, using this information to generate a 3D visualization of the observed surface.


### Missing Region Analysis

<img width="1030" height="730" alt="photo_5834758139167837872_w" src="https://github.com/user-attachments/assets/97d47a3c-06f3-436b-b03f-5cbd0a72182d" />

The extracted Missing Regions can be individually visualized and studied directly on the reconstructed anatomy.
