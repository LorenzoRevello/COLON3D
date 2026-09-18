"""
Pose Estimation Module for Bimodal Camera Pose Prediction.
Extracted and refactored from test_NUOVO.py for real-time integration.

This module encapsulates the same pose estimation logic as test_NUOVO.py
but in a form that can be called frame-by-frame from main.py
"""
import torch
import numpy as np
import os
from loss_functions import logq_to_quaternion, quat2mat
from scipy.spatial.transform import Rotation

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def compute_pose_with_inv(pose_net, tgt_img, ref_imgs):
    """
    Compute forward and inverse pose from image pair.
    
    Args:
        pose_net: Loaded PoseCorrNet model
        tgt_img: Target image tensor (B, 3, 256, 256), normalized with ImageNet stats
        ref_imgs: List of reference images [None, ref_img_tensor]
    
    Returns:
        pose: Forward pose predictions (B, 2, 6)
        pose_inv: Inverse pose predictions (B, 2, 6)
        conf: Forward confidence (B, 2)
        conf_inv: Inverse confidence (B, 2)
        scores_AB: Similarity scores
        max_indices: Max indices
    """
    i = 1  # ref_imgs is [None, ref_img], so index 1
    pose, conf, _, _ = pose_net(tgt_img, ref_imgs[i])
    pose_inv, conf_inv, scores_AB, max_indices = pose_net(ref_imgs[i], tgt_img)
    
    return pose, pose_inv, conf, conf_inv, scores_AB, max_indices


@torch.no_grad()
def process_frame_and_accumulate_pose(pose_net, tgt_img, ref_imgs, current_pose, bins, softmax, binned=True):
    """
    Process a single frame pair, estimate relative pose, and accumulate to global pose.
    
    Implements the core logic from the main inference loop of test_NUOVO.py:
    - Computes forward/inverse pose using the same network calls
    - Applies softmax and bins weighting  
    - Extracts translation and rotation using logq_to_quaternion and quat2mat
    - Accumulates relative pose into global pose matrix
    
    Args:
        pose_net: PoseCorrNet model (loaded and in eval mode)
        tgt_img: Target image tensor (B, 3, 256, 256), normalized with ImageNet stats
        ref_imgs: List of reference images [None, ref_img_tensor]
        current_pose: Current accumulated pose matrix (4, 4) in cm scale
        bins: Pose bins tensor (2, 6) from get_bins_quat()
        softmax: nn.Softmax(dim=1) layer instance
        binned: Whether to use binned pose estimates (default True)
    
    Returns:
        new_pose: Updated accumulated pose matrix (4, 4) in cm scale
        confidence: Maximum confidence score (float, 0-1)
        trans_rel: Relative translation vector (3,) in cm
        rot_rel: Relative rotation matrix (3, 3)
    """
    
    # STEP 1: Compute forward and inverse poses (same as test_NUOVO.py line 147)
    poses, poses_inv, confs_raw, confs_inv_raw, scores_AB, max_indices = compute_pose_with_inv(
        pose_net, tgt_img, ref_imgs
    )
    
    # STEP 2: Apply softmax (same as test_NUOVO.py lines 149-150)
    confs_sm = softmax(confs_raw)
    confs_inv_sm = softmax(confs_inv_raw)
    
    # ✅ STEP 3: Extract FORWARD pose using bins + confidence weighting
    if binned:
        binned_poses = bins + poses
        
        confidence_b0 = confs_sm[:, 0].view(-1, 1).repeat(1, 6)
        confidence_b1 = confs_sm[:, 1].view(-1, 1).repeat(1, 6)
        pred_poses = binned_poses[:, 0, :] * confidence_b0 + binned_poses[:, 1, :] * confidence_b1
    else:
        pred_poses = poses[:, 0, :]
    
    # ✅ STEP 4: Extract translation (FORWARD)
    trans_rel = pred_poses[0, :3].detach().cpu().numpy()
    
    # ✅ STEP 5: Extract and convert rotation from logq to matrix (FORWARD)
    rot_quat = logq_to_quaternion(pred_poses[0, 3:].unsqueeze(0)).detach().cpu().numpy()[0]
    rot_rel = quat2mat(rot_quat)
    
    # STEP 6: Build 4x4 transformation matrix (same as test_NUOVO.py lines 189-190)
    step = np.concatenate((rot_rel, trans_rel.reshape((3, 1))), 1)
    step = np.concatenate((step, np.array([0.0, 0.0, 0.0, 1.0]).reshape((1, 4))), 0)
    
    # STEP 7: Accumulate pose (same as test_NUOVO.py line 191)
    new_pose = np.matmul(current_pose, step)
    
    # ✅ STEP 8: Extract confidence for monitoring (FORWARD)
    confidence = float(torch.max(confs_sm).detach().cpu().numpy())
    
    return new_pose, confidence, trans_rel, rot_rel


def save_pose_trajectory(trajectory_4x4, output_dir, suffix=''):
    """
    Save trajectory positions and quaternions to text files.
    
    Args:
        trajectory_4x4: Array of 4x4 transformation matrices (N, 4, 4)
        output_dir: Directory to save files
        suffix: Optional suffix for filename (e.g., '_forward', '')
    
    Returns:
        None (saves to disk)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract positions
    positions = trajectory_4x4[:, :3, 3]
    
    # Extract quaternions
    quaternions = []
    for i in range(trajectory_4x4.shape[0]):
        rot_matrix = trajectory_4x4[i, :3, :3]
        quat = Rotation.from_matrix(rot_matrix).as_quat()  # [x, y, z, w]
        quaternions.append(quat)
    quaternions = np.array(quaternions)
    
    # Save positions
    pos_file = f'{output_dir}/SavedPosition{suffix}.txt'
    np.savetxt(pos_file, positions, fmt='%.6f')
    
    # Save quaternions (w, x, y, z format)
    quat_file = f'{output_dir}/SavedRotationQuaternion{suffix}.txt'
    quaternions_wxyz = quaternions[:, [3, 0, 1, 2]]
    np.savetxt(quat_file, quaternions_wxyz, fmt='%.6f')
    
    print(f'Saved {len(positions)} positions to {pos_file}')
    print(f'Saved {len(quaternions)} quaternions to {quat_file}')
