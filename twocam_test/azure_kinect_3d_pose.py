#!/usr/bin/env python3
"""
Azure Kinect Multi-view 3D Pose Estimation
Extracts 2D poses from two camera views and triangulates to 3D
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
import argparse

import mmdet  # Import mmdet first to register transforms
from mmdet.apis import init_detector, inference_detector
from mmpose.apis import init_model, inference_topdown


def load_calibration(calib_file):
    """Load camera calibration parameters"""
    with open(calib_file, 'r') as f:
        calib_data = json.load(f)
    
    cameras = {}
    for cam_data in calib_data['cameras']:
        cam_name = cam_data['name']
        cameras[cam_name] = {
            'K': np.array(cam_data['K']),
            'distCoef': np.array(cam_data['distCoef']),
            'R': np.array(cam_data['R']),
            't': np.array(cam_data['t']).reshape(3, 1),
            'resolution': cam_data['resolution']
        }
        
        # Compute projection matrix
        cameras[cam_name]['P'] = cameras[cam_name]['K'] @ np.hstack([
            cameras[cam_name]['R'], cameras[cam_name]['t']
        ])
    
    return cameras


def triangulate_points(kpts_2d_list, cameras, cam_names):
    """
    Triangulate 2D keypoints from multiple views to 3D
    
    Args:
        kpts_2d_list: List of 2D keypoints for each camera [N_cams, N_joints, 2]
        cameras: Camera calibration dict
        cam_names: List of camera names
    
    Returns:
        kpts_3d: 3D keypoints [N_joints, 3]
    """
    n_joints = len(kpts_2d_list[0])
    kpts_3d = np.zeros((n_joints, 3))
    
    # Get projection matrices
    P1 = cameras[cam_names[0]]['P']
    P2 = cameras[cam_names[1]]['P']
    
    for j in range(n_joints):
        # Get 2D points for this joint
        pt1 = kpts_2d_list[0][j][:2]  # [x, y]
        pt2 = kpts_2d_list[1][j][:2]  # [x, y]
        
        # Skip invalid points
        if pt1[0] <= 0 or pt1[1] <= 0 or pt2[0] <= 0 or pt2[1] <= 0:
            kpts_3d[j] = [0, 0, 0]
            continue
            
        # Triangulate using cv2.triangulatePoints
        points_4d = cv2.triangulatePoints(P1, P2, pt1.reshape(2, 1), pt2.reshape(2, 1))
        
        # Convert from homogeneous coordinates
        if points_4d[3, 0] != 0:
            kpts_3d[j] = points_4d[:3, 0] / points_4d[3, 0]
        else:
            kpts_3d[j] = [0, 0, 0]
    
    return kpts_3d


def extract_2d_poses(image_path, det_model, pose_model):
    """Extract 2D poses from image"""
    # Load image
    image = cv2.imread(image_path)
    
    # Run detection
    det_results = inference_detector(det_model, image)
    
    # Filter person detections
    if hasattr(det_results, 'pred_instances'):
        # mmdet 3.x format
        bboxes = det_results.pred_instances.bboxes.cpu().numpy()
        scores = det_results.pred_instances.scores.cpu().numpy()
        labels = det_results.pred_instances.labels.cpu().numpy()
        
        # Filter person class (class 0 in COCO)
        person_mask = labels == 0
        bboxes = bboxes[person_mask]
        scores = scores[person_mask]
    else:
        # mmdet 2.x format
        bboxes = det_results[0][:, :4]  # x1, y1, x2, y2
        scores = det_results[0][:, 4]
    
    if len(bboxes) == 0:
        return None
    
    # Use highest scoring detection
    best_idx = np.argmax(scores)
    best_bbox = bboxes[best_idx:best_idx+1]
    
    # Run pose estimation
    pose_results = inference_topdown(pose_model, image, best_bbox)
    
    if len(pose_results) > 0:
        keypoints = pose_results[0].pred_instances.keypoints[0]  # [17, 3] for COCO
        return keypoints
    
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--det-config', default='D:/mmpose/demo/mmdetection_cfg/rtmdet_m_8xb32-300e_coco.py')
    parser.add_argument('--det-checkpoint', default='D:/SelfPose3d/mmposecheckpoints/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth')
    parser.add_argument('--pose-config', default='D:/SelfPose3d/mmposecheckpoints/rtmpose-m_8xb256-420e_coco-256x192.py')
    parser.add_argument('--pose-checkpoint', default='D:/SelfPose3d/mmposecheckpoints/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth')
    parser.add_argument('--calib-file', default='D:/mmpose/calibration/azure_kinect_calibration.json')
    parser.add_argument('--cam1-dir', default='D:/mmpose/frames/00')
    parser.add_argument('--cam2-dir', default='D:/mmpose/frames/01')
    parser.add_argument('--output-dir', default='D:/mmpose/output_3d')
    args = parser.parse_args()
    
    # Load calibration
    print("Loading calibration...")
    cameras = load_calibration(args.calib_file)
    cam_names = ['00', '01']
    
    # Initialize models
    print("Loading detection model...")
    det_model = init_detector(args.det_config, args.det_checkpoint, device='cpu')
    
    print("Loading pose model...")
    pose_model = init_model(args.pose_config, args.pose_checkpoint, device='cpu')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get image files
    cam1_images = sorted(Path(args.cam1_dir).glob('*.png'))
    cam2_images = sorted(Path(args.cam2_dir).glob('*.png'))
    
    print(f"Found {len(cam1_images)} images in cam1, {len(cam2_images)} images in cam2")
    
    # Process matching frames
    results_3d = []
    
    for i, (img1_path, img2_path) in enumerate(zip(cam1_images, cam2_images)):
        print(f"Processing frame {i+1}/{len(cam1_images)}: {img1_path.name}, {img2_path.name}")
        
        # Extract 2D poses
        kpts_2d_1 = extract_2d_poses(str(img1_path), det_model, pose_model)
        kpts_2d_2 = extract_2d_poses(str(img2_path), det_model, pose_model)
        
        if kpts_2d_1 is not None and kpts_2d_2 is not None:
            # Triangulate to 3D
            kpts_3d = triangulate_points([kpts_2d_1, kpts_2d_2], cameras, cam_names)
            
            result = {
                'frame': i,
                'cam1_image': img1_path.name,
                'cam2_image': img2_path.name,
                'keypoints_2d_cam1': kpts_2d_1.tolist(),
                'keypoints_2d_cam2': kpts_2d_2.tolist(),
                'keypoints_3d': kpts_3d.tolist()
            }
            results_3d.append(result)
            
            print(f"  -> 3D keypoints shape: {kpts_3d.shape}")
        else:
            print(f"  -> Failed to extract 2D poses")
    
    # Save results
    output_file = os.path.join(args.output_dir, 'azure_kinect_3d_results.json')
    with open(output_file, 'w') as f:
        json.dump(results_3d, f, indent=2)
    
    print(f"Saved 3D results to {output_file}")
    print(f"Processed {len(results_3d)} frames successfully")


if __name__ == '__main__':
    main()