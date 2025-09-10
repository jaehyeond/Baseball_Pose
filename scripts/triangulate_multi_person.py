#!/usr/bin/env python3
"""
Multi-person triangulation using JSON data directly
"""
import json
import numpy as np
from pathlib import Path
import cv2

# Load calibration
def load_calibration():
    calib_path = "D:/mmpose/calibration/azure_kinect_calibration.json"
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    
    # Extract camera parameters  
    cam_00 = calib['cameras'][0]  # First camera (name: "00")
    cam_01 = calib['cameras'][1]  # Second camera (name: "01") 
    
    K1 = np.array(cam_00['K']).reshape(3, 3)
    K2 = np.array(cam_01['K']).reshape(3, 3) 
    R = np.array(cam_01['R']).reshape(3, 3)
    t = np.array(cam_01['t']).reshape(3, 1)
    
    # Build projection matrices
    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])  # [I | 0]
    P2 = K2 @ np.hstack([R, t])  # [R | t]
    
    return P1, P2, K1, K2

def triangulate_point(p1, p2, P1, P2):
    """Triangulate a single 3D point from 2D correspondences"""
    A = np.array([
        p1[0] * P1[2] - P1[0],
        p1[1] * P1[2] - P1[1], 
        p2[0] * P2[2] - P2[0],
        p2[1] * P2[2] - P2[1]
    ])
    
    _, _, V = np.linalg.svd(A)
    X = V[-1]
    X = X / X[3]  # Normalize
    return X[:3]

def compute_reprojection_error(X, p1, p2, P1, P2):
    """Compute reprojection error for a 3D point"""
    # Project back to both cameras
    X_hom = np.append(X, 1)
    p1_reproj = P1 @ X_hom
    p1_reproj = p1_reproj[:2] / p1_reproj[2]
    
    p2_reproj = P2 @ X_hom  
    p2_reproj = p2_reproj[:2] / p2_reproj[2]
    
    err1 = np.linalg.norm(p1 - p1_reproj)
    err2 = np.linalg.norm(p2 - p2_reproj)
    return (err1 + err2) / 2

def match_persons(inst1_list, inst2_list, P1, P2, max_error=80.0):
    """Match persons between cameras using triangulation cost"""
    n1, n2 = len(inst1_list), len(inst2_list)
    costs = np.full((n1, n2), 1e6)
    
    for i in range(n1):
        for j in range(n2):
            kpts1 = np.array(inst1_list[i]['keypoints']).reshape(-1, 2)
            kpts2 = np.array(inst2_list[j]['keypoints']).reshape(-1, 2)
            scores1 = np.array(inst1_list[i]['keypoint_scores'])
            scores2 = np.array(inst2_list[j]['keypoint_scores'])
            
            # Use more lenient thresholds for matching
            valid_idx = (scores1 > 0.2) & (scores2 > 0.2)
            if np.sum(valid_idx) < 2:
                continue
                
            errors = []
            for k in range(17):
                if not valid_idx[k]:
                    continue
                try:
                    X = triangulate_point(kpts1[k], kpts2[k], P1, P2)
                    err = compute_reprojection_error(X, kpts1[k], kpts2[k], P1, P2)
                    if err < max_error * 2:  # More lenient for initial matching
                        errors.append(err)
                except:
                    continue
            
            if len(errors) >= 2:
                costs[i, j] = np.mean(errors)
    
    # Find ALL possible matches, not just the best
    matches = []
    used_i = set()
    used_j = set()
    
    # Sort all valid pairs by cost
    valid_pairs = []
    for i in range(n1):
        for j in range(n2):
            if costs[i, j] < max_error:
                valid_pairs.append((costs[i, j], i, j))
    
    valid_pairs.sort()  # Sort by cost
    
    # Select non-overlapping matches
    for cost, i, j in valid_pairs:
        if i not in used_i and j not in used_j:
            matches.append((i, j))
            used_i.add(i)
            used_j.add(j)
            if len(matches) >= min(n1, n2):
                break
    
    return matches

def triangulate_skeleton(inst1, inst2, P1, P2):
    """Triangulate full skeleton from matched persons"""
    kpts1 = np.array(inst1['keypoints']).reshape(-1, 2)
    kpts2 = np.array(inst2['keypoints']).reshape(-1, 2)
    scores1 = np.array(inst1['keypoint_scores'])
    scores2 = np.array(inst2['keypoint_scores'])
    
    kpts3d = np.zeros((17, 3))
    conf3d = np.zeros(17)
    
    for i in range(17):
        if scores1[i] > 0.05 and scores2[i] > 0.05:
            try:
                X = triangulate_point(kpts1[i], kpts2[i], P1, P2)
                err = compute_reprojection_error(X, kpts1[i], kpts2[i], P1, P2)
                if err < 50.0:  # More lenient reprojection error
                    kpts3d[i] = X
                    conf3d[i] = min(scores1[i], scores2[i])
            except:
                continue
    
    return kpts3d, conf3d

def main():
    # Load calibration
    P1, P2, K1, K2 = load_calibration()
    
    # Load 2D poses
    with open("D:/mmpose/poses_2d_cam00/results_output_video.json", 'r') as f:
        cam00_data = json.load(f)
    with open("D:/mmpose/poses_2d_cam01/results_output_video.json", 'r') as f:
        cam01_data = json.load(f)
    
    # Find frames with exactly 2 people in both cameras
    good_frames = []
    for i in range(156):
        if (len(cam00_data['instance_info'][i]['instances']) == 2 and 
            len(cam01_data['instance_info'][i]['instances']) == 2):
            good_frames.append(i)
    
    print(f"Processing {len(good_frames)} frames with exactly 2 people in both cameras")
    
    results = []
    success_count = 0
    
    for frame_idx in good_frames[:20]:  # Test with first 20 good frames
        inst1_list = cam00_data['instance_info'][frame_idx]['instances']
        inst2_list = cam01_data['instance_info'][frame_idx]['instances']
        
        # Match persons between cameras
        matches = match_persons(inst1_list, inst2_list, P1, P2)
        
        print(f"Frame {frame_idx}: Found {len(matches)} person matches")
        
        if len(matches) == 0:
            continue
            
        persons = []
        for match_idx, (i, j) in enumerate(matches):
            kpts3d, conf3d = triangulate_skeleton(inst1_list[i], inst2_list[j], P1, P2)
            
            # Check if we got reasonable results
            valid_joints = np.sum(conf3d > 0)
            if valid_joints >= 5:  # Need at least 5 valid joints
                persons.append({
                    'pair': {'cam1_idx': int(i), 'cam2_idx': int(j)},
                    'keypoints_3d': kpts3d.tolist(),
                    'confidences': conf3d.tolist(),
                    'valid_joints': int(valid_joints)
                })
        
        if len(persons) > 0:
            results.append({
                'frame_index': frame_idx,
                'frame_idx': frame_idx, 
                'num_persons': len(persons),
                'persons': persons
            })
            success_count += 1
            print(f"  -> Successfully triangulated {len(persons)} persons")
        else:
            print(f"  -> Failed to triangulate any persons")
    
    # Save results
    output_data = {
        'meta': {
            'total_frames': len(results),
            'original_frames': len(good_frames),
            'keypoints_format': 'xyz'
        },
        'results': results
    }
    
    output_path = "D:/mmpose/output_3d/multi_person_3d.json"
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nCompleted: {success_count} frames with 3D poses saved to {output_path}")
    
    # Print summary
    if success_count > 0:
        total_persons = sum(result['num_persons'] for result in results)
        avg_persons = total_persons / success_count
        print(f"Average persons per frame: {avg_persons:.2f}")

if __name__ == "__main__":
    main()