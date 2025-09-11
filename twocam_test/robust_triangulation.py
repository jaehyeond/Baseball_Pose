#!/usr/bin/env python3
"""
Robust Multi-View 3D Triangulation
- RANSAC-based outlier rejection
- Iterative optimization refinement  
- Temporal consistency enforcement
- Multiple triangulation algorithms with quality scoring
- Proper handling of Azure Kinect calibration format
"""

import argparse
import json
import numpy as np
import cv2
from pathlib import Path
from scipy.optimize import minimize
from collections import defaultdict
import math

class RobustTriangulator:
    """
    Robust triangulation system with multiple algorithms and quality assessment
    """
    
    def __init__(self, calib_data, conf_threshold=0.5, reproj_threshold=5.0):
        self.conf_threshold = conf_threshold
        self.reproj_threshold = reproj_threshold
        
        # Parse Azure Kinect calibration format
        self.K1 = np.array(calib_data['primary_camera_intrinsics']['camera_matrix'])
        self.K2 = np.array(calib_data['secondary_camera_intrinsics']['camera_matrix'])
        
        self.d1 = np.array(calib_data['primary_camera_intrinsics']['distortion']).flatten()
        self.d2 = np.array(calib_data['secondary_camera_intrinsics']['distortion']).flatten()
        
        # Stereo calibration parameters
        self.R = np.array(calib_data['stereo_parameters']['rotation_matrix'])
        self.t = np.array(calib_data['stereo_parameters']['translation_vector']).reshape(3, 1) / 1000.0  # mm to meters
        
        # Build projection matrices
        # Camera 1: Reference (identity)
        self.P1 = self.K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
        
        # Camera 2: Relative to Camera 1
        self.P2 = self.K2 @ np.hstack([self.R, self.t])
        
        print(f"Camera setup:")
        print(f"  Baseline: {np.linalg.norm(self.t):.3f}m")
        print(f"  Rotation angle: {math.degrees(math.acos((np.trace(self.R) - 1) / 2)):.1f}°")
        
    def undistort_points(self, pts1, pts2):
        """Undistort 2D points from both cameras"""
        if len(pts1) == 0 or len(pts2) == 0:
            return pts1, pts2
            
        pts1_ud = cv2.undistortPoints(pts1.reshape(-1, 1, 2), self.K1, self.d1, P=self.K1).reshape(-1, 2)
        pts2_ud = cv2.undistortPoints(pts2.reshape(-1, 1, 2), self.K2, self.d2, P=self.K2).reshape(-1, 2)
        
        return pts1_ud, pts2_ud
    
    def triangulate_dlt(self, pts1, pts2):
        """Direct Linear Transform triangulation"""
        if len(pts1) != len(pts2) or len(pts1) == 0:
            return np.array([]), np.array([])
            
        # Undistort points
        pts1_ud, pts2_ud = self.undistort_points(pts1, pts2)
        
        points_3d = []
        reprojection_errors = []
        
        for p1, p2 in zip(pts1_ud, pts2_ud):
            # Triangulate single point
            X_h = cv2.triangulatePoints(self.P1, self.P2, 
                                       p1.reshape(2, 1).astype(np.float64),
                                       p2.reshape(2, 1).astype(np.float64))
            
            if abs(X_h[3, 0]) < 1e-12:
                points_3d.append([0, 0, 0])  # Invalid point
                reprojection_errors.append(float('inf'))
                continue
                
            X = (X_h[:3, 0] / X_h[3, 0])
            
            # Calculate reprojection error
            x1_reproj = self.P1 @ np.append(X, 1.0)
            x1_reproj = x1_reproj[:2] / x1_reproj[2]
            
            x2_reproj = self.P2 @ np.append(X, 1.0) 
            x2_reproj = x2_reproj[:2] / x2_reproj[2]
            
            error1 = np.linalg.norm(x1_reproj - p1)
            error2 = np.linalg.norm(x2_reproj - p2)
            avg_error = (error1 + error2) / 2.0
            
            points_3d.append(X)
            reprojection_errors.append(avg_error)
        
        return np.array(points_3d), np.array(reprojection_errors)
    
    def triangulate_optimal(self, pts1, pts2):
        """Optimal triangulation using algebraic correction"""
        if len(pts1) != len(pts2) or len(pts1) == 0:
            return np.array([]), np.array([])
            
        # Undistort points
        pts1_ud, pts2_ud = self.undistort_points(pts1, pts2)
        
        points_3d = []
        reprojection_errors = []
        
        for p1, p2 in zip(pts1_ud, pts2_ud):
            # Initial DLT solution
            X_h = cv2.triangulatePoints(self.P1, self.P2,
                                       p1.reshape(2, 1).astype(np.float64),
                                       p2.reshape(2, 1).astype(np.float64))
            
            if abs(X_h[3, 0]) < 1e-12:
                points_3d.append([0, 0, 0])
                reprojection_errors.append(float('inf'))
                continue
                
            X_init = X_h[:3, 0] / X_h[3, 0]
            
            # Refine with optimization
            def reprojection_cost(X):
                x1_proj = self.P1 @ np.append(X, 1.0)
                x1_proj = x1_proj[:2] / x1_proj[2]
                
                x2_proj = self.P2 @ np.append(X, 1.0)
                x2_proj = x2_proj[:2] / x2_proj[2]
                
                error1 = np.linalg.norm(x1_proj - p1) ** 2
                error2 = np.linalg.norm(x2_proj - p2) ** 2
                
                return error1 + error2
            
            try:
                result = minimize(reprojection_cost, X_init, method='BFGS')
                if result.success:
                    X_opt = result.x
                    final_error = math.sqrt(result.fun / 2.0)  # Average error
                else:
                    X_opt = X_init
                    final_error = math.sqrt(reprojection_cost(X_init) / 2.0)
            except:
                X_opt = X_init
                final_error = math.sqrt(reprojection_cost(X_init) / 2.0)
            
            points_3d.append(X_opt)
            reprojection_errors.append(final_error)
        
        return np.array(points_3d), np.array(reprojection_errors)
    
    def filter_outliers_ransac(self, pts1, pts2, confidences):
        """RANSAC-based outlier filtering"""
        if len(pts1) < 4:  # Need minimum points for RANSAC
            return list(range(len(pts1)))
        
        n_samples = min(8, len(pts1))  # Sample size
        n_iterations = 100
        best_inliers = []
        
        for _ in range(n_iterations):
            # Random sample
            sample_idx = np.random.choice(len(pts1), n_samples, replace=False)
            sample_pts1 = pts1[sample_idx]
            sample_pts2 = pts2[sample_idx]
            
            # Triangulate sample
            points_3d, errors = self.triangulate_dlt(sample_pts1, sample_pts2)
            
            if len(points_3d) == 0:
                continue
            
            # Find inliers in full set
            all_3d, all_errors = self.triangulate_dlt(pts1, pts2)
            inliers = []
            
            for i, (error, conf) in enumerate(zip(all_errors, confidences)):
                if (error < self.reproj_threshold and 
                    conf >= self.conf_threshold and
                    all_3d[i][2] > 0.1):  # Reasonable depth
                    inliers.append(i)
            
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
        
        return best_inliers
    
    def temporal_smoothing(self, current_3d, previous_3d, alpha=0.7):
        """Temporal smoothing for stability"""
        if previous_3d is None or len(previous_3d) != len(current_3d):
            return current_3d
        
        smoothed = []
        for curr, prev in zip(current_3d, previous_3d):
            if np.allclose(curr, [0, 0, 0]) or np.allclose(prev, [0, 0, 0]):
                smoothed.append(curr)
            else:
                # Distance check - if too far, don't smooth
                dist = np.linalg.norm(curr - prev)
                if dist > 0.5:  # 50cm threshold
                    smoothed.append(curr)
                else:
                    smoothed.append(alpha * curr + (1 - alpha) * prev)
        
        return np.array(smoothed)
    
    def triangulate_person(self, kpts1, kpts2, previous_3d=None):
        """Triangulate full skeleton for one person"""
        if len(kpts1) != len(kpts2):
            return None
        
        valid_joints = []
        pts1_list, pts2_list, conf_list = [], [], []
        
        # Collect valid joints
        for j, (kp1, kp2) in enumerate(zip(kpts1, kpts2)):
            if (len(kp1) >= 3 and len(kp2) >= 3 and 
                kp1[2] >= self.conf_threshold and kp2[2] >= self.conf_threshold):
                valid_joints.append(j)
                pts1_list.append(kp1[:2])
                pts2_list.append(kp2[:2])
                conf_list.append(min(kp1[2], kp2[2]))
        
        if len(valid_joints) < 3:  # Need minimum valid joints
            return None
        
        pts1 = np.array(pts1_list)
        pts2 = np.array(pts2_list)
        confidences = np.array(conf_list)
        
        # RANSAC outlier filtering
        inlier_indices = self.filter_outliers_ransac(pts1, pts2, confidences)
        
        if len(inlier_indices) < 3:
            return None
        
        # Triangulate with optimal method
        inlier_pts1 = pts1[inlier_indices]
        inlier_pts2 = pts2[inlier_indices]
        
        points_3d, errors = self.triangulate_optimal(inlier_pts1, inlier_pts2)
        
        # Build full skeleton (17 joints)
        skeleton_3d = np.zeros((17, 4))  # x, y, z, confidence
        
        for i, global_idx in enumerate([valid_joints[j] for j in inlier_indices]):
            if i < len(points_3d) and not np.allclose(points_3d[i], [0, 0, 0]):
                skeleton_3d[global_idx, :3] = points_3d[i]
                skeleton_3d[global_idx, 3] = max(0.1, confidences[inlier_indices[i]] * 
                                                (1.0 - min(1.0, errors[i] / 20.0)))
        
        # Temporal smoothing if previous frame available
        if previous_3d is not None:
            skeleton_3d[:, :3] = self.temporal_smoothing(skeleton_3d[:, :3], 
                                                        previous_3d[:, :3])
        
        return skeleton_3d

def load_pose_data(json_file):
    """Load 2D pose data from MMPose JSON - handles different formats"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    persons = []
    
    # Handle list format (direct person instances)
    if isinstance(data, list):
        for instance in data:
            kpts = np.array(instance['keypoints'])  # (17, 2)
            confs = np.array(instance['keypoint_scores'])  # (17,)
            
            # Combine into (17, 3) format
            person_kpts = np.zeros((17, 3))
            person_kpts[:, :2] = kpts
            person_kpts[:, 2] = confs
            
            persons.append(person_kpts)
    
    # Handle dict format with instance_info
    elif isinstance(data, dict):
        for instance in data.get('instance_info', []):
            kpts = np.array(instance['keypoints'])  # (17, 2)
            confs = np.array(instance['keypoint_scores'])  # (17,)
            
            # Combine into (17, 3) format
            person_kpts = np.zeros((17, 3))
            person_kpts[:, :2] = kpts
            person_kpts[:, 2] = confs
            
            persons.append(person_kpts)
    
    return persons

def match_persons_robust(persons1, persons2):
    """Robust person matching across views"""
    if not persons1 or not persons2:
        return []
    
    # Simple distance-based matching for now
    matches = []
    used_indices2 = set()
    
    for i, p1 in enumerate(persons1):
        best_match = None
        best_score = float('inf')
        
        for j, p2 in enumerate(persons2):
            if j in used_indices2:
                continue
            
            # Calculate matching score based on 2D distance
            valid_joints1 = p1[:, 2] > 0.3
            valid_joints2 = p2[:, 2] > 0.3
            common_joints = valid_joints1 & valid_joints2
            
            if np.sum(common_joints) < 5:  # Need enough common joints
                continue
            
            # Average 2D distance for common joints
            dist = np.mean(np.linalg.norm(p1[common_joints, :2] - p2[common_joints, :2], axis=1))
            
            if dist < best_score:
                best_score = dist
                best_match = j
        
        if best_match is not None and best_score < 100:  # Reasonable threshold
            matches.append((i, best_match))
            used_indices2.add(best_match)
    
    return matches

def main():
    parser = argparse.ArgumentParser(description='Robust 3D Triangulation')
    parser.add_argument('--calib-file', required=True, help='Azure Kinect calibration JSON')
    parser.add_argument('--cam1-poses', required=True, help='Camera 1 pose results folder')
    parser.add_argument('--cam2-poses', required=True, help='Camera 2 pose results folder') 
    parser.add_argument('--output-file', required=True, help='Output 3D results JSON')
    parser.add_argument('--conf-threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--reproj-threshold', type=float, default=5.0, help='Reprojection error threshold (px)')
    parser.add_argument('--start-frame', type=int, default=0, help='Start frame index')
    parser.add_argument('--end-frame', type=int, default=-1, help='End frame index (-1 for all)')
    
    args = parser.parse_args()
    
    # Load calibration
    print("Loading calibration...")
    with open(args.calib_file, 'r') as f:
        calib_data = json.load(f)
    
    triangulator = RobustTriangulator(calib_data, args.conf_threshold, args.reproj_threshold)
    
    # Get frame files
    cam1_files = sorted(Path(args.cam1_poses).glob('*.json'))
    cam2_files = sorted(Path(args.cam2_poses).glob('*.json'))
    
    n_frames = min(len(cam1_files), len(cam2_files))
    if args.end_frame != -1:
        n_frames = min(n_frames, args.end_frame - args.start_frame + 1)
    
    print(f"Processing {n_frames} frames...")
    
    # Process frames
    results = []
    previous_3d = {}  # Store previous 3D poses per person
    
    for frame_idx in range(args.start_frame, args.start_frame + n_frames):
        if frame_idx >= len(cam1_files) or frame_idx >= len(cam2_files):
            break
        
        print(f"Frame {frame_idx + 1}/{n_frames}", end=' ')
        
        # Load 2D poses
        persons1 = load_pose_data(cam1_files[frame_idx])
        persons2 = load_pose_data(cam2_files[frame_idx])
        
        # Match persons across views
        matches = match_persons_robust(persons1, persons2)
        
        if not matches:
            print("-> No matches")
            continue
        
        # Triangulate each matched pair
        frame_persons = []
        for person_id, (idx1, idx2) in enumerate(matches):
            prev_3d = previous_3d.get(person_id, None)
            
            skeleton_3d = triangulator.triangulate_person(
                persons1[idx1], persons2[idx2], prev_3d)
            
            if skeleton_3d is not None:
                # Store for next frame
                previous_3d[person_id] = skeleton_3d
                
                # Convert to output format
                keypoints_3d = skeleton_3d.tolist()
                
                frame_persons.append({
                    'keypoints_3d': keypoints_3d,
                    'person_id': person_id,
                    'cam1_idx': idx1,
                    'cam2_idx': idx2
                })
        
        results.append({
            'frame_index': frame_idx,
            'persons': frame_persons
        })
        
        print(f"-> {len(frame_persons)} persons")
    
    # Save results
    output_data = {
        'meta': {
            'dataset': 'azure_kinect_robust',
            'num_keypoints': 17,
            'keypoint_names': [
                'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
                'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
            ],
            'triangulation_method': 'robust_ransac_optimal',
            'conf_threshold': args.conf_threshold,
            'reproj_threshold': args.reproj_threshold
        },
        'results': results
    }
    
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nRobust triangulation completed!")
    print(f"Results saved to: {output_path}")
    print(f"Processed {len(results)} frames with robust methods")

if __name__ == '__main__':
    main()