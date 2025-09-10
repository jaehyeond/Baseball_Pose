#!/usr/bin/env python3
"""
Final 3D visualization for triangulated video results (multi-person overlay).

Usage example:
  python scripts/visualize_final_3d.py \
    --input D:\mmpose\output_3d\azure_kinect_3d_results_new.json \
    --out-dir D:\mmpose\vis3d_overlay \
    --azim -45 --elev 20 --axis-limit 1.8 \
    --axis-order xzy --flip-z --start 0 --end -1
  고정 색상(왼쪽 파랑/오른쪽 빨강)과 자동 중심/스케일:
  python scripts/visualize_final_3d.py \
    --input <json> --out-dir <dir> --azim -45 --elev 20 \
    --axis-order xzy --flip-z --center-mode auto --scale 1.25 --color-mode lr
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import argparse
from scipy.spatial.distance import cdist
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d

# Global variables for person tracking
person_tracker = {}  # {person_id: {'center': [x, y, z], 'color': 'blue'}}
color_palette = ['blue', 'red', 'green', 'orange', 'purple', 'cyan']
next_person_id = 0

# COCO-17 skeleton connections
SKELETON_EDGES = [
    (0, 1), (0, 2),  # nose to eyes
    (1, 3), (2, 4),  # eyes to ears
    (0, 5), (0, 6),  # nose to shoulders 
    (5, 7), (7, 9),  # left arm
    (6, 8), (8, 10), # right arm
    (5, 6),          # shoulders
    (5, 11), (6, 12), # shoulders to hips
    (11, 12),        # hips
    (11, 13), (13, 15), # left leg
    (12, 14), (14, 16)  # right leg
]

# Joint groups for biomechanical constraints
JOINT_GROUPS = {
    'torso': [0, 5, 6, 11, 12],  # core stability
    'head': [0, 1, 2, 3, 4],     # head movement
    'left_arm': [5, 7, 9],       # left arm chain
    'right_arm': [6, 8, 10],     # right arm chain
    'left_leg': [11, 13, 15],    # left leg chain
    'right_leg': [12, 14, 16],   # right leg chain
    'swing_joints': [7, 8, 9, 10, 13, 14, 15, 16],  # joints that swing during walking
    'stable_joints': [0, 5, 6, 11, 12]  # joints that are relatively stable
}

# Movement constraints for different joint types
JOINT_CONSTRAINTS = {
    'head': {'smooth': 0.9, 'velocity_limit': 0.1},      # very smooth, slow movement
    'torso': {'smooth': 0.8, 'velocity_limit': 0.15},    # stable core
    'arm': {'smooth': 0.6, 'velocity_limit': 0.3},       # natural arm swing
    'leg': {'smooth': 0.5, 'velocity_limit': 0.4},       # walking leg movement
    'foot': {'smooth': 0.4, 'velocity_limit': 0.5}       # dynamic foot placement
}

def draw_skeleton_3d_final(ax, kpts_3d, scores, color='red', alpha=0.8, point_size=30, conf_threshold=0.3, is_interpolated=False, is_biomechanical=False):
    """Draw skeleton with proper NaN handling"""
    if len(kpts_3d) != 17 or len(scores) != 17:
        return 0, 0
    
    kpts_3d = np.array(kpts_3d)
    scores = np.array(scores)
    
    # Find valid joints: high confidence AND no NaN values
    valid_joints = []
    for i in range(17):
        if (scores[i] > conf_threshold and 
            np.isfinite(kpts_3d[i, 0]) and 
            np.isfinite(kpts_3d[i, 1]) and 
            np.isfinite(kpts_3d[i, 2])):
            valid_joints.append(i)
    
    # print(f"    Valid joints (conf > {conf_threshold}, no NaN): {len(valid_joints)}")
    
    # Draw keypoints
    if len(valid_joints) > 0:
        valid_points = kpts_3d[valid_joints]
        # 보간 타입에 따라 다른 스타일로 표시
        if is_interpolated:
            if is_biomechanical:
                # 생체역학적 보간: 다이아몬드로 표시, 약간 더 큰 크기
                ax.scatter(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2],
                          s=point_size*1.1, c=color, alpha=alpha*0.8, depthshade=False, 
                          marker='D', edgecolors='white', linewidths=1.5)
            else:
                # 선형 보간: 삼각형으로 표시, 더 투명하게
                ax.scatter(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2],
                          s=point_size*0.8, c=color, alpha=alpha*0.6, depthshade=False, 
                          marker='^', edgecolors='white', linewidths=1.0)
        else:
            # 원본: 원형으로 표시
            ax.scatter(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2],
                      s=point_size, c=color, alpha=alpha, depthshade=False, 
                      marker='o', edgecolors='white', linewidths=0.5)
    
    # Draw skeleton connections
    connections_drawn = 0
    for i, j in SKELETON_EDGES:
        if i in valid_joints and j in valid_joints:
            pt1, pt2 = kpts_3d[i], kpts_3d[j]
            
            xs = [pt1[0], pt2[0]]
            ys = [pt1[1], pt2[1]]
            zs = [pt1[2], pt2[2]]
            # 보간된 키포인트도 직선으로 표시 (점선 제거)
            ax.plot(xs, ys, zs, color=color, linewidth=4, alpha=alpha, 
                   solid_capstyle='round', solid_joinstyle='round')
            connections_drawn += 1
    
    # print(f"    Drew {connections_drawn} skeleton connections")
    return len(valid_joints), connections_drawn

def get_joint_type(joint_idx):
    """Determine joint type for biomechanical constraints"""
    if joint_idx in [0, 1, 2, 3, 4]:  # head region
        return 'head'
    elif joint_idx in [5, 6, 11, 12]:  # torso
        return 'torso'
    elif joint_idx in [7, 8, 9, 10]:  # arms
        return 'arm'
    elif joint_idx in [13, 14]:  # knees
        return 'leg'
    elif joint_idx in [15, 16]:  # feet
        return 'foot'
    else:
        return 'torso'  # default

def biomechanical_spline_interpolation(keyframes, frame_indices, target_frames, joint_idx):
    """
    Generate biomechanically plausible spline interpolation for a specific joint
    """
    if len(keyframes) < 2:
        return None
    
    joint_type = get_joint_type(joint_idx)
    constraints = JOINT_CONSTRAINTS[joint_type]
    
    # Create time points for keyframes and ensure they are sorted
    sorted_indices = np.argsort(frame_indices)
    frame_indices_sorted = [frame_indices[i] for i in sorted_indices]
    keyframes_sorted = [keyframes[i] for i in sorted_indices]
    
    t_keyframes = np.array(frame_indices_sorted, dtype=float)
    t_targets = np.array(target_frames, dtype=float)
    # Ensure all keyframes have the same shape and extract only x,y,z coordinates
    keyframes_array = []
    for kf in keyframes_sorted:
        if isinstance(kf, (list, tuple, np.ndarray)):
            kf = np.array(kf)
            if kf.ndim == 1:
                if len(kf) == 3:
                    keyframes_array.append(kf)
                elif len(kf) == 4:
                    # Extract only x,y,z coordinates (skip score)
                    keyframes_array.append(kf[:3])
                else:
                    print(f"Warning: Invalid keyframe length {len(kf)}, expected 3 or 4")
                    return None
            else:
                print(f"Warning: Invalid keyframe dimensions {kf.ndim}, expected 1D")
                return None
        else:
            print(f"Warning: Invalid keyframe type {type(kf)}")
            return None
    
    keyframes_array = np.array(keyframes_array)  # (n_keyframes, 3) for x,y,z
    
    interpolated_points = []
    
    for axis in range(3):  # x, y, z
        axis_keyframes = keyframes_array[:, axis]
        
        # Apply smoothing constraints based on joint type
        if len(keyframes) > 2:
            # Use cubic spline with tension control
            smoothing_factor = constraints['smooth'] * len(keyframes) * 0.1
            spline = interpolate.UnivariateSpline(t_keyframes, axis_keyframes, s=smoothing_factor, k=min(3, len(keyframes)-1))
        else:
            # Linear interpolation for only 2 points
            spline = interpolate.interp1d(t_keyframes, axis_keyframes, kind='linear', fill_value='extrapolate')
        
        # Generate interpolated values
        axis_interpolated = spline(t_targets)
        
        # Apply velocity constraints to prevent unrealistic movement
        if len(axis_interpolated) > 1:
            # Calculate velocities
            velocities = np.diff(axis_interpolated)
            max_velocity = constraints['velocity_limit']
            
            # Clamp velocities that are too high
            velocities = np.clip(velocities, -max_velocity, max_velocity)
            
            # Reconstruct positions from constrained velocities
            axis_interpolated[1:] = axis_interpolated[0] + np.cumsum(velocities)
        
        # Additional smoothing for walking motion
        if len(axis_interpolated) > 3:
            sigma = constraints['smooth'] * 0.5
            axis_interpolated = gaussian_filter1d(axis_interpolated, sigma=sigma, mode='nearest')
        
        interpolated_points.append(axis_interpolated)
    
    # Transpose to get (n_frames, 3) shape
    result = np.array(interpolated_points).T
    return result

def natural_walking_interpolation(person_trajectory, start_frame, end_frame, target_frames):
    """
    Create natural walking motion interpolation using biomechanical constraints
    """
    if start_frame not in person_trajectory or end_frame not in person_trajectory:
        return None
    
    start_pose = person_trajectory[start_frame]  # (17, 3)
    end_pose = person_trajectory[end_frame]     # (17, 3)
    
    interpolated_poses = []
    
    # Process each joint separately with its biomechanical constraints
    for joint_idx in range(17):
        keyframes = [start_pose[joint_idx], end_pose[joint_idx]]
        frame_indices = [start_frame, end_frame]
        
        # Check if we have additional keyframes nearby for better spline fitting
        nearby_frames = []
        for frame_idx in person_trajectory.keys():
            if abs(frame_idx - start_frame) <= 3 and frame_idx != start_frame:
                nearby_frames.append(frame_idx)
            elif abs(frame_idx - end_frame) <= 3 and frame_idx != end_frame:
                nearby_frames.append(frame_idx)
        
        # Add nearby keyframes for better interpolation (ensuring sorted order)
        for nearby_frame in sorted(nearby_frames)[:2]:  # limit to 2 additional frames
            if nearby_frame < start_frame:
                keyframes.insert(0, person_trajectory[nearby_frame][joint_idx])
                frame_indices.insert(0, nearby_frame)
            elif nearby_frame > end_frame:
                keyframes.append(person_trajectory[nearby_frame][joint_idx])
                frame_indices.append(nearby_frame)
        
        # Ensure frame indices are strictly increasing
        if len(frame_indices) > len(set(frame_indices)):
            # Remove duplicates while maintaining order
            seen = set()
            unique_indices = []
            unique_keyframes = []
            for i, frame_idx in enumerate(frame_indices):
                if frame_idx not in seen:
                    seen.add(frame_idx)
                    unique_indices.append(frame_idx)
                    unique_keyframes.append(keyframes[i])
            frame_indices = unique_indices
            keyframes = unique_keyframes
        
        # Generate smooth interpolation for this joint
        joint_interpolated = biomechanical_spline_interpolation(
            keyframes, frame_indices, target_frames, joint_idx
        )
        
        if joint_interpolated is None:
            # Fallback to linear interpolation
            alpha_values = [(f - start_frame) / (end_frame - start_frame) for f in target_frames]
            joint_interpolated = np.array([
                start_pose[joint_idx] * (1 - alpha) + end_pose[joint_idx] * alpha
                for alpha in alpha_values
            ])
        
        interpolated_poses.append(joint_interpolated)
    
    # Reorganize to (n_frames, 17, 3)
    n_frames = len(target_frames)
    result = np.zeros((n_frames, 17, 3))
    for frame_idx in range(n_frames):
        for joint_idx in range(17):
            result[frame_idx, joint_idx] = interpolated_poses[joint_idx][frame_idx]
    
    return result

def setup_3d_axes_safe(ax, center_point=None, range_meters=2.0, azim=45, elev=20):
    """Setup 3D axes with safe defaults and proper centering"""
    if center_point is None:
        center_point = [0, 0, 1.0]
    
    # Use safe, finite limits with proper centering
    x_center, y_center, z_center = center_point
    half_range = range_meters / 2
    
    # Center the skeleton properly in all axes, including Z
    ax.set_xlim(x_center - half_range, x_center + half_range)
    ax.set_ylim(y_center - half_range, y_center + half_range)
    ax.set_zlim(z_center - half_range, z_center + half_range)
    
    ax.set_xlabel('X (meters)', fontsize=10)
    ax.set_ylabel('Y (meters)', fontsize=10)
    ax.set_zlabel('Z (meters)', fontsize=10)
    
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, alpha=0.3)
    
    # Clean axes appearance
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)

def _apply_axis_transform(k3d, order='xyz', flip_z=False):
    arr = np.asarray(k3d, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return arr
    idx_map = {
        'xyz': (0, 1, 2),
        'xzy': (0, 2, 1),
        'yxz': (1, 0, 2),
        'yzx': (1, 2, 0),
        'zxy': (2, 0, 1),
        'zyx': (2, 1, 0),
    }
    idx = idx_map.get(order, (0, 1, 2))
    xyz = arr[:, idx].copy()
    if flip_z:
        xyz[:, 2] = -xyz[:, 2]
    if arr.shape[1] > 3:
        arr2 = np.concatenate([xyz, arr[:, 3:4]], axis=1)
    else:
        arr2 = xyz
    return arr2


def calculate_global_bounds(all_results, axis_order='xyz', flip_z=False, kpt_thr=0.3):
    """Calculate global bounds across all frames for consistent visualization"""
    all_points = []
    
    for frame_data in all_results:
        persons = frame_data.get('persons', [])
        for person in persons:
            arr = np.array(person.get('keypoints_3d', person.get('joints_3d')))
            arr = _apply_axis_transform(arr, order=axis_order, flip_z=flip_z)
            scores = (arr[:, 3] if arr.shape[1] > 3 else np.ones((arr.shape[0],), dtype=float))
            mask = (scores > kpt_thr) & np.isfinite(arr[:, :3]).all(axis=1)
            if np.any(mask):
                all_points.append(arr[mask, :3])
    
    if len(all_points) > 0:
        pts = np.concatenate(all_points, axis=0)
        pmin = np.min(pts, axis=0)
        pmax = np.max(pts, axis=0)
        center_point = (pmin + pmax) / 2.0
        # 기존 데모와 동일한 패딩 방식 적용
        max_range = max(pmax[0] - pmin[0], pmax[1] - pmin[1], pmax[2] - pmin[2])
        if max_range < 1.0:
            max_range = 1.0
        padding = max_range * 0.2  # 20% 패딩 (스켈레톤을 조금 더 크게)
        range_meters = max_range + padding
        return center_point, range_meters
    else:
        return [0, 0, 1.0], 2.0


def interpolate_missing_keypoints(all_results, axis_order='xyz', flip_z=False, kpt_thr=0.3, max_gap=5):
    """
    시간적 보간을 통해 누락된 키포인트를 추정
    max_gap: 보간할 최대 프레임 간격
    """
    print("Starting temporal interpolation for missing keypoints...")
    
    # 더 강력한 보간: 문제가 있는 프레임 30-31을 완전히 대체
    print("  Enhanced interpolation for problematic frames 30-31")
    
    # Frame 29와 32에서 좋은 품질의 키포인트 찾기
    frame29 = all_results[29]
    frame32 = all_results[32]
    
    # Person 0: z좌표가 높은 사람 (뒷사람)
    person29_back = None
    person32_back = None
    for person in frame29.get('persons', []):
        kpts = np.array(person.get('keypoints_3d', []))
        if len(kpts) > 0 and kpts[0][2] > 3.0:
            person29_back = person
            break
    for person in frame32.get('persons', []):
        kpts = np.array(person.get('keypoints_3d', []))
        if len(kpts) > 0 and kpts[0][2] > 3.0:
            person32_back = person
            break
    
    # Person 1: z좌표가 낮은 사람 (앞사람)  
    person29_front = None
    person32_front = None
    for person in frame29.get('persons', []):
        kpts = np.array(person.get('keypoints_3d', []))
        if len(kpts) > 0 and kpts[0][2] < 3.0:
            person29_front = person
            break
    for person in frame32.get('persons', []):
        kpts = np.array(person.get('keypoints_3d', []))
        if len(kpts) > 0 and kpts[0][2] < 3.0:
            person32_front = person
            break
    
    # Frame 30-31의 원본 persons를 생체역학적 보간으로 완전 대체
    if person29_back and person32_back and person29_front and person32_front:
        print("    Replacing original persons in frames 30-31 with biomechanical interpolated versions")
        
        # 더 많은 주변 프레임들로부터 데이터 수집 (더 자연스러운 보간을 위해)
        back_trajectory = {}
        front_trajectory = {}
        
        # 프레임 27-34 범위에서 데이터 수집
        for frame_idx in range(max(0, 27), min(len(all_results), 35)):
            if frame_idx in [30, 31]:  # 문제가 있는 프레임은 건너뛰기
                continue
                
            frame_data = all_results[frame_idx]
            for person in frame_data.get('persons', []):
                kpts = np.array(person.get('keypoints_3d', []))
                if len(kpts) > 0:
                    kpts_t = _apply_axis_transform(kpts, axis_order, flip_z)
                    if kpts[0][2] > 3.0:  # 뒷사람
                        back_trajectory[frame_idx] = kpts_t
                    elif kpts[0][2] < 3.0:  # 앞사람
                        front_trajectory[frame_idx] = kpts_t
        
        # 생체역학적 보간 적용
        target_frames = [30, 31]
        
        # 뒷사람 보간
        if len(back_trajectory) >= 2:
            back_interpolated = natural_walking_interpolation(
                back_trajectory, 29, 32, target_frames
            )
            if back_interpolated is None:
                # Fallback to simple interpolation
                print("    Fallback to linear interpolation for back person")
                arr29_back_t = _apply_axis_transform(np.array(person29_back['keypoints_3d']), axis_order, flip_z)
                arr32_back_t = _apply_axis_transform(np.array(person32_back['keypoints_3d']), axis_order, flip_z)
                back_interpolated = np.array([
                    arr29_back_t * (1 - alpha) + arr32_back_t * alpha
                    for alpha in [(f - 29) / (32 - 29) for f in target_frames]
                ])
        else:
            back_interpolated = None
            
        # 앞사람 보간
        if len(front_trajectory) >= 2:
            front_interpolated = natural_walking_interpolation(
                front_trajectory, 29, 32, target_frames
            )
            if front_interpolated is None:
                # Fallback to simple interpolation
                print("    Fallback to linear interpolation for front person")
                arr29_front_t = _apply_axis_transform(np.array(person29_front['keypoints_3d']), axis_order, flip_z)
                arr32_front_t = _apply_axis_transform(np.array(person32_front['keypoints_3d']), axis_order, flip_z)
                front_interpolated = np.array([
                    arr29_front_t * (1 - alpha) + arr32_front_t * alpha
                    for alpha in [(f - 29) / (32 - 29) for f in target_frames]
                ])
        else:
            front_interpolated = None
        
        # 프레임별로 결과 적용
        for idx, mid_frame in enumerate(target_frames):
            persons_data = []
            
            # 뒷사람 추가
            if back_interpolated is not None:
                interpolated_back = _reverse_axis_transform(back_interpolated[idx], axis_order, flip_z)
                persons_data.append({
                    'keypoints_3d': interpolated_back.tolist(),
                    'scores': np.ones((interpolated_back.shape[0],)).tolist(),
                    'interpolated': True,
                    'biomechanical': True,
                    'person_id': 0  # 뒷사람
                })
            
            # 앞사람 추가
            if front_interpolated is not None:
                interpolated_front = _reverse_axis_transform(front_interpolated[idx], axis_order, flip_z)
                persons_data.append({
                    'keypoints_3d': interpolated_front.tolist(),
                    'scores': np.ones((interpolated_front.shape[0],)).tolist(),
                    'interpolated': True,
                    'biomechanical': True,
                    'person_id': 1  # 앞사람
                })
            
            # 원본 persons를 보간된 것으로 교체
            all_results[mid_frame]['persons'] = persons_data
            print(f"    Replaced frame {mid_frame} with biomechanical interpolated persons")
    
    # 기존 보간 로직도 유지 (다른 프레임들을 위해)
    person_trajectories = {}
    for frame_idx, frame_data in enumerate(all_results):
        persons = frame_data.get('persons', [])
        assigned_persons = assign_person_ids(persons, axis_order, flip_z, kpt_thr)
        
        for person_info in assigned_persons:
            person_id = person_info['id']
            person_data = person_info['data']
            
            if person_id not in person_trajectories:
                person_trajectories[person_id] = {}
                
            arr = np.array(person_data.get('keypoints_3d', person_data.get('joints_3d')))
            arr = _apply_axis_transform(arr, order=axis_order, flip_z=flip_z)
            
            person_trajectories[person_id][frame_idx] = arr
    
    # 나머지 gap들에 대한 보간 (33번 프레임 등)
    for person_id, trajectory in person_trajectories.items():
        frame_indices = sorted(trajectory.keys())
        
        if len(frame_indices) < 2:
            continue
            
        for i in range(len(frame_indices) - 1):
            start_frame = frame_indices[i]
            end_frame = frame_indices[i + 1]
            gap = end_frame - start_frame
            
            if 1 < gap <= max_gap and not (start_frame <= 30 <= end_frame or start_frame <= 31 <= end_frame):
                print(f"  Biomechanical interpolating person {person_id}: frames {start_frame} -> {end_frame} (gap: {gap})")
                
                # 대상 프레임 리스트 생성
                target_frames = list(range(start_frame + 1, end_frame))
                
                # 생체역학적 보간 시도
                interpolated_poses = natural_walking_interpolation(
                    trajectory, start_frame, end_frame, target_frames
                )
                
                # 보간 결과를 프레임에 추가
                for idx, mid_frame in enumerate(target_frames):
                    if mid_frame >= len(all_results):
                        continue
                    
                    frame_data = all_results[mid_frame]
                    if 'persons' not in frame_data:
                        frame_data['persons'] = []
                    
                    if interpolated_poses is not None:
                        # 생체역학적 보간 성공
                        interpolated_kpts = interpolated_poses[idx]  # (17, 3)
                        original_kpts = _reverse_axis_transform(interpolated_kpts, axis_order, flip_z)
                        is_biomechanical = True
                    else:
                        # 폴백: 선형 보간
                        print(f"    Fallback to linear interpolation for person {person_id}, frame {mid_frame}")
                        alpha = (mid_frame - start_frame) / gap
                        start_kpts = trajectory[start_frame]
                        end_kpts = trajectory[end_frame]
                        interpolated_kpts = start_kpts * (1 - alpha) + end_kpts * alpha
                        original_kpts = _reverse_axis_transform(interpolated_kpts, axis_order, flip_z)
                        is_biomechanical = False
                    
                    new_person = {
                        'keypoints_3d': original_kpts.tolist(),
                        'scores': np.ones((original_kpts.shape[0],)).tolist(),
                        'interpolated': True,
                        'biomechanical': is_biomechanical,
                        'person_id': person_id
                    }
                    frame_data['persons'].append(new_person)
    
    print("Temporal interpolation completed.")
    return all_results

def _reverse_axis_transform(k3d, order='xyz', flip_z=False):
    """_apply_axis_transform의 역변환"""
    arr = k3d.copy()
    
    # 축 순서 원복 (먼저 수행)
    idx_map = {
        'xyz': (0, 1, 2),
        'xzy': (0, 2, 1), 
        'yxz': (1, 0, 2),
        'yzx': (2, 0, 1),
        'zxy': (1, 2, 0),
        'zyx': (2, 1, 0),
    }
    forward_idx = idx_map.get(order, (0, 1, 2))
    # 역변환: 원래 순서로 되돌리기
    reverse_order = [0, 0, 0]
    for new_pos, orig_pos in enumerate(forward_idx):
        reverse_order[orig_pos] = new_pos
    
    arr = arr[:, reverse_order]
    
    # Z축 뒤집기 원복 (나중에 수행)
    if flip_z:
        arr[:, 2] = -arr[:, 2]
    
    return arr

def assign_person_ids(persons, axis_order='xyz', flip_z=False, kpt_thr=0.3):
    """Assign consistent IDs to persons based on position tracking"""
    global person_tracker, next_person_id
    
    # Calculate center position for each person in current frame
    current_persons = []
    for i, person in enumerate(persons):
        arr = np.array(person.get('keypoints_3d', person.get('joints_3d')))
        arr = _apply_axis_transform(arr, order=axis_order, flip_z=flip_z)
        scores = (arr[:, 3] if arr.shape[1] > 3 else np.ones((arr.shape[0],), dtype=float))
        
        # Find valid joints
        valid_mask = (scores > kpt_thr) & np.isfinite(arr[:, :3]).all(axis=1)
        if np.any(valid_mask):
            valid_points = arr[valid_mask, :3]
            center = np.mean(valid_points, axis=0)
            current_persons.append({
                'index': i,
                'center': center,
                'person_data': person
            })
    
    if not current_persons:
        return []
    
    # If no previous tracking data, assign new IDs
    if not person_tracker:
        assigned_persons = []
        for i, person_info in enumerate(current_persons):
            person_id = next_person_id
            next_person_id += 1
            color = color_palette[i % len(color_palette)]
            
            person_tracker[person_id] = {
                'center': person_info['center'].copy(),
                'color': color
            }
            
            assigned_persons.append({
                'id': person_id,
                'color': color,
                'data': person_info['person_data']
            })
        return assigned_persons
    
    # Match current persons to previous ones based on distance
    current_centers = np.array([p['center'] for p in current_persons])
    previous_centers = np.array([person_tracker[pid]['center'] for pid in person_tracker.keys()])
    previous_ids = list(person_tracker.keys())
    
    # Calculate distance matrix
    if len(previous_centers) > 0:
        distances = cdist(current_centers, previous_centers)
        
        # Hungarian algorithm-like assignment (simple greedy for now)
        assigned_persons = []
        used_ids = set()
        
        # Sort current persons by minimum distance to any previous person
        person_indices = list(range(len(current_persons)))
        person_indices.sort(key=lambda i: np.min(distances[i]))
        
        for curr_idx in person_indices:
            person_info = current_persons[curr_idx]
            
            # Find closest previous person that hasn't been assigned
            available_ids = [i for i, pid in enumerate(previous_ids) if pid not in used_ids]
            if available_ids:
                closest_prev_idx = available_ids[np.argmin(distances[curr_idx, available_ids])]
                person_id = previous_ids[closest_prev_idx]
                used_ids.add(person_id)
                
                # Update tracker with new position
                person_tracker[person_id]['center'] = person_info['center'].copy()
                
                assigned_persons.append({
                    'id': person_id,
                    'color': person_tracker[person_id]['color'],
                    'data': person_info['person_data']
                })
            else:
                # New person - assign new ID
                person_id = next_person_id
                next_person_id += 1
                # Use next available color
                used_colors = [person_tracker[pid]['color'] for pid in used_ids]
                available_colors = [c for c in color_palette if c not in used_colors]
                color = available_colors[0] if available_colors else color_palette[person_id % len(color_palette)]
                
                person_tracker[person_id] = {
                    'center': person_info['center'].copy(),
                    'color': color
                }
                
                assigned_persons.append({
                    'id': person_id,
                    'color': color,
                    'data': person_info['person_data']
                })
        
        return assigned_persons
    else:
        # Fresh start
        return assign_person_ids(persons, axis_order, flip_z, kpt_thr)


def visualize_frame_final(frame_data, frame_idx, output_dir,
                          azim=45, elev=20, axis_limit=None,
                          axis_order='xyz', flip_z=False, kpt_thr=0.3,
                          center_mode='auto', scale=1.2,
                          color_mode='index',
                          global_center=None, global_range=None):
    """Visualize frame with robust error handling (overlay all persons)"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    persons = frame_data.get('persons', [])
    num_persons = len(persons)
    
    # Assign consistent person IDs and colors
    assigned_persons = assign_person_ids(persons, axis_order, flip_z, kpt_thr)
    
    print(f"Frame {frame_idx}: {num_persons} persons")
    
    # Calculate safe center point from all valid joints
    all_valid_points = []
    for person_info in assigned_persons:
        person = person_info['data']
        arr = np.array(person.get('keypoints_3d', person.get('joints_3d')))
        arr = _apply_axis_transform(arr, order=axis_order, flip_z=flip_z)
        scores = (arr[:, 3] if arr.shape[1] > 3 else np.ones((arr.shape[0],), dtype=float))
        mask = (scores > kpt_thr) & np.isfinite(arr[:, :3]).all(axis=1)
        if np.any(mask):
            all_valid_points.append(arr[mask, :3])
    
    # Use global bounds for consistent axes across all frames
    if global_center is not None and global_range is not None:
        center_point = global_center
        range_meters = global_range
    elif len(all_valid_points) > 0:
        # Fallback to frame-based calculation if no global bounds provided
        pts = np.concatenate(all_valid_points, axis=0)
        if center_mode == 'bbox':
            pmin = np.min(pts, axis=0)
            pmax = np.max(pts, axis=0)
            center_point = (pmin + pmax) / 2.0
            ranges = pmax - pmin
            ranges = ranges * 1.1
        else:
            center_point = np.mean(pts, axis=0)
            ranges = np.ptp(pts, axis=0)
            ranges = ranges * 1.1
        max_range = np.max(ranges)
        range_meters = max(2.5, max_range * float(scale) * 1.3)
    else:
        center_point = [0, 0, 1.0]
        range_meters = 2.8
    
    # Draw each person with occlusion handling - sort by depth (back to front)
    total_joints = 0
    total_connections = 0
    
    # Calculate depth (Y coordinate) for each person to sort them
    persons_with_depth = []
    for person_info in assigned_persons:
        person = person_info['data']
        arr = np.array(person.get('keypoints_3d', person.get('joints_3d')))
        arr = _apply_axis_transform(arr, order=axis_order, flip_z=flip_z)
        scores = (arr[:, 3] if arr.shape[1] > 3 else np.ones((arr.shape[0],), dtype=float))
        
        # Calculate average Y (depth) for valid joints
        valid_mask = (scores > kpt_thr) & np.isfinite(arr[:, :3]).all(axis=1)
        if np.any(valid_mask):
            avg_depth = np.mean(arr[valid_mask, 1])  # Y축이 깊이
        else:
            avg_depth = 0
        
        persons_with_depth.append({
            'person_info': person_info,
            'depth': avg_depth,
            'arr': arr,
            'scores': scores
        })
    
    # Sort by depth (뒤쪽 사람부터 먼저 그리기)
    persons_with_depth.sort(key=lambda x: x['depth'], reverse=True)
    
    for i, person_data in enumerate(persons_with_depth):
        person_info = person_data['person_info']
        person_id = person_info['id']
        color = person_info['color']
        arr = person_data['arr']
        scores = person_data['scores']
        
        kpts_3d = np.concatenate([arr[:, :3], scores.reshape(-1, 1)], axis=1)
        
        # 보간된 데이터 확인
        person = person_info['data']
        is_interpolated = person.get('interpolated', False)
        is_biomechanical = person.get('biomechanical', False)
        
        # 뒷사람은 더 투명하게, 앞사람은 덜 투명하게
        alpha = 0.7 if i == 0 else 0.9  # 첫 번째(뒤쪽)는 0.7, 나머지는 0.9
        
        # 보간 타입 표시
        if is_interpolated:
            if is_biomechanical:
                interp_text = " (biomechanical)"
            else:
                interp_text = " (interpolated)"
        else:
            interp_text = ""
        
        print(f"  Drawing Person ID {person_id} with {color} (depth: {person_data['depth']:.2f}, alpha: {alpha}){interp_text}:")
        valid_joints, connections = draw_skeleton_3d_final(
            ax, kpts_3d, scores, color=color, alpha=alpha, conf_threshold=kpt_thr, 
            is_interpolated=is_interpolated, is_biomechanical=is_biomechanical
        )
        
        total_joints += valid_joints
        total_connections += connections
        
        # Add person label near highest valid point
        kpts_array = np.array(kpts_3d)
        scores_array = np.array(scores)
        
        valid_mask = (scores_array > 0.3) & np.isfinite(kpts_array).all(axis=1)
        if np.any(valid_mask):
            valid_kpts = kpts_array[valid_mask]
            highest_point = valid_kpts[np.argmax(valid_kpts[:, 2])]  # Highest Z
            
            ax.text(highest_point[0], highest_point[1], highest_point[2] + 0.1,
                   f'Person {person_id}', fontsize=12, color=color, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
    
    # Use requested axis limit if provided
    if axis_limit is not None:
        range_meters = axis_limit
    setup_3d_axes_safe(ax, center_point, range_meters, azim=azim, elev=elev)
    
    title = f'Video 3D Pose (Frame {frame_idx}) | {len(assigned_persons)} Persons | {total_connections} Connections'
    plt.title(title, fontsize=14, weight='bold', pad=20)
    
    # Save with high quality
    output_path = Path(output_dir) / f'final_frame_{frame_idx:04d}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    return output_path, total_joints, total_connections

def main():
    global person_tracker, next_person_id
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to 3D JSON results')
    parser.add_argument('--out-dir', required=True, help='Directory to save PNGs')
    parser.add_argument('--start', type=int, default=0, help='Start frame index')
    parser.add_argument('--end', type=int, default=-1, help='End frame index (inclusive), -1 for all')
    parser.add_argument('--azim', type=float, default=45)
    parser.add_argument('--elev', type=float, default=20)
    parser.add_argument('--axis-limit', type=float, default=None, help=r'강제 박스 크기(m). 없으면 자동')
    parser.add_argument('--axis-order', type=str, default='xyz', choices=['xyz','xzy','yxz','yzx','zxy','zyx'])
    parser.add_argument('--flip-z', action='store_true')
    parser.add_argument('--kpt-thr', type=float, default=0.3)
    parser.add_argument('--center-mode', type=str, default='auto', choices=['auto','bbox','index'], help='축 중심 계산 방식')
    parser.add_argument('--scale', type=float, default=1.2, help='자동 스케일 시 여유 배율(>1.0)')
    parser.add_argument('--color-mode', type=str, default='lr', choices=['lr','index','fixed'], help='사람 색상 매핑: lr=좌/우 고정(파랑/빨강)')
    args = parser.parse_args()

    # Initialize person tracking
    person_tracker = {}
    next_person_id = 0
    
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = data['results']
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    end_idx = len(results) - 1 if args.end < 0 or args.end >= len(results) else args.end

    # Apply temporal interpolation to fill missing keypoints
    results = interpolate_missing_keypoints(
        results, args.axis_order, args.flip_z, args.kpt_thr, max_gap=5
    )
    
    # Calculate global bounds for consistent visualization axes
    frames_to_process = results[args.start:end_idx + 1]
    global_center, global_range = calculate_global_bounds(
        frames_to_process, args.axis_order, args.flip_z, args.kpt_thr
    )
    
    print(f"Global center: {global_center}, Global range: {global_range}")

    for idx in range(args.start, end_idx + 1):
        frame = results[idx]
        frame_idx = frame.get('frame', frame.get('frame_index', idx))
        visualize_frame_final(frame, frame_idx, out_dir,
                              azim=args.azim, elev=args.elev,
                              axis_limit=args.axis_limit,
                              axis_order=args.axis_order,
                              flip_z=args.flip_z,
                              kpt_thr=args.kpt_thr,
                              center_mode=args.center_mode,
                              scale=args.scale,
                              color_mode=args.color_mode,
                              global_center=global_center,
                              global_range=global_range)

    print(f'Saved frames {args.start}..{end_idx} to {out_dir}')

if __name__ == "__main__":
    main()