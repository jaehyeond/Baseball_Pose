#!/usr/bin/env python3
"""
Final 3D visualization for triangulated video results (multi-person overlay).

Usage example:
  python scripts/visualize_final_3d.py \
    --input D:\mmpose\output_3d\azure_kinect_3d_results_new.json \
    --out-dir D:\mmpose\vis3d_final \
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

def setup_3d_axes_safe(ax, center_point=None, range_meters=2.0, azim=45, elev=20):
    """Setup 3D axes with proper standing pose orientation"""
    if center_point is None:
        center_point = [0, 0, 1.0]
    
    x_center, y_center, z_center = center_point
    half_range = range_meters / 2
    
    # Z축이 위쪽을 향하도록 설정 (서있는 자세)
    ax.set_xlim(x_center - half_range, x_center + half_range)
    ax.set_ylim(y_center - half_range, y_center + half_range)
    ax.set_zlim(max(0, z_center - half_range), z_center + half_range)  # Z축 최소값 0 보장
    
    # 축 비율을 1:1:1로 설정하여 정확한 비례 유지
    ax.set_box_aspect([1, 1, 1])
    
    ax.set_xlabel('X (meters)', fontsize=10)
    ax.set_ylabel('Y (meters)', fontsize=10) 
    ax.set_zlabel('Z (meters)', fontsize=10)
    
    # 시점을 서있는 사람을 보기 좋게 조정
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, alpha=0.3)
    
    # 축 배경 설정
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)

def fix_pose_orientation(kpts_3d):
    """포즈 방향을 서있는 자세로 수정"""
    if len(kpts_3d) < 17:
        return kpts_3d
    
    kpts = np.array(kpts_3d)
    
    # 주요 관절 인덱스 (COCO-17)
    nose = 0
    left_shoulder = 5
    right_shoulder = 6
    left_hip = 11
    right_hip = 12
    left_ankle = 15
    right_ankle = 16
    
    # 유효한 관절들만 선택
    valid_joints = []
    for i in [nose, left_shoulder, right_shoulder, left_hip, right_hip, left_ankle, right_ankle]:
        if i < len(kpts) and np.isfinite(kpts[i]).all():
            valid_joints.append(i)
    
    if len(valid_joints) < 3:
        return kpts
    
    # 어깨 중심과 엉덩이 중심 계산
    shoulder_joints = [i for i in [left_shoulder, right_shoulder] if i in valid_joints]
    hip_joints = [i for i in [left_hip, right_hip] if i in valid_joints]
    foot_joints = [i for i in [left_ankle, right_ankle] if i in valid_joints]
    
    if len(shoulder_joints) >= 1 and len(hip_joints) >= 1:
        shoulder_center = np.mean(kpts[shoulder_joints], axis=0)
        hip_center = np.mean(kpts[hip_joints], axis=0)
        
        # 몸통 벡터 (어깨 → 엉덩이)
        torso_vector = hip_center - shoulder_center
        
        # 발이 있다면 발 중심도 고려
        if len(foot_joints) >= 1:
            foot_center = np.mean(kpts[foot_joints], axis=0)
            # 엉덩이 → 발 벡터
            leg_vector = foot_center - hip_center
            # 전체 몸 벡터 (어깨 → 발)
            body_vector = foot_center - shoulder_center
        else:
            body_vector = torso_vector
        
        # Z축이 위쪽이 되도록 조정이 필요한지 확인
        # 현재 몸 벡터가 주로 어느 축을 향하고 있는지 확인
        abs_components = np.abs(body_vector)
        main_axis = np.argmax(abs_components)
        
        # 만약 주 방향이 Z축이 아니라면 회전 필요
        if main_axis != 2:  # Z축이 아닌 경우
            # 간단한 축 교환으로 수정
            if main_axis == 0:  # X축이 주 방향인 경우
                # X → Z, Y → Y, Z → X 교환
                kpts_fixed = kpts.copy()
                kpts_fixed[:, [0, 2]] = kpts[:, [2, 0]]
                # 방향 조정
                if body_vector[0] < 0:  # 아래쪽을 향하고 있다면 뒤집기
                    kpts_fixed[:, 2] = -kpts_fixed[:, 2]
                return kpts_fixed
            elif main_axis == 1:  # Y축이 주 방향인 경우  
                # Y → Z, X → X, Z → Y 교환
                kpts_fixed = kpts.copy()
                kpts_fixed[:, [1, 2]] = kpts[:, [2, 1]]
                # 방향 조정
                if body_vector[1] < 0:  # 아래쪽을 향하고 있다면 뒤집기
                    kpts_fixed[:, 2] = -kpts_fixed[:, 2]
                return kpts_fixed
        else:
            # Z축이 주 방향이지만 아래쪽을 향하고 있다면 뒤집기
            if body_vector[2] < 0:
                kpts_fixed = kpts.copy()
                kpts_fixed[:, 2] = -kpts_fixed[:, 2]
                return kpts_fixed
    
    return kpts

def simple_interpolate_missing_keypoints(all_results, axis_order='xyz', flip_z=False, kpt_thr=0.3, max_gap=3):
    """
    간단한 선형 보간을 통해 누락된 키포인트를 추정
    """
    print("Starting simple temporal interpolation for missing keypoints...")
    
    # 기본 person tracking 및 interpolation
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
    
    # 간단한 선형 보간으로 gap 채우기
    for person_id, trajectory in person_trajectories.items():
        frame_indices = sorted(trajectory.keys())
        
        if len(frame_indices) < 2:
            continue
            
        for i in range(len(frame_indices) - 1):
            start_frame = frame_indices[i]
            end_frame = frame_indices[i + 1]
            gap = end_frame - start_frame
            
            if 1 < gap <= max_gap:
                print(f"  Linear interpolating person {person_id}: frames {start_frame} -> {end_frame} (gap: {gap})")
                
                start_kpts = trajectory[start_frame]
                end_kpts = trajectory[end_frame]
                
                # 선형 보간으로 중간 프레임들 생성
                for mid_frame in range(start_frame + 1, end_frame):
                    alpha = (mid_frame - start_frame) / gap
                    interpolated_kpts = start_kpts * (1 - alpha) + end_kpts * alpha
                    original_kpts = _reverse_axis_transform(interpolated_kpts, axis_order, flip_z)
                    
                    # 해당 프레임에 보간된 person 추가
                    frame_data = all_results[mid_frame]
                    if 'persons' not in frame_data:
                        frame_data['persons'] = []
                    
                    new_person = {
                        'keypoints_3d': original_kpts.tolist(),
                        'scores': np.ones((original_kpts.shape[0],)).tolist(),
                        'interpolated': True,
                        'person_id': person_id
                    }
                    frame_data['persons'].append(new_person)
    
    print("Simple temporal interpolation completed.")
    return all_results

def draw_skeleton_3d_final(ax, kpts_3d, scores, color='red', alpha=0.8, point_size=30, conf_threshold=0.3, is_interpolated=False):
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
    
    # Draw keypoints
    if len(valid_joints) > 0:
        valid_points = kpts_3d[valid_joints]
        # 보간된 키포인트는 다른 스타일로 표시
        if is_interpolated:
            # 보간된 점: 삼각형으로 표시, 더 투명하게
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
            # 보간된 키포인트는 선도 다른 스타일로
            if is_interpolated:
                ax.plot(xs, ys, zs, color=color, linewidth=3, alpha=alpha*0.7, 
                       linestyle='--', solid_capstyle='round', solid_joinstyle='round')
            else:
                ax.plot(xs, ys, zs, color=color, linewidth=4, alpha=alpha, 
                       solid_capstyle='round', solid_joinstyle='round')
            connections_drawn += 1
    
    return len(valid_joints), connections_drawn

def _apply_axis_transform(k3d, order='xyz', flip_z=False):
    """Apply axis reordering and Z-flip transformations"""
    arr = k3d.copy()
    
    # Z축 뒤집기 먼저 수행
    if flip_z:
        arr[:, 2] = -arr[:, 2]
    
    # 축 순서 변경
    idx_map = {
        'xyz': (0, 1, 2),
        'xzy': (0, 2, 1), 
        'yxz': (1, 0, 2),
        'yzx': (2, 0, 1),
        'zxy': (1, 2, 0),
        'zyx': (2, 1, 0),
    }
    new_idx = idx_map.get(order, (0, 1, 2))
    arr = arr[:, new_idx]
    
    return arr

def _reverse_axis_transform(k3d, axis_order='xyz', flip_z=False):
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
    forward_idx = idx_map.get(axis_order, (0, 1, 2))
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
    
    assigned_persons = []
    current_centers = []
    
    # Calculate center points for all current persons
    for person in persons:
        arr = np.array(person.get('keypoints_3d', person.get('joints_3d')))
        arr = _apply_axis_transform(arr, order=axis_order, flip_z=flip_z)
        scores = (arr[:, 3] if arr.shape[1] > 3 else np.ones((arr.shape[0],), dtype=float))
        
        # Calculate center using valid joints
        valid_mask = (scores > kpt_thr) & np.isfinite(arr[:, :3]).all(axis=1)
        if np.any(valid_mask):
            center = np.mean(arr[valid_mask, :3], axis=0)
            current_centers.append(center)
        else:
            current_centers.append(np.array([0, 0, 0]))
    
    if len(current_centers) == 0:
        return assigned_persons
    
    current_centers = np.array(current_centers)
    
    if len(person_tracker) == 0:
        # First frame - assign new IDs
        for i, (person, center) in enumerate(zip(persons, current_centers)):
            color = color_palette[next_person_id % len(color_palette)]
            person_tracker[next_person_id] = {'center': center, 'color': color}
            assigned_persons.append({
                'id': next_person_id, 
                'color': color, 
                'data': person
            })
            next_person_id += 1
    else:
        # Match current persons to existing tracks
        existing_ids = list(person_tracker.keys())
        existing_centers = np.array([person_tracker[pid]['center'] for pid in existing_ids])
        
        if len(existing_centers) > 0:
            # Compute distance matrix
            distances = cdist(current_centers, existing_centers)
            
            # Simple greedy assignment (closest match)
            used_existing = set()
            for i in range(len(current_centers)):
                # Find closest unused existing person
                min_dist = float('inf')
                best_match = None
                
                for j, existing_id in enumerate(existing_ids):
                    if existing_id not in used_existing and distances[i, j] < min_dist:
                        min_dist = distances[i, j]
                        best_match = existing_id
                
                if best_match is not None and min_dist < 1.0:  # threshold for matching
                    # Update existing track
                    person_tracker[best_match]['center'] = current_centers[i]
                    assigned_persons.append({
                        'id': best_match,
                        'color': person_tracker[best_match]['color'],
                        'data': persons[i]
                    })
                    used_existing.add(best_match)
                else:
                    # Create new track
                    color = color_palette[next_person_id % len(color_palette)]
                    person_tracker[next_person_id] = {
                        'center': current_centers[i], 
                        'color': color
                    }
                    assigned_persons.append({
                        'id': next_person_id,
                        'color': color,
                        'data': persons[i]
                    })
                    next_person_id += 1
    
    return assigned_persons

def calculate_global_bounds(all_results, axis_order='xyz', flip_z=False, kpt_thr=0.3):
    """Calculate global bounds across all frames"""
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

def visualize_frame_final(frame_data, frame_idx, output_dir,
                          azim=45, elev=20, axis_limit=None,
                          axis_order='xyz', flip_z=False, kpt_thr=0.3,
                          center_mode='auto', scale=1.2,
                          color_mode='index',
                          global_center=None, global_range=None):
    """서있는 자세로 시각화하는 개선된 함수"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    persons = frame_data.get('persons', [])
    num_persons = len(persons)
    
    # 사람 ID 및 색상 할당
    assigned_persons = assign_person_ids(persons, axis_order, flip_z, kpt_thr)
    
    print(f"Frame {frame_idx}: {num_persons} persons")
    
    # 유효한 포인트 수집 및 자세 수정
    all_valid_points = []
    corrected_persons = []
    
    for person_info in assigned_persons:
        person = person_info['data']
        arr = np.array(person.get('keypoints_3d', person.get('joints_3d')))
        
        # 자세 방향 수정 적용
        arr_corrected = fix_pose_orientation(arr)
        
        # 축 변환 적용
        arr_transformed = _apply_axis_transform(arr_corrected, order=axis_order, flip_z=flip_z)
        scores = (arr_transformed[:, 3] if arr_transformed.shape[1] > 3 else np.ones((arr_transformed.shape[0],), dtype=float))
        
        mask = (scores > kpt_thr) & np.isfinite(arr_transformed[:, :3]).all(axis=1)
        if np.any(mask):
            all_valid_points.append(arr_transformed[mask, :3])
        
        corrected_persons.append({
            'person_info': person_info,
            'arr_transformed': arr_transformed,
            'scores': scores
        })
    
    # 전역 경계 사용 또는 프레임별 계산
    if global_center is not None and global_range is not None:
        center_point = global_center
        range_meters = global_range
    elif len(all_valid_points) > 0:
        pts = np.concatenate(all_valid_points, axis=0)
        center_point = np.mean(pts, axis=0)
        ranges = np.ptp(pts, axis=0)
        max_range = np.max(ranges)
        range_meters = max(2.5, max_range * float(scale) * 1.3)
    else:
        center_point = [0, 0, 1.0]
        range_meters = 2.8
    
    # 깊이별 정렬 및 그리기
    total_joints = 0
    total_connections = 0
    
    persons_with_depth = []
    for corrected_person in corrected_persons:
        person_info = corrected_person['person_info']
        arr_transformed = corrected_person['arr_transformed']
        scores = corrected_person['scores']
        
        # Y축을 깊이로 사용
        valid_mask = (scores > kpt_thr) & np.isfinite(arr_transformed[:, :3]).all(axis=1)
        if np.any(valid_mask):
            avg_depth = np.mean(arr_transformed[valid_mask, 1])
        else:
            avg_depth = 0
        
        persons_with_depth.append({
            'person_info': person_info,
            'depth': avg_depth,
            'arr_transformed': arr_transformed,
            'scores': scores
        })
    
    # 깊이별 정렬 (뒤에서 앞으로)
    persons_with_depth.sort(key=lambda x: x['depth'], reverse=True)
    
    for i, person_data in enumerate(persons_with_depth):
        person_info = person_data['person_info']
        person_id = person_info['id']
        color = person_info['color']
        arr_transformed = person_data['arr_transformed']
        scores = person_data['scores']
        
        kpts_3d = np.concatenate([arr_transformed[:, :3], scores.reshape(-1, 1)], axis=1)
        
        # 보간 정보
        person = person_info['data']
        is_interpolated = person.get('interpolated', False)
        
        # 투명도 설정 (뒷사람은 더 투명하게)
        alpha = 0.7 if i == 0 else 0.9
        
        # 스켈레톤 크기를 큐브에 맞게 조정
        point_size = max(80, int(range_meters * 20))
        
        # 보간 타입 표시
        if is_interpolated:
            interp_text = " (interpolated)"
        else:
            interp_text = ""
        
        print(f"  Drawing Person ID {person_id} with {color} (depth: {person_data['depth']:.2f}){interp_text}")
        
        valid_joints, connections = draw_skeleton_3d_final(
            ax, kpts_3d, scores, color=color, alpha=alpha, 
            point_size=point_size, conf_threshold=kpt_thr,
            is_interpolated=is_interpolated
        )
        
        total_joints += valid_joints
        total_connections += connections
        
        # 사람 라벨 (머리 위쪽에)
        valid_mask = (scores > 0.3) & np.isfinite(kpts_3d).all(axis=1)
        if np.any(valid_mask):
            valid_kpts = kpts_3d[valid_mask]
            highest_point = valid_kpts[np.argmax(valid_kpts[:, 2])]  # 가장 높은 Z
            
            label_offset = range_meters * 0.05
            ax.text(highest_point[0], highest_point[1], highest_point[2] + label_offset,
                   f'Person {person_id}', fontsize=12, color=color, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
    
    # 축 설정 (서있는 자세용)
    if axis_limit is not None:
        range_meters = axis_limit
    setup_3d_axes_safe(ax, center_point, range_meters, azim=azim, elev=elev)
    
    title = f'Video 3D Pose (Frame {frame_idx}) | {len(assigned_persons)} Persons | {total_connections} Connections'
    plt.title(title, fontsize=14, weight='bold', pad=20)
    
    # 고품질 저장 - vis3d_final로 고정
    output_path = Path("D:/mmpose/vis3d_final") / f'final_frame_{frame_idx:04d}.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    return output_path, total_joints, total_connections

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize 3D poses from triangulated data')
    
    parser.add_argument('--input', required=True, help='Path to triangulated JSON file')
    parser.add_argument('--out-dir', default='D:/mmpose/vis3d_final', help='Output directory (default: vis3d_final)')
    parser.add_argument('--start', type=int, default=0, help='Start frame (0-based)')
    parser.add_argument('--end', type=int, default=-1, help='End frame (exclusive, -1 for all)')
    parser.add_argument('--azim', type=float, default=-45.0, help='Azimuth angle')
    parser.add_argument('--elev', type=float, default=20.0, help='Elevation angle')
    parser.add_argument('--axis-limit', type=float, default=None, help='Fixed axis limits')
    parser.add_argument('--axis-order', choices=['xyz','xzy','yxz','yzx','zxy','zyx'], 
                       default='xzy', help='Axis order transformation')
    parser.add_argument('--flip-z', action='store_true', help='Flip Z axis')
    parser.add_argument('--kpt-thr', type=float, default=0.3, help='Keypoint confidence threshold')
    parser.add_argument('--center-mode', choices=['auto','bbox','index'], default='auto',
                       help='Center calculation mode')
    parser.add_argument('--scale', type=float, default=1.25, help='Scale factor for visualization')
    parser.add_argument('--color-mode', choices=['lr','index','fixed'], default='index',
                       help='Color assignment mode')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Reset global variables for multiple runs
    global person_tracker, next_person_id
    person_tracker = {}
    next_person_id = 0
    
    # Load data
    print(f'Loading poses from: {args.input}')
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    # Handle different JSON formats
    if isinstance(data, dict):
        if 'results' in data:
            all_results = data['results']
        elif 'frames' in data:
            all_results = data['frames']
        else:
            # Try to find the first list value
            for key, value in data.items():
                if isinstance(value, list):
                    all_results = value
                    break
            else:
                raise ValueError(f"Could not find frame data in JSON structure: {list(data.keys())}")
    elif isinstance(data, list):
        all_results = data
    else:
        raise ValueError(f"Unexpected JSON format: {type(data)}")
    
    print(f'Loaded {len(all_results)} frames')
    
    # Apply simple temporal interpolation (not biomechanical)
    results = simple_interpolate_missing_keypoints(
        all_results, axis_order=args.axis_order, flip_z=args.flip_z, 
        kpt_thr=args.kpt_thr, max_gap=3
    )
    
    # Calculate global bounds for consistent visualization
    global_center, global_range = calculate_global_bounds(
        results, axis_order=args.axis_order, flip_z=args.flip_z, kpt_thr=args.kpt_thr
    )
    print(f'Global center: {global_center}, Global range: {global_range}')
    
    # Determine frame range
    if args.end == -1 or args.end > len(results):
        end_frame = len(results)
    else:
        end_frame = args.end
    
    start_frame = max(0, args.start)
    
    print(f'Processing frames {start_frame} to {end_frame-1}...')
    print(f'Results will be saved to: D:/mmpose/vis3d_final')
    
    # Process all frames
    for frame_idx in range(start_frame, end_frame):
        if frame_idx >= len(results):
            break
            
        frame_data = results[frame_idx]
        
        output_path, total_joints, total_connections = visualize_frame_final(
            frame_data, frame_idx, args.out_dir,
            azim=args.azim, elev=args.elev, axis_limit=args.axis_limit,
            axis_order=args.axis_order, flip_z=args.flip_z, kpt_thr=args.kpt_thr,
            center_mode=args.center_mode, scale=args.scale, color_mode=args.color_mode,
            global_center=global_center, global_range=global_range
        )
    
    print(f'Saved frames {start_frame}..{end_frame-1} to D:/mmpose/vis3d_final')

if __name__ == '__main__':
    main()