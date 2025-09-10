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

def draw_skeleton_3d_final(ax, kpts_3d, scores, color='red', alpha=0.8, point_size=30, conf_threshold=0.3):
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
        ax.scatter(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2],
                  s=point_size, c=color, alpha=alpha, depthshade=True, marker='o')
    
    # Draw skeleton connections
    connections_drawn = 0
    for i, j in SKELETON_EDGES:
        if i in valid_joints and j in valid_joints:
            pt1, pt2 = kpts_3d[i], kpts_3d[j]
            
            xs = [pt1[0], pt2[0]]
            ys = [pt1[1], pt2[1]]
            zs = [pt1[2], pt2[2]]
            ax.plot(xs, ys, zs, color=color, linewidth=4, alpha=alpha)
            connections_drawn += 1
    
    # print(f"    Drew {connections_drawn} skeleton connections")
    return len(valid_joints), connections_drawn

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
    
    # Draw each person with consistent colors
    total_joints = 0
    total_connections = 0

    for person_info in assigned_persons:
        person = person_info['data']
        person_id = person_info['id']
        color = person_info['color']
        
        arr = np.array(person.get('keypoints_3d', person.get('joints_3d')))
        arr = _apply_axis_transform(arr, order=axis_order, flip_z=flip_z)
        scores = (arr[:, 3] if arr.shape[1] > 3 else np.ones((arr.shape[0],), dtype=float))
        kpts_3d = np.concatenate([arr[:, :3], scores.reshape(-1, 1)], axis=1)

        print(f"  Drawing Person ID {person_id} with {color}:")
        valid_joints, connections = draw_skeleton_3d_final(
            ax, kpts_3d, scores, color=color, alpha=0.8, conf_threshold=kpt_thr
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
