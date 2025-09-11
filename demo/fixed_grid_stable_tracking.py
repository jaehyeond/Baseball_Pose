#!/usr/bin/env python3
"""
Fixed Grid + Stable Person Tracking 3D Visualization
- 완전 고정된 Grid (프레임간 변화 없음)
- 안정적인 사람 ID 추적 (색상 일관성 유지)
- 정확한 좌표 변환
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import argparse

# COCO-17 skeleton connections
SKELETON_EDGES = [
    (0,1),(1,2),(2,3),(3,4),
    (1,5),(5,7),(7,9),
    (1,6),(6,8),(8,10),
    (5,6),(5,11),(6,12),
    (11,12),(11,13),(13,15),
    (12,14),(14,16)
]

# 고정 Grid 설정 (미터 단위) - 데이터 범위에 맞게 조정
FIXED_GRID_SIZE = {
    'X_RANGE': (-2.0, 1.0),  # 3m 폭 (좌우)
    'Y_RANGE': (2.0, 6.0),   # 4m 깊이 (전후) - 데이터가 3~5 범위  
    'Z_RANGE': (-2.0, 1.0),  # 3m 높이 (상하) - 데이터가 -1.5~0.5 범위
}

# 고정 카메라 뷰
FIXED_VIEW = {
    'azim': -60,  # 고정 방위각
    'elev': 20,   # 고정 고도각
}

# 고정 색상 (사람별 일관성 유지)
PERSON_COLORS = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']

# Global person tracking
prev_centers = None

def ensure_j3(kpts):
    """Convert any format to (J,3) shape robustly"""
    k = np.array(kpts, dtype=np.float32)
    if k.ndim == 1 and (k.size % 3 == 0):
        k = k.reshape(-1, 3)
    elif k.ndim == 2 and k.shape[1] == 3:
        pass
    elif k.ndim == 2 and k.shape[0] == 3:
        k = k.T
    elif k.ndim == 2 and k.shape[1] > 3:
        k = k[:, :3]
    else:
        return np.empty((0, 3), dtype=np.float32)
    return k

def axis_indices(order):
    """Convert axis order string to indices"""
    idx = {'x': 0, 'y': 1, 'z': 2}
    order = order.lower()
    if len(order) != 3 or any(c not in 'xyz' for c in order):
        raise ValueError(f'Invalid axis order: {order}')
    return [idx[c] for c in order]

def permute_kpts(kpts, order_idx):
    """Reorder coordinate axes"""
    k = ensure_j3(kpts)
    if k.size == 0:
        return k
    return k[:, order_idx]

def flip_axis_kpts(kpts, flip_z=False):
    """Flip Z axis if needed"""
    k = kpts.copy()
    if flip_z and k.size > 0:
        k[:, 2] = -k[:, 2]
    return k

def person_centroid(kpts):
    """Calculate person center from valid joints"""
    if kpts.size == 0:
        return np.array([0, 0, 0])
    
    # Filter valid joints (not 0,0,0)
    valid_mask = ~((kpts[:, 0] == 0) & (kpts[:, 1] == 0) & (kpts[:, 2] == 0))
    if valid_mask.sum() == 0:
        return np.array([0, 0, 0])
    
    valid_kpts = kpts[valid_mask]
    return np.mean(valid_kpts, axis=0)

def stable_assign_persons(prev_centers, curr_centers):
    """Stable person assignment to maintain color consistency"""
    if prev_centers is None or len(prev_centers) == 0:
        return list(range(len(curr_centers)))
    
    if len(curr_centers) == 0:
        return []
    
    # Calculate distance matrix
    distances = np.zeros((len(prev_centers), len(curr_centers)))
    for i, prev_center in enumerate(prev_centers):
        for j, curr_center in enumerate(curr_centers):
            distances[i, j] = np.linalg.norm(prev_center - curr_center)
    
    # Hungarian-like assignment (greedy for simplicity)
    assigned = []
    used_curr = set()
    
    for prev_idx in range(len(prev_centers)):
        if len(used_curr) >= len(curr_centers):
            break
            
        # Find closest unassigned current person
        best_curr_idx = -1
        best_dist = float('inf')
        
        for curr_idx in range(len(curr_centers)):
            if curr_idx not in used_curr and distances[prev_idx, curr_idx] < best_dist:
                best_dist = distances[prev_idx, curr_idx]
                best_curr_idx = curr_idx
        
        if best_curr_idx != -1 and best_dist < 1.0:  # max 1m distance threshold (더 엄격)
            assigned.append((prev_idx, best_curr_idx))
            used_curr.add(best_curr_idx)
    
    # Create mapping
    mapping = [-1] * len(curr_centers)
    for prev_idx, curr_idx in assigned:
        mapping[curr_idx] = prev_idx
    
    # Assign new IDs to unassigned persons
    next_new_id = len(prev_centers)
    for curr_idx in range(len(curr_centers)):
        if mapping[curr_idx] == -1:
            mapping[curr_idx] = next_new_id
            next_new_id += 1
    
    return mapping

def draw_person_stable(ax, kpts, color, person_id, radius=50, thickness=6):
    """Draw skeleton with stable styling and person ID"""
    if kpts.size == 0:
        return
    
    # Filter valid joints (not 0,0,0)
    valid_joints = ~((kpts[:, 0] == 0) & (kpts[:, 1] == 0) & (kpts[:, 2] == 0))
    
    # Draw valid joints as points
    if valid_joints.sum() > 0:
        valid_kpts = kpts[valid_joints]
        ax.scatter(valid_kpts[:, 0], valid_kpts[:, 1], valid_kpts[:, 2], 
                  s=radius, c=color, depthshade=True, alpha=0.9, 
                  edgecolors='black', linewidth=2)
    
    # Draw skeleton connections (only if both joints are valid)
    for i, j in SKELETON_EDGES:
        if (i < len(kpts) and j < len(kpts) and 
            valid_joints[i] and valid_joints[j]):
            xs = [kpts[i, 0], kpts[j, 0]]
            ys = [kpts[i, 1], kpts[j, 1]]
            zs = [kpts[i, 2], kpts[j, 2]]
            ax.plot(xs, ys, zs, linewidth=thickness, c=color, alpha=0.9)
    
    # Add person ID label above head
    if valid_joints.sum() > 0:
        head_joints = [0, 1, 2, 3, 4]  # nose, eyes, ears
        head_valid = [i for i in head_joints if i < len(kpts) and valid_joints[i]]
        
        if head_valid:
            head_center = np.mean(kpts[head_valid], axis=0)
            ax.text(head_center[0], head_center[1], head_center[2] + 0.2, 
                   f'P{person_id}', fontsize=12, fontweight='bold', 
                   color=color, ha='center', va='bottom')

def setup_absolute_fixed_axes(ax, title="Stable Tracking Fixed Grid"):
    """Setup absolutely fixed 3D axes - NEVER CHANGES"""
    # 절대 고정 Grid 범위
    ax.set_xlim(FIXED_GRID_SIZE['X_RANGE'])
    ax.set_ylim(FIXED_GRID_SIZE['Y_RANGE'])
    ax.set_zlim(FIXED_GRID_SIZE['Z_RANGE'])
    
    # 완전히 고정된 비율 (실제 미터 기준)
    x_range = FIXED_GRID_SIZE['X_RANGE'][1] - FIXED_GRID_SIZE['X_RANGE'][0]
    y_range = FIXED_GRID_SIZE['Y_RANGE'][1] - FIXED_GRID_SIZE['Y_RANGE'][0]
    z_range = FIXED_GRID_SIZE['Z_RANGE'][1] - FIXED_GRID_SIZE['Z_RANGE'][0]
    ax.set_box_aspect([x_range, y_range, z_range])
    
    # 절대 고정 카메라 뷰 (회전 없음)
    ax.view_init(elev=FIXED_VIEW['elev'], azim=FIXED_VIEW['azim'])
    
    # 축 라벨
    ax.set_xlabel('X (meters)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y (meters)', fontsize=14, fontweight='bold')
    ax.set_zlabel('Z (meters)', fontsize=14, fontweight='bold')
    
    # 고정 Grid 스타일
    ax.grid(True, alpha=0.5, linewidth=1)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.zaxis.set_major_locator(plt.MultipleLocator(0.5))
    
    # 제목
    ax.set_title(title, fontsize=16, fontweight='bold', pad=25)
    
    # 배경 스타일
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.xaxis.pane.set_alpha(0.2)
    ax.yaxis.pane.set_alpha(0.2)
    ax.zaxis.pane.set_alpha(0.2)

def visualize_frame_stable(data_frame, frame_idx, axis_order_idx, flip_z, out_path):
    """Visualize single frame with stable person tracking"""
    global prev_centers
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 프레임 데이터 처리
    persons = data_frame.get('persons', [])
    if not persons and 'keypoints_3d' in data_frame:
        persons = [{'keypoints_3d': data_frame['keypoints_3d']}]
    
    # 좌표 변환 및 중심점 계산
    transformed_persons = []
    curr_centers = []
    
    for person in persons:
        # 🔥 CRITICAL: 정확한 좌표 변환
        k = permute_kpts(person.get('keypoints_3d', []), axis_order_idx)
        k = flip_axis_kpts(k, flip_z)
        
        if k.size == 0:
            continue
            
        # 유효한 키포인트 체크
        valid_mask = ~((k[:, 0] == 0) & (k[:, 1] == 0) & (k[:, 2] == 0))
        if valid_mask.sum() < 5:  # 최소 5개 키포인트 필요
            continue
            
        transformed_persons.append(k)
        curr_centers.append(person_centroid(k))
    
    # 안정적인 사람 ID 할당
    person_mapping = stable_assign_persons(prev_centers, curr_centers)
    
    # 시각화
    displayed_persons = 0
    for i, (kpts, person_id) in enumerate(zip(transformed_persons, person_mapping)):
        color = PERSON_COLORS[person_id % len(PERSON_COLORS)]
        draw_person_stable(ax, kpts, color, person_id)
        displayed_persons += 1
    
    # 다음 프레임을 위한 중심점 업데이트
    if curr_centers:
        # 새로운 중심점 배열 생성 (ID 순서로 정렬)
        max_id = max(person_mapping) if person_mapping else 0
        new_centers = [None] * (max_id + 1)
        
        for curr_center, person_id in zip(curr_centers, person_mapping):
            new_centers[person_id] = curr_center
            
        # None 값 제거
        prev_centers = [c for c in new_centers if c is not None]
    
    # 절대 고정 축 설정
    title = f"Stable Track Fixed Grid (Frame {frame_idx}) | {displayed_persons} Persons"
    setup_absolute_fixed_axes(ax, title)
    
    # 저장
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return out_path

def main():
    parser = argparse.ArgumentParser(description='Fixed Grid + Stable Tracking 3D Visualization')
    parser.add_argument('--input', required=True, help='Path to 3D JSON results')
    parser.add_argument('--out-dir', required=True, help='Directory to save PNGs')
    parser.add_argument('--axis-order', default='xzy', help='Axis order (default: xzy)')
    parser.add_argument('--flip-z', action='store_true', help='Flip Z axis')
    parser.add_argument('--start', type=int, default=0, help='Start frame')
    parser.add_argument('--end', type=int, default=-1, help='End frame (-1 for all)')
    
    args = parser.parse_args()
    
    # Load data
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    frames = data['results']
    axis_order_idx = axis_indices(args.axis_order)
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Process frames
    start_frame = args.start
    end_frame = len(frames) - 1 if args.end == -1 else args.end
    
    print(f"Processing frames {start_frame} to {end_frame} with STABLE TRACKING...")
    print(f"Fixed Grid: X{FIXED_GRID_SIZE['X_RANGE']}, Y{FIXED_GRID_SIZE['Y_RANGE']}, Z{FIXED_GRID_SIZE['Z_RANGE']}")
    print(f"Fixed View: azim={FIXED_VIEW['azim']}, elev={FIXED_VIEW['elev']}")
    
    global prev_centers
    prev_centers = None  # Reset tracking
    
    for frame_idx in range(start_frame, end_frame + 1):
        if frame_idx >= len(frames):
            break
            
        out_path = out_dir / f"stable_frame_{frame_idx:04d}.png"
        visualize_frame_stable(frames[frame_idx], frame_idx, axis_order_idx, args.flip_z, out_path)
        print(f"Saved: {out_path}")
    
    print(f"Completed: {end_frame - start_frame + 1} frames with STABLE TRACKING")

if __name__ == '__main__':
    main()