#!/usr/bin/env python3
"""
Fixed Grid 3D Visualization for Azure Kinect
- 고정된 그리드 크기 (실제 방 크기 기반)
- 회전 없는 고정 카메라 뷰
- 캘리브레이션 데이터 기반 좌표계
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

# 고정 Grid 설정 (미터 단위)
FIXED_GRID_SIZE = {
    'X_RANGE': (-3.0, 3.0),  # 6m 폭 (좌우)
    'Y_RANGE': (0.0, 6.0),   # 6m 깊이 (전후)
    'Z_RANGE': (-1.0, 2.5),  # 3.5m 높이 (상하)
}

# 고정 카메라 뷰
FIXED_VIEW = {
    'azim': -60,  # 고정 방위각
    'elev': 20,   # 고정 고도각
}

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

def draw_person_fixed(ax, kpts, color, radius=40, thickness=5):
    """Draw skeleton with fixed styling"""
    if kpts.size == 0:
        return
    
    # Filter valid joints (not 0,0,0)
    valid_joints = ~((kpts[:, 0] == 0) & (kpts[:, 1] == 0) & (kpts[:, 2] == 0))
    
    # Draw valid joints as points
    if valid_joints.sum() > 0:
        valid_kpts = kpts[valid_joints]
        ax.scatter(valid_kpts[:, 0], valid_kpts[:, 1], valid_kpts[:, 2], 
                  s=radius, c=color, depthshade=True, alpha=0.9, edgecolors='black', linewidth=1)
    
    # Draw skeleton connections (only if both joints are valid)
    for i, j in SKELETON_EDGES:
        if (i < len(kpts) and j < len(kpts) and 
            valid_joints[i] and valid_joints[j]):
            xs = [kpts[i, 0], kpts[j, 0]]
            ys = [kpts[i, 1], kpts[j, 1]]
            zs = [kpts[i, 2], kpts[j, 2]]
            ax.plot(xs, ys, zs, linewidth=thickness, c=color, alpha=0.9)

def setup_fixed_axes(ax, title="Fixed Grid 3D Pose"):
    """Setup fixed 3D axes - NO ROTATION"""
    # 고정 Grid 범위 설정
    ax.set_xlim(FIXED_GRID_SIZE['X_RANGE'])
    ax.set_ylim(FIXED_GRID_SIZE['Y_RANGE'])
    ax.set_zlim(FIXED_GRID_SIZE['Z_RANGE'])
    
    # 🔥 CRITICAL: 완전히 동일한 비율로 고정 (1:1:1)
    x_range = FIXED_GRID_SIZE['X_RANGE'][1] - FIXED_GRID_SIZE['X_RANGE'][0]  # 6.0
    y_range = FIXED_GRID_SIZE['Y_RANGE'][1] - FIXED_GRID_SIZE['Y_RANGE'][0]  # 6.0
    z_range = FIXED_GRID_SIZE['Z_RANGE'][1] - FIXED_GRID_SIZE['Z_RANGE'][0]  # 3.5
    ax.set_box_aspect([x_range, y_range, z_range])  # 실제 미터 비율로 고정
    
    # 고정 카메라 뷰
    ax.view_init(elev=FIXED_VIEW['elev'], azim=FIXED_VIEW['azim'])
    
    # 축 라벨
    ax.set_xlabel('X (meters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (meters)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z (meters)', fontsize=12, fontweight='bold')
    
    # Grid 스타일링
    ax.grid(True, alpha=0.4)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.zaxis.set_major_locator(plt.MultipleLocator(0.5))
    
    # 제목
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # 배경색
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)

def visualize_frame_fixed(data_frame, frame_idx, axis_order_idx, flip_z, out_path):
    """Visualize single frame with fixed grid"""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # 프레임 데이터 처리
    persons = data_frame.get('persons', [])
    if not persons and 'keypoints_3d' in data_frame:
        persons = [{'keypoints_3d': data_frame['keypoints_3d']}]
    
    # 색상 설정
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan']
    
    valid_persons = 0
    for i, person in enumerate(persons):
        k = permute_kpts(person.get('keypoints_3d', []), axis_order_idx)
        k = flip_axis_kpts(k, flip_z)
        
        if k.size == 0:
            continue
            
        # 유효한 키포인트가 있는지 확인
        valid_mask = ~((k[:, 0] == 0) & (k[:, 1] == 0) & (k[:, 2] == 0))
        if valid_mask.sum() < 3:  # 최소 3개 키포인트 필요
            continue
            
        color = colors[valid_persons % len(colors)]
        draw_person_fixed(ax, k, color)
        valid_persons += 1
    
    # 고정 축 설정
    title = f"Fixed Grid 3D Pose (Frame {frame_idx}) | {valid_persons} Persons"
    setup_fixed_axes(ax, title)
    
    # 저장
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return out_path

def main():
    parser = argparse.ArgumentParser(description='Fixed Grid 3D Visualization')
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
    
    print(f"Processing frames {start_frame} to {end_frame} with FIXED GRID...")
    print(f"Grid Size: X{FIXED_GRID_SIZE['X_RANGE']}, Y{FIXED_GRID_SIZE['Y_RANGE']}, Z{FIXED_GRID_SIZE['Z_RANGE']}")
    print(f"Fixed View: azim={FIXED_VIEW['azim']}, elev={FIXED_VIEW['elev']}")
    
    for frame_idx in range(start_frame, end_frame + 1):
        if frame_idx >= len(frames):
            break
            
        out_path = out_dir / f"fixed_frame_{frame_idx:04d}.png"
        visualize_frame_fixed(frames[frame_idx], frame_idx, axis_order_idx, args.flip_z, out_path)
        print(f"Saved: {out_path}")
    
    print(f"Completed: {end_frame - start_frame + 1} frames with FIXED GRID")

if __name__ == '__main__':
    main()