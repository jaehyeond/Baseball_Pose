#!/usr/bin/env python3
"""
Robust 3D Visualization - 최종 안정화 버전
- 강력한 ID 추적 (confidence 기반 + 히스토리 추적)
- 음수 좌표 보정
- 완전 고정 Grid
- 데이터 품질 필터링
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

# 완전 고정 Grid (실제 방 크기 기준)
FIXED_GRID = {
    'X_RANGE': (-2.0, 1.5),  # 3.5m 폭
    'Y_RANGE': (2.5, 6.0),   # 3.5m 깊이
    'Z_RANGE': (-0.5, 2.5),  # 3m 높이 - 음수 최소화
}

# 고정 뷰
FIXED_VIEW = {'azim': -60, 'elev': 20}

# 고정 색상 (절대 변경되지 않음)
PERSON_COLORS = ['red', 'blue']  # 딱 2명만

# Global ID 추적
global_persons = {}  # {person_id: history_info}

def ensure_j3(kpts):
    k = np.array(kpts, dtype=np.float32)
    if k.ndim == 1 and (k.size % 3 == 0):
        k = k.reshape(-1, 3)
    elif k.ndim == 2 and k.shape[1] >= 3:
        k = k[:, :3]
    elif k.ndim == 2 and k.shape[0] == 3:
        k = k.T
    else:
        return np.empty((0, 3), dtype=np.float32)
    return k

def permute_and_flip(kpts, axis_order='xzy', flip_z=True):
    """좌표 변환 + 음수 보정"""
    k = ensure_j3(kpts)
    if k.size == 0:
        return k
    
    # XZY 변환
    if axis_order.lower() == 'xzy':
        k_trans = k[:, [0, 2, 1]].copy()
    else:
        k_trans = k.copy()
    
    # Z축 뒤집기
    if flip_z:
        k_trans[:, 2] = -k_trans[:, 2]
    
    # 🔥 음수 Z값 보정 (바닥을 Z=0으로)
    if k_trans.size > 0:
        # 발목 관절들로 바닥 높이 추정
        ankle_indices = [15, 16]  # left_ankle, right_ankle
        ankle_z_vals = []
        for idx in ankle_indices:
            if idx < len(k_trans):
                ankle_z_vals.append(k_trans[idx, 2])
        
        if ankle_z_vals:
            floor_z = min(ankle_z_vals)
            if floor_z < -0.2:  # 바닥이 -0.2보다 낮으면 보정
                k_trans[:, 2] -= floor_z  # 전체를 위로 이동
    
    return k_trans

def get_person_features(kpts):
    """사람 특징 추출 (ID 추적용)"""
    if kpts.size == 0:
        return None
    
    # 유효한 키포인트만 필터링
    valid_mask = ~((kpts[:, 0] == 0) & (kpts[:, 1] == 0) & (kpts[:, 2] == 0))
    if valid_mask.sum() < 8:  # 최소 8개 키포인트 필요
        return None
    
    valid_kpts = kpts[valid_mask]
    
    # 특징 계산
    features = {
        'center': np.mean(valid_kpts, axis=0),
        'height': valid_kpts[:, 2].max() - valid_kpts[:, 2].min(),
        'width': valid_kpts[:, 0].max() - valid_kpts[:, 0].min(),
        'valid_count': valid_mask.sum(),
        'bbox': [valid_kpts[:, i].min() for i in range(3)] + [valid_kpts[:, i].max() for i in range(3)]
    }
    
    return features

def robust_person_matching(frame_persons, frame_idx):
    """강력한 사람 매칭 알고리즘"""
    global global_persons
    
    # 현재 프레임 특징들 추출
    current_features = []
    valid_persons = []
    
    for person_data in frame_persons:
        kpts = permute_and_flip(person_data.get('keypoints_3d', []))
        features = get_person_features(kpts)
        
        if features is not None:
            current_features.append(features)
            valid_persons.append({'kpts': kpts, 'features': features})
    
    if not current_features:
        return []
    
    # 첫 프레임: 순서대로 ID 할당
    if not global_persons or frame_idx == 0:
        global_persons.clear()
        assignments = []
        for i, person in enumerate(valid_persons):
            person_id = i
            global_persons[person_id] = {
                'center_history': [person['features']['center']],
                'last_seen': frame_idx
            }
            assignments.append((person, person_id))
        return assignments
    
    # 기존 사람들과 매칭
    assignments = []
    used_ids = set()
    
    # 각 현재 사람에 대해 가장 가까운 기존 사람 찾기
    for curr_person in valid_persons:
        curr_center = curr_person['features']['center']
        
        best_id = None
        best_dist = float('inf')
        
        for person_id, history in global_persons.items():
            if person_id in used_ids:
                continue
                
            # 최근 중심점과의 거리 계산
            last_center = history['center_history'][-1]
            dist = np.linalg.norm(curr_center - last_center)
            
            # 거리 임계값: 0.8m
            if dist < 0.8 and dist < best_dist:
                best_dist = dist
                best_id = person_id
        
        if best_id is not None:
            # 기존 사람에 매칭
            global_persons[best_id]['center_history'].append(curr_center)
            global_persons[best_id]['last_seen'] = frame_idx
            assignments.append((curr_person, best_id))
            used_ids.add(best_id)
        else:
            # 새로운 사람 (하지만 2명 제한)
            if len(used_ids) < 2:
                new_id = 0 if 0 not in used_ids else 1
                global_persons[new_id] = {
                    'center_history': [curr_center],
                    'last_seen': frame_idx
                }
                assignments.append((curr_person, new_id))
                used_ids.add(new_id)
    
    # 오래된 기록 정리 (10프레임 이상 안 보인 사람)
    to_remove = []
    for person_id, history in global_persons.items():
        if frame_idx - history['last_seen'] > 10:
            to_remove.append(person_id)
    
    for person_id in to_remove:
        del global_persons[person_id]
    
    return assignments

def draw_robust_person(ax, kpts, person_id, radius=60, thickness=7):
    """강력한 사람 그리기"""
    if kpts.size == 0:
        return
    
    color = PERSON_COLORS[person_id % len(PERSON_COLORS)]
    
    # 유효한 관절만 표시
    valid_joints = ~((kpts[:, 0] == 0) & (kpts[:, 1] == 0) & (kpts[:, 2] == 0))
    
    # 관절점 그리기
    if valid_joints.sum() > 0:
        valid_kpts = kpts[valid_joints]
        ax.scatter(valid_kpts[:, 0], valid_kpts[:, 1], valid_kpts[:, 2], 
                  s=radius, c=color, alpha=0.9, edgecolors='black', linewidth=2)
    
    # 골격 연결선 그리기
    for i, j in SKELETON_EDGES:
        if (i < len(kpts) and j < len(kpts) and 
            valid_joints[i] and valid_joints[j]):
            
            # Grid 범위 체크
            p1, p2 = kpts[i], kpts[j]
            if (FIXED_GRID['X_RANGE'][0] <= p1[0] <= FIXED_GRID['X_RANGE'][1] and
                FIXED_GRID['Y_RANGE'][0] <= p1[1] <= FIXED_GRID['Y_RANGE'][1] and
                FIXED_GRID['Z_RANGE'][0] <= p1[2] <= FIXED_GRID['Z_RANGE'][1] and
                FIXED_GRID['X_RANGE'][0] <= p2[0] <= FIXED_GRID['X_RANGE'][1] and
                FIXED_GRID['Y_RANGE'][0] <= p2[1] <= FIXED_GRID['Y_RANGE'][1] and
                FIXED_GRID['Z_RANGE'][0] <= p2[2] <= FIXED_GRID['Z_RANGE'][1]):
                
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                       linewidth=thickness, c=color, alpha=0.9)
    
    # ID 라벨
    if valid_joints.sum() > 0:
        head_center = np.mean(valid_kpts[:3], axis=0) if len(valid_kpts) >= 3 else valid_kpts[0]
        ax.text(head_center[0], head_center[1], head_center[2] + 0.15, 
               f'Person{person_id}', fontsize=14, fontweight='bold', 
               color=color, ha='center', va='bottom')

def setup_absolute_grid(ax, frame_idx, person_count):
    """절대 고정 Grid"""
    ax.set_xlim(FIXED_GRID['X_RANGE'])
    ax.set_ylim(FIXED_GRID['Y_RANGE'])
    ax.set_zlim(FIXED_GRID['Z_RANGE'])
    
    # 고정 비율
    x_range = FIXED_GRID['X_RANGE'][1] - FIXED_GRID['X_RANGE'][0]
    y_range = FIXED_GRID['Y_RANGE'][1] - FIXED_GRID['Y_RANGE'][0]
    z_range = FIXED_GRID['Z_RANGE'][1] - FIXED_GRID['Z_RANGE'][0]
    ax.set_box_aspect([x_range, y_range, z_range])
    
    # 고정 뷰
    ax.view_init(elev=FIXED_VIEW['elev'], azim=FIXED_VIEW['azim'])
    
    # 축 라벨
    ax.set_xlabel('X (meters)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y (meters)', fontsize=14, fontweight='bold') 
    ax.set_zlabel('Z (meters)', fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.6)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.zaxis.set_major_locator(plt.MultipleLocator(0.5))
    
    # 제목
    ax.set_title(f'Robust Fixed Grid (Frame {frame_idx}) | {person_count} Persons', 
                fontsize=16, fontweight='bold', pad=25)

def visualize_robust_frame(data_frame, frame_idx, out_path):
    """강력한 프레임 시각화"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    persons = data_frame.get('persons', [])
    if not persons and 'keypoints_3d' in data_frame:
        persons = [{'keypoints_3d': data_frame['keypoints_3d']}]
    
    # 강력한 사람 매칭
    assignments = robust_person_matching(persons, frame_idx)
    
    # 시각화
    for person_data, person_id in assignments:
        draw_robust_person(ax, person_data['kpts'], person_id)
    
    # 절대 고정 Grid
    setup_absolute_grid(ax, frame_idx, len(assignments))
    
    # 저장
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return out_path

def main():
    parser = argparse.ArgumentParser(description='Robust 3D Visualization')
    parser.add_argument('--input', required=True, help='3D JSON results')
    parser.add_argument('--out-dir', required=True, help='Output directory')
    parser.add_argument('--start', type=int, default=0, help='Start frame')
    parser.add_argument('--end', type=int, default=-1, help='End frame')
    
    args = parser.parse_args()
    
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    frames = data['results']
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    start_frame = args.start
    end_frame = len(frames) - 1 if args.end == -1 else args.end
    
    print(f"ROBUST Processing frames {start_frame} to {end_frame}...")
    print(f"Fixed Grid: X{FIXED_GRID['X_RANGE']}, Y{FIXED_GRID['Y_RANGE']}, Z{FIXED_GRID['Z_RANGE']}")
    
    global global_persons
    global_persons = {}  # Reset
    
    for frame_idx in range(start_frame, end_frame + 1):
        if frame_idx >= len(frames):
            break
            
        out_path = out_dir / f"robust_frame_{frame_idx:04d}.png"
        visualize_robust_frame(frames[frame_idx], frame_idx, out_path)
        print(f"Saved: {out_path}")
    
    print(f"ROBUST Completed: {end_frame - start_frame + 1} frames")

if __name__ == '__main__':
    main()