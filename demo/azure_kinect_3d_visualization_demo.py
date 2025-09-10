# demo/azure_kinect_3d_visualization_demo.py
import os
import json
import argparse
import warnings
from typing import List, Tuple, Dict, Any

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 파일 저장용
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# -------------------- 유틸 --------------------
def load_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def to_numpy(a):
    return np.array(a, dtype=np.float32)

def nanmean_axis(a, axis=0):
    return np.nanmean(a, axis=axis)

def person_centroid(kpts: np.ndarray) -> np.ndarray:
    """kpts: (J,3) -> (3,) 평균 좌표(결측값 NaN 무시)"""
    return nanmean_axis(kpts, axis=0)

def axis_indices(order: str) -> List[int]:
    """order 문자열(예: 'yzx')을 인덱스 [1,2,0]로 변환"""
    idx = {'x': 0, 'y': 1, 'z': 2}
    order = order.lower()
    if len(order) != 3 or any(c not in 'xyz' for c in order):
        raise ValueError(f'잘못된 axis order: {order}')
    return [idx[c] for c in order]

def ensure_j3(kpts: np.ndarray) -> np.ndarray:
    """
    어떤 형태로 들어와도 (J,3)로 강건하게 변환:
    - 1D(flat, 길이=3J) -> (-1,3)
    - (J,4) -> 앞 3컬럼 사용
    - (3,J) -> 전치
    - (J,3) -> 그대로
    그 외는 NaN으로 채워 건너뜀.
    """
    k = to_numpy(kpts)
    if k.ndim == 1 and (k.size % 3 == 0):
        k = k.reshape(-1, 3)
    elif k.ndim == 2 and k.shape[1] == 3:
        pass
    elif k.ndim == 2 and k.shape[0] == 3:
        k = k.T
    elif k.ndim == 2 and k.shape[1] > 3:
        k = k[:, :3]
    else:
        # 형식을 끝내 해석 못하면 비어있는 것으로 처리
        return np.empty((0, 3), dtype=np.float32)
    return k

def permute_kpts(kpts: np.ndarray, order_idx: List[int]) -> np.ndarray:
    """(J,3) 좌표축 재정렬 (안전하게 ensure_j3 후 적용)"""
    k = ensure_j3(kpts)
    if k.size == 0:
        return k
    return k[:, order_idx]

def flip_axis_kpts(kpts: np.ndarray, flip_z: bool = False) -> np.ndarray:
    """Z축 뒤집기 (머리가 위로 향하게)"""
    k = kpts.copy()
    if flip_z and k.size > 0:
        k[:, 2] = -k[:, 2]  # Z축 뒤집기
    return k

# azure_kinect_3d_visualization_demo.py에서 이 함수를 찾아서 교체하세요

def compute_global_limits_adaptive(frames: List[Dict], axis_order_idx: List[int], flip_z: bool = False) -> Tuple[Tuple[float,float],Tuple[float,float],Tuple[float,float]]:
    """적응적 축 범위 계산 - 유효한 좌표만 사용"""
    xs, ys, zs = [], [], []
    
    for fr in frames:
        persons = fr.get('persons', [])
        if not persons and 'keypoints_3d' in fr:
            persons = [{'keypoints_3d': fr['keypoints_3d']}]
        
        for p in persons:
            k = permute_kpts(p.get('keypoints_3d', []), axis_order_idx)
            k = flip_axis_kpts(k, flip_z)
            if k.size == 0:
                continue
            
            # 0,0,0이 아닌 유효한 좌표만 수집
            valid_mask = ~((k[:, 0] == 0) & (k[:, 1] == 0) & (k[:, 2] == 0))
            if valid_mask.sum() > 0:
                valid_k = k[valid_mask]
                xs.extend(valid_k[:, 0])
                ys.extend(valid_k[:, 1])
                zs.extend(valid_k[:, 2])

    if not xs:
        return (-1, 1), (-1, 1), (-1, 1)

    # 실제 데이터 범위 계산
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    z_min, z_max = min(zs), max(zs)
    
    # 중심점
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2
    
    # 최대 범위 계산
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    
    # 최소 범위 보장 (너무 작으면 1.0으로)
    if max_range < 1.0:
        max_range = 1.0
    
    # 패딩 추가
    padding = max_range * 0.3
    half_range = (max_range + padding) / 2
    
    x_lim = (x_center - half_range, x_center + half_range)
    y_lim = (y_center - half_range, y_center + half_range)
    z_lim = (z_center - half_range, z_center + half_range)
    
    return x_lim, y_lim, z_lim

# -------------------- 그리기 --------------------
# COCO-17 기준 예시 엣지
SKELETON_EDGES = [
    (0,1),(1,2),(2,3),(3,4),
    (1,5),(5,7),(7,9),
    (1,6),(6,8),(8,10),
    (5,6),(5,11),(6,12),
    (11,12),(11,13),(13,15),
    (12,14),(14,16)
]

def draw_person(ax, kpts: np.ndarray, color: str, radius=20, thickness=3):  # 크기 증가
    """개선된 스켈레톤 그리기 - 0,0,0 관절 필터링 포함"""
    if kpts.size == 0:
        return
    
    # 유효한 관절만 필터링 (0,0,0이 아닌 관절)
    valid_joints = ~((kpts[:, 0] == 0) & (kpts[:, 1] == 0) & (kpts[:, 2] == 0))
    
    # 유효한 관절만 점으로 그리기
    if valid_joints.sum() > 0:
        valid_kpts = kpts[valid_joints]
        ax.scatter(valid_kpts[:, 0], valid_kpts[:, 1], valid_kpts[:, 2], 
                  s=radius, c=color, depthshade=True)
    
    # 스켈레톤 연결선 그리기 (양쪽 관절이 모두 유효할 때만)
    for i, j in SKELETON_EDGES:
        if (i < len(kpts) and j < len(kpts) and 
            valid_joints[i] and valid_joints[j]):
            xs = [kpts[i,0], kpts[j,0]]
            ys = [kpts[i,1], kpts[j,1]]
            zs = [kpts[i,2], kpts[j,2]]
            ax.plot(xs, ys, zs, linewidth=thickness, c=color)

def setup_axes(ax, x_lim, y_lim, z_lim, elev, azim, labels=('X','Y','Z'), show_labels=True):
    ax.set_xlim(x_lim); ax.set_ylim(y_lim); ax.set_zlim(z_lim)
    ax.set_box_aspect((x_lim[1]-x_lim[0], y_lim[1]-y_lim[0], z_lim[1]-z_lim[0]))
    ax.view_init(elev=elev, azim=azim)
    if show_labels:
        ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1]); ax.set_zlabel(labels[2])
    else:
        ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
    ax.grid(True, alpha=0.3)

# -------------------- 정렬/아이디 고정 --------------------
def sort_two_by_depth(persons_k: List[np.ndarray], depth_axis: str = 'y', front_is_min=True) -> List[int]:
    axis = {'x':0, 'y':1, 'z':2}[depth_axis.lower()]
    depths = [nanmean_axis(k[:, axis]) for k in persons_k]
    order = np.argsort(depths)
    if not front_is_min:
        order = order[::-1]
    return list(order)

def stable_assign_two(prev_centers: List[np.ndarray], curr_centers: List[np.ndarray]) -> List[int]:
    if len(prev_centers) != 2 or len(curr_centers) != 2:
        return list(range(len(curr_centers)))
    d00 = np.linalg.norm(prev_centers[0] - curr_centers[0])
    d11 = np.linalg.norm(prev_centers[1] - curr_centers[1])
    sum_a = d00 + d11
    d01 = np.linalg.norm(prev_centers[0] - curr_centers[1])
    d10 = np.linalg.norm(prev_centers[1] - curr_centers[0])
    sum_b = d01 + d10
    return [0,1] if sum_a <= sum_b else [1,0]

# -------------------- JSON Normalizer --------------------
def _as_persons_from_frame_dict(frame_dict: Dict[str, Any]) -> Dict[str, Any]:
    persons = frame_dict.get('persons', None)
    if persons is None:
        k3d = frame_dict.get('keypoints_3d', None)
        sc  = frame_dict.get('scores', None)
        if k3d is not None:
            k3d = to_numpy(k3d)
            if k3d.ndim == 3:
                out = []
                for p in range(k3d.shape[0]):
                    item = {'keypoints_3d': k3d[p].tolist()}
                    if sc is not None:
                        s = to_numpy(sc)
                        if s.ndim == 2 and s.shape[0] == k3d.shape[0]:
                            item['scores'] = s[p].tolist()
                    out.append(item)
                persons = out
            elif k3d.ndim == 2:
                persons = [{'keypoints_3d': k3d.tolist()}]
    if persons is None:
        persons = []
    return {
        'frame_idx': frame_dict.get('frame_idx', None),
        'persons': persons
    }

def _normalize_list(lst: List[Any]) -> List[Dict]:
    frames = []
    for i, fr in enumerate(lst):
        if isinstance(fr, dict):
            norm = _as_persons_from_frame_dict(fr)
            if norm['frame_idx'] is None:
                norm['frame_idx'] = i
            frames.append(norm)
        else:
            k = to_numpy(fr)
            if k.ndim == 2 and k.shape[1] == 3:
                frames.append({'frame_idx': i, 'persons':[{'keypoints_3d': k.tolist()}]})
            elif k.ndim == 3 and k.shape[2] == 3:
                persons = [{'keypoints_3d': k[p].tolist()} for p in range(k.shape[0])]
                frames.append({'frame_idx': i, 'persons': persons})
            else:
                raise ValueError('리스트 요소 형식을 해석할 수 없습니다.')
    frames.sort(key=lambda x: x.get('frame_idx', 0))
    return frames

def normalize_frames(raw) -> List[Dict]:
    # list 형태: 바로 정규화
    if isinstance(raw, list):
        return _normalize_list(raw)

    # dict 형태: 여러 키 후보를 검사
    if isinstance(raw, dict):
        for key in ('frames', 'results', 'data', 'predictions', 'items'):
            if key in raw and isinstance(raw[key], list):
                return _normalize_list(raw[key])

        # { "0": {...}, "1": {...}, ... } 같은 매핑
        if all(isinstance(k, (str, int)) for k in raw.keys()):
            frames = []
            try:
                sorted_items = sorted(raw.items(), key=lambda kv: int(kv[0]))
            except Exception:
                sorted_items = sorted(raw.items(), key=lambda kv: str(kv[0]))

            for i, (k, v) in enumerate(sorted_items):
                if isinstance(v, dict):
                    norm = _as_persons_from_frame_dict(v)
                    if norm['frame_idx'] is None:
                        try:
                            norm['frame_idx'] = int(k)
                        except Exception:
                            norm['frame_idx'] = i
                    frames.append(norm)
                else:
                    k3d = ensure_j3(v)
                    if k3d.size > 0:
                        frames.append({'frame_idx': i, 'persons':[{'keypoints_3d': k3d.tolist()}]})
                    else:
                        # 끝내 해석할 수 없는 값은 건너뜀
                        continue
            frames.sort(key=lambda x: x.get('frame_idx', 0))
            return frames

    raise ValueError('알 수 없는 JSON 포맷입니다.')

# -------------------- 메인 --------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='triangulation json')
    ap.add_argument('--output-root', default='./vis3d')
    ap.add_argument('--frame', type=int, default=None, help='단일 프레임 인덱스 (0-based)')
    ap.add_argument('--all-frames', action='store_true')
    ap.add_argument('--save-predictions', action='store_true')
    ap.add_argument('--kpt-thr', type=float, default=0.0)
    ap.add_argument('--elev', type=float, default=25.0)
    ap.add_argument('--azim', type=float, default=-135.0)

    # 좌표축 재배열 (기본: yzx = y→X, z→Y, x→Z)
    ap.add_argument('--axis-order',
                    choices=['xyz','xzy','yxz','yzx','zxy','zyx'],
                    default='yzx',
                    help='그릴 때 (X,Y,Z)로 사용할 원본 축 순서. 기본 yzx(= y→X, z→Y, x→Z).')

    # Z축 뒤집기 옵션 추가
    ap.add_argument('--flip-z', action='store_true',
                    help='Z축을 뒤집어서 머리가 위로 향하게 함')

    # 축 라벨 숨기기 옵션
    ap.add_argument('--hide-axis-labels', action='store_true',
                    help='X, Y, Z 축 라벨을 숨김')

    # 앞/뒤 판단 축(색 고정용)
    ap.add_argument('--depth-axis', choices=['x','y','z'], default='y',
                    help='앞/뒤를 판단할 축(기본: y)')
    ap.add_argument('--front-is-min-depth', action='store_true',
                    help='깊이가 작을수록 카메라에 가깝다고 가정(기본 True)')
    ap.add_argument('--no-front-is-min-depth', dest='front_is_min_depth', action='store_false')
    ap.set_defaults(front_is_min_depth=True)

    args = ap.parse_args()
    if not args.all_frames and args.frame is None:
        args.frame = 0
    return args

def main():
    args = parse_args()

    print(f'Loading 3D poses from: {args.input}')
    raw = load_json(args.input)
    frames = normalize_frames(raw)
    print(f'Loaded {len(frames)} frames')

    ensure_dir(args.output_root)

    # 축 재정렬 인덱스 계산
    order_idx = axis_indices(args.axis_order)

    # 1) 전역 축 범위 계산(바닥 고정) - 재정렬 적용
    x_lim, y_lim, z_lim = compute_global_limits_adaptive(frames, order_idx, args.flip_z)

    # 2) 렌더 대상 프레임
    if args.all_frames:
        indices = list(range(len(frames)))
    else:
        if args.frame < 0 or args.frame >= len(frames):
            raise IndexError(f'frame index {args.frame} is out of range [0, {len(frames)-1}]')
        indices = [args.frame]

    # 3) 색상/아이디 고정용 이전 프레임 중심 저장
    prev_centers = None  # [center_red, center_blue]

    for idx in indices:
        fr = frames[idx]
        persons = fr.get('persons', [])
        if not persons and 'keypoints_3d' in fr:
            persons = [{'keypoints_3d': fr['keypoints_3d']}]

        # numpy 변환 + 스코어 필터 + 축 재정렬
        persons_k = []
        for p in persons:
            k = ensure_j3(p.get('keypoints_3d', []))
            if k.size == 0:
                continue
            if args.kpt_thr > 0 and p.get('scores') is not None:
                s = to_numpy(p['scores'])
                if s.ndim == 1:
                    s = s[:, None]
                m = s >= args.kpt_thr
                k = np.where(m, k, np.nan)
            k = k[:, order_idx]   # 좌표축 재배열
            k = flip_axis_kpts(k, args.flip_z)  # Z축 뒤집기
            persons_k.append(k)

        # 두 사람일 때: 깊이/이전 프레임 기반으로 빨강/파랑 고정
        draw_order = list(range(len(persons_k)))
        if len(persons_k) == 2:
            curr_centers = [person_centroid(k) for k in persons_k]
            if prev_centers is None:
                order = sort_two_by_depth(persons_k, depth_axis=args.depth_axis,
                                          front_is_min=args.front_is_min_depth)
                draw_order = order          # order[0]=앞(RED), order[1]=뒤(BLUE)
                prev_centers = [curr_centers[order[0]], curr_centers[order[1]]]
            else:
                perm = stable_assign_two(prev_centers, curr_centers)
                draw_order = perm           # perm[0]=RED, perm[1]=BLUE
                prev_centers = [curr_centers[perm[0]], curr_centers[perm[1]]]

        # 그림
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        setup_axes(ax, x_lim, y_lim, z_lim, elev=args.elev, azim=args.azim,
                   labels=('X','Y','Z'), show_labels=not args.hide_axis_labels)
        ax.set_title(f'3D Pose (frame {fr.get("frame_idx", idx)}) | persons={len(persons_k)}')

        colors = ['#e41a1c', '#377eb8', '#999999', '#777777']  # 빨강=앞, 파랑=뒤
        if len(persons_k) == 2:
            draw_person(ax, persons_k[draw_order[0]], colors[0], radius=30, thickness=4)
            draw_person(ax, persons_k[draw_order[1]], colors[1], radius=30, thickness=4)
        else:
            for i, k in enumerate(persons_k):
                c = colors[i] if i < len(colors) else '#999999'
                draw_person(ax, k, c, radius=30, thickness=4)

        out_png = os.path.join(args.output_root, f'frame_{idx:04d}.png')
        fig.savefig(out_png, dpi=120, bbox_inches='tight')
        plt.close(fig)

        if args.save_predictions:
            out_json = os.path.join(args.output_root, f'frame_{idx:04d}.json')
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(fr, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
