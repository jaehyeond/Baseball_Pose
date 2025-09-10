# demo/azure_kinect_3d_visualization_demo.py
import os
import json
import argparse
import warnings
from typing import List, Tuple, Dict, Any

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 이미지 파일 저장용
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ---------- 유틸 ----------

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
    """kpts: (J,3) -> (3,) 평균 좌표(결측값 NaN은 무시)"""
    return nanmean_axis(kpts, axis=0)

def compute_global_limits(frames: List[Dict], pad: float = 0.05) -> Tuple[Tuple[float,float],Tuple[float,float],Tuple[float,float]]:
    """모든 프레임 모든 사람의 좌표를 모아 전역 축 범위를 계산"""
    xs, ys, zs = [], [], []
    for fr in frames:
        persons = fr.get('persons', [])
        if not persons and 'keypoints_3d' in fr:
            persons = [{'keypoints_3d': fr['keypoints_3d']}]
        for p in persons:
            k = to_numpy(p['keypoints_3d'])  # (J,3)
            if k.size == 0:
                continue
            xs.append(k[:, 0]); ys.append(k[:, 1]); zs.append(k[:, 2])

    if not xs:
        # 좌표가 하나도 없으면 기본 범위 제공
        return (-1, 1), (-1, 1), (-1, 1)

    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    z_min, z_max = float(np.min(zs)), float(np.max(zs))

    xr = x_max - x_min; yr = y_max - y_min; zr = z_max - z_min
    x_lim = (x_min - xr * pad, x_max + xr * pad)
    y_lim = (y_min - yr * pad, y_max + yr * pad)
    z_lim = (z_min - zr * pad, z_max + zr * pad)
    return x_lim, y_lim, z_lim


# ---------- 그리기 ----------

# COCO-17을 가정한 예시 엣지(필요시 바꾸세요)
SKELETON_EDGES = [
    (0,1),(1,2),(2,3),(3,4),
    (1,5),(5,7),(7,9),
    (1,6),(6,8),(8,10),
    (5,6),(5,11),(6,12),
    (11,12),(11,13),(13,15),
    (12,14),(14,16)
]

def draw_person(ax, kpts: np.ndarray, color: str, radius=10, thickness=2):
    ax.scatter(kpts[:, 0], kpts[:, 1], kpts[:, 2], s=radius, c=color, depthshade=True)
    for i, j in SKELETON_EDGES:
        if i < len(kpts) and j < len(kpts):
            xs = [kpts[i,0], kpts[j,0]]
            ys = [kpts[i,1], kpts[j,1]]
            zs = [kpts[i,2], kpts[j,2]]
            ax.plot(xs, ys, zs, linewidth=thickness, c=color)

def setup_axes(ax, x_lim, y_lim, z_lim, elev, azim):
    ax.set_xlim(x_lim); ax.set_ylim(y_lim); ax.set_zlim(z_lim)
    ax.set_box_aspect((x_lim[1]-x_lim[0], y_lim[1]-y_lim[0], z_lim[1]-z_lim[0]))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.grid(True, alpha=0.3)


# ---------- 정렬/아이디 고정 ----------

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


# ---------- JSON Normalizer (관대한 로더) ----------

def _as_persons_from_frame_dict(frame_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    frame_dict 안에서 persons 리스트를 표준화해 반환.
    - persons가 이미 있으면 그대로 사용
    - 없고 keypoints_3d가 (P,J,3)라면 persons로 분해
    - 없고 keypoints_3d가 (J,3)라면 1명으로 간주
    """
    persons = frame_dict.get('persons', None)
    if persons is None:
        k3d = frame_dict.get('keypoints_3d', None)
        sc  = frame_dict.get('scores', None)
        if k3d is not None:
            k3d = to_numpy(k3d)
            if k3d.ndim == 3:      # (P,J,3)
                out = []
                for p in range(k3d.shape[0]):
                    item = {'keypoints_3d': k3d[p].tolist()}
                    if sc is not None:
                        s = to_numpy(sc)
                        if s.ndim == 2 and s.shape[0] == k3d.shape[0]:
                            item['scores'] = s[p].tolist()
                    out.append(item)
                persons = out
            elif k3d.ndim == 2:    # (J,3)
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
            # 예상 밖 포맷: 한 프레임에 한 사람(k3d)만 들어있는 경우 등
            k = to_numpy(fr)
            if k.ndim == 2 and k.shape[1] == 3:
                frames.append({'frame_idx': i, 'persons':[{'keypoints_3d': k.tolist()}]})
            else:
                raise ValueError('리스트 요소 형식을 해석할 수 없습니다.')
    frames.sort(key=lambda x: x.get('frame_idx', 0))
    return frames

def normalize_frames(raw) -> List[Dict]:
    """다양한 포맷을 허용하여 {frame_idx, persons:[{keypoints_3d, scores}]} 리스트로 정규화"""
    # 1) 최상위가 리스트
    if isinstance(raw, list):
        return _normalize_list(raw)

    # 2) 최상위가 딕셔너리
    if isinstance(raw, dict):
        # 흔한 키들 시도
        for key in ('frames', 'results', 'data', 'predictions', 'items'):
            if key in raw and isinstance(raw[key], list):
                return _normalize_list(raw[key])

        # 프레임 인덱스 -> 프레임 딕셔너리 (예: {"0": {...}, "1": {...}})
        if all(isinstance(k, (str,int)) for k in raw.keys()):
            frames = []
            # 키를 정수로 정렬 가능하면 정수로, 아니면 문자열로
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
                    # 값이 바로 (J,3) 혹은 (P,J,3)인 경우
                    k3d = to_numpy(v)
                    if k3d.ndim == 2 and k3d.shape[1] == 3:
                        frames.append({'frame_idx': i, 'persons':[{'keypoints_3d': k3d.tolist()}]})
                    elif k3d.ndim == 3 and k3d.shape[2] == 3:
                        persons = [{'keypoints_3d': k3d[p].tolist()} for p in range(k3d.shape[0])]
                        frames.append({'frame_idx': i, 'persons': persons})
                    else:
                        raise ValueError('프레임 매핑 값 형식을 해석할 수 없습니다.')
            frames.sort(key=lambda x: x.get('frame_idx', 0))
            return frames

    # 전부 실패
    raise ValueError('알 수 없는 JSON 포맷입니다.')


# ---------- 메인 ----------

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
    ap.add_argument('--depth-axis', choices=['x','y','z'], default='y',
                    help='앞/뒤를 판단할 축(기본: y)')
    ap.add_argument('--front-is-min-depth', action='store_true',
                    help='깊이값이 작은 쪽이 카메라에 더 가깝다고 가정(기본 True)')
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

    # 1) 전역 축 범위 계산(바닥 고정)
    x_lim, y_lim, z_lim = compute_global_limits(frames, pad=0.05)

    # 2) 렌더 대상 프레임 인덱스
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

        # numpy 변환 & 스코어 필터
        persons_k = []
        for p in persons:
            k = to_numpy(p['keypoints_3d'])
            if args.kpt_thr > 0 and p.get('scores') is not None:
                s = to_numpy(p['scores'])
                m = s >= args.kpt_thr
                k = np.where(m[:, None], k, np.nan)
            persons_k.append(k)

        draw_order = list(range(len(persons_k)))
        if len(persons_k) == 2:
            curr_centers = [person_centroid(k) for k in persons_k]
            if prev_centers is None:
                order = sort_two_by_depth(persons_k, depth_axis=args.depth_axis,
                                          front_is_min=args.front_is_min_depth)
                draw_order = order  # order[0]=앞(RED), order[1]=뒤(BLUE)
                prev_centers = [curr_centers[order[0]], curr_centers[order[1]]]
            else:
                perm = stable_assign_two(prev_centers, curr_centers)
                draw_order = perm   # perm[0]=RED, perm[1]=BLUE
                prev_centers = [curr_centers[perm[0]], curr_centers[perm[1]]]

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        setup_axes(ax, x_lim, y_lim, z_lim, elev=args.elev, azim=args.azim)
        ax.set_title(f'3D Pose (frame {fr.get("frame_idx", idx)}) | persons={len(persons_k)}')

        colors = ['#e41a1c', '#377eb8', '#999999', '#777777']
        if len(persons_k) == 2:
            draw_person(ax, persons_k[draw_order[0]], colors[0])
            draw_person(ax, persons_k[draw_order[1]], colors[1])
        else:
            for i, k in enumerate(persons_k):
                c = colors[i] if i < len(colors) else '#999999'
                draw_person(ax, k, c)

        out_png = os.path.join(args.output_root, f'frame_{idx:04d}.png')
        fig.savefig(out_png, dpi=120, bbox_inches='tight')
        plt.close(fig)

        if args.save_predictions:
            out_json = os.path.join(args.output_root, f'frame_{idx:04d}.json')
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(fr, f, ensure_ascii=False, indent=2)

    try:
        import imageio
        import imageio.v2 as iio
        pngs = [os.path.join(args.output_root, f) for f in sorted(os.listdir(args.output_root)) if f.endswith('.png')]
        if len(pngs) > 1:
            mp4_path = os.path.join(args.output_root, 'vis3d.mp4')
            writer = iio.get_writer(mp4_path, fps=10)
            for p in pngs:
                writer.append_data(iio.imread(p))
            writer.close()
    except Exception:
        warnings.warn("imageio가 없어 MP4를 만들 수 없습니다. 'pip install imageio imageio-ffmpeg' 후 재시도하세요.")

if __name__ == '__main__':
    main()
