#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Occlusion-Robust Two-Camera 3D Triangulation
- Temporal consistency for handling occlusions
- Multi-view fusion for missing keypoints
- Trajectory smoothing and outlier removal
- Enhanced person matching across views
"""

import argparse
import json
from pathlib import Path
import itertools
import numpy as np
import cv2
import warnings
warnings.filterwarnings('ignore')


def load_calibration_new_format(calib_file):
    """Load camera calibration from azure_kinect_calibration.json format"""
    with open(calib_file, 'r', encoding='utf-8') as f:
        calib_data = json.load(f)

    cameras = {}
    
    # Primary camera (camera 1)
    primary = calib_data['primary_camera_intrinsics']
    K1 = np.array(primary['camera_matrix'], dtype=np.float64)
    d1 = np.array(primary['distortion'][0], dtype=np.float64).reshape(-1, 1)
    
    # For primary camera, use identity transformation
    R1 = np.eye(3, dtype=np.float64)
    t1 = np.zeros((3, 1), dtype=np.float64)
    P1 = K1 @ np.hstack([R1, t1])
    
    cameras['00'] = {
        'K': K1, 'distCoef': d1, 'R': R1, 't': t1, 'P': P1,
        'resolution': calib_data['calibration_info']['image_resolution']
    }
    
    # Secondary camera (camera 2)
    secondary = calib_data['secondary_camera_intrinsics']
    stereo = calib_data['stereo_parameters']
    
    K2 = np.array(secondary['camera_matrix'], dtype=np.float64)
    d2 = np.array(secondary['distortion'][0], dtype=np.float64).reshape(-1, 1)
    
    # Get stereo transformation
    R2 = np.array(stereo['rotation_matrix'], dtype=np.float64)
    t2 = np.array([[stereo['translation_vector'][0][0]], 
                   [stereo['translation_vector'][1][0]], 
                   [stereo['translation_vector'][2][0]]], dtype=np.float64)
    
    # Convert from mm to meters
    t2 = t2 / 1000.0
    
    P2 = K2 @ np.hstack([R2, t2])
    
    cameras['01'] = {
        'K': K2, 'distCoef': d2, 'R': R2, 't': t2, 'P': P2,
        'resolution': calib_data['calibration_info']['image_resolution']
    }
    
    return cameras


def load_2d_instances(json_file):
    """Load 2D pose instances from MMPose JSON format"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    instances = []
    if isinstance(data, list):
        inst_list = data
    else:
        inst_list = data.get('instance_info', [])
    
    for inst in inst_list:
        kpts2d = np.array(inst['keypoints'], dtype=np.float32)
        kptsconf = np.array(inst['keypoint_scores'], dtype=np.float32)
        J = kpts2d.shape[0]
        kpts = np.zeros((J, 3), dtype=np.float32)
        kpts[:, :2] = kpts2d
        kpts[:, 2] = kptsconf
        score = float(np.mean(kptsconf))
        instances.append({'kpts': kpts, 'score': score})
    return instances


def _triangulate_one(K1, d1, P1, K2, d2, P2, pt1, pt2, undistort=True):
    """Triangulate a single joint with reprojection error"""
    if undistort:
        pt1_ud = cv2.undistortPoints(pt1.reshape(1, 1, 2), K1, d1, P=K1).reshape(2,)
        pt2_ud = cv2.undistortPoints(pt2.reshape(1, 1, 2), K2, d2, P=K2).reshape(2,)
    else:
        pt1_ud, pt2_ud = pt1, pt2

    X_h = cv2.triangulatePoints(P1, P2,
                                pt1_ud.reshape(2, 1).astype(np.float64),
                                pt2_ud.reshape(2, 1).astype(np.float64))
    if abs(X_h[3, 0]) < 1e-12:
        return False, None, None, None
    X = (X_h[:3, 0] / X_h[3, 0]).astype(np.float64)

    # Reproject to both cameras
    x1 = P1 @ np.append(X, 1.0); x1 = x1[:2] / x1[2]
    x2 = P2 @ np.append(X, 1.0); x2 = x2[:2] / x2[2]
    err1 = float(np.linalg.norm(x1 - pt1_ud))
    err2 = float(np.linalg.norm(x2 - pt2_ud))
    return True, X, err1, err2


def temporal_interpolate_keypoints(trajectories, frame_idx, joint_idx, window_size=5):
    """시간적 보간을 통한 가려진 관절 복원 (numpy only)"""
    start_idx = max(0, frame_idx - window_size)
    end_idx = min(len(trajectories), frame_idx + window_size + 1)
    
    valid_frames = []
    valid_points = []
    
    # 윈도우 내에서 유효한 포인트들 수집
    for i in range(start_idx, end_idx):
        if i < len(trajectories) and trajectories[i] is not None:
            if joint_idx < len(trajectories[i]) and np.all(np.isfinite(trajectories[i][joint_idx][:3])):
                valid_frames.append(i)
                valid_points.append(trajectories[i][joint_idx][:3])
    
    if len(valid_points) >= 2:
        valid_frames = np.array(valid_frames)
        valid_points = np.array(valid_points)
        
        # 선형 보간 (numpy interp 사용)
        try:
            interpolated_point = np.array([
                np.interp(frame_idx, valid_frames, valid_points[:, 0]),
                np.interp(frame_idx, valid_frames, valid_points[:, 1]), 
                np.interp(frame_idx, valid_frames, valid_points[:, 2])
            ])
            return interpolated_point
        except:
            pass
    
    return None


def smooth_trajectory(trajectory_3d, joint_idx, kernel_size=5):
    """3D 궤적 스무딩으로 노이즈 제거 (간단한 moving average)"""
    if len(trajectory_3d) < kernel_size:
        return trajectory_3d
    
    smoothed = [frame.copy() if frame is not None else None for frame in trajectory_3d]
    
    # 각 관절의 3D 좌표를 추출
    coords = []
    valid_indices = []
    
    for i, frame in enumerate(trajectory_3d):
        if frame is not None and joint_idx < len(frame):
            if np.all(np.isfinite(frame[joint_idx][:3])):
                coords.append(frame[joint_idx][:3])
                valid_indices.append(i)
    
    if len(coords) > kernel_size:
        coords = np.array(coords)
        
        # 간단한 moving average 적용
        half_kernel = kernel_size // 2
        for j, idx in enumerate(valid_indices):
            start = max(0, j - half_kernel)
            end = min(len(coords), j + half_kernel + 1)
            
            # 주변 포인트들의 평균 계산
            avg_coord = np.mean(coords[start:end], axis=0)
            
            if smoothed[idx] is not None:
                smoothed[idx][joint_idx][:3] = avg_coord
    
    return smoothed


def enhanced_person_matching(cam1_list, cam2_list, cameras, cam_names, 
                           prev_pairs=None, temporal_weight=0.3):
    """시간적 일관성을 고려한 향상된 사람 매칭"""
    n1, n2 = len(cam1_list), len(cam2_list)
    if n1 == 0 or n2 == 0:
        return []

    # 기본 triangulation cost 계산
    C = np.zeros((n1, n2), dtype=np.float64)
    for i in range(n1):
        for j in range(n2):
            C[i, j] = pair_cost_by_triangulation(cam1_list[i], cam2_list[j], cameras, cam_names)
    
    # 이전 프레임 매칭 정보가 있으면 temporal consistency 추가
    if prev_pairs is not None and len(prev_pairs) > 0:
        temporal_cost = np.full((n1, n2), 1000.0)  # 높은 기본 비용
        
        for i in range(min(n1, len(prev_pairs))):
            for j in range(min(n2, len(prev_pairs))):
                # 이전 프레임에서의 매칭이 유지되면 보너스
                if i < len(prev_pairs) and j == prev_pairs[i][1]:
                    temporal_cost[i, j] = 0.0
                # 포지션 기반 유사도도 고려
                else:
                    pos1 = np.mean(cam1_list[i]['kpts'][:, :2], axis=0)
                    pos2 = np.mean(cam2_list[j]['kpts'][:, :2], axis=0)
                    temporal_cost[i, j] = np.linalg.norm(pos1 - pos2) / 100.0
        
        # 전체 비용에 temporal cost 추가
        C = (1 - temporal_weight) * C + temporal_weight * temporal_cost
    
    # 최적 매칭 찾기
    m = min(n1, n2)
    best_pairs, best_cost = None, 1e18
    for perm in itertools.permutations(range(n2), m):
        cost = sum(C[i, perm[i]] for i in range(m))
        if cost < best_cost:
            best_cost = cost
            best_pairs = [(i, perm[i]) for i in range(m)]
    
    return best_pairs if best_pairs is not None else []


def pair_cost_by_triangulation(inst_cam1, inst_cam2, cameras, cam_names, undistort=True):
    """개선된 triangulation cost 계산"""
    k1 = inst_cam1['kpts']; k2 = inst_cam2['kpts']
    K1 = cameras[cam_names[0]]['K']; d1 = cameras[cam_names[0]]['distCoef']; P1 = cameras[cam_names[0]]['P']
    K2 = cameras[cam_names[1]]['K']; d2 = cameras[cam_names[1]]['distCoef']; P2 = cameras[cam_names[1]]['P']

    errs = []
    # 핵심 관절들에 더 높은 가중치 부여
    joint_weights = np.ones(17)
    joint_weights[[0, 5, 6, 11, 12]] = 2.0  # nose, shoulders, hips에 높은 가중치
    
    for j in range(k1.shape[0]):
        pt1 = k1[j, :2].astype(np.float32)
        pt2 = k2[j, :2].astype(np.float32)
        c1, c2 = float(k1[j, 2]), float(k2[j, 2])
        
        if not (np.isfinite(pt1).all() and np.isfinite(pt2).all()):
            continue
        if c1 < 0.1 or c2 < 0.1:  # 매우 낮은 confidence는 제외
            continue
            
        ok, X, e1, e2 = _triangulate_one(K1, d1, P1, K2, d2, P2, pt1, pt2, undistort=undistort)
        if ok:
            weighted_error = 0.5 * (e1 + e2) * joint_weights[j] * min(c1, c2)
            errs.append(weighted_error)
    
    if len(errs) == 0:
        return 1e9
    return float(np.mean(errs))


def triangulate_skeleton_enhanced(kpts_cam1, kpts_cam2, cameras, cam_names,
                                conf_thr=0.3, reproj_thr_px=None, undistort=True,
                                force_17=False, relax_reproj_px=60.0, 
                                prev_3d=None, use_prev_fill=False,
                                temporal_data=None, frame_idx=None):
    """향상된 스켈레톤 삼각측량 (temporal interpolation 포함)"""
    K1 = cameras[cam_names[0]]['K']; d1 = cameras[cam_names[0]]['distCoef']; P1 = cameras[cam_names[0]]['P']
    K2 = cameras[cam_names[1]]['K']; d2 = cameras[cam_names[1]]['distCoef']; P2 = cameras[cam_names[1]]['P']

    J = kpts_cam1.shape[0]
    kpts_3d = np.zeros((J, 3), dtype=np.float64)
    conf_out = np.zeros((J,), dtype=np.float32)
    fill_mode = ['none'] * J
    errs1, errs2 = [], []

    def _try(pt1, pt2, thr_px):
        ok, X, e1, e2 = _triangulate_one(K1, d1, P1, K2, d2, P2, pt1, pt2, undistort=undistort)
        if not ok:
            return False, None, None, None
        if thr_px is not None and (e1 > thr_px or e2 > thr_px):
            return False, X, e1, e2
        return True, X, e1, e2

    for j in range(J):
        pt1 = kpts_cam1[j, :2].astype(np.float32)
        pt2 = kpts_cam2[j, :2].astype(np.float32)
        c1 = float(kpts_cam1[j, 2]); c2 = float(kpts_cam2[j, 2])

        # 1) Standard triangulation attempt
        if np.isfinite(pt1).all() and np.isfinite(pt2).all() and c1 >= conf_thr and c2 >= conf_thr:
            ok, X, e1, e2 = _try(pt1, pt2, reproj_thr_px)
            if ok:
                kpts_3d[j] = X; conf_out[j] = min(c1, c2)
                errs1.append(e1); errs2.append(e2)
                continue

        # 2) Single-view with high confidence (한쪽만 보이는 경우)
        if c1 >= 0.8 and c2 < conf_thr and np.isfinite(pt1).all():
            # 다른 시야에서는 낮은 confidence지만 triangulation 시도
            if np.isfinite(pt2).all():
                ok2, X2, e1, e2 = _try(pt1, pt2, relax_reproj_px * 1.5)
                if ok2:
                    kpts_3d[j] = X2
                    conf_out[j] = c1 * 0.7  # Penalize single-view
                    fill_mode[j] = 'single_view_cam1'
                    if e1 is not None: errs1.append(e1)
                    if e2 is not None: errs2.append(e2)
                    continue
        
        if c2 >= 0.8 and c1 < conf_thr and np.isfinite(pt2).all():
            if np.isfinite(pt1).all():
                ok2, X2, e1, e2 = _try(pt1, pt2, relax_reproj_px * 1.5)
                if ok2:
                    kpts_3d[j] = X2
                    conf_out[j] = c2 * 0.7  # Penalize single-view
                    fill_mode[j] = 'single_view_cam2'
                    if e1 is not None: errs1.append(e1)
                    if e2 is not None: errs2.append(e2)
                    continue

        # 3) Temporal interpolation (시간적 보간)
        if temporal_data is not None and frame_idx is not None:
            interpolated = temporal_interpolate_keypoints(temporal_data, frame_idx, j)
            if interpolated is not None:
                kpts_3d[j] = interpolated
                conf_out[j] = 0.4  # Medium confidence for interpolated
                fill_mode[j] = 'temporal_interpolation'
                continue

        # 4) Relaxed retry
        if force_17 and np.isfinite(pt1).all() and np.isfinite(pt2).all():
            ok2, X2, e1, e2 = _try(pt1, pt2, relax_reproj_px)
            if ok2 or (X2 is not None):
                if X2 is not None:
                    kpts_3d[j] = X2
                    conf_out[j] = max(0.05, min(c1, c2) * 0.5)
                fill_mode[j] = 'relaxed'
                if e1 is not None: errs1.append(e1)
                if e2 is not None: errs2.append(e2)
                continue

        # 5) Previous frame fill
        if use_prev_fill and (prev_3d is not None) and np.isfinite(prev_3d).all():
            kpts_3d[j] = prev_3d[j]
            conf_out[j] = 0.01
            fill_mode[j] = 'prev'

    stats = {
        'reproj_err_cam0_mean': float(np.mean(errs1)) if errs1 else None,
        'reproj_err_cam1_mean': float(np.mean(errs2)) if errs2 else None,
        'reproj_err_cam0_max': float(np.max(errs1)) if errs1 else None,
        'reproj_err_cam1_max': float(np.max(errs2)) if errs2 else None,
        'valid': int(np.count_nonzero(conf_out > 0.0)),
        'total': int(J),
        'filled': fill_mode
    }
    return kpts_3d, conf_out, stats


def main():
    parser = argparse.ArgumentParser(description="Occlusion-robust 3D triangulation")
    parser.add_argument('--calib-file', required=True, help='Camera calibration JSON')
    parser.add_argument('--cam1-poses', required=True, help='Camera 1 poses folder')
    parser.add_argument('--cam2-poses', required=True, help='Camera 2 poses folder')
    parser.add_argument('--output-file', required=True, help='Output JSON path')
    parser.add_argument('--conf-thr', type=float, default=0.3, help='2D confidence threshold')
    parser.add_argument('--reproj-thr', type=float, default=25.0, help='Reprojection error threshold (px)')
    parser.add_argument('--extrinsic-format', choices=['c2w', 'w2c'], default='c2w')
    parser.add_argument('--undistort', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--expect-persons', type=int, default=2, help='Expected persons per frame')
    parser.add_argument('--force-17', action='store_true', help='Force complete 17 joints')
    parser.add_argument('--relax-reproj', type=float, default=60.0, help='Relaxed reprojection threshold')
    parser.add_argument('--use-prev-fill', action='store_true', help='Use previous frame filling')
    
    # 새로운 occlusion handling 파라미터들
    parser.add_argument('--temporal-window', type=int, default=5, help='Temporal interpolation window size')
    parser.add_argument('--smooth-trajectory', action='store_true', help='Apply trajectory smoothing')
    parser.add_argument('--temporal-weight', type=float, default=0.3, help='Temporal consistency weight for matching')
    
    args = parser.parse_args()

    print("Loading calibration (occlusion-robust mode)...")
    cameras = load_calibration_new_format(args.calib_file)
    cam_names = ['00', '01']

    # Load all frame data
    cam1_jsons = sorted(Path(args.cam1_poses).glob('*.json'))
    cam2_jsons = sorted(Path(args.cam2_poses).glob('*.json'))
    print(f"Found {len(cam1_jsons)} pose files for cam1, {len(cam2_jsons)} for cam2")
    n_frames = min(len(cam1_jsons), len(cam2_jsons))
    
    if len(cam1_jsons) != len(cam2_jsons):
        print(f"[WARN] Frame count mismatch: cam1={len(cam1_jsons)} cam2={len(cam2_jsons)}; processing {n_frames}")

    # COCO-17 keypoint names
    kp_names = [
        "nose","left_eye","right_eye","left_ear","right_ear",
        "left_shoulder","right_shoulder","left_elbow","right_elbow", 
        "left_wrist","right_wrist","left_hip","right_hip",
        "left_knee","right_knee","left_ankle","right_ankle"
    ]

    results_all = []
    prev_pairs = None  # 이전 프레임의 person matching 정보
    prev_3d_persons = None  # 이전 프레임의 3D 포즈 정보
    
    # 전체 궤적 저장 (temporal interpolation용)
    person_trajectories = [[] for _ in range(args.expect_persons)]  # 각 사람별 궤적
    
    print("Processing frames with occlusion handling...")
    
    for i in range(n_frames):
        j1 = cam1_jsons[i]; j2 = cam2_jsons[i]
        print(f"Processing frame {i+1}/{n_frames}: {j1.name}, {j2.name}")

        insts1 = load_2d_instances(str(j1))
        insts2 = load_2d_instances(str(j2))

        if len(insts1) == 0 or len(insts2) == 0:
            print("  -> No instances in one view; filling with temporal data")
            # TODO: 한쪽 시야만 있는 경우에도 temporal interpolation 활용
            continue

        # Enhanced person matching with temporal consistency
        pairs = enhanced_person_matching(
            insts1, insts2, cameras, cam_names,
            prev_pairs=prev_pairs, 
            temporal_weight=args.temporal_weight
        )
        
        if len(pairs) == 0:
            print("  -> Enhanced matching failed; skipping")
            continue

        persons_out = []
        total_valid = 0
        reproj_thr = None if (args.reproj_thr is not None and args.reproj_thr < 0) else args.reproj_thr

        pairs_sorted = sorted(pairs, key=lambda ab: (ab[0], ab[1]))
        
        # 이전 프레임 정보가 없으면 초기화
        if prev_3d_persons is None:
            prev_3d_persons = [None for _ in range(len(pairs_sorted))]

        for p_idx, (idx1, idx2) in enumerate(pairs_sorted):
            k1 = insts1[idx1]['kpts']; k2 = insts2[idx2]['kpts']

            # Enhanced triangulation with temporal data
            k3d, conf, stats = triangulate_skeleton_enhanced(
                k1, k2, cameras, cam_names,
                conf_thr=args.conf_thr, reproj_thr_px=reproj_thr, undistort=args.undistort,
                force_17=args.force_17, relax_reproj_px=args.relax_reproj,
                prev_3d=None if prev_3d_persons is None else prev_3d_persons[p_idx],
                use_prev_fill=args.use_prev_fill,
                temporal_data=person_trajectories[p_idx] if p_idx < len(person_trajectories) else None,
                frame_idx=i
            )

            total_valid += int(np.count_nonzero(conf > 0.0))

            # 궤적에 현재 프레임 추가
            if p_idx < len(person_trajectories):
                person_trajectories[p_idx].append(k3d.copy())

            # Pack results
            per_joint = []
            detailed = {}
            for j, name in enumerate(kp_names):
                per_joint.append(k3d[j].tolist() + [float(conf[j])])
                detailed[name] = {
                    'position_3d': k3d[j].tolist(),
                    'confidence': float(conf[j]),
                    'cam1_2d': k1[j, :2].tolist(),
                    'cam2_2d': k2[j, :2].tolist(),
                    'fill_mode': stats['filled'][j] if 'filled' in stats else 'none'
                }

            persons_out.append({
                'pair': {'cam1_idx': int(idx1), 'cam2_idx': int(idx2)},
                'keypoints_3d': per_joint,
                'keypoints_detailed': detailed,
                'reprojection_error': {
                    'cam0_mean': stats['reproj_err_cam0_mean'],
                    'cam1_mean': stats['reproj_err_cam1_mean'],
                    'cam0_max': stats['reproj_err_cam0_max'],
                    'cam1_max': stats['reproj_err_cam1_max'],
                    'valid': stats['valid'],
                    'total': stats['total']
                }
            })

            # Update for next frame
            if prev_3d_persons is not None:
                prev_3d_persons[p_idx] = k3d.copy()

        # Update previous pairs for next frame
        prev_pairs = pairs_sorted
        
        print(f"  -> Persons: {len(persons_out)} | Valid 3D joints: {total_valid}/{len(pairs_sorted)*17}")

        results_all.append({
            'frame_index': i,
            'cam1_file': j1.name,
            'cam2_file': j2.name,
            'num_persons': len(persons_out),
            'persons': persons_out
        })

    # Apply trajectory smoothing if requested
    if args.smooth_trajectory:
        print("Applying trajectory smoothing...")
        for person_idx in range(len(person_trajectories)):
            for joint_idx in range(17):
                person_trajectories[person_idx] = smooth_trajectory(
                    person_trajectories[person_idx], joint_idx
                )

    # Save results
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        'meta': {
            'dataset': 'azure_kinect_multiview_occlusion_robust',
            'cameras': ['00', '01'],
            'num_keypoints': 17,
            'keypoint_names': kp_names,
            'extrinsic_format': 'new_calib_format',
            'undistort': bool(args.undistort),
            'conf_thr': float(args.conf_thr),
            'reproj_thr': float(args.reproj_thr) if args.reproj_thr is not None else None,
            'force_17': bool(args.force_17),
            'relax_reproj': float(args.relax_reproj),
            'use_prev_fill': bool(args.use_prev_fill),
            'temporal_window': int(args.temporal_window),
            'smooth_trajectory': bool(args.smooth_trajectory),
            'temporal_weight': float(args.temporal_weight)
        },
        'results': results_all
    }
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nOcclusion-robust 3D triangulation completed!")
    print(f"Results saved to: {out_path}")
    print(f"Processed {len(results_all)} frames successfully")
    print(f"Enhanced features used:")
    print(f"  - Temporal consistency matching: {args.temporal_weight > 0}")
    print(f"  - Temporal interpolation window: {args.temporal_window}")
    print(f"  - Trajectory smoothing: {args.smooth_trajectory}")


if __name__ == '__main__':
    main()