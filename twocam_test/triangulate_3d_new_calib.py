#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-Camera 3D Triangulation (Multi-person, supports Force-17) - New Calibration Format
- Cross-view matching by minimizing triangulation reprojection error
- Per-person 17-joint triangulation with thresholds
- Optional force-complete of all 17 joints using relaxed retry / prev-frame fill
- Saves per-frame, per-person results
- Updated to work with azure_kinect_calibration.json format

Usage (example):
python triangulate_3d_new_calib.py ^
  --calib-file D:\mmpose\calibration\azure_kinect_calibration.json ^
  --cam1-poses D:\mmpose\results_2d_camera1 ^
  --cam2-poses D:\mmpose\results_2d_camera2 ^
  --output-file D:\mmpose\output_3d\azure_kinect_3d_results_new.json ^
  --conf-thr 0.3 --reproj-thr 25.0 --extrinsic-format c2w --force-17 --relax-reproj 60.0 --use-prev-fill
"""

import argparse
import json
from pathlib import Path
import itertools
import numpy as np
import cv2


# ----------------------------
# Calibration / I/O
# ----------------------------
def load_calibration_new_format(calib_file):
    """Load camera intrinsics/extrinsics from new azure_kinect_calibration.json format
       and build projection matrices P=K[R|t].
    """
    with open(calib_file, 'r', encoding='utf-8') as f:
        calib_data = json.load(f)

    cameras = {}
    
    # Primary camera (camera 1)
    primary = calib_data['primary_camera_intrinsics']
    K1 = np.array(primary['camera_matrix'], dtype=np.float64)
    d1 = np.array(primary['distortion'][0], dtype=np.float64).reshape(-1, 1)
    
    # For primary camera, use identity transformation (world origin at primary camera)
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
    
    # Get stereo transformation (secondary camera relative to primary)
    R2 = np.array(stereo['rotation_matrix'], dtype=np.float64)
    t2 = np.array([[stereo['translation_vector'][0][0]], 
                   [stereo['translation_vector'][1][0]], 
                   [stereo['translation_vector'][2][0]]], dtype=np.float64)
    
    # Convert from mm to meters for consistency
    t2 = t2 / 1000.0
    
    P2 = K2 @ np.hstack([R2, t2])
    
    cameras['01'] = {
        'K': K2, 'distCoef': d2, 'R': R2, 't': t2, 'P': P2,
        'resolution': calib_data['calibration_info']['image_resolution']
    }
    
    return cameras


def load_2d_instances(json_file):
    """
    Load all person instances from a MMPose pose JSON.
    Expected structure (per frame): List of instances
      [
        {"keypoints": [[x,y], ... 17], "keypoint_scores": [c, ... 17]},
        ...
      ]
    Returns: list of dicts with 'kpts' (J,3) and 'score' (float)
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    instances = []
    
    # Handle both formats: list directly or dict with instance_info
    if isinstance(data, list):
        inst_list = data
    else:
        inst_list = data.get('instance_info', [])
    
    for inst in inst_list:
        kpts2d = np.array(inst['keypoints'], dtype=np.float32)        # (J,2)
        kptsconf = np.array(inst['keypoint_scores'], dtype=np.float32) # (J,)
        J = kpts2d.shape[0]
        kpts = np.zeros((J, 3), dtype=np.float32)
        kpts[:, :2] = kpts2d
        kpts[:, 2] = kptsconf
        score = float(np.mean(kptsconf))
        instances.append({'kpts': kpts, 'score': score})
    return instances


# ----------------------------
# Core triangulation utilities
# ----------------------------
def _triangulate_one(K1, d1, P1, K2, d2, P2, pt1, pt2, undistort=True):
    """Triangulate a single joint and compute reprojection errors (px) in both cameras."""
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

    # Reproject to both cams (pixel domain)
    x1 = P1 @ np.append(X, 1.0); x1 = x1[:2] / x1[2]
    x2 = P2 @ np.append(X, 1.0); x2 = x2[:2] / x2[2]
    err1 = float(np.linalg.norm(x1 - pt1_ud))
    err2 = float(np.linalg.norm(x2 - pt2_ud))
    return True, X, err1, err2


def triangulate_skeleton(kpts_cam1, kpts_cam2, cameras, cam_names,
                         conf_thr=0.3, reproj_thr_px=None, undistort=True,
                         force_17=False, relax_reproj_px=60.0, prev_3d=None, use_prev_fill=False):
    """
    Triangulate full skeleton (J joints) for one matched person pair.

    Returns:
      kpts_3d: (J,3) float64
      conf_out: (J,) float32 (0 means invalid/filled)
      stats: dict with mean/max reprojection errors and 'filled' list (per joint mode)
    """
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
        c1  = float(kpts_cam1[j, 2]); c2 = float(kpts_cam2[j, 2])

        # 1) Standard attempt
        if np.isfinite(pt1).all() and np.isfinite(pt2).all() and c1 >= conf_thr and c2 >= conf_thr:
            ok, X, e1, e2 = _try(pt1, pt2, reproj_thr_px)
            if ok:
                kpts_3d[j] = X; conf_out[j] = min(c1, c2)
                errs1.append(e1); errs2.append(e2)
                continue

        if not force_17:
            # Not forcing — leave as invalid (0-conf)
            continue

        # 2) Relaxed retry (ignore conf, relax reprojection threshold)
        if np.isfinite(pt1).all() and np.isfinite(pt2).all():
            ok2, X2, e1, e2 = _try(pt1, pt2, relax_reproj_px)
            if ok2 or (X2 is not None):
                if X2 is not None:
                    kpts_3d[j] = X2
                    conf_out[j] = max(0.05, min(c1, c2) * 0.5)  # mark as low confidence
                fill_mode[j] = 'relaxed'
                if e1 is not None: errs1.append(e1)
                if e2 is not None: errs2.append(e2)
                continue

        # 3) Prev-frame fill (if allowed)
        if use_prev_fill and (prev_3d is not None) and np.isfinite(prev_3d).all():
            kpts_3d[j] = prev_3d[j]
            conf_out[j] = 0.01
            fill_mode[j] = 'prev'
            continue
        # else: remains invalid (0)

    stats = {
        'reproj_err_cam0_mean': float(np.mean(errs1)) if errs1 else None,
        'reproj_err_cam1_mean': float(np.mean(errs2)) if errs2 else None,
        'reproj_err_cam0_max':  float(np.max(errs1))  if errs1 else None,
        'reproj_err_cam1_max':  float(np.max(errs2))  if errs2 else None,
        'valid': int(np.count_nonzero(conf_out > 0.0)),
        'total': int(J),
        'filled': fill_mode
    }
    return kpts_3d, conf_out, stats


def pair_cost_by_triangulation(inst_cam1, inst_cam2, cameras, cam_names, undistort=True):
    """
    Matching cost between a cam1 person and a cam2 person:
    - Triangulate all possible joints (no thresholds)
    - Use mean reprojection error (px) across valid joints as cost
    """
    k1 = inst_cam1['kpts']; k2 = inst_cam2['kpts']
    K1 = cameras[cam_names[0]]['K']; d1 = cameras[cam_names[0]]['distCoef']; P1 = cameras[cam_names[0]]['P']
    K2 = cameras[cam_names[1]]['K']; d2 = cameras[cam_names[1]]['distCoef']; P2 = cameras[cam_names[1]]['P']

    errs = []
    for j in range(k1.shape[0]):
        pt1 = k1[j, :2].astype(np.float32)
        pt2 = k2[j, :2].astype(np.float32)
        if not (np.isfinite(pt1).all() and np.isfinite(pt2).all()):
            continue
        ok, X, e1, e2 = _triangulate_one(K1, d1, P1, K2, d2, P2, pt1, pt2, undistort=undistort)
        if ok:
            errs.append(0.5 * (e1 + e2))
    if len(errs) == 0:
        return 1e9  # huge cost if nothing valid
    return float(np.mean(errs))


def match_instances(cam1_list, cam2_list, cameras, cam_names, undistort=True):
    """
    One-to-one matching by minimizing triangulation reprojection cost.
    For small N, exhaustive permutation over cam2 indices is used.
    Returns: list of (idx_cam1, idx_cam2)
    """
    n1, n2 = len(cam1_list), len(cam2_list)
    if n1 == 0 or n2 == 0:
        return []

    # Build cost matrix
    C = np.zeros((n1, n2), dtype=np.float64)
    for i in range(n1):
        for j in range(n2):
            C[i, j] = pair_cost_by_triangulation(cam1_list[i], cam2_list[j], cameras, cam_names, undistort=undistort)

    m = min(n1, n2)
    best_pairs, best_cost = None, 1e18
    for perm in itertools.permutations(range(n2), m):
        cost = 0.0
        for i in range(m):
            cost += C[i, perm[i]]
        if cost < best_cost:
            best_cost = cost
            best_pairs = [(i, perm[i]) for i in range(m)]
    return best_pairs if best_pairs is not None else []


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calib-file', required=True, help='New format calibration JSON')
    parser.add_argument('--cam1-poses', required=True, help='Folder of results_*.json for camera 1')
    parser.add_argument('--cam2-poses', required=True, help='Folder of results_*.json for camera 2')
    parser.add_argument('--output-file', required=True, help='Output JSON path')
    parser.add_argument('--conf-thr', type=float, default=0.3, help='2D keypoint confidence threshold')
    parser.add_argument('--reproj-thr', type=float, default=25.0,
                        help='Reprojection error threshold (px); negative to disable')
    parser.add_argument('--extrinsic-format', choices=['c2w', 'w2c'], default='c2w',
                        help='Extrinsic convention in calibration JSON (ignored for new format)')
    # BooleanOptionalAction is available in Python 3.9+ (your env is 3.9)
    parser.add_argument('--undistort', action=argparse.BooleanOptionalAction, default=True,
                        help='Apply undistortion (use --no-undistort to disable)')
    parser.add_argument('--expect-persons', type=int, default=2, help='Expected persons per frame (for logging only)')

    # Force-17 options
    parser.add_argument('--force-17', action='store_true',
                        help='Always output 17 joints per person using relaxed retry / prev fill')
    parser.add_argument('--relax-reproj', type=float, default=60.0,
                        help='Fallback reprojection threshold (px) used in force-17 mode')
    parser.add_argument('--use-prev-fill', action='store_true',
                        help='If relaxed retry fails, copy previous frame 3D joint')

    args = parser.parse_args()

    # Load calibration
    print("Loading calibration (new format)...")
    cameras = load_calibration_new_format(args.calib_file)
    cam_names = ['00', '01']

    # Gather frames
    cam1_jsons = sorted(Path(args.cam1_poses).glob('*.json'))
    cam2_jsons = sorted(Path(args.cam2_poses).glob('*.json'))
    print(f"Found {len(cam1_jsons)} pose files for cam1, {len(cam2_jsons)} for cam2")
    n_frames = min(len(cam1_jsons), len(cam2_jsons))
    if len(cam1_jsons) != len(cam2_jsons):
        print(f"[WARN] frame count mismatch: cam1={len(cam1_jsons)} cam2={len(cam2_jsons)}; processing {n_frames} frames.")

    # COCO-17 keypoint names
    kp_names = [
        "nose","left_eye","right_eye","left_ear","right_ear",
        "left_shoulder","right_shoulder","left_elbow","right_elbow",
        "left_wrist","right_wrist","left_hip","right_hip",
        "left_knee","right_knee","left_ankle","right_ankle"
    ]

    results_all = []
    # For prev-frame fill per person slot (kept in matched order each frame)
    prev_3d_persons = None  # will be list of np.ndarray(J,3)

    for i in range(n_frames):
        j1 = cam1_jsons[i]; j2 = cam2_jsons[i]
        print(f"Processing frame {i+1}/{n_frames}: {j1.name}, {j2.name}")

        insts1 = load_2d_instances(str(j1))
        insts2 = load_2d_instances(str(j2))

        if len(insts1) == 0 or len(insts2) == 0:
            print("  -> No instances in one of the views; skipping")
            continue

        if args.expect_persons and (len(insts1) != args.expect_persons or len(insts2) != args.expect_persons):
            print(f"  [WARN] #persons mismatch: cam1={len(insts1)} cam2={len(insts2)} (expected {args.expect_persons})")

        # Cross-view matching
        pairs = match_instances(insts1, insts2, cameras, cam_names, undistort=args.undistort)
        if len(pairs) == 0:
            print("  -> Matching failed; skipping")
            continue

        # prepare prev-person list
        if prev_3d_persons is None:
            prev_3d_persons = [None for _ in range(len(pairs))]

        persons_out = []
        total_valid = 0
        reproj_thr = None if (args.reproj_thr is not None and args.reproj_thr < 0) else args.reproj_thr

        # (optional) keep a deterministic pair order for prev_3d mapping:
        pairs_sorted = sorted(pairs, key=lambda ab: (ab[0], ab[1]))

        for p_idx, (idx1, idx2) in enumerate(pairs_sorted):
            k1 = insts1[idx1]['kpts']; k2 = insts2[idx2]['kpts']

            k3d, conf, stats = triangulate_skeleton(
                k1, k2, cameras, cam_names,
                conf_thr=args.conf_thr, reproj_thr_px=reproj_thr, undistort=args.undistort,
                force_17=args.force_17, relax_reproj_px=args.relax_reproj,
                prev_3d=None if prev_3d_persons is None else prev_3d_persons[p_idx],
                use_prev_fill=args.use_prev_fill
            )

            total_valid += int(np.count_nonzero(conf > 0.0))

            # pack joint-wise details
            per_joint = []
            detailed = {}
            for j, name in enumerate(kp_names):
                per_joint.append(k3d[j].tolist() + [float(conf[j])])
                detailed[name] = {
                    'position_3d': k3d[j].tolist(),
                    'confidence': float(conf[j]),
                    'cam1_2d': k1[j, :2].tolist(),
                    'cam2_2d': k2[j, :2].tolist()
                }
            # include fill_mode if present
            if 'filled' in stats:
                for j, name in enumerate(kp_names):
                    detailed[name]['fill_mode'] = stats['filled'][j]

            persons_out.append({
                'pair': {'cam1_idx': int(idx1), 'cam2_idx': int(idx2)},
                'keypoints_3d': per_joint,
                'keypoints_detailed': detailed,
                'reprojection_error': {
                    'cam0_mean': stats['reproj_err_cam0_mean'],
                    'cam1_mean': stats['reproj_err_cam1_mean'],
                    'cam0_max':  stats['reproj_err_cam0_max'],
                    'cam1_max':  stats['reproj_err_cam1_max'],
                    'valid':     stats['valid'],
                    'total':     stats['total']
                }
            })

            # store for next frame fill
            if prev_3d_persons is not None:
                prev_3d_persons[p_idx] = k3d.copy()

        print(f"  -> Persons: {len(persons_out)} | Valid 3D joints total: {total_valid}/{len(pairs_sorted)*len(kp_names)}")

        results_all.append({
            'frame_index': i,
            'cam1_file': j1.name,
            'cam2_file': j2.name,
            'num_persons': len(persons_out),
            'persons': persons_out
        })

    # Save JSON
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        'meta': {
            'dataset': 'azure_kinect_multiview_new_calib',
            'cameras': ['00', '01'],
            'num_keypoints': 17,
            'keypoint_names': [
                "nose","left_eye","right_eye","left_ear","right_ear",
                "left_shoulder","right_shoulder","left_elbow","right_elbow",
                "left_wrist","right_wrist","left_hip","right_hip",
                "left_knee","right_knee","left_ankle","right_ankle"
            ],
            'extrinsic_format': 'new_calib_format',
            'undistort': bool(args.undistort),
            'conf_thr': float(args.conf_thr),
            'reproj_thr': float(args.reproj_thr) if args.reproj_thr is not None else None,
            'force_17': bool(args.force_17),
            'relax_reproj': float(args.relax_reproj),
            'use_prev_fill': bool(args.use_prev_fill)
        },
        'results': results_all
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\n3D triangulation completed!")
    print(f"Results saved to: {out_path}")
    print(f"Processed {len(results_all)} frames successfully")


if __name__ == '__main__':
    main()