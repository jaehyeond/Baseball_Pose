#!/usr/bin/env python3
"""
Azure Kinect 3D Triangulation
Load 2D poses from JSON files and triangulate to 3D using camera calibration
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
import argparse


def load_calibration(calib_file):
    """Load camera calibration parameters"""
    with open(calib_file, 'r') as f:
        calib_data = json.load(f)

    cameras = {}
    for cam_data in calib_data['cameras']:
        cam_name = cam_data['name']
        K = np.array(cam_data['K'], dtype=np.float64)
        dist = np.array(cam_data['distCoef'], dtype=np.float64).reshape(-1, 1)
        R = np.array(cam_data['R'], dtype=np.float64)
        t = np.array(cam_data['t'], dtype=np.float64).reshape(3, 1)

        cameras[cam_name] = {
            'K': K,
            'distCoef': dist,
            'R': R,
            't': t,
            'resolution': cam_data.get('resolution', None)
        }
        # Projection matrix P = K [R|t]
        cameras[cam_name]['P'] = K @ np.hstack([R, t])

    return cameras


def load_2d_poses(json_file):
    """Load 2D poses from JSON file (first instance) -> [J,3]: x,y,conf"""
    with open(json_file, 'r') as f:
        data = json.load(f)

    if 'instance_info' in data and len(data['instance_info']) > 0:
        inst = data['instance_info'][0]
        keypoints_2d = np.array(inst['keypoints'], dtype=np.float32)      # [J,2]
        keypoint_scores = np.array(inst['keypoint_scores'], dtype=np.float32)  # [J]
        J = keypoints_2d.shape[0]
        keypoints = np.zeros((J, 3), dtype=np.float32)
        keypoints[:, :2] = keypoints_2d
        keypoints[:, 2] = keypoint_scores
        return keypoints
    return None


def triangulate_points(kpts_2d_cam1, kpts_2d_cam2, cameras, cam_names,
                       conf_thr=0.3, reproj_thr_px=None, undistort=True):
    """
    Triangulate 2D keypoints from two cameras to 3D.

    Args:
        kpts_2d_cam1: [J,3] (x,y,conf)
        kpts_2d_cam2: [J,3] (x,y,conf)
        cameras: dict from load_calibration()
        cam_names: ['00','01']
        conf_thr: confidence threshold
        reproj_thr_px: reject joint if reprojection error > this (pixels). None to disable
        undistort: apply undistortion with each camera K,dist

    Returns:
        kpts_3d: [J,3] (float64)
        conf_out: [J] (float32)
        stats: dict with mean/max reprojection errors per camera
    """
    K1, d1 = cameras[cam_names[0]]['K'], cameras[cam_names[0]]['distCoef']
    K2, d2 = cameras[cam_names[1]]['K'], cameras[cam_names[1]]['distCoef']
    P1 = cameras[cam_names[0]]['P'].astype(np.float64)
    P2 = cameras[cam_names[1]]['P'].astype(np.float64)

    n_joints = len(kpts_2d_cam1)
    kpts_3d = np.zeros((n_joints, 3), dtype=np.float64)
    conf_out = np.zeros((n_joints,), dtype=np.float32)

    err1_list, err2_list = [], []

    for j in range(n_joints):
        pt1 = kpts_2d_cam1[j][:2].astype(np.float32)  # (x,y)
        pt2 = kpts_2d_cam2[j][:2].astype(np.float32)
        c1 = float(kpts_2d_cam1[j][2])
        c2 = float(kpts_2d_cam2[j][2])

        # skip invalid/low confidence
        if (not np.isfinite(pt1).all()) or (not np.isfinite(pt2).all()):
            continue
        if c1 < conf_thr or c2 < conf_thr:
            continue

        # per-joint undistortion to pixel coords consistent with P=K[R|t]
        if undistort:
            pt1_ud = cv2.undistortPoints(pt1.reshape(1, 1, 2), K1, d1, P=K1).reshape(2,)
            pt2_ud = cv2.undistortPoints(pt2.reshape(1, 1, 2), K2, d2, P=K2).reshape(2,)
        else:
            pt1_ud, pt2_ud = pt1, pt2

        # triangulate (expects px coords consistent with P)
        X_h = cv2.triangulatePoints(P1, P2,
                                    pt1_ud.reshape(2, 1).astype(np.float64),
                                    pt2_ud.reshape(2, 1).astype(np.float64))
        if abs(X_h[3, 0]) < 1e-12:
            continue
        X = (X_h[:3, 0] / X_h[3, 0]).astype(np.float64)  # 3D point

        # reprojection in pixels
        x1 = P1 @ np.append(X, 1.0)
        x1 = x1[:2] / x1[2]
        x2 = P2 @ np.append(X, 1.0)
        x2 = x2[:2] / x2[2]

        err1 = float(np.linalg.norm(x1 - pt1_ud))
        err2 = float(np.linalg.norm(x2 - pt2_ud))

        # optional rejection by reprojection error
        if reproj_thr_px is not None and (err1 > reproj_thr_px or err2 > reproj_thr_px):
            continue

        kpts_3d[j] = X
        conf_out[j] = min(c1, c2)
        err1_list.append(err1)
        err2_list.append(err2)

    stats = {
        'reproj_err_cam0_mean': float(np.mean(err1_list)) if err1_list else None,
        'reproj_err_cam1_mean': float(np.mean(err2_list)) if err2_list else None,
        'reproj_err_cam0_max':  float(np.max(err1_list))  if err1_list else None,
        'reproj_err_cam1_max':  float(np.max(err2_list))  if err2_list else None,
        'valid': int(np.count_nonzero(conf_out > 0.0)),
        'total': int(n_joints),
    }
    return kpts_3d, conf_out, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calib-file', default='D:/mmpose/calibration/azure_kinect_calibration.json')
    parser.add_argument('--cam1-poses', default='D:/mmpose/poses_2d_cam00')
    parser.add_argument('--cam2-poses', default='D:/mmpose/poses_2d_cam01')
    parser.add_argument('--output-file', default='D:/mmpose/azure_kinect_3d_results.json')
    parser.add_argument('--conf-thr', type=float, default=0.3)
    parser.add_argument('--reproj-thr', type=float, default=5.0, help='px; set negative to disable')
    parser.add_argument('--undistort', action=argparse.BooleanOptionalAction, default=True, help='apply undistortion (use --no-undistort to disable)')
    args = parser.parse_args()

    # Load calibration
    print("Loading calibration...")
    cameras = load_calibration(args.calib_file)
    cam_names = ['00', '01']

    # Collect JSON files
    cam1_jsons = sorted(Path(args.cam1_poses).glob('results_*.json'))
    cam2_jsons = sorted(Path(args.cam2_poses).glob('results_*.json'))

    print(f"Found {len(cam1_jsons)} pose files for cam1, {len(cam2_jsons)} for cam2")
    if len(cam1_jsons) != len(cam2_jsons):
        print(f"[WARN] frame count mismatch: cam1={len(cam1_jsons)} cam2={len(cam2_jsons)}; processing common prefix only.")

    # COCO-17 keypoint names
    coco_keypoints = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    n_frames = min(len(cam1_jsons), len(cam2_jsons))
    results_3d = []

    for i in range(n_frames):
        json1_path = cam1_jsons[i]
        json2_path = cam2_jsons[i]
        print(f"Processing frame {i+1}/{n_frames}: {json1_path.name}, {json2_path.name}")

        kpts_2d_cam1 = load_2d_poses(str(json1_path))
        kpts_2d_cam2 = load_2d_poses(str(json2_path))

        if kpts_2d_cam1 is None or kpts_2d_cam2 is None:
            print("  -> Failed to load 2D poses; skipping")
            continue

        # Triangulate
        reproj_thr = None if (args.reproj_thr is not None and args.reproj_thr < 0) else args.reproj_thr
        kpts_3d, confidence, stats = triangulate_points(
            kpts_2d_cam1, kpts_2d_cam2, cameras, cam_names,
            conf_thr=args.conf_thr, reproj_thr_px=reproj_thr, undistort=args.undistort
        )

        # Build result per frame
        keypoints_with_names = {}
        keypoints_3d_list = []
        for j, name in enumerate(coco_keypoints):
            keypoints_with_names[name] = {
                'position_3d': kpts_3d[j].tolist(),
                'confidence': float(confidence[j]),
                'cam1_2d': kpts_2d_cam1[j][:2].tolist(),
                'cam2_2d': kpts_2d_cam2[j][:2].tolist()
            }
            keypoints_3d_list.append(kpts_3d[j].tolist() + [float(confidence[j])])

        result = {
            'frame': i,
            'cam1_file': json1_path.name,
            'cam2_file': json2_path.name,  # ← 콤마가 꼭 필요!
            'keypoints_3d': keypoints_3d_list,
            'keypoints_detailed': keypoints_with_names,
            'valid_keypoints': int(np.sum(confidence > 0)),
            'total_keypoints': len(coco_keypoints),
            'reprojection_error': stats
        }
        results_3d.append(result)

        print(f"  -> Valid 3D keypoints: {result['valid_keypoints']}/{len(coco_keypoints)}")

    # Save results
    output_data = {
        'meta_info': {
            'dataset': 'azure_kinect_multiview',
            'num_keypoints': 17,
            'keypoint_names': coco_keypoints,
            'num_frames': len(results_3d),
            'cameras': cam_names,
            'coordinate_system': 'world_coordinates_meters'
        },
        'results': results_3d
    }

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\n3D triangulation completed!")
    print(f"Results saved to: {out_path}")
    print(f"Processed {len(results_3d)} frames successfully")

    if results_3d:
        valid_counts = [r['valid_keypoints'] for r in results_3d]
        print(f"Average valid keypoints per frame: {np.mean(valid_counts):.1f}/{len(coco_keypoints)}")
        print(f"Best frame:  {int(np.max(valid_counts))} keypoints")
        print(f"Worst frame: {int(np.min(valid_counts))} keypoints")


if __name__ == '__main__':
    main()
