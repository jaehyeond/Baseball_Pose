# -*- coding: utf-8 -*-
"""
Improved Two-cam triangulation with better matching and occlusion handling.
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def load_json_video(path: str) -> List[List[Dict[str, np.ndarray]]]:
    """영상 전체 JSON을 프레임 리스트로 변환. MMPose 영상 형식도 지원."""
    p = Path(path)
    if p.is_dir():
        cand = p / "results_output_video.json"
        if not cand.exists():
            raise FileNotFoundError(f"Video JSON not found in directory: {p}")
        path = str(cand)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # MMPose video format: list of frame dictionaries
    if isinstance(data, list):
        frames: List[List[Dict[str, np.ndarray]]] = []
        for frame_data in data:
            frame_id = frame_data.get('frame_id', 0)
            instances = frame_data.get('instances', [])
            
            people = []
            for inst in instances:
                k = inst.get('keypoints', [])
                if not k:
                    continue
                k = np.asarray(k, dtype=np.float32)
                
                # Ensure proper shape: (17, 2)
                if k.ndim == 1:
                    if k.size == 34:  # 17 joints * 2 coordinates
                        k = k.reshape(17, 2)
                    else:
                        continue
                elif k.ndim == 2:
                    if k.shape[0] == 17 and k.shape[1] >= 2:
                        k = k[:, :2]
                    else:
                        continue
                else:
                    continue
                
                # Handle scores
                s = inst.get('keypoint_scores', [])
                if not s:
                    scores = np.ones(17, dtype=np.float32)
                else:
                    s = np.asarray(s, dtype=np.float32)
                    if s.size != 17:
                        scores = np.ones(17, dtype=np.float32)
                    else:
                        scores = s
                
                people.append({"keypoints": k, "scores": scores})
            frames.append(people)
        
        return frames

    # Original format support
    raw_frames = None
    if isinstance(data, dict):
        if "instance_info" in data:
            raw_frames = data["instance_info"]
        elif "frames" in data:
            raw_frames = data["frames"]
    if raw_frames is None:
        raise ValueError("Unsupported JSON format.")

    frames: List[List[Dict[str, np.ndarray]]] = []
    for fr in raw_frames:
        insts = fr.get("instances", fr.get("objects", []))
        people = []
        for inst in insts:
            k = inst.get("keypoints") or inst.get("keypoints_2d")
            if k is None:
                continue
            k = np.asarray(k, dtype=np.float32)

            if k.ndim == 1:
                J = 17 if (k.size % 17 == 0) else (k.size // 3)
                if k.size == J * 3:
                    k = k.reshape(J, 3)[:, :2]
                elif k.size == J * 2:
                    k = k.reshape(J, 2)
                else:
                    continue
            elif k.ndim == 2:
                if k.shape[1] >= 2:
                    k = k[:, :2]
                else:
                    continue
            else:
                continue

            s = inst.get("keypoint_scores", inst.get("scores"))
            if s is None:
                scores = np.ones((k.shape[0],), dtype=np.float32)
            else:
                s = np.asarray(s, dtype=np.float32)
                if s.ndim == 1:
                    scores = s
                elif s.ndim == 2 and s.shape[1] >= 1:
                    scores = s[:, 0]
                else:
                    scores = np.ones((k.shape[0],), dtype=np.float32)

            people.append({"keypoints": k, "scores": scores})
        frames.append(people)

    return frames


def parse_calib(calib_path: str, extrinsic_format: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """캘리브 JSON → (K0,d0,K1,d1,R_rel,t_rel)"""
    def _arr(d, *keys):
        for k in keys:
            if k in d:
                return np.asarray(d[k], dtype=np.float64)
        raise KeyError(keys)

    with open(calib_path, "r", encoding="utf-8") as f:
        cj = json.load(f)

    cams = cj.get("cameras", cj.get("Cameras"))
    if cams is None or len(cams) < 2:
        raise ValueError("Calibration JSON must have at least two cameras.")

    c0, c1 = cams[0], cams[1]
    K0 = _arr(c0, "K", "camera_matrix", "intrinsic")
    K1 = _arr(c1, "K", "camera_matrix", "intrinsic")
    d0 = _arr(c0, "dist", "distCoef", "distCoeffs").ravel()
    d1 = _arr(c1, "dist", "distCoef", "distCoeffs").ravel()
    R0 = _arr(c0, "R")
    t0 = _arr(c0, "t", "T").ravel()
    R1 = _arr(c1, "R")
    t1 = _arr(c1, "t", "T").ravel()

    if extrinsic_format.lower() == "c2w":
        def c2w_to_w2c(Rcw, tcw):
            Rwc = Rcw.T
            twc = -Rcw.T @ tcw
            return Rwc, twc
        R0, t0 = c2w_to_w2c(R0, t0)
        R1, t1 = c2w_to_w2c(R1, t1)

    Rrel = R1 @ np.linalg.inv(R0)
    trel = t1 - R1 @ np.linalg.inv(R0) @ t0
    return K0, d0, K1, d1, Rrel, trel


def parse_calib_compat(calib_path: str, extrinsic_format: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility parser for various calibration JSON schemas.

    Tries Azure Kinect-like stereo schema first; falls back to the
    existing {"cameras": [...]} schema.
    Returns (K0, d0, K1, d1, R_rel, t_rel), with translation in meters.
    """
    with open(calib_path, "r", encoding="utf-8") as f:
        cj = json.load(f)

    # Azure Kinect-like stereo schema
    pri = cj.get("primary_camera_intrinsics")
    sec = cj.get("secondary_camera_intrinsics")
    ster = cj.get("stereo_parameters")
    if pri is not None and sec is not None and ster is not None:
        def _arr(node, *keys):
            for k in keys:
                if k in node:
                    return np.asarray(node[k], dtype=np.float64)
            raise KeyError(keys)

        K0 = _arr(pri, "camera_matrix", "K", "intrinsic")
        K1 = _arr(sec, "camera_matrix", "K", "intrinsic")

        def _dist(dnode):
            dist = dnode.get("distortion") or dnode.get("dist") or dnode.get("distCoef")
            if dist is None:
                return np.zeros((5,), dtype=np.float64)
            return np.asarray(dist, dtype=np.float64).ravel()

        d0 = _dist(pri)
        d1 = _dist(sec)

        R_rel = _arr(ster, "rotation_matrix", "R", "R_rel")
        t_rel = _arr(ster, "translation_vector", "t", "t_rel").ravel()
        # Convert mm->m if magnitudes look like millimeters
        if np.linalg.norm(t_rel) > 100.0:
            t_rel = t_rel / 1000.0

        return K0, d0, K1, d1, R_rel.astype(np.float64), t_rel.astype(np.float64)

    # Fallback: use original parser
    return parse_calib(calib_path, extrinsic_format)


def undistort_pts(pts: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    if pts.size == 0:
        return pts.copy()
    pts = pts.reshape(-1, 1, 2).astype(np.float32)
    und = cv2.undistortPoints(pts, cameraMatrix=K, distCoeffs=dist.astype(np.float64), P=K)
    return und.reshape(-1, 2)


def triangulate_robust(P0: np.ndarray, P1: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """Robust triangulation with outlier filtering"""
    if p0.shape[0] == 0:
        return np.array([]).reshape(0, 3)
    
    X_h = cv2.triangulatePoints(P0, P1, p0.T, p1.T)
    X = (X_h[:3] / X_h[3]).T
    
    # Filter out points that are too far behind the camera or too far away
    valid_mask = (X_h[3] > 0.1) & (np.linalg.norm(X, axis=1) < 10.0)
    
    return X, valid_mask


def compute_pose_similarity(pose1: np.ndarray, pose2: np.ndarray, scores1: np.ndarray, scores2: np.ndarray) -> float:
    """두 포즈 간의 유사도 계산 (높은 confidence를 가진 관절 위주로)"""
    if pose1.shape[0] == 0 or pose2.shape[0] == 0:
        return 0.0
    
    # 양쪽 모두 confidence가 높은 관절만 사용
    valid_mask = (scores1 > 0.3) & (scores2 > 0.3)
    if valid_mask.sum() < 3:  # 최소 3개 관절은 필요
        return 0.0
    
    p1_valid = pose1[valid_mask]
    p2_valid = pose2[valid_mask]
    
    # 포즈의 중심점으로 정규화
    c1 = np.mean(p1_valid, axis=0)
    c2 = np.mean(p2_valid, axis=0)
    
    p1_centered = p1_valid - c1
    p2_centered = p2_valid - c2
    
    # 크기로 정규화
    scale1 = np.std(p1_centered)
    scale2 = np.std(p2_centered)
    
    if scale1 < 1e-6 or scale2 < 1e-6:
        return 0.0
    
    p1_norm = p1_centered / scale1
    p2_norm = p2_centered / scale2
    
    # 유클리드 거리 기반 유사도
    distances = np.linalg.norm(p1_norm - p2_norm, axis=1)
    similarity = np.exp(-np.mean(distances))
    
    return similarity


def match_people_advanced(people0: List[Dict[str, np.ndarray]], people1: List[Dict[str, np.ndarray]]) -> List[Tuple[int, int]]:
    """개선된 사람 매칭 알고리즘"""
    if len(people0) == 0 or len(people1) == 0:
        return []
    
    n0, n1 = len(people0), len(people1)
    
    # 유사도 행렬 계산
    similarity_matrix = np.zeros((n0, n1))
    
    for i, p0 in enumerate(people0):
        for j, p1 in enumerate(people1):
            sim = compute_pose_similarity(
                p0["keypoints"], p1["keypoints"],
                p0["scores"], p1["scores"]
            )
            similarity_matrix[i, j] = sim
    
    # 헝가리안 알고리즘으로 최적 매칭
    # similarity를 cost로 바꾸기 위해 1에서 빼기 (단, 너무 작은 값은 제외)
    cost_matrix = 1 - similarity_matrix
    
    # 너무 낮은 유사도는 매칭에서 제외
    cost_matrix[similarity_matrix < 0.1] = 1e6
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # 유효한 매칭만 반환
    matches = []
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < 1e5:  # 유효한 매칭
            matches.append((r, c))
    
    return matches


def smooth_3d_pose(current_pose: np.ndarray, prev_pose: Optional[np.ndarray], 
                   current_scores: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    """3D 포즈 시간적 스무딩"""
    if prev_pose is None:
        return current_pose
    
    # confidence가 낮은 관절은 이전 프레임 값을 더 많이 사용
    weights = np.clip(current_scores, 0.1, 1.0).reshape(-1, 1)
    smoothed = alpha * current_pose + (1 - alpha) * prev_pose
    
    # confidence에 따라 가중 평균
    final_pose = weights * current_pose + (1 - weights) * smoothed
    
    return final_pose


def filter_outlier_joints(pose_3d: np.ndarray, scores: np.ndarray, threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """통계적 이상치 제거"""
    if pose_3d.shape[0] < 5:  # 관절이 너무 적으면 필터링 안함
        return pose_3d, scores
    
    # 포즈 중심점 계산
    valid_mask = scores > 0.3
    if valid_mask.sum() < 3:
        return pose_3d, scores
    
    center = np.mean(pose_3d[valid_mask], axis=0)
    distances = np.linalg.norm(pose_3d - center, axis=1)
    
    # 중위수와 MAD 기반 이상치 검출
    median_dist = np.median(distances[valid_mask])
    mad = np.median(np.abs(distances[valid_mask] - median_dist))
    
    if mad < 1e-6:
        return pose_3d, scores
    
    z_scores = (distances - median_dist) / (1.4826 * mad)  # MAD to std conversion
    outlier_mask = z_scores > threshold
    
    # 이상치는 confidence 크게 감소
    scores_filtered = scores.copy()
    scores_filtered[outlier_mask] *= 0.1
    
    return pose_3d, scores_filtered


def build_projection(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    Rt = np.hstack([R, t.reshape(3, 1)])
    return K @ Rt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-file", required=True)
    ap.add_argument("--cam1-json", dest="cam1_json", required=True)
    ap.add_argument("--cam2-json", dest="cam2_json", required=True)
    ap.add_argument("--output-file", required=True)
    ap.add_argument("--extrinsic-format", choices=["w2c", "c2w"], default="w2c")
    ap.add_argument("--conf-thr", type=float, default=0.4)  # 높인 기본값
    ap.add_argument("--no-undistort", action="store_true")
    ap.add_argument("--frame-offset", type=int, default=0)
    ap.add_argument("--max-persons", type=int, default=3)
    ap.add_argument("--reproj-stat", action="store_true")
    ap.add_argument("--smooth-alpha", type=float, default=0.7, 
                    help="시간적 스무딩 계수 (0-1, 높을수록 현재 프레임 중시)")
    ap.add_argument("--outlier-threshold", type=float, default=2.5,
                    help="이상치 검출 임계값")
    args = ap.parse_args()

    print("Loading calibration...")
    K0, d0, K1, d1, Rrel, trel = parse_calib_compat(args.calib_file, args.extrinsic_format)

    print("Loading video JSONs...")
    frames0 = load_json_video(args.cam1_json)
    frames1 = load_json_video(args.cam2_json)

    n = min(len(frames0), len(frames1) - args.frame_offset)
    if n <= 0:
        raise RuntimeError("No overlapping frames.")
    print(f"Total overlapping frames: {n}")

    P0 = build_projection(K0, np.eye(3, dtype=np.float64), np.zeros(3))
    P1 = build_projection(K1, Rrel.astype(np.float64), trel.astype(np.float64))

    results_out: List[Dict[str, Any]] = []
    prev_poses_3d: List[np.ndarray] = []  # 이전 프레임 3D 포즈들

    for i in range(n):
        people0 = frames0[i]
        people1 = frames1[i + args.frame_offset]

        # 개선된 매칭 알고리즘 사용
        pairs = match_people_advanced(people0, people1)
        if args.max_persons > 0:
            pairs = pairs[:args.max_persons]

        persons_out = []
        current_poses_3d = []

        for pair_idx, (i0, i1) in enumerate(pairs):
            k0 = people0[i0]["keypoints"].copy()
            s0 = people0[i0]["scores"].copy()
            k1 = people1[i1]["keypoints"].copy()
            s1 = people1[i1]["scores"].copy()

            # 더 엄격한 confidence 필터링
            conf = np.minimum(s0, s1)
            mask = (s0 > args.conf_thr) & (s1 > args.conf_thr)

            J = k0.shape[0]
            joints3d = np.full((J, 3), np.nan, dtype=np.float32)
            scores = np.zeros((J,), dtype=np.float32)

            # 삼각측량
            if mask.sum() > 0:
                p0 = k0[mask]
                p1 = k1[mask]
                if not args.no_undistort:
                    p0 = undistort_pts(p0, K0, d0)
                    p1 = undistort_pts(p1, K1, d1)
                
                X, valid_mask = triangulate_robust(P0, P1, p0, p1)
                
                idxs = np.where(mask)[0]
                if valid_mask.sum() > 0:
                    joints3d[idxs[valid_mask], :] = X[valid_mask]
                    scores[idxs[valid_mask]] = conf[idxs[valid_mask]]

            # NaN 처리 및 스무딩
            valid_joints = ~np.isnan(joints3d).any(axis=1)
            if valid_joints.sum() > 0:
                # 이상치 필터링
                joints3d, scores = filter_outlier_joints(joints3d, scores, args.outlier_threshold)
                
                # 시간적 스무딩
                prev_pose = prev_poses_3d[pair_idx] if pair_idx < len(prev_poses_3d) else None
                if prev_pose is not None and valid_joints.sum() > 3:
                    joints3d[valid_joints] = smooth_3d_pose(
                        joints3d[valid_joints], prev_pose[valid_joints], 
                        scores[valid_joints], args.smooth_alpha
                    )

            # 최종 NaN 처리
            joints3d = np.nan_to_num(joints3d, nan=0.0)
            scores = np.nan_to_num(scores, nan=0.0)

            current_poses_3d.append(joints3d.copy())
            persons_out.append({
                "keypoints_3d": joints3d.tolist(),
                "scores": scores.tolist(),
                "ok": True
            })

        # 이전 프레임 정보 업데이트
        prev_poses_3d = current_poses_3d

        results_out.append({
            "frame": i,
            "num_persons": len(persons_out),
            "persons": persons_out
        })

        if i % 50 == 0:
            print(f"Processed frame {i}/{n}")

    out = {
        "meta": {
            "skeleton": "coco17",
            "ref_cam": "cam0",
            "extrinsic_format": args.extrinsic_format,
            "improved_matching": True,
            "temporal_smoothing": args.smooth_alpha
        },
        "results": results_out
    }

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved: {args.output_file}")


if __name__ == "__main__":
    main()

