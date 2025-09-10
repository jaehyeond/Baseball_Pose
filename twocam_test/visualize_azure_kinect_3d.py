#!/usr/bin/env python3
"""
Azure Kinect 3D Pose Visualization using MMPose visualization tools
Based on mmpose/demo/body3d_pose_lifter_demo.py and local_visualizer_3d.py
"""

import json
import numpy as np
import cv2
import os
from argparse import ArgumentParser
from pathlib import Path

import mmcv
from mmengine.structures import InstanceData
from mmpose.structures import PoseDataSample
from mmpose.registry import VISUALIZERS
from mmpose.visualization import Pose3dLocalVisualizer


# COCO 17 keypoint skeleton connections (same as mmpose format)
COCO_SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],  # legs
    [5, 11], [6, 12], [5, 6],  # torso
    [5, 7], [6, 8], [7, 9], [8, 10],  # arms
    [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]  # head
]

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


def load_3d_results(json_file):
    """Load 3D pose results from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def _apply_axis_transform(keypoints_3d, axis_order='xyz', flip_z=False):
    """Reorder axes and optionally flip Z for visualization.

    axis_order: one of {xyz,xzy,yxz,yzx,zxy,zyx}
    flip_z: if True, multiply Z by -1 after reordering.
    """
    arr = np.asarray(keypoints_3d, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return keypoints_3d
    xyz = arr[:, :3]
    conf = arr[:, 3:] if arr.shape[1] > 3 else None
    order_map = {
        'xyz': (0, 1, 2),
        'xzy': (0, 2, 1),
        'yxz': (1, 0, 2),
        'yzx': (1, 2, 0),
        'zxy': (2, 0, 1),
        'zyx': (2, 1, 0),
    }
    idx = order_map.get(axis_order, (0, 1, 2))
    xyz2 = xyz[:, idx].copy()
    if flip_z:
        xyz2[:, 2] = -xyz2[:, 2]
    out = np.concatenate([xyz2, conf] if conf is not None else [xyz2], axis=1)
    return out.tolist()


def create_pose_data_sample(keypoints_3d, frame_idx=0):
    """Create PoseDataSample from 3D keypoints for visualization"""
    # Extract coordinates and confidence
    keypoints = np.array(keypoints_3d)  # [17, 4] -> x, y, z, conf
    kpts_3d = keypoints[:, :3]  # [17, 3]
    kpt_scores = keypoints[:, 3]  # [17]
    
    # Create instance data
    pred_instances = InstanceData()
    pred_instances.keypoints = kpts_3d.reshape(1, 17, 3)  # [1, 17, 3]
    pred_instances.keypoint_scores = kpt_scores.reshape(1, 17)  # [1, 17]
    
    # Create pose data sample
    pose_data_sample = PoseDataSample()
    pose_data_sample.pred_instances = pred_instances
    
    return pose_data_sample


def create_pose_data_sample_multi(kpts3d_list, frame_idx=0):
    """Create PoseDataSample from a list of 3D keypoints (multi-person)."""
    arrs = [np.asarray(k, dtype=float) for k in kpts3d_list]
    kpts = np.stack([a[:, :3] for a in arrs], axis=0)  # [N,17,3]
    scores = np.stack([a[:, 3] if a.shape[1] > 3 else np.ones(a.shape[0]) for a in arrs], axis=0)  # [N,17]

    pred_instances = InstanceData()
    pred_instances.keypoints = kpts
    pred_instances.keypoint_scores = scores

    pose_data_sample = PoseDataSample()
    pose_data_sample.pred_instances = pred_instances
    return pose_data_sample


def setup_visualizer():
    """Setup 3D pose visualizer with COCO format"""
    # Define colors for keypoints (BGR format)
    kpt_color = [
        [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255],  # head
        [0, 255, 0], [255, 128, 0], [0, 255, 0], [255, 128, 0], [0, 255, 0], [255, 128, 0],  # arms
        [0, 255, 0], [255, 128, 0], [0, 255, 0], [255, 128, 0], [0, 255, 0], [255, 128, 0]  # legs
    ]
    
    # Define colors for skeleton links
    link_color = [
        [0, 255, 0], [0, 255, 0], [255, 128, 0], [255, 128, 0], [51, 153, 255],  # legs
        [51, 153, 255], [51, 153, 255], [51, 153, 255],  # torso
        [0, 255, 0], [255, 128, 0], [0, 255, 0], [255, 128, 0],  # arms  
        [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255]  # head
    ]

    # Ensure numpy arrays for boolean indexing inside visualizer
    kpt_color = np.asarray(kpt_color, dtype=np.uint8)
    link_color = np.asarray(link_color, dtype=np.uint8)

    visualizer = Pose3dLocalVisualizer(
        name='azure_kinect_3d',
        kpt_color=kpt_color,
        link_color=link_color,
        skeleton=COCO_SKELETON,
        line_width=2,
        radius=4
    )
    
    # Set dataset meta info
    dataset_meta = {
        'dataset_name': 'azure_kinect_coco',
        'keypoint_names': KEYPOINT_NAMES,
        'skeleton_links': COCO_SKELETON,
        'num_keypoints': 17
    }
    visualizer.set_dataset_meta(dataset_meta)
    
    return visualizer


def _auto_axis_limit_from_persons(persons, kpt_thr=0.3, min_limit=1.5, scale=1.2):
    """Compute a reasonable axis_limit from valid joints across persons."""
    all_pts = []
    for person in persons:
        arr = np.asarray(person['keypoints_3d'], dtype=float)
        conf = arr[:, 3] if arr.shape[1] >= 4 else np.ones(arr.shape[0])
        mask = (conf > kpt_thr) & np.isfinite(arr[:, :3]).all(axis=1)
        if np.any(mask):
            all_pts.append(arr[mask, :3])
    if not all_pts:
        return min_limit
    all_pts = np.concatenate(all_pts, axis=0)
    ranges = np.ptp(all_pts, axis=0)
    limit = max(min_limit, float(np.max(ranges) * scale))
    return limit


def visualize_single_frame(data, frame_idx=0, save_path=None, show=True,
                           axis_limit=None, axis_dist=10.0,
                           azim=70, elev=15, kpt_thr=0.3,
                           axis_order='xyz', flip_z=False):
    """Visualize single 3D pose frame"""
    if frame_idx >= len(data['results']):
        print(f"Frame {frame_idx} not found. Available frames: 0-{len(data['results'])-1}")
        return
    
    frame_data = data['results'][frame_idx]
    persons = frame_data.get('persons')

    # Setup visualizer
    visualizer = setup_visualizer()

    # Create a blank image for visualization
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255

    # Auto axis limit if not provided
    if axis_limit is None and 'persons' in frame_data:
        axis_limit = _auto_axis_limit_from_persons(frame_data['persons'], kpt_thr)
    if axis_limit is None:
        axis_limit = 2.0
    
    # Draw 3D pose
    if persons is None:
        # Legacy single-person format: frame_data contains keypoints_3d directly
        keypoints_3d = _apply_axis_transform(frame_data['keypoints_3d'], axis_order, flip_z)
        pose_sample = create_pose_data_sample(keypoints_3d, frame_idx)
        vis_img = visualizer._draw_3d_data_samples(
            img,
            pose_sample,
            draw_gt=False,
            kpt_thr=kpt_thr,
            num_instances=1,
            axis_azimuth=azim,
            axis_elev=elev,
            axis_limit=axis_limit,
            axis_dist=axis_dist
        )
    else:
        # Multi-person: draw each person sequentially to avoid width reshape issues
        vis_img = img
        for p in persons:
            k3d = _apply_axis_transform(p['keypoints_3d'], axis_order, flip_z)
            pose_sample = create_pose_data_sample(k3d, frame_idx)
            vis_img = visualizer._draw_3d_data_samples(
                vis_img,
                pose_sample,
                draw_gt=False,
                kpt_thr=kpt_thr,
                num_instances=1,
                axis_azimuth=azim,
                axis_elev=elev,
                axis_limit=axis_limit,
                axis_dist=axis_dist
            )
    
    # Get visualization result already returned as vis_img
    if save_path:
        mmcv.imwrite(vis_img, save_path)
        print(f"Saved: {save_path}")
    
    if show:
        cv2.imshow(f'3D Pose - Frame {frame_idx}', mmcv.rgb2bgr(vis_img))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return vis_img


def visualize_animation(data, save_dir=None, show=True, delay=500,
                        axis_limit=None, axis_dist=10.0, azim=70, elev=15,
                        kpt_thr=0.3, axis_order='xyz', flip_z=False):
    """Create animation from multiple frames"""
    if save_dir:
        Path(save_dir).mkdir(exist_ok=True)
    
    visualizer = setup_visualizer()
    
    print(f"Creating animation from {len(data['results'])} frames...")
    
    for i, frame_data in enumerate(data['results']):
        # Create image
        img = np.ones((600, 800, 3), dtype=np.uint8) * 255

        if isinstance(frame_data, dict) and 'persons' in frame_data and isinstance(frame_data['persons'], list):
            # Multi-person frame
            persons = frame_data['persons']
            axis_limit_i = axis_limit or _auto_axis_limit_from_persons(persons, kpt_thr)
            vis_img = img
            for p in persons:
                k3d = _apply_axis_transform(p['keypoints_3d'], axis_order, flip_z)
                pose_sample = create_pose_data_sample(k3d, i)
                vis_img = visualizer._draw_3d_data_samples(
                    vis_img,
                    pose_sample,
                    draw_gt=False,
                    kpt_thr=kpt_thr,
                    num_instances=1,
                    axis_azimuth=azim + i * 2,
                    axis_elev=elev,
                    axis_limit=axis_limit_i,
                    axis_dist=axis_dist
                )
        else:
            # Single-person legacy format
            keypoints_3d = frame_data['keypoints_3d']
            axis_limit_i = axis_limit or 2.0
            k3d = _apply_axis_transform(keypoints_3d, axis_order, flip_z)
            pose_sample = create_pose_data_sample(k3d, i)
            vis_img = visualizer._draw_3d_data_samples(
                img,
                pose_sample,
                draw_gt=False,
                kpt_thr=kpt_thr,
                num_instances=1,
                axis_azimuth=azim + i * 2,
                axis_elev=elev,
                axis_limit=axis_limit_i,
                axis_dist=axis_dist
            )
        
        if save_dir:
            save_path = os.path.join(save_dir, f"frame_{i:04d}.png")
            mmcv.imwrite(vis_img, save_path)
        
        if show:
            cv2.imshow('3D Pose Animation', mmcv.rgb2bgr(vis_img))
            key = cv2.waitKey(delay)
            if key == 27:  # ESC key
                break
    
    if show:
        cv2.destroyAllWindows()
    
    print(f"Animation complete! Frames saved to: {save_dir}" if save_dir else "Animation complete!")


def create_multi_view_comparison(data, frame_indices=None, save_path=None,
                                 axis_limit=None, axis_dist=10.0,
                                 azim=70, elev=15, kpt_thr=0.3,
                                 axis_order='xyz', flip_z=False):
    """Create comparison view of multiple frames"""
    if frame_indices is None:
        frame_indices = list(range(min(4, len(data['results']))))
    
    n_frames = len(frame_indices)
    cols = min(2, n_frames)
    rows = (n_frames + cols - 1) // cols
    
    # Create combined image
    img_h, img_w = 400, 600
    combined_img = np.ones((img_h * rows, img_w * cols, 3), dtype=np.uint8) * 255
    
    visualizer = setup_visualizer()
    
    for i, frame_idx in enumerate(frame_indices):
        if frame_idx >= len(data['results']):
            continue
            
        frame_data = data['results'][frame_idx]
        # Create individual image
        img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255

        if isinstance(frame_data, dict) and 'persons' in frame_data and isinstance(frame_data['persons'], list):
            persons = frame_data['persons']
            axis_limit_i = axis_limit or _auto_axis_limit_from_persons(persons, kpt_thr)
            vis_img = img
            for p in persons:
                k3d = _apply_axis_transform(p['keypoints_3d'], axis_order, flip_z)
                pose_sample = create_pose_data_sample(k3d, frame_idx)
                vis_img = visualizer._draw_3d_data_samples(
                    vis_img,
                    pose_sample,
                    draw_gt=False,
                    kpt_thr=kpt_thr,
                    num_instances=1,
                    axis_azimuth=azim + i * 30,
                    axis_elev=elev,
                    axis_limit=axis_limit_i,
                    axis_dist=axis_dist
                )
        else:
            keypoints_3d = frame_data['keypoints_3d']
            axis_limit_i = axis_limit or 2.0
            k3d = _apply_axis_transform(keypoints_3d, axis_order, flip_z)
            pose_sample = create_pose_data_sample(k3d, frame_idx)
            vis_img = visualizer._draw_3d_data_samples(
                img,
                pose_sample,
                draw_gt=False,
                kpt_thr=kpt_thr,
                num_instances=1,
                axis_azimuth=azim + i * 30,
                axis_elev=elev,
                axis_limit=axis_limit_i,
                axis_dist=axis_dist
            )
        
        # Place in combined image
        row = i // cols
        col = i % cols
        combined_img[row*img_h:(row+1)*img_h, col*img_w:(col+1)*img_w] = vis_img
    
    if save_path:
        mmcv.imwrite(combined_img, save_path)
        print(f"Multi-view saved: {save_path}")
    
    cv2.imshow('Multi-view 3D Poses', mmcv.rgb2bgr(combined_img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return combined_img


def main():
    parser = ArgumentParser(description='Visualize Azure Kinect 3D poses using MMPose')
    parser.add_argument('--input', default='D:/mmpose/azure_kinect_3d_results.json',
                       help='Input JSON file with 3D poses')
    parser.add_argument('--frame', type=int, default=0,
                       help='Frame index to visualize')
    parser.add_argument('--animation', action='store_true',
                       help='Create animation from all frames')
    parser.add_argument('--multi-view', action='store_true',
                       help='Show multiple frames comparison')
    parser.add_argument('--save-dir', type=str, default='D:/mmpose/visualization_output',
                       help='Directory to save outputs')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display windows')
    parser.add_argument('--delay', type=int, default=500,
                       help='Animation delay in milliseconds')
    parser.add_argument('--axis-limit', type=float, default=None,
                        help='Axis limit (auto if omitted)')
    parser.add_argument('--axis-dist', type=float, default=10.0,
                        help='Axis distance (camera distance)')
    parser.add_argument('--azim', type=float, default=70,
                        help='Azimuth angle')
    parser.add_argument('--elev', type=float, default=15,
                        help='Elevation angle')
    parser.add_argument('--kpt-thr', type=float, default=0.3,
                        help='Keypoint confidence threshold')
    # Compatibility/quality-of-life flags
    parser.add_argument('--output-root', type=str, default=None,
                        help='Alias of --save-dir')
    parser.add_argument('--all-frames', action='store_true',
                        help='Alias of --animation (export all frames)')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Accepted for compatibility (no effect)')
    parser.add_argument('--axis-order', type=str, default='xyz',
                        choices=['xyz','xzy','yxz','yzx','zxy','zyx'],
                        help='Axis order for visualization (reorders coordinates)')
    parser.add_argument('--flip-z', action='store_true',
                        help='Flip Z axis sign after reordering')
    parser.add_argument('--depth-axis', type=str, default='z',
                        choices=['x','y','z'],
                        help='Compatibility only (use --axis-order/--flip-z instead)')
    parser.add_argument('--hide-axis-labels', action='store_true',
                        help='Compatibility only (labels are minimal)')
    
    args = parser.parse_args()

    # Map compatibility flags
    if args.output_root and not args.save_dir:
        args.save_dir = args.output_root
    if args.all_frames:
        args.animation = True
    
    # Load 3D pose data
    print(f"Loading 3D poses from: {args.input}")
    data = load_3d_results(args.input)
    print(f"Loaded {len(data['results'])} frames")
    
    # Create save directory
    if args.save_dir:
        Path(args.save_dir).mkdir(exist_ok=True)
    
    show = not args.no_show
    
    if args.animation:
        # Create animation
        visualize_animation(data, args.save_dir, show, args.delay,
                            axis_limit=args.axis_limit, axis_dist=args.axis_dist,
                            azim=args.azim, elev=args.elev, kpt_thr=args.kpt_thr,
                            axis_order=args.axis_order, flip_z=args.flip_z)
    elif args.multi_view:
        # Multi-view comparison
        save_path = os.path.join(args.save_dir, "multi_view_comparison.png") if args.save_dir else None
        create_multi_view_comparison(data, save_path=save_path,
                                     axis_limit=args.axis_limit, axis_dist=args.axis_dist,
                                     azim=args.azim, elev=args.elev, kpt_thr=args.kpt_thr,
                                     axis_order=args.axis_order, flip_z=args.flip_z)
    else:
        # Single frame visualization
        save_path = os.path.join(args.save_dir, f"3d_pose_frame_{args.frame}.png") if args.save_dir else None
        visualize_single_frame(data, args.frame, save_path, show,
                               axis_limit=args.axis_limit, axis_dist=args.axis_dist,
                               azim=args.azim, elev=args.elev, kpt_thr=args.kpt_thr,
                               axis_order=args.axis_order, flip_z=args.flip_z)


if __name__ == '__main__':
    main()
