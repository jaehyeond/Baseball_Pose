#!/usr/bin/env python3
"""
Simple 3D Pose Visualization for Azure Kinect
Using matplotlib for 3D skeleton visualization
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path
import cv2

# COCO 17 keypoint skeleton connections
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

# Color scheme for different body parts
KEYPOINT_COLORS = [
    'red',      # nose
    'blue', 'blue', 'cyan', 'cyan',  # eyes, ears
    'green', 'orange', 'green', 'orange', 'green', 'orange',  # shoulders, elbows, wrists
    'green', 'orange', 'green', 'orange', 'green', 'orange'   # hips, knees, ankles
]


def load_3d_results(json_file):
    """Load 3D pose results from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def plot_3d_skeleton(ax, keypoints_3d, title="3D Pose", show_labels=True):
    """Plot 3D skeleton from keypoints"""
    # Extract coordinates and confidence
    x = [kpt[0] for kpt in keypoints_3d]
    y = [kpt[1] for kpt in keypoints_3d] 
    z = [kpt[2] for kpt in keypoints_3d]
    confidence = [kpt[3] for kpt in keypoints_3d]
    
    # Plot keypoints
    for i, (xi, yi, zi, conf, color) in enumerate(zip(x, y, z, confidence, KEYPOINT_COLORS)):
        if conf > 0.3:  # Only show high-confidence keypoints
            ax.scatter(xi, yi, zi, c=color, s=50, alpha=0.8)
            if show_labels and conf > 0.7:
                ax.text(xi, yi, zi, f' {i}', fontsize=8, alpha=0.7)
    
    # Plot skeleton connections
    for connection in COCO_SKELETON:
        start_idx, end_idx = connection
        if (start_idx < len(keypoints_3d) and end_idx < len(keypoints_3d) and 
            confidence[start_idx] > 0.3 and confidence[end_idx] > 0.3):
            ax.plot([x[start_idx], x[end_idx]], 
                   [y[start_idx], y[end_idx]], 
                   [z[start_idx], z[end_idx]], 
                   'b-', alpha=0.6, linewidth=2)
    
    # Set labels and title
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)') 
    ax.set_zlabel('Z (meters)')
    ax.set_title(title)
    
    # Set equal aspect ratio
    max_range = max([max(x)-min(x), max(y)-min(y), max(z)-min(z)]) / 2
    mid_x = (max(x) + min(x)) * 0.5
    mid_y = (max(y) + min(y)) * 0.5
    mid_z = (max(z) + min(z)) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    return ax


def visualize_single_frame(data, frame_idx=0, save_path=None, show=True):
    """Visualize single frame"""
    if frame_idx >= len(data['results']):
        print(f"Frame {frame_idx} not found. Available frames: 0-{len(data['results'])-1}")
        return
    
    frame_data = data['results'][frame_idx]
    keypoints_3d = frame_data['keypoints_3d']
    
    fig = plt.figure(figsize=(12, 10))
    
    # Main 3D plot
    ax = fig.add_subplot(111, projection='3d')
    title = f"Azure Kinect 3D Pose - Frame {frame_idx}\nValid keypoints: {frame_data['valid_keypoints']}/17"
    plot_3d_skeleton(ax, keypoints_3d, title)
    
    # Add confidence info
    avg_conf = np.mean([kpt[3] for kpt in keypoints_3d])
    plt.figtext(0.02, 0.02, f"Average confidence: {avg_conf:.3f}", fontsize=10, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
    
    # Add view angle controls text
    plt.figtext(0.02, 0.95, "Use mouse to rotate view", fontsize=10, bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.5))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def visualize_multiple_frames(data, frame_indices=None, save_path=None, show=True):
    """Visualize multiple frames in grid"""
    if frame_indices is None:
        frame_indices = list(range(min(6, len(data['results']))))
    
    n_frames = len(frame_indices)
    cols = min(3, n_frames)
    rows = (n_frames + cols - 1) // cols
    
    fig = plt.figure(figsize=(5*cols, 4*rows))
    fig.suptitle(f'Azure Kinect 3D Poses - Multiple Frames Comparison', fontsize=16, y=0.98)
    
    for i, frame_idx in enumerate(frame_indices):
        if frame_idx >= len(data['results']):
            continue
            
        frame_data = data['results'][frame_idx]
        keypoints_3d = frame_data['keypoints_3d']
        
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        title = f"Frame {frame_idx} ({frame_data['valid_keypoints']}/17)"
        plot_3d_skeleton(ax, keypoints_3d, title, show_labels=False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Multi-view saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def create_animation_frames(data, output_dir, view_angle_step=5):
    """Create individual frames for animation"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Creating animation frames in {output_dir}")
    
    for i, frame_data in enumerate(data['results']):
        keypoints_3d = frame_data['keypoints_3d']
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        title = f"Frame {i+1}/{len(data['results'])} - Azure Kinect 3D Pose"
        plot_3d_skeleton(ax, keypoints_3d, title, show_labels=False)
        
        # Rotate view for dynamic effect
        ax.view_init(elev=15, azim=70 + i * view_angle_step)
        
        # Save frame
        save_path = output_path / f"frame_{i:04d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if (i + 1) % 5 == 0:
            print(f"  Created {i+1}/{len(data['results'])} frames")
    
    print(f"Animation frames complete! {len(data['results'])} frames saved to {output_dir}")
    print("To create video: ffmpeg -r 5 -i frame_%04d.png -vcodec libx264 -pix_fmt yuv420p animation.mp4")


def analyze_pose_quality(data):
    """Analyze and print pose quality statistics"""
    print("\n" + "="*50)
    print("Azure Kinect 3D Pose Quality Analysis")
    print("="*50)
    
    all_confidences = []
    valid_counts = []
    keypoint_stats = np.zeros(17)
    
    for frame_data in data['results']:
        keypoints_3d = frame_data['keypoints_3d']
        confidences = [kpt[3] for kpt in keypoints_3d]
        all_confidences.extend(confidences)
        valid_counts.append(frame_data['valid_keypoints'])
        
        # Per-keypoint statistics
        for i, kpt in enumerate(keypoints_3d):
            keypoint_stats[i] += kpt[3]
    
    # Overall statistics
    print(f"Total frames processed: {len(data['results'])}")
    print(f"Average confidence: {np.mean(all_confidences):.3f}")
    print(f"Min confidence: {np.min(all_confidences):.3f}")
    print(f"Max confidence: {np.max(all_confidences):.3f}")
    print(f"Average valid keypoints per frame: {np.mean(valid_counts):.1f}/17")
    print(f"Best frame: {max(valid_counts)} valid keypoints")
    print(f"Worst frame: {min(valid_counts)} valid keypoints")
    
    # Per-keypoint average confidence
    keypoint_stats /= len(data['results'])
    print(f"\nPer-keypoint average confidence:")
    print("-" * 40)
    for i, (name, avg_conf) in enumerate(zip(KEYPOINT_NAMES, keypoint_stats)):
        status = "OK" if avg_conf > 0.8 else "WARN" if avg_conf > 0.5 else "FAIL"
        print(f"[{status:4}] {name:15}: {avg_conf:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Visualize Azure Kinect 3D poses')
    parser.add_argument('--input', default='D:/mmpose/azure_kinect_3d_results.json',
                       help='Input JSON file with 3D poses')
    parser.add_argument('--frame', type=int, default=0,
                       help='Frame index to visualize (default: 0)')
    parser.add_argument('--multiple', action='store_true',
                       help='Visualize multiple frames in grid')
    parser.add_argument('--frames', type=int, nargs='+',
                       help='Specific frame indices to visualize')
    parser.add_argument('--animation', action='store_true',
                       help='Create animation frame sequence')
    parser.add_argument('--save-dir', type=str, default='D:/mmpose/3d_visualization',
                       help='Directory to save visualization')
    parser.add_argument('--analyze', action='store_true',
                       help='Show detailed pose quality analysis')
    parser.add_argument('--no-show', action='store_true',
                       help='Save only, do not display plots')
    
    args = parser.parse_args()
    
    # Load 3D pose data
    print(f"Loading 3D poses from: {args.input}")
    data = load_3d_results(args.input)
    print(f"Loaded {len(data['results'])} frames")
    
    if args.analyze:
        analyze_pose_quality(data)
    
    # Create save directory
    if args.save_dir:
        Path(args.save_dir).mkdir(exist_ok=True)
    
    show = not args.no_show
    
    if args.animation:
        # Create animation frames
        create_animation_frames(data, args.save_dir)
    elif args.multiple or args.frames:
        # Multi-frame visualization
        frame_indices = args.frames if args.frames else None
        save_path = f"{args.save_dir}/multiple_frames.png" if args.save_dir else None
        visualize_multiple_frames(data, frame_indices, save_path, show)
    else:
        # Single frame visualization
        save_path = f"{args.save_dir}/frame_{args.frame}.png" if args.save_dir else None
        visualize_single_frame(data, args.frame, save_path, show)


if __name__ == '__main__':
    main()