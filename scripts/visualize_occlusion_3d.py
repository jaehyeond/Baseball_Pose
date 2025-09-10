#!/usr/bin/env python3
"""
Azure Kinect style 3D visualization for occlusion-robust results
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# COCO-17 skeleton connections
SKELETON_EDGES = [
    (0, 1), (0, 2),  # nose to eyes
    (1, 3), (2, 4),  # eyes to ears
    (0, 5), (0, 6),  # nose to shoulders 
    (5, 7), (7, 9),  # left arm
    (6, 8), (8, 10), # right arm
    (5, 6),          # shoulders
    (5, 11), (6, 12), # shoulders to hips
    (11, 12),        # hips
    (11, 13), (13, 15), # left leg
    (12, 14), (14, 16)  # right leg
]

def draw_skeleton_3d(ax, kpts_3d, color='red', alpha=0.8, point_size=50):
    """Draw a single person skeleton in 3D"""
    if len(kpts_3d) != 17:
        return
    
    kpts_3d = np.array(kpts_3d)
    
    # Find valid joints (non-zero coordinates)
    valid_joints = []
    for i, point in enumerate(kpts_3d):
        if not (abs(point[0]) < 1e-6 and abs(point[1]) < 1e-6 and abs(point[2]) < 1e-6):
            valid_joints.append(i)
    
    print(f"    Drawing {len(valid_joints)} valid joints with color {color}")
    
    # Draw keypoints as scatter plot
    if len(valid_joints) > 0:
        valid_points = kpts_3d[valid_joints]
        ax.scatter(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2],
                  s=point_size, c=color, alpha=alpha, depthshade=True, marker='o')
    
    # Draw skeleton connections with more robust checking
    connections_drawn = 0
    for i, j in SKELETON_EDGES:
        # Check if both joints are valid
        if i in valid_joints and j in valid_joints:
            pt1, pt2 = kpts_3d[i], kpts_3d[j]
            
            # Double check they're not zero
            pt1_valid = not (abs(pt1[0]) < 1e-6 and abs(pt1[1]) < 1e-6 and abs(pt1[2]) < 1e-6)
            pt2_valid = not (abs(pt2[0]) < 1e-6 and abs(pt2[1]) < 1e-6 and abs(pt2[2]) < 1e-6)
            
            if pt1_valid and pt2_valid:
                xs = [pt1[0], pt2[0]]
                ys = [pt1[1], pt2[1]]
                zs = [pt1[2], pt2[2]]
                ax.plot(xs, ys, zs, color=color, linewidth=3, alpha=alpha)
                connections_drawn += 1
    
    print(f"    Drew {connections_drawn} skeleton connections")
    return len(valid_joints), connections_drawn

def setup_3d_axes(ax, center_point=None, range_meters=2.0):
    """Setup 3D axes with proper scaling and labels"""
    if center_point is None:
        center_point = [0, 0, 1.5]
    
    # Set equal aspect ratio and range
    ax.set_xlim(center_point[0] - range_meters/2, center_point[0] + range_meters/2)
    ax.set_ylim(center_point[1] - range_meters/2, center_point[1] + range_meters/2) 
    ax.set_zlim(center_point[2] - range_meters/2, center_point[2] + range_meters/2)
    
    # Labels and grid
    ax.set_xlabel('X (meters)', fontsize=10)
    ax.set_ylabel('Y (meters)', fontsize=10)
    ax.set_zlabel('Z (meters)', fontsize=10)
    
    # Nice viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Grid and background
    ax.grid(True, alpha=0.3)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False  
    ax.zaxis.pane.fill = False
    
    # Make pane edges more transparent
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)

def visualize_frame(frame_data, frame_idx, output_dir):
    """Visualize a single frame with multiple persons"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    persons = frame_data.get('persons', [])
    num_persons = len(persons)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple'][:num_persons]
    
    # Find center point for better visualization
    all_points = []
    for person in persons:
        kpts = person['keypoints_3d']
        for kpt in kpts:
            if not (kpt[0] == 0 and kpt[1] == 0 and kpt[2] == 0):
                all_points.append(kpt)
    
    if len(all_points) > 0:
        all_points = np.array(all_points)
        center_point = np.mean(all_points, axis=0)
        range_meters = max(2.0, np.max(np.ptp(all_points, axis=0)) * 1.5)
    else:
        center_point = [0, 0, 1.5]
        range_meters = 2.0
    
    # Draw each person
    for i, person in enumerate(persons):
        kpts_3d = person['keypoints_3d']
        color = colors[i % len(colors)]
        
        print(f"  Drawing Person {i+1} with {color}:")
        valid_joints, connections = draw_skeleton_3d(ax, kpts_3d, color=color, alpha=0.8)
        
        # Add person label
        valid_points = [kpt for kpt in kpts_3d if not (abs(kpt[0]) < 1e-6 and abs(kpt[1]) < 1e-6 and abs(kpt[2]) < 1e-6)]
        if valid_points:
            head_point = valid_points[0] if len(valid_points) > 0 else center_point
            ax.text(head_point[0], head_point[1], head_point[2] + 0.2,
                   f'Person {i+1}', fontsize=12, color=color, weight='bold')
    
    setup_3d_axes(ax, center_point, range_meters)
    
    # Title with occlusion info
    occlusion_info = " (Occlusion)" if frame_data.get('occlusion_detected', False) else ""
    title = f'3D Pose (Frame {frame_idx}) | Persons: {num_persons}{occlusion_info}'
    plt.title(title, fontsize=14, weight='bold')
    
    # Save
    output_path = Path(output_dir) / f'frame_{frame_idx:04d}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    return output_path

def main():
    # Load 3D results
    with open("D:/mmpose/output_3d/occlusion_robust_3d.json", 'r') as f:
        data = json.load(f)
    
    output_dir = Path("D:/mmpose/vis3d_occlusion_robust")
    output_dir.mkdir(exist_ok=True)
    
    results = data['results']
    print(f"Visualizing {len(results)} frames with occlusion-robust 3D poses")
    
    # Statistics
    single_person_frames = sum(1 for r in results if r['num_persons'] == 1)
    multi_person_frames = sum(1 for r in results if r['num_persons'] > 1)
    occlusion_frames = sum(1 for r in results if r.get('occlusion_detected', False))
    
    print(f"Single person frames: {single_person_frames}")
    print(f"Multi-person frames: {multi_person_frames}")
    print(f"Occlusion cases: {occlusion_frames}")
    
    # Visualize all frames
    for i, frame_data in enumerate(results):
        frame_idx = frame_data['frame_index']
        output_path = visualize_frame(frame_data, frame_idx, output_dir)
        
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{len(results)} frames")
    
    print(f"Visualization complete! Saved to: {output_dir}")
    
    # Create a summary of multi-person frames
    multi_frames = [r for r in results if r['num_persons'] > 1]
    if multi_frames:
        print(f"\nMulti-person frames:")
        for frame in multi_frames[:10]:  # Show first 10
            frame_idx = frame['frame_index']
            num_persons = frame['num_persons']
            occlusion = frame.get('occlusion_detected', False)
            print(f"  Frame {frame_idx}: {num_persons} persons {'(occlusion)' if occlusion else ''}")

if __name__ == "__main__":
    main()