#!/usr/bin/env python3
"""
Visualize video-based 3D results with proper skeleton rendering
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# COCO-17 skeleton connections (same as before)
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

def draw_skeleton_3d_video(ax, joints_3d, color='red', alpha=0.8, point_size=50, conf_threshold=0.3):
    """Draw skeleton from video-based 3D results (format: [x,y,z,confidence])"""
    if len(joints_3d) != 17:
        return 0, 0
    
    joints_3d = np.array(joints_3d)
    
    # Extract coordinates and confidence
    coords = joints_3d[:, :3]  # x, y, z
    confs = joints_3d[:, 3]    # confidence
    
    # Find valid joints based on confidence
    valid_joints = []
    for i in range(17):
        if confs[i] > conf_threshold:
            valid_joints.append(i)
    
    print(f"    Valid joints (conf > {conf_threshold}): {len(valid_joints)} joints")
    
    # Draw keypoints
    if len(valid_joints) > 0:
        valid_points = coords[valid_joints]
        ax.scatter(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2],
                  s=point_size, c=color, alpha=alpha, depthshade=True, marker='o')
    
    # Draw skeleton connections
    connections_drawn = 0
    for i, j in SKELETON_EDGES:
        if i in valid_joints and j in valid_joints:
            pt1, pt2 = coords[i], coords[j]
            
            xs = [pt1[0], pt2[0]]
            ys = [pt1[1], pt2[1]]
            zs = [pt1[2], pt2[2]]
            ax.plot(xs, ys, zs, color=color, linewidth=3, alpha=alpha)
            connections_drawn += 1
    
    print(f"    Drew {connections_drawn} skeleton connections")
    return len(valid_joints), connections_drawn

def setup_3d_axes(ax, center_point=None, range_meters=2.0):
    """Setup 3D axes with proper scaling"""
    if center_point is None:
        center_point = [0, 0, 1.5]
    
    ax.set_xlim(center_point[0] - range_meters/2, center_point[0] + range_meters/2)
    ax.set_ylim(center_point[1] - range_meters/2, center_point[1] + range_meters/2)
    ax.set_zlim(center_point[2] - range_meters/2, center_point[2] + range_meters/2)
    
    ax.set_xlabel('X (meters)', fontsize=10)
    ax.set_ylabel('Y (meters)', fontsize=10)
    ax.set_zlabel('Z (meters)', fontsize=10)
    
    ax.view_init(elev=20, azim=45)
    ax.grid(True, alpha=0.3)
    
    # Make axes cleaner
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)

def visualize_frame(frame_data, frame_idx, output_dir):
    """Visualize single frame from video results"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    persons = frame_data.get('persons', [])
    num_persons = len(persons)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple'][:num_persons]
    
    print(f"Frame {frame_idx}: {num_persons} persons")
    
    # Calculate center point from all valid joints
    all_points = []
    for person in persons:
        joints = np.array(person['joints_3d'])
        coords = joints[:, :3]
        confs = joints[:, 3]
        valid_coords = coords[confs > 0.3]
        if len(valid_coords) > 0:
            all_points.extend(valid_coords.tolist())
    
    if len(all_points) > 0:
        all_points = np.array(all_points)
        center_point = np.mean(all_points, axis=0)
        range_meters = max(2.0, np.max(np.ptp(all_points, axis=0)) * 1.5)
    else:
        center_point = [0, 0, 1.5]
        range_meters = 2.0
    
    # Draw each person
    total_joints = 0
    total_connections = 0
    
    for i, person in enumerate(persons):
        joints_3d = person['joints_3d']
        color = colors[i % len(colors)]
        
        print(f"  Drawing Person {i+1} with {color}:")
        valid_joints, connections = draw_skeleton_3d_video(ax, joints_3d, color=color, alpha=0.8)
        
        total_joints += valid_joints
        total_connections += connections
        
        # Add person label near head
        if valid_joints > 0:
            head_joint = np.array(joints_3d[0][:3])  # nose joint
            ax.text(head_joint[0], head_joint[1], head_joint[2] + 0.15,
                   f'Person {i+1}', fontsize=12, color=color, weight='bold')
    
    setup_3d_axes(ax, center_point, range_meters)
    
    title = f'Video-based 3D Pose (Frame {frame_idx}) | {num_persons} Persons | {total_connections} Connections'
    plt.title(title, fontsize=14, weight='bold')
    
    # Save
    output_path = Path(output_dir) / f'video_frame_{frame_idx:04d}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    return output_path, total_joints, total_connections

def main():
    # Load video-based 3D results
    with open("D:/mmpose/output_3d/video_based_3d.json", 'r') as f:
        data = json.load(f)
    
    output_dir = Path("D:/mmpose/vis3d_video_based")
    output_dir.mkdir(exist_ok=True)
    
    results = data['results']
    print(f"Visualizing {len(results)} frames from video-based 3D processing")
    
    # Statistics
    multi_person_frames = [r for r in results if r['num_persons'] > 1]
    print(f"Multi-person frames: {len(multi_person_frames)}")
    
    # Process first 20 frames to verify quality
    total_joints = 0
    total_connections = 0
    processed_frames = 0
    
    for i, frame_data in enumerate(results[:20]):  # First 20 frames for testing
        frame_idx = frame_data['frame']
        
        output_path, joints, connections = visualize_frame(frame_data, frame_idx, output_dir)
        total_joints += joints
        total_connections += connections
        processed_frames += 1
        
        if (i + 1) % 5 == 0:
            print(f"Processed {i + 1}/20 test frames")
    
    print(f"\nTest visualization complete!")
    print(f"Average joints per frame: {total_joints/processed_frames:.1f}")
    print(f"Average connections per frame: {total_connections/processed_frames:.1f}")
    print(f"Results saved to: {output_dir}")
    
    # Show summary of best multi-person frames
    best_frames = []
    for frame in multi_person_frames[:10]:
        frame_idx = frame['frame']
        num_persons = frame['num_persons']
        
        # Calculate total valid joints
        total_valid_joints = 0
        for person in frame['persons']:
            joints = np.array(person['joints_3d'])
            valid_joints = sum(1 for conf in joints[:, 3] if conf > 0.3)
            total_valid_joints += valid_joints
        
        best_frames.append((frame_idx, num_persons, total_valid_joints))
    
    best_frames.sort(key=lambda x: x[2], reverse=True)  # Sort by total valid joints
    
    print(f"\nBest multi-person frames (by joint quality):")
    for frame_idx, num_persons, valid_joints in best_frames[:5]:
        print(f"  Frame {frame_idx}: {num_persons} persons, {valid_joints} valid joints")

if __name__ == "__main__":
    main()