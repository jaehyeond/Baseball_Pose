#!/usr/bin/env python3
"""
Fix scale issues in triangulate_video.py results and visualize properly
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

def fix_scale_and_format(data):
    """Fix scale issues and convert to vis3d_front format"""
    
    # Analyze current scale by looking at a few good frames
    sample_coords = []
    for frame in data['results'][:20]:
        if frame['num_persons'] >= 2:
            for person in frame['persons']:
                kpts_3d = np.array(person['keypoints_3d'])
                scores = np.array(person['scores'])
                
                # Use high-confidence joints
                valid_mask = scores > 0.5
                if np.sum(valid_mask) > 10:
                    valid_coords = kpts_3d[valid_mask]
                    sample_coords.extend(valid_coords.tolist())
    
    if len(sample_coords) > 50:
        sample_coords = np.array(sample_coords)
        
        # Estimate scale based on Z-coordinate range (should be around 1.5-2.5m for standing people)
        z_range = np.ptp(sample_coords[:, 2])
        current_z_mean = np.mean(sample_coords[:, 2])
        
        print(f"Current Z range: {z_range:.3f}m, mean Z: {current_z_mean:.3f}m")
        
        # Ideal Z should be around 1.5-2.5m for people
        if current_z_mean < 1.2:  # Scale is too small
            scale_factor = 2.0 / current_z_mean  # Scale to put people at ~2m height
            print(f"Applying scale factor: {scale_factor:.2f}")
        else:
            scale_factor = 1.0
            print("Scale appears correct")
    else:
        # Default scale factor based on previous observation
        scale_factor = 10.0
        print(f"Not enough data for auto-scale, using factor: {scale_factor}")
    
    # Apply scale correction and format conversion
    fixed_results = []
    
    for frame in data['results']:
        fixed_frame = {
            "frame_idx": frame['frame'],
            "persons": []
        }
        
        for person_idx, person in enumerate(frame['persons']):
            kpts_3d = np.array(person['keypoints_3d'])
            scores = np.array(person['scores'])
            
            # Apply scale correction
            kpts_3d_scaled = kpts_3d * scale_factor
            
            # Convert to vis3d_front format: [x, y, z, confidence]
            kpts_3d_with_conf = []
            for j in range(17):
                kpts_3d_with_conf.append([
                    float(kpts_3d_scaled[j, 0]),
                    float(kpts_3d_scaled[j, 1]), 
                    float(kpts_3d_scaled[j, 2]),
                    float(scores[j])
                ])
            
            fixed_person = {
                "pair": {"cam1_idx": person_idx, "cam2_idx": person_idx},
                "keypoints_3d": kpts_3d_with_conf
            }
            
            fixed_frame["persons"].append(fixed_person)
        
        if len(fixed_frame["persons"]) > 0:
            fixed_results.append(fixed_frame)
    
    return fixed_results

def draw_skeleton_3d_correct(ax, kpts_3d_with_conf, color='red', alpha=0.8, point_size=50, conf_threshold=0.3):
    """Draw skeleton using vis3d_front format: [x,y,z,conf]"""
    if len(kpts_3d_with_conf) != 17:
        return 0, 0
    
    kpts_3d_with_conf = np.array(kpts_3d_with_conf)
    
    # Extract coordinates and confidence
    coords = kpts_3d_with_conf[:, :3]  # x, y, z
    confs = kpts_3d_with_conf[:, 3]    # confidence
    
    # Find valid joints
    valid_joints = []
    for i in range(17):
        if (confs[i] > conf_threshold and 
            np.isfinite(coords[i, 0]) and 
            np.isfinite(coords[i, 1]) and 
            np.isfinite(coords[i, 2])):
            valid_joints.append(i)
    
    print(f"    Drawing {len(valid_joints)} valid joints with {color}")
    
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
    
    print(f"    Drew {connections_drawn} connections")
    return len(valid_joints), connections_drawn

def visualize_corrected_frame(frame_data, frame_idx, output_dir):
    """Visualize corrected frame"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    persons = frame_data.get('persons', [])
    num_persons = len(persons)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple'][:num_persons]
    
    print(f"Frame {frame_idx}: {num_persons} persons")
    
    # Calculate reasonable center and range
    all_valid_points = []
    for person in persons:
        kpts_data = np.array(person['keypoints_3d'])
        coords = kpts_data[:, :3]
        confs = kpts_data[:, 3]
        
        valid_mask = (confs > 0.3) & np.isfinite(coords).all(axis=1)
        if np.any(valid_mask):
            all_valid_points.extend(coords[valid_mask].tolist())
    
    if len(all_valid_points) > 5:
        all_valid_points = np.array(all_valid_points)
        center_point = np.mean(all_valid_points, axis=0)
        range_meters = max(2.5, np.max(np.ptp(all_valid_points, axis=0)) * 1.2)
    else:
        center_point = [0, 0, 2.0]
        range_meters = 2.5
    
    # Draw each person
    total_joints = 0
    total_connections = 0
    
    for i, person in enumerate(persons):
        kpts_3d_with_conf = person['keypoints_3d']
        color = colors[i % len(colors)]
        
        print(f"  Drawing Person {i+1} with {color}:")
        valid_joints, connections = draw_skeleton_3d_correct(
            ax, kpts_3d_with_conf, color=color, alpha=0.8
        )
        
        total_joints += valid_joints
        total_connections += connections
        
        # Add label
        coords = np.array(kpts_3d_with_conf)[:, :3]
        confs = np.array(kpts_3d_with_conf)[:, 3]
        
        valid_mask = (confs > 0.3) & np.isfinite(coords).all(axis=1)
        if np.any(valid_mask):
            valid_coords = coords[valid_mask]
            head_point = valid_coords[np.argmax(valid_coords[:, 2])]  # Highest point
            
            ax.text(head_point[0], head_point[1], head_point[2] + 0.2,
                   f'Person {i+1}', fontsize=12, color=color, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
    
    # Set up axes like vis3d_front
    ax.set_xlim(center_point[0] - range_meters/2, center_point[0] + range_meters/2)
    ax.set_ylim(center_point[1] - range_meters/2, center_point[1] + range_meters/2)
    ax.set_zlim(max(0.1, center_point[2] - range_meters/2), center_point[2] + range_meters/2)
    
    ax.set_xlabel('X (meters)', fontsize=10)
    ax.set_ylabel('Y (meters)', fontsize=10)
    ax.set_zlabel('Z (meters)', fontsize=10)
    
    ax.view_init(elev=15, azim=45)  # Similar to vis3d_front
    ax.grid(True, alpha=0.3)
    
    # Clean axes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    title = f'3D Pose (Frame {frame_idx}) | Persons: {num_persons} | {total_connections} Connections'
    plt.title(title, fontsize=14, weight='bold', pad=20)
    
    # Save
    output_path = Path(output_dir) / f'corrected_frame_{frame_idx:04d}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    return output_path, total_joints, total_connections

def main():
    # Load triangulate_video.py results
    with open("D:/mmpose/output_3d/full_video.json", 'r') as f:
        data = json.load(f)
    
    print("Fixing scale and format issues...")
    fixed_results = fix_scale_and_format(data)
    
    output_dir = Path("D:/mmpose/vis3d_corrected_scale")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Visualizing {len(fixed_results)} corrected frames")
    
    # Find best multi-person frames
    best_frames = []
    for frame in fixed_results:
        frame_idx = frame['frame_idx']
        num_persons = len(frame['persons'])
        
        if num_persons >= 2:  # Focus on multi-person frames
            # Count total valid joints
            total_valid_joints = 0
            for person in frame['persons']:
                kpts_data = np.array(person['keypoints_3d'])
                confs = kpts_data[:, 3]
                valid_joints = sum(1 for conf in confs if conf > 0.3)
                total_valid_joints += valid_joints
            
            best_frames.append((frame_idx, num_persons, total_valid_joints))
    
    # Sort by quality
    best_frames.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\nTop multi-person frames:")
    for frame_idx, num_persons, valid_joints in best_frames[:10]:
        print(f"  Frame {frame_idx}: {num_persons} persons, {valid_joints} valid joints")
    
    # Process top 5 frames
    total_joints = 0
    total_connections = 0
    
    for i, (frame_idx, _, _) in enumerate(best_frames[:5]):
        # Find frame data
        frame_data = None
        for frame in fixed_results:
            if frame['frame_idx'] == frame_idx:
                frame_data = frame
                break
        
        if frame_data:
            output_path, joints, connections = visualize_corrected_frame(frame_data, frame_idx, output_dir)
            total_joints += joints
            total_connections += connections
            
            print(f"Processed corrected frame {frame_idx} ({i+1}/5)")
    
    print(f"\nCorrected visualization complete!")
    print(f"Average joints per frame: {total_joints/5:.1f}")
    print(f"Average connections per frame: {total_connections/5:.1f}")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()