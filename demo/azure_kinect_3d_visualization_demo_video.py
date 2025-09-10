#!/usr/bin/env python3
"""
Improved 3D visualization based on azure_kinect_3d_visualization_demo.py
Handles occlusion better with stable person tracking and proper sizing.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import argparse

# Global variables for person tracking
prev_centers = None  # [center_person0, center_person1]
color_palette = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']

# COCO-17 skeleton connections (demo.py 방식)
SKELETON_EDGES = [
    (0,1),(1,2),(2,3),(3,4),
    (1,5),(5,7),(7,9),
    (1,6),(6,8),(8,10),
    (5,6),(5,11),(6,12),
    (11,12),(11,13),(13,15),
    (12,14),(14,16)
]

def ensure_j3(kpts):
    """Convert any format to (J,3) shape robustly"""
    k = np.array(kpts, dtype=np.float32)
    if k.ndim == 1 and (k.size % 3 == 0):
        k = k.reshape(-1, 3)
    elif k.ndim == 2 and k.shape[1] == 3:
        pass
    elif k.ndim == 2 and k.shape[0] == 3:
        k = k.T
    elif k.ndim == 2 and k.shape[1] > 3:
        k = k[:, :3]
    else:
        return np.empty((0, 3), dtype=np.float32)
    return k

def person_centroid(kpts):
    """Calculate person center from valid joints (ignoring NaN)"""
    return np.nanmean(kpts, axis=0)

def axis_indices(order):
    """Convert axis order string to indices"""
    idx = {'x': 0, 'y': 1, 'z': 2}
    order = order.lower()
    if len(order) != 3 or any(c not in 'xyz' for c in order):
        raise ValueError(f'Invalid axis order: {order}')
    return [idx[c] for c in order]

def permute_kpts(kpts, order_idx):
    """Reorder coordinate axes"""
    k = ensure_j3(kpts)
    if k.size == 0:
        return k
    return k[:, order_idx]

def flip_axis_kpts(kpts, flip_z=False):
    """Flip Z axis if needed"""
    k = kpts.copy()
    if flip_z and k.size > 0:
        k[:, 2] = -k[:, 2]
    return k

def sort_two_by_depth(persons_k, depth_axis='y', front_is_min=True):
    """Sort two persons by depth (front/back)"""
    axis = {'x': 0, 'y': 1, 'z': 2}[depth_axis.lower()]
    depths = [np.nanmean(k[:, axis]) for k in persons_k]
    order = np.argsort(depths)
    if not front_is_min:
        order = order[::-1]
    return list(order)

def stable_assign_two(prev_centers, curr_centers):
    """Stable assignment for two persons - prevents ID swapping"""
    if len(prev_centers) != 2 or len(curr_centers) != 2:
        return list(range(len(curr_centers)))
    
    # Calculate total distance for both possible assignments
    d00 = np.linalg.norm(prev_centers[0] - curr_centers[0])
    d11 = np.linalg.norm(prev_centers[1] - curr_centers[1])
    sum_a = d00 + d11  # Keep current assignment
    
    d01 = np.linalg.norm(prev_centers[0] - curr_centers[1])
    d10 = np.linalg.norm(prev_centers[1] - curr_centers[0])
    sum_b = d01 + d10  # Swap assignment
    
    return [0, 1] if sum_a <= sum_b else [1, 0]

def compute_global_limits_adaptive(frames, axis_order_idx, flip_z=False):
    """Calculate adaptive axis limits for consistent visualization"""
    xs, ys, zs = [], [], []
    
    for fr in frames:
        persons = fr.get('persons', [])
        if not persons and 'keypoints_3d' in fr:
            persons = [{'keypoints_3d': fr['keypoints_3d']}]
        
        for p in persons:
            k = permute_kpts(p.get('keypoints_3d', []), axis_order_idx)
            k = flip_axis_kpts(k, flip_z)
            if k.size == 0:
                continue
            
            # Collect valid coordinates (not 0,0,0)
            valid_mask = ~((k[:, 0] == 0) & (k[:, 1] == 0) & (k[:, 2] == 0))
            if valid_mask.sum() > 0:
                valid_k = k[valid_mask]
                xs.extend(valid_k[:, 0])
                ys.extend(valid_k[:, 1])
                zs.extend(valid_k[:, 2])

    if not xs:
        return (-1, 1), (-1, 1), (-1, 1)

    # Calculate data range
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    z_min, z_max = min(zs), max(zs)
    
    # Center points
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2
    
    # Maximum range
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    
    # Minimum range guarantee
    if max_range < 1.0:
        max_range = 1.0
    
    # Add padding
    padding = max_range * 0.3
    half_range = (max_range + padding) / 2
    
    x_lim = (x_center - half_range, x_center + half_range)
    y_lim = (y_center - half_range, y_center + half_range)
    z_lim = (z_center - half_range, z_center + half_range)
    
    return x_lim, y_lim, z_lim

def draw_person(ax, kpts, color, radius=30, thickness=4):
    """Draw skeleton with proper filtering (no 0,0,0 joints)"""
    if kpts.size == 0:
        return
    
    # Filter valid joints (not 0,0,0)
    valid_joints = ~((kpts[:, 0] == 0) & (kpts[:, 1] == 0) & (kpts[:, 2] == 0))
    
    # Draw valid joints as points
    if valid_joints.sum() > 0:
        valid_kpts = kpts[valid_joints]
        ax.scatter(valid_kpts[:, 0], valid_kpts[:, 1], valid_kpts[:, 2], 
                  s=radius, c=color, depthshade=True, alpha=0.8)
    
    # Draw skeleton connections (only if both joints are valid)
    for i, j in SKELETON_EDGES:
        if (i < len(kpts) and j < len(kpts) and 
            valid_joints[i] and valid_joints[j]):
            xs = [kpts[i, 0], kpts[j, 0]]
            ys = [kpts[i, 1], kpts[j, 1]]
            zs = [kpts[i, 2], kpts[j, 2]]
            ax.plot(xs, ys, zs, linewidth=thickness, c=color, alpha=0.8)

def setup_axes(ax, x_lim, y_lim, z_lim, elev, azim, labels=('X', 'Y', 'Z')):
    """Setup 3D axes with proper proportions"""
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)
    ax.set_box_aspect((x_lim[1]-x_lim[0], y_lim[1]-y_lim[0], z_lim[1]-z_lim[0]))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.grid(True, alpha=0.3)

def temporal_interpolation_simple(all_results, order_idx, flip_z=False, max_gap=3):
    """Simple temporal interpolation for missing keypoints"""
    print("Applying simple temporal interpolation...")
    
    # Build person trajectories
    person_trajectories = {}
    
    for frame_idx, frame_data in enumerate(all_results):
        persons = frame_data.get('persons', [])
        
        # Convert and transform keypoints
        persons_k = []
        for p in persons:
            k = ensure_j3(p.get('keypoints_3d', []))
            if k.size == 0:
                continue
            k = k[:, order_idx]  # Reorder axes
            k = flip_axis_kpts(k, flip_z)  # Flip Z if needed
            persons_k.append(k)
        
        if len(persons_k) == 0:
            continue
            
        # Calculate centers for tracking
        centers = [person_centroid(k) for k in persons_k]
        
        # Assign person IDs (simple approach)
        if frame_idx == 0:
            # First frame: assign based on depth
            person_trajectories = {i: {} for i in range(len(persons_k))}
            for i, k in enumerate(persons_k):
                person_trajectories[i][frame_idx] = k
        else:
            # Match with previous frame
            if len(centers) == 2 and len(person_trajectories) == 2:
                prev_centers_list = []
                for pid in sorted(person_trajectories.keys()):
                    if frame_idx - 1 in person_trajectories[pid]:
                        prev_centers_list.append(person_centroid(person_trajectories[pid][frame_idx - 1]))
                
                if len(prev_centers_list) == 2:
                    assignment = stable_assign_two(prev_centers_list, centers)
                    for i, assigned_idx in enumerate(assignment):
                        person_trajectories[i][frame_idx] = persons_k[assigned_idx]
                else:
                    # Fallback: assign by order
                    for i, k in enumerate(persons_k):
                        if i in person_trajectories:
                            person_trajectories[i][frame_idx] = k
            else:
                # Handle variable number of persons
                for i, k in enumerate(persons_k):
                    if i not in person_trajectories:
                        person_trajectories[i] = {}
                    person_trajectories[i][frame_idx] = k
    
    # Fill gaps with linear interpolation
    for person_id, trajectory in person_trajectories.items():
        frame_indices = sorted(trajectory.keys())
        
        if len(frame_indices) < 2:
            continue
        
        for i in range(len(frame_indices) - 1):
            start_frame = frame_indices[i]
            end_frame = frame_indices[i + 1]
            gap = end_frame - start_frame
            
            if 1 < gap <= max_gap:
                print(f"  Interpolating person {person_id}: frames {start_frame} -> {end_frame}")
                
                start_kpts = trajectory[start_frame]
                end_kpts = trajectory[end_frame]
                
                # Linear interpolation for missing frames
                for mid_frame in range(start_frame + 1, end_frame):
                    alpha = (mid_frame - start_frame) / gap
                    interpolated_kpts = start_kpts * (1 - alpha) + end_kpts * alpha
                    
                    # Convert back to original format and add to results
                    # Reverse transformations
                    k_reversed = interpolated_kpts.copy()
                    if flip_z:
                        k_reversed[:, 2] = -k_reversed[:, 2]
                    
                    # Reverse axis order
                    reverse_order = [0, 0, 0]
                    for new_pos, orig_pos in enumerate(order_idx):
                        reverse_order[orig_pos] = new_pos
                    k_original = k_reversed[:, reverse_order]
                    
                    # Add interpolated person to frame
                    if 'persons' not in all_results[mid_frame]:
                        all_results[mid_frame]['persons'] = []
                    
                    all_results[mid_frame]['persons'].append({
                        'keypoints_3d': k_original.tolist(),
                        'scores': np.ones(k_original.shape[0]).tolist(),
                        'interpolated': True,
                        'person_id': person_id
                    })
    
    return all_results

def visualize_frame_improved(frame_data, frame_idx, output_dir,
                           azim=-45, elev=20, axis_order='xzy', flip_z=True,
                           depth_axis='y', front_is_min=True,
                           global_limits=None):
    """Improved frame visualization with stable person tracking"""
    global prev_centers
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    persons = frame_data.get('persons', [])
    if not persons and 'keypoints_3d' in frame_data:
        persons = [{'keypoints_3d': frame_data['keypoints_3d']}]
    
    # Convert axis order to indices
    order_idx = axis_indices(axis_order)
    
    # Process persons: convert + transform keypoints
    persons_k = []
    for p in persons:
        k = ensure_j3(p.get('keypoints_3d', []))
        if k.size == 0:
            continue
        k = k[:, order_idx]  # Reorder axes
        k = flip_axis_kpts(k, flip_z)  # Flip Z if needed
        persons_k.append(k)
    
    if len(persons_k) == 0:
        print(f"Frame {frame_idx}: No valid persons")
        plt.close(fig)
        return None
    
    # Stable person assignment for consistent colors
    draw_order = list(range(len(persons_k)))
    if len(persons_k) == 2:
        curr_centers = [person_centroid(k) for k in persons_k]
        
        if prev_centers is None:
            # First frame: sort by depth
            order = sort_two_by_depth(persons_k, depth_axis=depth_axis, front_is_min=front_is_min)
            draw_order = order  # order[0]=front(RED), order[1]=back(BLUE)
            prev_centers = [curr_centers[order[0]], curr_centers[order[1]]]
        else:
            # Subsequent frames: stable assignment
            assignment = stable_assign_two(prev_centers, curr_centers)
            draw_order = assignment  # assignment[0]=RED, assignment[1]=BLUE
            prev_centers = [curr_centers[assignment[0]], curr_centers[assignment[1]]]
    
    # Use frame-specific limits instead of global limits for larger view
    if global_limits:
        # Calculate frame-specific limits for this frame only
        all_points = []
        for k in persons_k:
            valid_mask = ~((k[:, 0] == 0) & (k[:, 1] == 0) & (k[:, 2] == 0))
            valid_mask = valid_mask & ~np.isnan(k).any(axis=1)
            if valid_mask.sum() > 0:
                all_points.append(k[valid_mask])
        
        if all_points:
            pts = np.concatenate(all_points, axis=0)
            # Use percentile to exclude outliers
            x_min, x_max = np.percentile(pts[:, 0], [2, 98])
            y_min, y_max = np.percentile(pts[:, 1], [2, 98])
            z_min, z_max = np.percentile(pts[:, 2], [2, 98])
            
            # Small padding
            x_pad = max((x_max - x_min) * 0.15, 0.1)
            y_pad = max((y_max - y_min) * 0.15, 0.1)
            z_pad = max((z_max - z_min) * 0.15, 0.1)
            
            x_lim = (x_min - x_pad, x_max + x_pad)
            y_lim = (y_min - y_pad, y_max + y_pad)
            z_lim = (z_min - z_pad, z_max + z_pad)
        else:
            x_lim = y_lim = z_lim = (-1, 1)
    else:
        # Calculate frame-specific limits
        all_points = []
        for k in persons_k:
            valid_mask = ~((k[:, 0] == 0) & (k[:, 1] == 0) & (k[:, 2] == 0))
            if valid_mask.sum() > 0:
                all_points.append(k[valid_mask])
        
        if all_points:
            pts = np.concatenate(all_points, axis=0)
            margin = 0.5
            x_lim = (pts[:, 0].min() - margin, pts[:, 0].max() + margin)
            y_lim = (pts[:, 1].min() - margin, pts[:, 1].max() + margin)
            z_lim = (pts[:, 2].min() - margin, pts[:, 2].max() + margin)
        else:
            x_lim = y_lim = z_lim = (-1, 1)
    
    # Setup axes
    setup_axes(ax, x_lim, y_lim, z_lim, elev=elev, azim=azim, labels=('X', 'Y', 'Z'))
    
    # Draw persons with consistent colors
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00']  # red, blue, green, orange
    total_connections = 0
    
    if len(persons_k) == 2:
        # Two persons: use stable color assignment
        draw_person(ax, persons_k[draw_order[0]], colors[0], radius=30, thickness=4)  # RED (front)
        draw_person(ax, persons_k[draw_order[1]], colors[1], radius=30, thickness=4)  # BLUE (back)
        total_connections = len(SKELETON_EDGES) * 2
    else:
        # Variable number of persons
        for i, k in enumerate(persons_k):
            color = colors[i] if i < len(colors) else '#999999'
            draw_person(ax, k, color, radius=80, thickness=8)
        total_connections = len(SKELETON_EDGES) * len(persons_k)
    
    # Title
    title = f'Video 3D Pose (Frame {frame_idx}) | {len(persons_k)} Persons | {total_connections} Connections'
    plt.title(title, fontsize=14, weight='bold', pad=20)
    
    # Save
    output_path = Path(output_dir) / f'final_frame_{frame_idx:04d}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return output_path

def main():
    global prev_centers
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to 3D JSON results')
    parser.add_argument('--out-dir', required=True, help='Directory to save PNGs')
    parser.add_argument('--all-frames', action='store_true', help='Process all frames')
    parser.add_argument('--azim', type=float, default=135, help='Camera azimuth angle')
    parser.add_argument('--elev', type=float, default=20, help='Camera elevation angle')
    parser.add_argument('--axis-order', type=str, default='xzy', help='Axis order (e.g., xzy, xyz, yxz)')
    parser.add_argument('--flip-z', action='store_true', help='Flip Z axis for proper orientation')
    parser.add_argument('--depth-axis', choices=['x','y','z'], default='y', help='Axis for front/back determination')
    parser.add_argument('--front-is-min-depth', action='store_true', help='Smaller depth = closer to camera')
    parser.add_argument('--interpolate', action='store_true', help='Apply temporal interpolation')
    
    args = parser.parse_args()
    
    # demo.py와 동일한 로직
    if not args.all_frames:
        print("Use --all-frames to process all frames")
        return
    
    # Initialize tracking
    prev_centers = None
    
    # Load data
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = data['results']
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all frames
    frames_to_process = results
    
    # Apply interpolation if requested
    if args.interpolate:
        order_idx = axis_indices(args.axis_order)
        results = temporal_interpolation_simple(results, order_idx, args.flip_z)
        frames_to_process = results
    
    # Calculate global limits for consistent visualization
    order_idx = axis_indices(args.axis_order)
    global_limits = compute_global_limits_adaptive(frames_to_process, order_idx, args.flip_z)
    
    print(f"Processing all frames: 0 to {len(results) - 1}")
    print(f"Global limits: X{global_limits[0]}, Y{global_limits[1]}, Z{global_limits[2]}")
    
    # Process each frame
    for idx in range(len(results)):
        frame = results[idx]
        frame_idx = frame.get('frame', frame.get('frame_index', idx))
        
        output_path = visualize_frame_improved(
            frame, frame_idx, out_dir,
            azim=args.azim, elev=args.elev,
            axis_order=args.axis_order, flip_z=args.flip_z,
            depth_axis=args.depth_axis, front_is_min=args.front_is_min_depth,
            global_limits=global_limits
        )
        
        if output_path:
            print(f"Saved: {output_path}")
    
    print(f'Completed: {len(results)} frames saved to {out_dir}')

if __name__ == "__main__":
    main()