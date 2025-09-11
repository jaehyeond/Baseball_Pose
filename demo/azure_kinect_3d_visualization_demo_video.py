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
from scipy import interpolate

# Global variables for person tracking
prev_centers = None  # [center_person0, center_person1]
color_palette = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']

# COCO-17 skeleton connections (demo.py 방식 성공)
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

def _apply_axis_transform(k3d, order='xyz', flip_z=False):
    """Apply coordinate axis transformation for proper orientation"""
    arr = np.asarray(k3d, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return arr
    
    # Axis order mapping
    idx_map = {
        'xyz': (0, 1, 2),
        'xzy': (0, 2, 1),
        'yxz': (1, 0, 2),
        'yzx': (1, 2, 0),
        'zxy': (2, 0, 1),
        'zyx': (2, 1, 0),
    }
    idx = idx_map.get(order, (0, 1, 2))
    xyz = arr[:, idx].copy()
    
    if flip_z:
        xyz[:, 2] = -xyz[:, 2]
    
    if arr.shape[1] > 3:
        arr2 = np.concatenate([xyz, arr[:, 3:4]], axis=1)
    else:
        arr2 = xyz
    return arr2

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

def calculate_global_bounds(all_results, axis_order='xyz', flip_z=False, kpt_thr=0.3, reference_frames=[31, 33]):
    """Calculate global bounds using specific reference frames for ground level"""
    all_points = []
    reference_foot_points = []  # Foot points from reference frames only
    
    # First, collect foot points from reference frames to establish ground level
    for frame_idx, frame_data in enumerate(all_results):
        if frame_idx in reference_frames:
            print(f"  Using frame {frame_idx} as ground reference")
            persons = frame_data.get('persons', [])
            for person in persons:
                arr = np.array(person.get('keypoints_3d', person.get('joints_3d')))
                arr = _apply_axis_transform(arr, order=axis_order, flip_z=flip_z)
                scores = (arr[:, 3] if arr.shape[1] > 3 else np.ones((arr.shape[0],), dtype=float))
                
                # Collect foot points (ankle keypoints: 15, 16 in COCO-17)
                foot_indices = [15, 16]  # left ankle, right ankle
                for fi in foot_indices:
                    if (fi < len(arr) and fi < len(scores) and 
                        scores[fi] > kpt_thr and np.isfinite(arr[fi]).all()):
                        reference_foot_points.append(arr[fi, :3])
                        print(f"    Frame {frame_idx} foot point: Z = {arr[fi, 2]:.3f}")
    
    # Calculate ground level from reference frames
    ground_level = 0.0
    if len(reference_foot_points) > 0:
        foot_pts = np.array(reference_foot_points)
        # Use median of reference foot Z coordinates for stability
        ground_level = np.median(foot_pts[:, 2])
        print(f"  Reference ground level from frames {reference_frames}: Z = {ground_level:.3f}")
    else:
        print(f"  Warning: No foot points found in reference frames {reference_frames}, using all frames")
        # Fallback to all frames if reference frames don't have good foot data
        for frame_data in all_results:
            persons = frame_data.get('persons', [])
            for person in persons:
                arr = np.array(person.get('keypoints_3d', person.get('joints_3d')))
                arr = _apply_axis_transform(arr, order=axis_order, flip_z=flip_z)
                scores = (arr[:, 3] if arr.shape[1] > 3 else np.ones((arr.shape[0],), dtype=float))
                
                foot_indices = [15, 16]
                for fi in foot_indices:
                    if (fi < len(arr) and fi < len(scores) and 
                        scores[fi] > kpt_thr and np.isfinite(arr[fi]).all()):
                        reference_foot_points.append(arr[fi, :3])
        
        if len(reference_foot_points) > 0:
            foot_pts = np.array(reference_foot_points)
            ground_level = np.percentile(foot_pts[:, 2], 10)
            print(f"  Fallback ground level: Z = {ground_level:.3f}")
    
    # Apply ground level adjustment to ALL frames
    print(f"  Applying ground adjustment {ground_level:.3f} to all frames...")
    print("  Note: Forward positioning will be applied during visualization")
    
    if ground_level != 0.0:
        for frame_idx, frame_data in enumerate(all_results):
            persons = frame_data.get('persons', [])
            for person_idx, person in enumerate(persons):
                keypoints_3d = person.get('keypoints_3d', [])
                if len(keypoints_3d) > 0:
                    # Convert to numpy array - work in original coordinate system first
                    arr = np.array(keypoints_3d)
                    
                    # Skip forward offset here - will be done in visualization
                    # if forward_offset != 0.0:
                    #     arr[:, 2] -= forward_offset  # Reduce Z = move toward camera
                    
                    # Now apply axis transformation for ground level adjustment
                    arr_transformed = _apply_axis_transform(arr, order=axis_order, flip_z=flip_z)
                    
                    # Apply ground level adjustment (in transformed coordinates)
                    if ground_level != 0.0:
                        arr_transformed[:, 2] -= ground_level  # Height adjustment
                    
                    # Convert back to original coordinate system
                    if flip_z:
                        arr_transformed[:, 2] = -arr_transformed[:, 2]
                    
                    if axis_order != 'xyz':
                        idx_map = {
                            'xyz': (0, 1, 2), 'xzy': (0, 2, 1), 'yxz': (1, 0, 2),
                            'yzx': (2, 0, 1), 'zxy': (1, 2, 0), 'zyx': (2, 1, 0),
                        }
                        forward_idx = idx_map.get(axis_order, (0, 1, 2))
                        reverse_order = [0, 0, 0]
                        for new_pos, orig_pos in enumerate(forward_idx):
                            reverse_order[orig_pos] = new_pos
                        arr_final = arr_transformed[:, reverse_order]
                    else:
                        arr_final = arr_transformed
                    
                    # Update the original data
                    all_results[frame_idx]['persons'][person_idx]['keypoints_3d'] = arr_final.tolist()
    
    # Now calculate bounds from adjusted data
    for frame_data in all_results:
        persons = frame_data.get('persons', [])
        for person in persons:
            arr = np.array(person.get('keypoints_3d', person.get('joints_3d')))
            arr = _apply_axis_transform(arr, order=axis_order, flip_z=flip_z)
            scores = (arr[:, 3] if arr.shape[1] > 3 else np.ones((arr.shape[0],), dtype=float))
            mask = (scores > kpt_thr) & np.isfinite(arr[:, :3]).all(axis=1)
            if np.any(mask):
                valid_points = arr[mask, :3]
                all_points.append(valid_points)
    
    if len(all_points) > 0:
        pts = np.concatenate(all_points, axis=0)
        
        # Calculate bounds with adjusted coordinates
        pmin = np.min(pts, axis=0)
        pmax = np.max(pts, axis=0)
        
        # Ensure Z minimum starts at ground level (0) or slightly below
        pmin[2] = min(pmin[2], -0.1)  # Allow slight below-ground for stability
        pmax[2] = max(pmax[2], 2.0)   # Ensure reasonable height ceiling
        
        center_point = (pmin + pmax) / 2.0
        max_range = max(pmax[0] - pmin[0], pmax[1] - pmin[1], pmax[2] - pmin[2])
        
        if max_range < 1.0:
            max_range = 1.0
        padding = max_range * 0.15  # Slightly less padding for tighter view
        range_meters = max_range + padding
        
        return center_point, range_meters
    else:
        return [0, 0, 1.0], 2.0

def draw_person(ax, kpts, color, radius=30, thickness=4):
    """Draw skeleton with proper filtering (no 0,0,0 joints and no NaN)"""
    if kpts.size == 0:
        return
    
    # Filter valid joints (not 0,0,0 and no NaN)
    valid_joints = ~((kpts[:, 0] == 0) & (kpts[:, 1] == 0) & (kpts[:, 2] == 0))
    valid_joints = valid_joints & np.isfinite(kpts).all(axis=1)
    
    # Draw valid joints as points
    if valid_joints.sum() > 0:
        valid_kpts = kpts[valid_joints]
        ax.scatter(valid_kpts[:, 0], valid_kpts[:, 1], valid_kpts[:, 2], 
                  s=radius, c=color, depthshade=True, alpha=0.8, marker='o')
    
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

def setup_3d_axes_safe(ax, center_point=None, range_meters=2.0, azim=45, elev=20):
    """Setup 3D axes with proper floor placement and ground reference"""
    if center_point is None:
        center_point = [0, 0, 1.0]
    
    # Use safe, finite limits with proper centering
    x_center, y_center, z_center = center_point
    half_range = range_meters / 2
    
    # Set axis limits with floor at Z=0 or below
    ax.set_xlim(x_center - half_range, x_center + half_range)
    ax.set_ylim(y_center - half_range, y_center + half_range)
    
    # Ensure Z axis shows ground level properly
    z_min = min(z_center - half_range, -0.1)  # Show some below-ground
    z_max = max(z_center + half_range, 2.0)   # Ensure reasonable ceiling
    ax.set_zlim(z_min, z_max)
    
    ax.set_xlabel('X (meters)', fontsize=10)
    ax.set_ylabel('Y (meters)', fontsize=10)
    ax.set_zlabel('Z (Height)', fontsize=10)
    
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, alpha=0.3)
    
    # Clean axes appearance
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)

def advanced_temporal_interpolation(all_results, axis_order='xyz', flip_z=False, max_gap=5, problematic_frames=[30, 31, 33]):
    """Advanced temporal interpolation - NO ground adjustment here, will be done globally later"""
    print("Starting advanced temporal interpolation for problematic frames...")
    print(f"Target frames: {problematic_frames}")
    
    # Skip ground level adjustment in interpolation - will be handled globally later
    print("  Skipping ground level adjustment in interpolation (will be applied globally)")
    
    # Build person trajectories using depth-based consistent tracking
    person_trajectories = {0: {}, 1: {}}  # front person (0), back person (1)
    
    # First pass: collect all good quality frames (excluding problematic ones)
    good_frames_data = {}
    for frame_idx, frame_data in enumerate(all_results):
        if frame_idx in problematic_frames:
            print(f"  Skipping problematic frame {frame_idx} in trajectory building")
            continue
            
        persons = frame_data.get('persons', [])
        if len(persons) < 1:  # Allow single person frames for better tracking
            continue
            
        # Convert and transform keypoints
        persons_k = []
        z_avgs = []
        for p in persons:
            k = ensure_j3(p.get('keypoints_3d', []))
            if k.size == 0:
                continue
            k = _apply_axis_transform(k, order=axis_order, flip_z=flip_z)
            
            # Calculate quality score
            valid_mask = np.isfinite(k[:, :3]).all(axis=1)
            quality_score = valid_mask.sum()
            
            if quality_score > 5:  # Need at least 5 valid joints
                persons_k.append(k)
                z_avg = np.mean(k[valid_mask, 2])
                z_avgs.append(z_avg)
            
        if len(persons_k) > 0:
            good_frames_data[frame_idx] = {
                'persons_k': persons_k,
                'z_avgs': z_avgs
            }
    
    # Second pass: assign consistent person IDs across all good frames
    print(f"Found {len(good_frames_data)} good quality frames for tracking")
    
    for frame_idx in sorted(good_frames_data.keys()):
        data = good_frames_data[frame_idx]
        persons_k = data['persons_k']
        z_avgs = data['z_avgs']
        
        if len(persons_k) == 2:
            # Two persons: assign by depth
            sorted_indices = np.argsort(z_avgs)
            person_trajectories[0][frame_idx] = persons_k[sorted_indices[0]]  # front (smaller Z)
            person_trajectories[1][frame_idx] = persons_k[sorted_indices[1]]  # back (larger Z)
            print(f"  Frame {frame_idx}: Front person Z={z_avgs[sorted_indices[0]]:.2f}, Back person Z={z_avgs[sorted_indices[1]]:.2f}")
            
        elif len(persons_k) == 1:
            # Single person: try to assign based on proximity to existing trajectory
            person_k = persons_k[0]
            z_avg = z_avgs[0]
            
            # Find which trajectory this person likely belongs to
            best_match_id = None
            best_distance = float('inf')
            
            for person_id in [0, 1]:
                trajectory = person_trajectories[person_id]
                if len(trajectory) == 0:
                    continue
                    
                # Find the most recent frame for this person
                recent_frames = [f for f in trajectory.keys() if f < frame_idx]
                if recent_frames:
                    recent_frame = max(recent_frames)
                    recent_pose = trajectory[recent_frame]
                    
                    # Calculate distance between current and recent pose
                    distance = np.mean(np.linalg.norm(person_k - recent_pose, axis=1))
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_match_id = person_id
            
            if best_match_id is not None and best_distance < 0.5:  # reasonable threshold
                person_trajectories[best_match_id][frame_idx] = person_k
                print(f"  Frame {frame_idx}: Assigned single person to trajectory {best_match_id} (distance: {best_distance:.3f})")
            else:
                # If we can't match, assign to the trajectory that would maintain depth consistency
                if z_avg < 3.0:  # probably front person
                    person_trajectories[0][frame_idx] = person_k
                    print(f"  Frame {frame_idx}: Assigned to front person by depth (Z={z_avg:.2f})")
                else:  # probably back person
                    person_trajectories[1][frame_idx] = person_k
                    print(f"  Frame {frame_idx}: Assigned to back person by depth (Z={z_avg:.2f})")
    
    # Now interpolate the problematic frames
    for target_frame in problematic_frames:
        print(f"\n  Interpolating frame {target_frame}:")
        interpolated_persons = []
        
        for person_id in [0, 1]:  # front, back
            person_name = "Front" if person_id == 0 else "Back"
            trajectory = person_trajectories[person_id]
            available_frames = sorted(trajectory.keys())
            
            if len(available_frames) < 2:
                print(f"    {person_name} person: Not enough trajectory data")
                continue
            
            # Find the best frames before and after target frame
            before_frames = [f for f in available_frames if f < target_frame]
            after_frames = [f for f in available_frames if f > target_frame]
            
            if not before_frames or not after_frames:
                print(f"    {person_name} person: No suitable before/after frames")
                continue
                
            # Use closest frames for interpolation
            before_frame = max(before_frames)
            after_frame = min(after_frames)
            gap = after_frame - before_frame
            
            # Be more permissive with gap size for continuity
            if gap > max_gap + 2:  # Allow slightly larger gaps
                print(f"    {person_name} person: Gap too large ({gap} > {max_gap + 2})")
                # Try to use a different frame pair if available
                alternative_before = [f for f in before_frames if f >= before_frame - 3]
                alternative_after = [f for f in after_frames if f <= after_frame + 3]
                
                if alternative_before and alternative_after:
                    before_frame = max(alternative_before)
                    after_frame = min(alternative_after)
                    gap = after_frame - before_frame
                    print(f"    {person_name} person: Using alternative frames {before_frame} -> {after_frame} (gap: {gap})")
                else:
                    continue
                
            print(f"    {person_name} person: Interpolating between frames {before_frame} and {after_frame}")
            
            # Perform cubic spline interpolation for smoother results
            before_pose = trajectory[before_frame]  # (17, 3)
            after_pose = trajectory[after_frame]    # (17, 3)
            
            # Check if we have more context frames for better interpolation
            context_frames = []
            context_poses = []
            
            # Add frames within reasonable distance for cubic spline
            for f in available_frames:
                if abs(f - target_frame) <= 4 and f != before_frame and f != after_frame:
                    context_frames.append(f)
                    context_poses.append(trajectory[f])
            
            # Perform joint-by-joint interpolation
            interpolated_pose = np.zeros((17, 3))
            
            for joint_idx in range(17):
                # Collect keyframes for this joint
                keyframe_times = [before_frame, after_frame]
                keyframe_poses = [before_pose[joint_idx], after_pose[joint_idx]]
                
                # Add context frames if available
                for cf, cp in zip(context_frames, context_poses):
                    keyframe_times.append(cf)
                    keyframe_poses.append(cp[joint_idx])
                
                # Sort by time
                sorted_indices = np.argsort(keyframe_times)
                keyframe_times = [keyframe_times[i] for i in sorted_indices]
                keyframe_poses = [keyframe_poses[i] for i in sorted_indices]
                
                # Interpolate each coordinate (X, Y, Z) separately
                for coord_idx in range(3):
                    coords = [pose[coord_idx] for pose in keyframe_poses]
                    
                    if len(keyframe_times) >= 2:
                        # Use scipy interpolation
                        from scipy import interpolate
                        if len(keyframe_times) > 3:
                            # Cubic spline for smooth motion
                            f_interp = interpolate.interp1d(keyframe_times, coords, kind='cubic', 
                                                          bounds_error=False, fill_value='extrapolate')
                        else:
                            # Linear interpolation for insufficient data
                            f_interp = interpolate.interp1d(keyframe_times, coords, kind='linear',
                                                          bounds_error=False, fill_value='extrapolate')
                        
                        interpolated_pose[joint_idx, coord_idx] = f_interp(target_frame)
                    else:
                        # Fallback: simple linear interpolation
                        alpha = (target_frame - before_frame) / (after_frame - before_frame)
                        interpolated_pose[joint_idx, coord_idx] = (
                            before_pose[joint_idx, coord_idx] * (1 - alpha) + 
                            after_pose[joint_idx, coord_idx] * alpha
                        )
            
            # Convert back to original coordinate system
            # First reverse flip_z
            if flip_z:
                interpolated_pose[:, 2] = -interpolated_pose[:, 2]
            
            # Then reverse axis transformation
            if axis_order != 'xyz':
                idx_map = {
                    'xyz': (0, 1, 2),
                    'xzy': (0, 2, 1),
                    'yxz': (1, 0, 2),
                    'yzx': (2, 0, 1),
                    'zxy': (1, 2, 0),
                    'zyx': (2, 1, 0),
                }
                forward_idx = idx_map.get(axis_order, (0, 1, 2))
                reverse_order = [0, 0, 0]
                for new_pos, orig_pos in enumerate(forward_idx):
                    reverse_order[orig_pos] = new_pos
                interpolated_pose = interpolated_pose[:, reverse_order]
            
            # No ground adjustment here - will be done globally later
            
            # Add to interpolated persons
            interpolated_persons.append({
                'keypoints_3d': interpolated_pose.tolist(),
                'scores': np.ones(17).tolist(),
                'interpolated': True,
                'person_id': person_id
            })
        
        # Replace the problematic frame's persons with interpolated ones
        if len(interpolated_persons) > 0:
            all_results[target_frame]['persons'] = interpolated_persons
            print(f"    Replaced frame {target_frame} with {len(interpolated_persons)} interpolated persons")
            
            # Ensure we always have both persons for continuity
            if len(interpolated_persons) == 1:
                print(f"    Warning: Only interpolated 1 person for frame {target_frame}, trying to add missing person...")
                
                # Find which person is missing
                existing_ids = [p['person_id'] for p in interpolated_persons]
                missing_id = 0 if 1 in existing_ids else 1
                missing_name = "Front" if missing_id == 0 else "Back"
                
                # Try to interpolate the missing person using nearby frames
                missing_trajectory = person_trajectories[missing_id]
                available_frames = sorted(missing_trajectory.keys())
                
                if len(available_frames) >= 2:
                    # Find frames close to target frame
                    nearby_before = [f for f in available_frames if f < target_frame and abs(f - target_frame) <= 6]
                    nearby_after = [f for f in available_frames if f > target_frame and abs(f - target_frame) <= 6]
                    
                    if nearby_before and nearby_after:
                        before_f = max(nearby_before)
                        after_f = min(nearby_after)
                        
                        # Simple linear interpolation for missing person
                        before_pose = missing_trajectory[before_f]
                        after_pose = missing_trajectory[after_f]
                        alpha = (target_frame - before_f) / (after_f - before_f)
                        missing_pose = before_pose * (1 - alpha) + after_pose * alpha
                        
                        # No ground adjustment here - will be done globally later
                        
                        # Convert back to original coordinate system
                        if flip_z:
                            missing_pose[:, 2] = -missing_pose[:, 2]
                        
                        if axis_order != 'xyz':
                            idx_map = {
                                'xyz': (0, 1, 2), 'xzy': (0, 2, 1), 'yxz': (1, 0, 2),
                                'yzx': (2, 0, 1), 'zxy': (1, 2, 0), 'zyx': (2, 1, 0),
                            }
                            forward_idx = idx_map.get(axis_order, (0, 1, 2))
                            reverse_order = [0, 0, 0]
                            for new_pos, orig_pos in enumerate(forward_idx):
                                reverse_order[orig_pos] = new_pos
                            missing_pose = missing_pose[:, reverse_order]
                        
                        # Add the missing person
                        all_results[target_frame]['persons'].append({
                            'keypoints_3d': missing_pose.tolist(),
                            'scores': np.ones(17).tolist(),
                            'interpolated': True,
                            'person_id': missing_id
                        })
                        
                        print(f"      Added missing {missing_name} person using frames {before_f}-{after_f}")
                    else:
                        print(f"      Could not find suitable frames for missing {missing_name} person")
        else:
            print(f"    Failed to interpolate frame {target_frame}")
            # As a last resort, try to copy from the nearest good frame
            if target_frame > 0:
                for offset in [1, 2, 3, -1, -2, -3]:
                    src_frame = target_frame + offset
                    if (0 <= src_frame < len(all_results) and 
                        src_frame not in problematic_frames and
                        len(all_results[src_frame].get('persons', [])) == 2):
                        
                        print(f"      Copying from frame {src_frame} as fallback")
                        all_results[target_frame]['persons'] = [
                            {**p, 'interpolated': True, 'fallback': True} 
                            for p in all_results[src_frame]['persons']
                        ]
                        break
    
    print("Advanced temporal interpolation completed.")
    return all_results

def apply_ground_normalization(persons_k, ground_level_adjustment=None):
    """Apply ground level normalization to place feet at Z=0"""
    if not persons_k or ground_level_adjustment is None:
        return persons_k
    
    normalized_persons = []
    for k in persons_k:
        if k.size == 0:
            normalized_persons.append(k)
            continue
            
        k_normalized = k.copy()
        k_normalized[:, 2] -= ground_level_adjustment  # Adjust Z coordinates
        normalized_persons.append(k_normalized)
    
    return normalized_persons

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
    
    # Process persons: convert + transform keypoints using improved transformation
    persons_k = []
    forward_view_offset = 0.8  # Move forward in viewing direction
    
    for p in persons:
        k = ensure_j3(p.get('keypoints_3d', []))
        if k.size == 0:
            continue
        k = _apply_axis_transform(k, order=axis_order, flip_z=flip_z)  # Use improved transformation
        
        # Move skeleton forward in viewing direction (Y axis in transformed coordinates)
        k[:, 1] -= forward_view_offset  # Move toward viewer
        
        persons_k.append(k)
    
    if len(persons_k) > 0:
        print(f"Frame {frame_idx}: Applied forward offset {forward_view_offset:.1f}m to {len(persons_k)} persons")
    
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
    
    # Calculate safe center point and range using improved method from visualize_final_3d_copy.py
    all_valid_points = []
    for k in persons_k:
        valid_mask = ~((k[:, 0] == 0) & (k[:, 1] == 0) & (k[:, 2] == 0))
        valid_mask = valid_mask & np.isfinite(k).all(axis=1)
        if valid_mask.sum() > 0:
            all_valid_points.append(k[valid_mask])
    
    # Use global limits if available, otherwise calculate frame-specific
    if global_limits and len(global_limits) == 2:
        center_point, range_meters = global_limits
    elif len(all_valid_points) > 0:
        pts = np.concatenate(all_valid_points, axis=0)
        pmin = np.min(pts, axis=0)
        pmax = np.max(pts, axis=0)
        center_point = (pmin + pmax) / 2.0
        max_range = max(pmax[0] - pmin[0], pmax[1] - pmin[1], pmax[2] - pmin[2])
        if max_range < 1.0:
            max_range = 1.0
        padding = max_range * 0.2
        range_meters = max_range + padding
    else:
        center_point = [0, 0, 1.0]
        range_meters = 2.0
    
    # Setup axes using improved method
    setup_3d_axes_safe(ax, center_point, range_meters, azim=azim, elev=elev)
    
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
        results = advanced_temporal_interpolation(results, args.axis_order, args.flip_z, 
                                                problematic_frames=[30, 31, 33])
        frames_to_process = results
    
    # Calculate global limits using frames 31 and 33 as ground reference
    print("Calculating global bounds using frames 31 and 33 as ground reference...")
    global_center, global_range = calculate_global_bounds(
        frames_to_process, args.axis_order, args.flip_z, reference_frames=[31, 33]
    )
    global_limits = (global_center, global_range)
    
    print(f"Processing all frames: 0 to {len(results) - 1}")
    print(f"Global center: {global_center}, Global range: {global_range}")
    
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