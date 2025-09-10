#!/usr/bin/env python3
"""
Debug script to understand why triangulation only produces 1 person per frame
"""
import json
import numpy as np
from pathlib import Path

def load_2d_poses_from_json(json_path):
    """Load 2D poses from MMPose output JSON"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    instances_per_frame = []
    for frame_data in data['instance_info']:
        frame_idx = frame_data['frame_id']
        instances = frame_data['instances']
        instances_per_frame.append({
            'frame_idx': frame_idx,
            'num_instances': len(instances),
            'instances': instances
        })
    
    return instances_per_frame

def main():
    # Load both camera poses
    cam00_data = load_2d_poses_from_json("D:/mmpose/poses_2d_cam00/results_output_video.json")
    cam01_data = load_2d_poses_from_json("D:/mmpose/poses_2d_cam01/results_output_video.json")
    
    print(f"Loaded {len(cam00_data)} frames from cam00")
    print(f"Loaded {len(cam01_data)} frames from cam01")
    
    # Analyze frame by frame
    both_have_2_persons = 0
    cam00_only_1 = 0
    cam01_only_1 = 0
    other_cases = 0
    
    print("\nFrame analysis:")
    for i in range(min(len(cam00_data), len(cam01_data))):
        c00 = cam00_data[i]['num_instances']
        c01 = cam01_data[i]['num_instances']
        
        if c00 == 2 and c01 == 2:
            both_have_2_persons += 1
        elif c00 == 1 and c01 >= 2:
            cam00_only_1 += 1
        elif c01 == 1 and c00 >= 2:
            cam01_only_1 += 1
        else:
            other_cases += 1
        
        if i < 20:  # Show first 20 frames
            print(f"Frame {i}: cam00={c00} instances, cam01={c01} instances")
    
    print(f"\nSummary:")
    print(f"Both cameras have 2 persons: {both_have_2_persons} frames")
    print(f"Cam00 has 1, Cam01 has ≥2: {cam00_only_1} frames")
    print(f"Cam01 has 1, Cam00 has ≥2: {cam01_only_1} frames")
    print(f"Other cases: {other_cases} frames")
    
    # Find frames where both cameras reliably detect 2 people
    good_frames = []
    for i in range(min(len(cam00_data), len(cam01_data))):
        if cam00_data[i]['num_instances'] == 2 and cam01_data[i]['num_instances'] == 2:
            good_frames.append(i)
    
    print(f"\nFrames with exactly 2 people in both cameras: {len(good_frames)}")
    if len(good_frames) > 0:
        print(f"First few good frames: {good_frames[:10]}")
        
        # Test with a single good frame
        test_frame = good_frames[50] if len(good_frames) > 50 else good_frames[0]
        print(f"\nTesting frame {test_frame}:")
        
        c00_instances = cam00_data[test_frame]['instances']
        c01_instances = cam01_data[test_frame]['instances']
        
        print(f"Cam00 instances: {len(c00_instances)}")
        for idx, inst in enumerate(c00_instances):
            kpts = np.array(inst['keypoints']).reshape(-1, 2)  # 2D keypoints are (x,y) pairs
            bbox = inst['bbox'][0] if isinstance(inst['bbox'][0], list) else inst['bbox']
            scores = inst['keypoint_scores']
            print(f"  Person {idx}: bbox={bbox}, keypoints shape={kpts.shape}, avg_score={np.mean(scores):.3f}")
            
        print(f"Cam01 instances: {len(c01_instances)}")
        for idx, inst in enumerate(c01_instances):
            kpts = np.array(inst['keypoints']).reshape(-1, 2)  # 2D keypoints are (x,y) pairs
            bbox = inst['bbox'][0] if isinstance(inst['bbox'][0], list) else inst['bbox']
            scores = inst['keypoint_scores']
            print(f"  Person {idx}: bbox={bbox}, keypoints shape={kpts.shape}, avg_score={np.mean(scores):.3f}")

if __name__ == "__main__":
    main()