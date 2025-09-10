#!/usr/bin/env python3
"""
Rename jpg files in Camera1 and Camera2 directories to sequential numbering
starting from 00000.jpg
"""
import os
import glob
from pathlib import Path

def rename_files_in_directory(directory_path):
    """Rename all jpg files in directory to sequential numbering starting from 00000.jpg"""
    
    # Get all jpg files and sort them
    jpg_files = glob.glob(os.path.join(directory_path, "*.jpg"))
    jpg_files.sort()  # Sort to maintain order
    
    print(f"\nProcessing directory: {directory_path}")
    print(f"Found {len(jpg_files)} jpg files")
    
    # Rename files
    for i, old_path in enumerate(jpg_files):
        new_filename = f"{i:05d}.jpg"  # 5-digit zero-padded number
        new_path = os.path.join(directory_path, new_filename)
        
        # Skip if already has correct name
        if os.path.basename(old_path) == new_filename:
            print(f"  {os.path.basename(old_path)} -> {new_filename} (already correct)")
            continue
            
        # Check if new filename already exists
        if os.path.exists(new_path):
            print(f"  Warning: {new_filename} already exists, skipping {os.path.basename(old_path)}")
            continue
            
        # Rename file
        os.rename(old_path, new_path)
        print(f"  {os.path.basename(old_path)} -> {new_filename}")
    
    print(f"Completed processing {directory_path}")

def main():
    # Define directories
    camera1_dir = "D:/mmpose/frames/Camera1"
    camera2_dir = "D:/mmpose/frames/Camera2"
    
    # Check if directories exist
    if not os.path.exists(camera1_dir):
        print(f"Error: Directory {camera1_dir} does not exist")
        return
        
    if not os.path.exists(camera2_dir):
        print(f"Error: Directory {camera2_dir} does not exist")
        return
    
    print("Starting file renaming process...")
    
    # Process both directories
    rename_files_in_directory(camera1_dir)
    rename_files_in_directory(camera2_dir)
    
    print("\nFile renaming completed!")
    
    # Verify results
    print("\nVerification:")
    for dir_path, dir_name in [(camera1_dir, "Camera1"), (camera2_dir, "Camera2")]:
        jpg_files = sorted(glob.glob(os.path.join(dir_path, "*.jpg")))
        print(f"{dir_name}: {len(jpg_files)} files")
        if jpg_files:
            print(f"  First: {os.path.basename(jpg_files[0])}")
            print(f"  Last: {os.path.basename(jpg_files[-1])}")

if __name__ == "__main__":
    main()