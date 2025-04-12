#!/usr/bin/env python3

import os
import h5py
import numpy as np
import argparse
from tqdm import tqdm
import json

# Import the to_mp4 function from eval.py
from robomimic.act.eval import to_mp4

def visualize_dataset(dataset_path="/share/data/ripl/tianchong/vista/data/low_dim_v141_obs.hdf5", 
                      output_dir="dataset_visualizations", 
                      num_demos=5, 
                      camera_name=None):
    """
    Render observations from a dataset created by dataset_states_to_obs.py to MP4 videos.
    
    Args:
        dataset_path (str): Path to the HDF5 dataset
        output_dir (str): Directory to save the MP4 files
        num_demos (int): Number of demonstrations to render
        camera_name (str): Name of the camera view to render, or None to auto-select
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the dataset
    with h5py.File(dataset_path, 'r') as f:
        # Get demo keys
        demos = [k for k in f["data"].keys() if k.startswith("demo_")]
        num_demos = min(num_demos, len(demos))
        
        print(f"Rendering {num_demos} demonstrations from {dataset_path}")
        
        # Detect available camera views from first demo
        if demos:
            first_demo = demos[0]
            available_cameras = []
            
            if "obs" in f["data"][first_demo]:
                for key in f["data"][first_demo]["obs"].keys():
                    if key.endswith("_image"):
                        # Extract camera name by removing "_image" suffix
                        camera = key[:-6]
                        available_cameras.append(camera)
            
            print(f"Available camera views: {available_cameras}")
            
            # Show camera info if available
            if "camera_info" in f["data"][first_demo].attrs:
                try:
                    camera_info = json.loads(f["data"][first_demo].attrs["camera_info"])
                    print("\nCamera information:")
                    for cam, info in camera_info.items():
                        print(f"  {cam}:")
                        if "intrinsics" in info:
                            print(f"    Intrinsics: {info['intrinsics']}")
                        if "extrinsics" in info:
                            print(f"    Extrinsics shape: {np.array(info['extrinsics']).shape}")
                except:
                    print("Could not parse camera info")
            
            # Auto-select camera if not specified
            if not camera_name:
                if available_cameras:
                    camera_name = available_cameras[0]
                    print(f"Auto-selected camera view: {camera_name}")
                else:
                    print("No camera views found in dataset!")
                    return
            else:
                if camera_name not in available_cameras:
                    print(f"Warning: Specified camera '{camera_name}' not found!")
                    if available_cameras:
                        camera_name = available_cameras[0]
                        print(f"Using '{camera_name}' instead.")
                    else:
                        print("No camera views found in dataset!")
                        return
        
        for i, demo_key in enumerate(demos[:num_demos]):
            print(f"Processing demo {i+1}/{num_demos}: {demo_key}")
            
            # Extract observations
            frames = []
            success = []
            rewards = []
            
            # Check if images exist in the dataset
            camera_image_key = f"{camera_name}_image"
            if f"data/{demo_key}/obs/{camera_image_key}" in f:
                # Extract RGB frames
                for j in tqdm(range(len(f[f"data/{demo_key}/obs/{camera_image_key}"]))):
                    # Get RGB observation (shape: H x W x C)
                    rgb = f[f"data/{demo_key}/obs/{camera_image_key}"][j]
                    # Convert from uint8 if needed (some datasets store as float [0,1])
                    if rgb.dtype != np.uint8:
                        rgb = (rgb * 255).astype(np.uint8)
                    frames.append(rgb)
                    
                    # Get reward if available
                    if f"data/{demo_key}/rewards" in f:
                        rewards.append(f[f"data/{demo_key}/rewards"][j])
                    else:
                        rewards.append(0.0)
                        
                    # Get success status (from reward == 1)
                    if len(rewards) > 0 and rewards[-1] == 1.0:
                        success.append(True)
                    else:
                        success.append(False)
                
                # Save the video
                video_filename = f"{demo_key}_camera_{camera_name}.mp4"
                video_path = os.path.join(output_dir, video_filename)
                to_mp4(video_path, frames, reward_list=rewards, success_list=success)
                print(f"Saved video to {video_path}")
            else:
                print(f"No image observations found for {camera_name} in demo {demo_key}")

if __name__ == "__main__":
    # Simple script with default arguments - no need for argument parsing
    visualize_dataset()
