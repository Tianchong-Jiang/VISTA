import os
import h5py
import numpy as np
import cv2
import argparse
from tqdm import tqdm
import random
import json

def visualize_dataset(dataset_path, output_dir, num_images=100):
    """
    Loads camera images from an HDF5 dataset and saves them as PNG files.
    """
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving images to: {output_dir}")

    with h5py.File(dataset_path, 'r') as f:
        # Get demos
        demo_keys = list(f['data'].keys())
        print(f"Dataset: {dataset_path} | {len(demo_keys)} demos")
        
        print("Discovering camera keys per demo…")
        demo_to_keys = {}
        for d in demo_keys:
            d_obs = f'data/{d}/obs'
            if d_obs in f:
                keys = [k for k in f[d_obs].keys() if k.endswith('_image') and len(f[f"{d_obs}/{k}"].shape) >= 3]
                if keys:
                    demo_to_keys[d] = keys
        print(f"Demos with camera data: {len(demo_to_keys)} / {len(demo_keys)}")
        
        # Collect valid samples
        valid_samples = []
        for demo_key in tqdm(demo_keys, desc="Scanning demos"):
            demo_path = f'data/{demo_key}'
            
            # Skip demos without camera keys discovered
            if demo_key not in demo_to_keys:
                continue

            image_keys_for_demo = demo_to_keys[demo_key]

            # Get camera poses
            if 'camera_poses' not in f[demo_path].attrs:
                continue
            camera_poses_data = json.loads(f[demo_path].attrs['camera_poses'])

            # Choose a single timestep uniformly for this demo
            any_img_path = f'data/{demo_key}/obs/{image_keys_for_demo[0]}'
            num_steps = f[any_img_path].shape[0]
            chosen_t = random.randint(0, num_steps - 1)

            # Add sample for each available camera key
            for img_key in image_keys_for_demo:
                valid_samples.append({
                    'demo': demo_key,
                    'timestep': chosen_t,
                    'key': img_key,
                    'camera_poses': camera_poses_data
                })
        
        print(f"Found {len(valid_samples)} valid images")
        
        # Sample and save images
        selected_samples = random.sample(valid_samples, min(num_images, len(valid_samples)))
        
        for sample in tqdm(selected_samples, desc="Saving images"):
            # Extract data
            demo_key = sample['demo']
            timestep = sample['timestep']
            img_key = sample['key']
            camera_poses_data = sample['camera_poses']
            
            # Get demo number
            demo_num = int(demo_key.split('_')[-1])
            
            # Load image
            img_rgb = f[f'data/{demo_key}/obs/{img_key}'][timestep]
            
            # Extract camera info using global camera ID from image key
            global_cam_id = int(img_key.split("_")[1])  # e.g., cam_2_image -> 2
            cam_data = next(cd for cd in camera_poses_data if cd['id'] == global_cam_id)
            pos = cam_data['position']
            camera_xyz = f"xyz_{pos[0]:.2f}_{pos[1]:.2f}_{pos[2]:.2f}"
            
            # Convert image
            if img_rgb.dtype != np.uint8:
                img_rgb = (img_rgb * 255).clip(0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            
            # Save image
            filename = f"demo{demo_num:03d}_step{timestep:03d}_cam{global_cam_id:02d}_{camera_xyz}.png"
            base_path = os.path.join(output_dir, filename[:-4])  # strip .png
            cv2.imwrite(base_path + ".png", img_bgr)

            # ------------------------------------------------------------------
            # Save Plücker embedding visualizations (first 3 dims and last 3 dims)
            # ------------------------------------------------------------------

            # Helper to convert float image (H, W, 3) to uint8 RGB
            def to_uint8(arr: np.ndarray) -> np.ndarray:
                mn = arr.min(axis=(0, 1), keepdims=True)
                mx = arr.max(axis=(0, 1), keepdims=True)
                rng = np.where(mx - mn == 0, 1, mx - mn)
                return np.clip((arr - mn) / rng * 255.0, 0, 255).astype(np.uint8)

            # Load Plücker map (H, W, 6) – stored once per camera per demo
            plk_key = img_key.replace("_image", "_plucker")
            plk = f[f'data/{demo_key}/plucker/{plk_key}'][()]
            # Split into first three and last three channels
            plk012 = to_uint8(plk[..., 0:3])
            plk345 = to_uint8(plk[..., 3:6])

            # OpenCV expects BGR
            cv2.imwrite(base_path + "_plk012.png", cv2.cvtColor(plk012, cv2.COLOR_RGB2BGR))
            cv2.imwrite(base_path + "_plk345.png", cv2.cvtColor(plk345, cv2.COLOR_RGB2BGR))

    print(f"Finished saving images to {output_dir}")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Visualize images from an HDF5 dataset.")
    parser.add_argument("--dataset", type=str, 
                        default=os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")), 
                                            "data", "low_dim_v141_randcam.hdf5"))
    parser.add_argument("--output_dir", type=str, 
                        default=os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")), 
                                            "rendered_dataset_vis"))
    parser.add_argument("--num_images", type=int, default=100)

    args = parser.parse_args()
    visualize_dataset(args.dataset, args.output_dir, args.num_images)
