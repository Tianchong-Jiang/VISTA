import h5py
import torch
import os
import numpy as np
import random
import re
import math
import glob
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

# --- Dataset Loading ---

def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_dict_mean(dict_list):
    """Compute the mean value for each key in a list of dictionaries."""
    if len(dict_list) == 0:
        return {}
    
    mean_dict = {}
    for key in dict_list[0].keys():
        if not isinstance(dict_list[0][key], torch.Tensor):
            continue  # Skip non-tensor values
        mean_dict[key] = torch.stack([d[key] for d in dict_list]).mean()
    return mean_dict

def detach_dict(dictionary):
    """Detach all tensors in a dictionary."""
    result = {}
    for k, v in dictionary.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.detach()
        else:
            result[k] = v
    return result

def cleanup_ckpt(ckpt_dir, keep=1):
    """Keep only the latest N checkpoints."""
    ckpts = glob.glob(os.path.join(ckpt_dir, "epoch_*.pth"))
    if len(ckpts) <= keep:
        return
    
    # Extract epoch numbers and sort
    epoch_nums = []
    for ckpt in ckpts:
        match = re.search(r"epoch_(\d+).pth", ckpt)
        if match:
            epoch_nums.append((int(match.group(1)), ckpt))
    
    # Sort by epoch number
    epoch_nums.sort(reverse=True)
    
    # Remove all but the latest K checkpoints
    for _, ckpt in epoch_nums[keep:]:
        os.remove(ckpt)

def get_last_ckpt(ckpt_dir):
    """Get the latest checkpoint in the directory."""
    if not os.path.exists(ckpt_dir):
        return None
    
    ckpts = glob.glob(os.path.join(ckpt_dir, "epoch_*.pth"))
    if not ckpts:
        return None
    
    # Extract epoch numbers and find the latest
    latest_epoch = -1
    latest_ckpt = None
    for ckpt in ckpts:
        match = re.search(r"epoch_(\d+).pth", ckpt)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_ckpt = ckpt
    
    return latest_ckpt

def cosine_schedule(optimizer, total_steps, eta_min=0.0):
    """Cosine learning rate schedule."""
    def lr_lambda(step):
        return eta_min + 0.5 * (1 - eta_min) * (1 + math.cos(math.pi * step / total_steps))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, eta_min=0.0):
    """Cosine learning rate schedule with warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return eta_min + 0.5 * (1 - eta_min) * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def get_norm_stats(dataset_path, num_demos):
    """
    Compute normalization statistics for actions and states from the dataset.
    
    Args:
        dataset_path (str): Path to the HDF5 dataset
        num_demos (int): Number of demonstrations to use for computing stats.
    Returns:
        dict: Dictionary containing normalization statistics
    """
    all_states_data = []
    all_action_data = []
    
    with h5py.File(dataset_path, 'r') as dataset_file:
        # Get total number of demos
        demo_keys = [k for k in dataset_file['data'].keys() if k.startswith('demo_')]
        num_demos = min(num_demos, len(demo_keys))
            
        print(f"Computing normalization statistics using {num_demos} demonstrations...")
        
        # Collect data from demonstrations
        for i in range(num_demos):
            demo_key = f'data/demo_{i}'
            
            # Read states and actions
            states = dataset_file[f'{demo_key}/states'][()]
            actions = dataset_file[f'{demo_key}/actions'][()]
            
            # For robosuite environments, states are typically joint positions
            # You may need to adjust this extraction based on your specific environment
            all_states_data.append(states)
            all_action_data.append(actions)
    
    # Stack all data
    states_array = np.vstack([s for s in all_states_data])
    actions_array = np.vstack([a for a in all_action_data])
    
    # Compute statistics
    state_mean = np.mean(states_array, axis=0, keepdims=True)
    state_std = np.std(states_array, axis=0, keepdims=True)
    state_std = np.clip(state_std, 1e-2, np.inf)  # Prevent division by very small numbers
    
    action_mean = np.mean(actions_array, axis=0, keepdims=True)
    action_std = np.std(actions_array, axis=0, keepdims=True)
    action_std = np.clip(action_std, 1e-2, np.inf)
    
    stats = {
        "state_mean": state_mean.squeeze(),
        "state_std": state_std.squeeze(),
        "action_mean": action_mean.squeeze(),
        "action_std": action_std.squeeze()
    }
    
    return stats


class RandomCrop(object):
    def __init__(self, min_side=224, max_side=256, output_size=256):
        self.min_side = min_side
        self.max_side = max_side
        self.output_size = output_size

    def __call__(self, img):
        # If batched: img has shape (B, C, H, W)
        if img.dim() == 4:
            B, C, H, W = img.shape
            processed = []
            for i in range(B):
                processed.append(self.crop_and_resize_single(img[i], H, W))
            return torch.stack(processed)
        # Single image: img has shape (C, H, W)
        elif img.dim() == 3:
            C, H, W = img.shape
            return self.crop_and_resize_single(img, H, W)
        else:
            raise ValueError("Input tensor must be 3D or 4D.")

    def crop_and_resize_single(self, image, H, W):
        # Choose a random crop size between min_side and max_side
        crop_size = random.randint(self.min_side, self.max_side)
        if crop_size > H or crop_size > W:
            raise ValueError("Crop size is larger than the image dimensions.")
        top = random.randint(0, H - crop_size)
        left = random.randint(0, W - crop_size)
        # Crop the image
        cropped = TF.crop(image, top, left, crop_size, crop_size)
        # Resize the cropped image back to output_size x output_size (256x256)
        resized = TF.resize(cropped, [self.output_size, self.output_size])
        return resized


class EpisodicDataset(Dataset):
    """
    Dataset for loading episodic data from the robomimic HDF5 format.
    Allows for random sampling of trajectories from demos.
    """
    def __init__(self, dataset_path, demo_indices, norm_stats, camera_names=["frontview"], 
                 max_seq_length=None, transform="id", image_size=256):
        """
        Args:
            dataset_path (str): Path to the HDF5 dataset
            demo_indices (list): List of demonstration indices to use
            norm_stats (dict): Normalization statistics for actions and states
            camera_names (list): List of camera names to use for observations
            max_seq_length (int, optional): Maximum sequence length for actions
            transform (str): Transform to apply to images - "id" or "crop"
            image_size (int): Size of image observations
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.demo_indices = demo_indices
        self.norm_stats = norm_stats
        self.camera_names = camera_names
        self.image_size = image_size
        
        # Identify available image observation keys
        self.image_keys = {}
        print("\nChecking available image keys in dataset...")
        with h5py.File(self.dataset_path, "r") as dataset_file:
            # Check the first demo's observation keys
            if len(self.demo_indices) > 0:
                demo_key = f'data/demo_{self.demo_indices[0]}'
                if 'obs' in dataset_file[demo_key]:
                    print(f"Available observation keys: {list(dataset_file[demo_key]['obs'].keys())}")
                    
                    # Find actual keys for each requested camera
                    for camera in self.camera_names:
                        # Check for exact match
                        if f"{camera}_image" in dataset_file[demo_key]['obs']:
                            self.image_keys[camera] = f"{camera}_image"
                        else:
                            # Look for any key containing this camera name
                            for key in dataset_file[demo_key]['obs'].keys():
                                if "_image" in key:
                                    print(f"Found image key: {key}")
                                    # Use the first image key we find if no exact match
                                    if not self.image_keys:
                                        self.image_keys[camera] = key
                                        print(f"Using {key} for camera {camera}")
                                        break
        
        # Warn if we couldn't find keys for all requested cameras
        for camera in self.camera_names:
            if camera not in self.image_keys:
                print(f"WARNING: Could not find image key for camera {camera}. Available keys: {list(self.image_keys.values())}")
        
        # Load all episode data
        self.demo_states = []
        self.demo_actions = []
        self.demo_lengths = []
        
        with h5py.File(self.dataset_path, "r") as dataset_file:
            for idx in self.demo_indices:
                demo_key = f'data/demo_{idx}'
                states = dataset_file[f'{demo_key}/states'][()]
                actions = dataset_file[f'{demo_key}/actions'][()]
                
                self.demo_states.append(states)
                self.demo_actions.append(actions)
                self.demo_lengths.append(len(actions))
        
        # Set maximum sequence length
        if max_seq_length is None:
            self.max_seq_length = max(self.demo_lengths)
        else:
            self.max_seq_length = max_seq_length
            
        # Set up image transforms
        if transform == "id":
            self.transforms = T.Lambda(lambda x: x)
        elif transform == "crop":
            self.transforms = RandomCrop(min_side=224, max_side=256, output_size=image_size)
        else:
            raise ValueError("Invalid transform type.")
            
    def __len__(self):
        return len(self.demo_indices)
    
    def __getitem__(self, index):
        """
        Get an item from the dataset. Randomly samples a starting point in the trajectory.
        
        Returns:
            tuple: (image_data, state_data, action_data, is_pad, cam_pose)
                - image_data: image from the dataset
                - state_data: state vector (normalized)
                - action_data: sequence of actions (normalized and padded)
                - is_pad: boolean mask indicating padding
                - cam_pose: camera pose (if available)
        """
        # Get demo data
        demo_length = self.demo_lengths[index]
        start_ts = np.random.randint(demo_length)
        
        # Get states and actions
        states = self.demo_states[index]
        actions = self.demo_actions[index][start_ts:]
        
        # Pad actions if needed
        padded_actions = np.zeros((self.max_seq_length, actions.shape[1]), dtype=np.float32)
        seq_length = min(len(actions), self.max_seq_length)
        padded_actions[:seq_length] = actions[:seq_length]
        
        # Create padding mask
        is_pad = np.zeros(self.max_seq_length, dtype=np.bool_)
        is_pad[seq_length:] = True
        
        # Get current state
        state = states[start_ts]
        
        # Normalize state and actions
        state_normalized = (state - self.norm_stats["state_mean"]) / self.norm_stats["state_std"]
        actions_normalized = (padded_actions - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        
        # Convert to torch tensors
        state_tensor = torch.from_numpy(state_normalized).float()
        actions_tensor = torch.from_numpy(actions_normalized).float()
        is_pad_tensor = torch.from_numpy(is_pad)
        
        # Initialize camera pose to default
        cam_pose = torch.zeros(7, dtype=torch.float32)  # Default camera pose
        
        # Load image directly from dataset
        with h5py.File(self.dataset_path, "r") as dataset_file:
            demo_key = f'data/demo_{self.demo_indices[index]}'
            camera_name = self.camera_names[0]  # Use the first camera
            
            # Get the actual image key for this camera
            image_key = f'{demo_key}/obs/{self.image_keys[camera_name]}'
            
            # Get RGB image at the specific timestep
            image = dataset_file[image_key][start_ts]
            
            # Convert to torch tensor and normalize
            if image.dtype == np.uint8:
                image_tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
            else:
                # If already float, assume in range [0,1]
                image_tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float()
        
        # Apply transforms
        image_tensor = self.transforms(image_tensor)
        
        return image_tensor, state_tensor, actions_tensor, is_pad_tensor, cam_pose


def load_data(dataset_path, num_demos, batch_size_train, batch_size_val, 
              camera_names=["frontview"], transform="id"):
    """
    Load datasets and create dataloaders for training and validation.
    
    Args:
        dataset_path (str): Path to the HDF5 dataset
        num_demos (int): Number of demonstrations to use
        batch_size_train (int): Batch size for training
        batch_size_val (int): Batch size for validation
        camera_names (list): List of camera names to use for observations
        transform (str): Transform to apply to images - "id" or "crop"
        
    Returns:
        tuple: (train_dataloader, val_dataloader, norm_stats)
    """
    assert num_demos > 5, "Number of demonstrations must be greater than 5"
    
    # Create indices and split into train/val
    indices = list(range(num_demos))
    train_indices = indices[:num_demos-5]
    val_indices = indices[num_demos-5:num_demos]
    
    print("Computing normalization statistics...")
    norm_stats = get_norm_stats(dataset_path, num_demos=num_demos)
    print("Computing normalization statistics... Done!")
    
    print("Loading datasets...")
    train_dataset = EpisodicDataset(
        dataset_path, 
        train_indices, 
        norm_stats, 
        camera_names=camera_names,
        transform=transform
    )
    
    val_dataset = EpisodicDataset(
        dataset_path, 
        val_indices, 
        norm_stats, 
        camera_names=camera_names,
        transform=transform
    )
    print("Loading datasets... Done!")
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size_train, 
        shuffle=True, 
        num_workers=0
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size_val, 
        shuffle=False, 
        num_workers=0
    )
    
    return train_dataloader, val_dataloader, norm_stats


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    from PIL import Image
    import robomimic.utils.file_utils as FileUtils
    
    # Create output directory for images
    output_dir = "dataset_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    # Example usage
    dataset_path = '/share/data/ripl/tianchong/vista/data/low_dim_v141_obs.hdf5'
    
    # Compute normalization statistics using first 10 demos
    print("Computing normalization statistics...")
    norm_stats = get_norm_stats(dataset_path, num_demos=10)
    print("Done computing statistics!")
    
    # Create dataset with first 10 demos
    demo_indices = list(range(10))
    dataset = EpisodicDataset(
        dataset_path, 
        demo_indices, 
        norm_stats, 
        camera_names=["frontview"],
        transform="id"
    )
    
    # Create a dataloader with batch size 1 for simplicity
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    print(f"Dataset contains {len(dataset)} demonstrations")
    
    # Get 10 samples from the dataloader and save their images
    print("Loading 10 samples using the dataloader...")
    for i, batch in enumerate(dataloader):
        if i >= 10:  # Only get 10 samples
            break
            
        # Unpack the batch
        image_data, state_data, action_data, is_pad, cam_pose = batch
        
        # The batch dimension is first, so we take the first item
        image = image_data[0].numpy()  # Shape: (C, H, W)
        state = state_data[0].numpy()
        action = action_data[0, 0].numpy()  # First action in sequence
        
        # Transpose from (C, H, W) to (H, W, C) for visualization
        image = np.transpose(image, (1, 2, 0))
        
        # Create a figure to display the image and some data
        plt.figure(figsize=(10, 10))
        
        # First subplot for the image
        plt.subplot(2, 1, 1)
        if image.shape[2] == 3:  # RGB image
            plt.imshow(np.clip(image, 0, 1))
        else:  # More channels (e.g., with depth or plucker)
            plt.imshow(np.clip(image[:, :, :3], 0, 1))  # Just show RGB channels
        plt.title(f"Sample {i+1}")
        plt.axis('off')
        
        # Second subplot for state and action data
        plt.subplot(2, 1, 2)
        plt.axis('off')
        plt.text(0.1, 0.8, f"State (first 5): {state[:5]}", fontsize=10)
        plt.text(0.1, 0.6, f"Action (first 5): {action[:5]}", fontsize=10)
        plt.text(0.1, 0.4, f"State shape: {state.shape}", fontsize=10)
        plt.text(0.1, 0.2, f"Action sequence shape: {action_data[0].shape}", fontsize=10)
        
        # Save the figure
        plt.savefig(f"{output_dir}/sample_{i+1}.jpg", bbox_inches='tight')
        plt.close()
        
        # Also save the raw image
        image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        if image.shape[2] >= 3:  # Ensure we have at least 3 channels for RGB
            img = Image.fromarray(image_uint8[:, :, :3])
            img.save(f"{output_dir}/sample_{i+1}_raw.jpg")
        
        print(f"Saved sample {i+1}")
    
    print(f"\nSamples saved to {output_dir}")
    print("Each sample includes a visualization of the image and associated state/action data")

