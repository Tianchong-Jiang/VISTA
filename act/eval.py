import os
import numpy as np
import torch
import cv2
import h5py

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils

def dummy_cam_fn(idx, total):
    # Return some default camera config
    return np.array([0, 0, 1, 0, 0, 0, 90])  # Example camera config format

def to_mp4(save_path, image_list, reward_list=None, success_list=None, info_list=None):
    """
    Save a list of images as an MP4 video with reward and success overlaid.
    
    Args:
        save_path (str): Path to save the MP4 file
        image_list (list): List of images (numpy arrays)
        reward_list (list, optional): List of rewards for each frame
        success_list (list, optional): List of success flags for each frame
        info_list (list, optional): List of additional info dictionaries
    """
    if not image_list:
        return
        
    height, width, _ = image_list[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, 10, (width, height))

    for i, img in enumerate(image_list):
        # Convert to BGR for OpenCV
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Add overlay text for rewards and success if available
        y_pos = 30
        if reward_list is not None and i < len(reward_list):
            reward_text = f"Reward: {reward_list[i]:.4f}"
            cv2.putText(frame, reward_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
            y_pos += 30
            
        if success_list is not None and i < len(success_list):
            success_text = f"Success: {success_list[i]}"
            cv2.putText(frame, success_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
            y_pos += 30
            
        if info_list is not None and i < len(info_list):
            # Display the first few items from the info dict
            for j, (key, value) in enumerate(info_list[i].items()):
                if j >= 3:  # Limit to 3 info items to avoid cluttering
                    break
                if isinstance(value, (int, float, bool, str)):
                    info_text = f"{key}: {value}"
                    cv2.putText(frame, info_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 255), 1, cv2.LINE_AA)
                    y_pos += 25

        out.write(frame)

    out.release()

def replay_dataset(dataset_path, num_demos=5, camera_name="frontview", output_dir="replay_videos"):
    """
    Replay actions from a robomimic dataset using the Evaluator class.
    
    Args:
        dataset_path (str): Path to the HDF5 dataset
        num_demos (int): Number of demonstrations to replay
        camera_name (str): Name of the camera view to render
        output_dir (str): Directory to save the MP4 files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get environment metadata from dataset
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)
    
    # Create environment for data processing
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=[camera_name],
        camera_height=256, 
        camera_width=256, 
        reward_shaping=False,
        use_depth_obs=False,
        use_image_obs=True
    )
    
    # Compute normalization statistics for the dataset
    from robomimic.act.utils import get_norm_stats
    norm_stats = get_norm_stats(dataset_path, num_demos=20)
    
    # Create a dummy policy that returns actions from the dataset
    class DatasetPolicy:
        def __init__(self, dataset_path, chunk_size=1):
            self.dataset_path = dataset_path
            self.current_demo = 0
            self.actions = None
            self.action_index = 0
            self.chunk_size = chunk_size
            self._load_next_demo()
            
        def _load_next_demo(self):
            with h5py.File(self.dataset_path, 'r') as dataset_file:
                demo_keys = [k for k in dataset_file['data'].keys() if k.startswith('demo_')]
                if self.current_demo >= len(demo_keys):
                    return False
                
                demo_key = f'data/demo_{self.current_demo}'
                self.actions = dataset_file[f'{demo_key}/actions'][()]
                self.action_index = 0
                self.current_demo += 1
                return True
                
        def __call__(self, state, obs):
            if self.action_index >= len(self.actions):
                return torch.zeros((1, self.chunk_size, self.actions.shape[1]), device="cuda")
            
            action = self.actions[self.action_index:self.action_index+1]
            self.action_index += 1
            
            # Return shape: [batch=1, chunk_size, action_dim]
            # Create a batch with the right chunk size dimension
            return torch.tensor(action, dtype=torch.float32, device="cuda").unsqueeze(0)
    
    # Create evaluator with chunk_size=1 for exact replay
    evaluator = Evaluator(
        env=env,
        norm_stats=norm_stats,
        chunk_size=1,  # Use chunk_size=1 for exact replay
        max_steps=200,  # Set high to replay full demos
        use_plucker=False
    )
    
    # Open the dataset to get info for each demo
    with h5py.File(dataset_path, 'r') as dataset_file:
        demo_keys = [k for k in dataset_file['data'].keys() if k.startswith('demo_')]
        num_demos = min(num_demos, len(demo_keys))
        
        print(f"Replaying {num_demos} demonstrations...")
        
        # Run evaluation for each demo
        for i in range(num_demos):
            # Create a new dummy policy for this demo
            dummy_policy = DatasetPolicy(dataset_path, chunk_size=evaluator.chunk_size)
            dummy_policy.current_demo = i
            dummy_policy._load_next_demo()
            
            # Reset environment to initial state of this demo
            demo_key = f'data/demo_{i}'
            states = dataset_file[f'{demo_key}/states'][()]
            init_state = dict(states=states[0])
            if "model_file" in dataset_file[demo_key].attrs:
                init_state["model"] = dataset_file[demo_key].attrs["model_file"]
            
            # Don't reset the environment here - let evaluate handle it
            
            # Run a single episode and get results
            results, success_rate, episode_length = evaluator.evaluate(
                policy=dummy_policy,
                save_path=output_dir,
                video_prefix=f"demo_{i}",
                camera_name=camera_name,
                init_state=init_state,  # Pass the initial state
                episode_num=i  # Pass episode number for printing
            )
            
            # Save statistics
            stats_path = os.path.join(output_dir, f"demo_{i}_stats.txt")
            with open(stats_path, 'w') as f:
                f.write(f"Demo {i} statistics:\n")
                for k, v in results.items():
                    f.write(f"{k}: {v}\n")
    
    print(f"All replays completed. Videos saved to {output_dir}")

class Evaluator:
    """
    Class to evaluate policies on robomimic environments
    """
    def __init__(self, env, norm_stats, chunk_size=15, max_steps=200, render=True, use_plucker=False):
        """
        Initialize the evaluator.
        
        Args:
            env: The environment to evaluate in
            norm_stats (dict): Normalization statistics for states and actions
            chunk_size (int): Number of actions to predict and execute in a single chunk
            max_steps (int): Maximum number of steps per episode
            render (bool): Whether to render and save frames
            use_plucker (bool): Whether to use plucker coordinates
        """
        self.env = env
        self.norm_stats = {k: torch.tensor(v).float() for k, v in norm_stats.items()}
        self.chunk_size = chunk_size
        self.max_steps = max_steps
        self.render = render
        self.use_plucker = use_plucker
        
    def evaluate(self, policy, save_path, video_prefix, camera_name, init_state=None, episode_num=0):
        """
        Run a single episode with the given policy
        
        Args:
            policy: The policy to use
            save_path: Path to save video
            video_prefix: Prefix for video filenames
            camera_name: Name of the camera view to render
            init_state: Initial state to reset the environment to (if None, use env.reset())
            episode_num: Episode number for printing
        Returns:
            dict: Evaluation metrics compatible with what callers expect
        """
        if init_state is not None:
            self.env.reset_to(init_state)
        else:
            self.env.reset()
        
        frames = []
        success_labels = []
        rewards = []
        success = []
        done = False
        step = 0
        has_succeeded = False
        
        obs = self.env.render(
            mode="rgb_array", 
            camera_name=camera_name,
            height=256,
            width=256
        ).copy()
        
        if self.render:
            frames.append(obs)
            success_labels.append(False)
        
        while not done and step < self.max_steps:
            # Convert to tensor and normalize
            rgb_obs = torch.from_numpy(obs).float().cuda() / 255.0  # Shape: [H, W, C]
            
            # Reshape to [B, N, C, H, W] format for ACT policy
            # First move channels to first dimension, then add batch and camera dimensions
            rgb_obs = rgb_obs.permute(2, 0, 1)  # [C, H, W]
            rgb_obs = rgb_obs.unsqueeze(0).unsqueeze(0)  # [1, 1, C, H, W]
            
            state = self.env.get_state()
            state_vector = state["states"]
            
            normalized_state = (state_vector - self.norm_stats["state_mean"].cpu().numpy()) / self.norm_stats["state_std"].cpu().numpy()
            state_tensor = torch.tensor(normalized_state[:9], device="cuda").float().unsqueeze(0) 
            
            with torch.no_grad():
                action_chunk = policy(state_tensor, rgb_obs)  # Shape: [batch=1, chunk_size, action_dim]
            
            # Denormalize the entire action chunk at once
            action_chunk = action_chunk[0].cpu().numpy()  # Remove batch dimension: [chunk_size, action_dim]
            action_chunk = action_chunk * self.norm_stats["action_std"].cpu().numpy() + self.norm_stats["action_mean"].cpu().numpy()
            
            for i in range(action_chunk.shape[0]):
                if done or step >= self.max_steps:
                    break
                
                action = action_chunk[i]
                
                next_obs, reward, done, info = self.env.step(action)
                
                current_success = reward == 1
                has_succeeded = has_succeeded or current_success
                
                rewards.append(float(reward))
                success.append(current_success)
                step += 1
                
                if self.render:
                    obs = self.env.render(
                        mode="rgb_array", 
                        camera_name=camera_name,
                        height=256,
                        width=256
                    ).copy()
                    frames.append(obs)
                    success_labels.append(has_succeeded)
                
                if done:
                    break
        
        final_success = any(success)
        print(f"Episode {episode_num}: Success = {final_success}")
        
        if save_path is not None and self.render and frames:
            video_path = os.path.join(save_path, f"{video_prefix}_success_{final_success}.mp4")
            to_mp4(video_path, frames, success_list=success_labels)
        
        results = {
            "success_rate": float(final_success),
            "mean_episode_length": float(step),
            "max_rewards": rewards
        }
        
        return results, float(final_success), step

def main():
    """Main function to run dataset replay or policy evaluation"""
    dataset_path = '/share/data/ripl/tianchong/vista/data/low_dim_v141.hdf5'
    output_dir = 'evaluation_results'
    camera_name = "frontview"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Replaying demonstrations from {dataset_path}")
    replay_dataset(
        dataset_path=dataset_path, 
        num_demos=5, 
        camera_name=camera_name,
        output_dir=output_dir
    )

if __name__ == '__main__':
    main()
