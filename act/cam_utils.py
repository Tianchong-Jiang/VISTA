import numpy as np
import robosuite.utils.transform_utils as T

def sample_lookat_poses(target_pos, radius, num_samples=10, world_up=np.array([0., 0., 1.])):
    """
    Samples random camera poses on a sphere around a target point,
    with each camera oriented to look at the target, maintaining world up.
    Calculates rotation matrix compatible with MuJoCo cam_xmat.

    Args:
        target_pos (np.ndarray): The (x, y, z) position the camera should look at.
        radius (float): The distance of the camera from the target point.
        num_samples (int): The number of camera poses to sample.
        world_up (np.ndarray): The world's up direction vector.

    Returns:
        list[tuple(np.ndarray, np.ndarray)]: A list of sampled poses,
            where each pose is a tuple containing:
            - camera_pos (np.ndarray): The (x, y, z) position of the camera.
            - camera_rot (np.ndarray): The 3x3 rotation matrix defining the
                                       orientation of the MuJoCo camera frame in world coordinates.
                                       (Columns are MuJoCo X, Y, Z axes in World).
    """
    poses = []
    target_pos = np.array(target_pos)
    world_up = np.array(world_up) / np.linalg.norm(world_up) # Ensure world_up is normalized
    
    for _ in range(num_samples):
        # 1. Sample random direction (unit vector) for camera offset
        offset_direction = np.random.randn(3)
        norm = np.linalg.norm(offset_direction)
        if norm < 1e-6:
            offset_direction = np.array([0.0, 0.0, 1.0]) # Default to Z-axis if random fails
        else:
            offset_direction = offset_direction / norm
        
        # Ensure the direction is in the upper hemisphere (Z >= 0 relative to target)
        if offset_direction[2] < 0:
            offset_direction *= -1
            norm = np.linalg.norm(offset_direction)
            if norm < 1e-6: offset_direction = np.array([0.0, 0.0, 1.0])
            else: offset_direction = offset_direction / norm
            
        # 2. Calculate camera position
        camera_pos = target_pos + offset_direction * radius
        
        # 3. Calculate orientation matrix for MuJoCo cam_xmat (Look-At)
        forward = target_pos - camera_pos # Vector from camera to target
        forward_norm = np.linalg.norm(forward)
        
        if forward_norm < 1e-6:
            # Camera is at the target, use a default orientation
            # Pointing down Z with Y pointing towards -X (consistent with MuJoCo default)
            camera_rot = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., -1.]]) @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            # This calculation needs verification, but provides a fallback.
            # A simpler fallback might be identity or previous valid rotation.
            if len(poses) > 0: camera_rot = poses[-1][1] # Use last rotation
            else: camera_rot = np.eye(3) # Fallback to identity
        else:
            forward = forward / forward_norm
            backward = -forward
            
            # Calculate MuJoCo Right vector (World X relative to camera) 
            # Use cross(world_up, backward) for stability when forward is near world_up
            right = np.cross(world_up, backward)
            right_norm = np.linalg.norm(right)
            if right_norm < 1e-6:
                # Handle case where backward vector is parallel to world_up
                if abs(backward[0]) > 0.99: # If backward is approx X or -X
                    right = np.cross(np.array([0., 1., 0.]), backward) # Use world Y 
                else:
                    right = np.cross(np.array([1., 0., 0.]), backward) # Use world X
                right_norm = np.linalg.norm(right)
                if right_norm < 1e-6: right = np.array([0., 1., 0.]) if abs(backward[0]) > 0.99 else np.array([1., 0., 0.]) # Fallback
                else: right = right / right_norm 
            else:
                right = right / right_norm

            # Calculate MuJoCo Down vector (World Y relative to camera)
            down = np.cross(backward, right) # Already normalized
            
            # Construct the rotation matrix (columns are MuJoCo axes in World)
            camera_rot = np.column_stack((right, down, backward))
            
        poses.append((camera_pos, camera_rot))
        
    return poses

# --- Potentially add other camera utility functions below ---
