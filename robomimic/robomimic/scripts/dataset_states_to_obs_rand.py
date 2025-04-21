"""
This script generates a robomimic dataset with random camera views per demo, 
including Plücker ray maps for each unique view.

 Example output file layout (defaults: num_demos=3, num_views=3, k=1):

 low_dim_v141_randcam.hdf5
 /
 ├─ data
 │   ├─ demo_0
 │   │   ├─ actions          (T-1 , A)
 │   │   ├─ states           (T   , S)
 │   │   ├─ rewards          (T-1)
 │   │   ├─ dones            (T-1)
 │   │   ├─ obs/
 │   │   │   ├─ cam_0_image   (T-1 , H, W, 3)  uint8
 │   │   │   ├─ cam_1_image   (...)
 │   │   │   └─ cam_2_image   (...)
 │   │   ├─ next_obs/         (same three *_image datasets)
 │   │   └─ attrs {cam_inds, model_file, num_samples}
 │   ├─ demo_1               # cameras 1,2,3 (sliding window)
 │   ├─ demo_2               # cameras 2,3,4
 │   ├─ train_plucker/       # Plücker maps for train cameras (IDs < test_start)
 │   │   ├─ cam_0_plucker    (H, W, 6)  float32
 │   │   └─ ...              (...)
 │   ├─ test_plucker/        # 100 held‑out cameras (IDs ≥ test_start)
 │   │   ├─ cam_123_plucker  (H, W, 6)  float32
 │   │   └─ ...              (...)
 │   └─ attrs  {total, env_args, all_camera_poses, test_cam_inds, train_cam_inds, num_total_cams}
 └─ mask (optional; copied from source dataset)

 Notes
 -----
 • Cameras follow a sliding-window pattern: demo 0 → 0,1,2 ; demo 1 → 1,2,3 ; etc.
 • Each Plücker dataset is stored **once per camera per demo** (shape HxWx6) because
   the ray map depends only on camera geometry, not on timestep.
 • `total_cameras = num_views + (num_demos-1) * k + n_test_cams`.
"""
import os
import json
import h5py
import argparse
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import xml.etree.ElementTree as ET
import torch
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_base import EnvBase
from robomimic.scripts.cam_embedding import PluckerEmbedder  # for Plücker ray maps

def sample_lookat_poses(target_pos, radius, num_samples=1, world_up=np.array([0., 0., 1.])):
    """
    Randomly sample camera poses (position and rotation) around a target position.
    Cameras are placed on a sphere of given radius around target_pos, oriented to look at target_pos.
    """
    poses = []
    target_pos = np.array(target_pos)
    world_up = np.array(world_up) / np.linalg.norm(world_up)
    for _ in range(num_samples):
        # Random direction on sphere (biased upward)
        offset_dir = np.random.randn(3)
        offset_dir /= (np.linalg.norm(offset_dir) + 1e-6)
        if offset_dir[2] < 0:
            offset_dir *= -1  # flip to ensure camera is above ground
        cam_pos = target_pos + offset_dir * radius
        # Compute rotation matrix that orients camera toward target_pos
        forward = (target_pos - cam_pos)
        forward /= (np.linalg.norm(forward) + 1e-6)
        backward = -forward
        right = np.cross(world_up, backward)
        right /= (np.linalg.norm(right) + 1e-6)
        down = np.cross(backward, right)
        cam_rot = np.column_stack((right, down, backward))
        poses.append((cam_pos, cam_rot))
    return poses

def calculate_frustum_corners(cam_pos, corrected_cam_rot, fovy, width, height, depth=0.5):
    """
    Calculate the 3D coordinates of the four corner points of a camera's view frustum at a given depth.
    Used for visualization (drawing frustum edges).
    """
    aspect = width / float(height)
    fovy_rad = fovy * np.pi / 180.0
    h = 2 * depth * np.tan(fovy_rad / 2.0)
    w = h * aspect
    # Frustum corners in camera-local coordinates (centered at cam_pos)
    corners_cam = [
        np.array([ w/2,  h/2, depth]),
        np.array([-w/2,  h/2, depth]),
        np.array([-w/2, -h/2, depth]),
        np.array([ w/2, -h/2, depth]),
    ]
    # Transform corners to world coordinates
    return [cam_pos + corrected_cam_rot @ corner for corner in corners_cam]

def add_visualization_to_xml(env, camera_poses, camera_height, camera_width, frustum_depth=0.0625, time_id=0):
    """
    Modify the environment's XML to add visual markers for camera frustums and example rays.
    Draws a red sphere at camera origin, cyan spheres at frustum corners, yellow lines for frustum edges,
    and colored spheres along 5 sample rays (for sanity check of ray directions).
    """
    xml_string = env.env.sim.model.get_xml()  # current XML
    root = ET.fromstring(xml_string)
    worldbody = root.find(".//worldbody")
    # Use intrinsic parameters from the reference camera (index 0 in model camera list)
    ref_cam_id = 0
    fovy = env.env.sim.model.cam_fovy[ref_cam_id]
    # Colors for visualization
    origin_color = "1 0 0 1"   # red
    corner_color = "0 1 1 1"   # cyan
    edge_color = "1 1 0 1"     # yellow
    # Helper functions to format coordinates
    def point_str(p): return f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}"
    def segment_str(p1, p2): return f"{point_str(p1)} {point_str(p2)}"
    # Add geometry for each camera pose in camera_poses list
    for idx, (cam_pos, cam_rot) in enumerate(camera_poses):
        prefix = f"t{time_id}_c{idx}"  # unique prefix to avoid name collisions
        # Camera origin (red sphere)
        origin_sphere = ET.SubElement(worldbody, "geom", {
            "name": f"cam_origin_{prefix}", "type": "sphere", "size": "0.005",
            "pos": point_str(cam_pos), "rgba": origin_color, "group": "1",
            "contype": "0", "conaffinity": "0"
        })
        # Rotate camera frame for frustum calculation (flip Y-axis for MuJoCo camera orientation)
        R_y180 = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
        frustum_rot = cam_rot @ R_y180
        # Compute frustum corner points in world coordinates
        corners_world = calculate_frustum_corners(cam_pos, frustum_rot, fovy, camera_width, camera_height, depth=frustum_depth)
        # Corner spheres (cyan)
        for i, corner in enumerate(corners_world):
            ET.SubElement(worldbody, "geom", {
                "name": f"frustum_corner_{prefix}_{i}", "type": "sphere", "size": "0.0025",
                "pos": point_str(corner), "rgba": corner_color, "group": "1",
                "contype": "0", "conaffinity": "0"
            })
        # Sample 5 pixel coordinates for ray visualization (corners and one arbitrary interior point)
        sample_pixels = [(0, 0), (0, camera_width-1), (camera_height-1, 0), (camera_height-1, camera_width-1), (int(camera_height*0.4), int(camera_width*0.8))]
        sample_colors = ["1 0 0 1", "0 1 0 1", "0 0 1 1", "1 1 0 1", "1 0 1 1"]  # colors for the 5 rays
        # Build camera intrinsics matrix
        fovy_rad = fovy * np.pi / 180.0
        focal = (camera_height / 2.0) / np.tan(fovy_rad / 2.0)
        cx, cy = (camera_width - 1) / 2.0, (camera_height - 1) / 2.0
        K = np.array([[focal, 0.0, cx],
                      [0.0, focal, cy],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
        # Camera-to-world transform matrix
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = cam_rot
        c2w[:3, 3] = cam_pos
        # Use PluckerEmbedder to get ray origins and directions for the sample pixels
        embedder = PluckerEmbedder(img_size=(camera_height, camera_width), device=torch.device("cpu"))
        rays = embedder(torch.tensor(K, device=torch.device("cpu")).unsqueeze(0), torch.tensor(c2w).unsqueeze(0))
        origins = rays["origins"][0].numpy()   # shape (H, W, 3)
        viewdirs = rays["viewdirs"][0].numpy() # shape (H, W, 3)
        # Place a small sphere along each sample ray direction (colored points in the image)
        depth_along_ray = 0.2  # distance along ray to place the sphere
        for (py, px), rgba in zip(sample_pixels, sample_colors):
            point_world = origins[py, px] + depth_along_ray * viewdirs[py, px]
            ET.SubElement(worldbody, "geom", {
                "name": f"ray_point_{prefix}_{py}_{px}", "type": "sphere", "size": "0.01",
                "pos": point_str(point_world), "rgba": rgba, "group": "1",
                "contype": "0", "conaffinity": "0"
            })
        # Frustum edge lines (yellow cylinders connecting origin to each corner, and between corners)
        edge_radius = 0.00075
        for i, corner in enumerate(corners_world):
            # origin to corner
            ET.SubElement(worldbody, "geom", {
                "name": f"frustum_edge_origin_{prefix}_{i}", "type": "cylinder", "size": f"{edge_radius:.4f}",
                "fromto": segment_str(cam_pos, corner), "rgba": edge_color, "group": "1",
                "contype": "0", "conaffinity": "0"
            })
        for i in range(4):
            p1, p2 = corners_world[i], corners_world[(i+1) % 4]
            ET.SubElement(worldbody, "geom", {
                "name": f"frustum_edge_corner_{prefix}_{i}", "type": "cylinder", "size": f"{edge_radius:.4f}",
                "fromto": segment_str(p1, p2), "rgba": edge_color, "group": "1",
                "contype": "0", "conaffinity": "0"
            })
    # Return the modified XML string
    return ET.tostring(root, encoding='unicode')

def dataset_states_to_obs_rand(dataset_path, output_name, num_demos=None, camera_names=["agentview"],
                               camera_height=256, camera_width=256, num_views=3, k=1, done_mode=1,
                               sampling_radius=0.5, visualize=False, frustum_depth=0.0625):
    """
    Generate a new dataset with random camera images and corresponding Plücker ray maps.
    - num_demos: how many demos from the input dataset to process (if None or -1, use all).
    - num_views: number of camera views per demo.
    - k: number of views to replace with new ones for each subsequent demo.
    - visualize: if True, draw camera frustums and sample rays in the images.
    """
    # Create environment for data processing
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=camera_names, 
        camera_height=camera_height, 
        camera_width=camera_width,
        reward_shaping=False
    )

    # Prepare output files
    input_file = h5py.File(dataset_path, "r")
    demos = list(input_file["data"].keys())

    # Sort demos by index (assuming names like 'demo_0', 'demo_1', ...)
    demos.sort(key=lambda x: int(x.split('_')[-1]))
    if num_demos is not None and num_demos > 0:
        demos = demos[:num_demos]
    output_path = os.path.join(os.path.dirname(dataset_path), output_name)
    output_file = h5py.File(output_path, "w")
    data_grp = output_file.create_group("data")
    
    # -------------------------------------------------------------
    # Precompute all camera poses and Plücker maps (train + test)
    # -------------------------------------------------------------
    n_test_cams = 100  # held‑out cameras for evaluation
    total_needed_cams = num_views + max(0, len(demos) - 1) * k + n_test_cams

    target_pos = np.array([0.0, 0.0, 0.85])  # target the cameras look at
    embedder = PluckerEmbedder(img_size=(camera_height, camera_width), device=torch.device("cpu"))

    # Build shared intrinsics once (from reference camera)
    ref_cam_id_env = env.env.sim.model.camera_name2id(camera_names[0])
    fovy_ref = env.env.sim.model.cam_fovy[ref_cam_id_env]
    fovy_rad = fovy_ref * np.pi / 180.0
    focal_len = (camera_height / 2.0) / np.tan(fovy_rad / 2.0)
    cx_ref, cy_ref = (camera_width - 1) / 2.0, (camera_height - 1) / 2.0
    K_ref = np.array([[focal_len, 0.0, cx_ref],
                      [0.0, focal_len, cy_ref],
                      [0.0, 0.0, 1.0]], dtype=np.float32)

    global_cam_poses = {}
    global_plk_maps = {}

    sampled_poses = sample_lookat_poses(target_pos, sampling_radius, num_samples=total_needed_cams)
    for cam_id, (cam_pos, cam_rot) in enumerate(sampled_poses):
        global_cam_poses[cam_id] = (cam_pos, cam_rot)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = cam_rot
        c2w[:3, 3] = cam_pos
        rays = embedder(torch.tensor(K_ref).unsqueeze(0), torch.tensor(c2w).unsqueeze(0))
        global_plk_maps[cam_id] = rays["plucker"][0].numpy().astype(np.float32)

    total_samples = 0

    # Prepare mask filtering if exists
    new_mask_entries = {mask_key: np.array([], dtype='|S8') for mask_key in input_file.get("mask", {})}

    # Process each selected demo
    for idx, ep in enumerate(tqdm(demos, desc="Demos")):
        ep_group = input_file[f"data/{ep}"]
        states = ep_group["states"][()]       # shape (T, ...) low-dimensional states
        actions = ep_group["actions"][()]     # shape (T-1, ...) actions

        current_cam_indices = list(range(idx * k, idx * k + num_views))

        # Reset environment to initial state of this demo
        initial_state = {"states": states[0], "model": ep_group.attrs["model_file"]}
        env.reset_to(initial_state)  # load model and set initial simulation state

        # Store the base XML for resets
        base_xml = env.env.sim.model.get_xml()

        traj_obs_list = []
        traj_next_obs_list = []
        traj_rewards = []
        traj_dones = []
        obs = {}

        # Add non-image observations if available
        if hasattr(env, "observe"):
            obs = env.observe()
        elif hasattr(env.env, "get_obs"):
            obs = env.env.get_obs()
        
        # # Remove any default image keys from env obs (we will add our own)
        # for key in list(obs.keys()):
        #     if key.endswith("_image"):
        #         obs.pop(key)
        # assert no image keys in obs
        assert not any(key.endswith("_image") for key in obs.keys()), "Image keys should not be present in obs"
                
        # Render images for each camera view at initial state
        state_index = 0  # time step index
        for j, cam_id in enumerate(current_cam_indices):
            cam_pos, cam_rot = global_cam_poses[cam_id]

            # If visualization of frustums is enabled, add markers
            if visualize:
                if j > 0:
                    # Reset to base XML to remove previous frustum geoms, and restore state
                    env.reset()
                    env.env.reset_from_xml_string(base_xml)
                    env.reset_to({"states": states[state_index]})

                # Inject visualization geoms for this camera
                mod_xml = add_visualization_to_xml(env, [(cam_pos, cam_rot)], camera_height, camera_width,
                                                   frustum_depth=frustum_depth, time_id=state_index)
                
                # Save current sim state, load modified XML, then restore state
                qpos = env.env.sim.data.qpos.copy()
                qvel = env.env.sim.data.qvel.copy()
                env.reset()
                env.env.reset_from_xml_string(mod_xml)
                env.env.sim.data.qpos[:] = qpos[:env.env.sim.model.nq]
                env.env.sim.data.qvel[:] = qvel[:env.env.sim.model.nv]
                env.env.sim.forward()

            # Position the camera (overwriting any default position)
            # Important: there cannot be env.env.sim.forward() between the camera position and rendering
            # Otherwise, the camera pose will be overwritten by the default camera pose
            ref_cam_id = env.env.sim.model.camera_name2id(camera_names[0])
            env.env.sim.data.cam_xpos[ref_cam_id] = cam_pos
            env.env.sim.data.cam_xmat[ref_cam_id] = cam_rot.flatten()

            # Render RGB image from this camera
            img = env.env.sim.render(camera_name=camera_names[0], width=camera_width, height=camera_height, depth=False)
            img = np.flipud(img)  # flip vertical if needed (MuJoCo often returns images upside-down)
            obs[f"cam_{cam_id}_image"] = img  # attach image to observation
        
        # for i in range(min(len(actions), len(states) - 1)):
        for i in range(3):  # TEMP
            # Apply action or set next state
            # If not the last step, we directly reset to the next state (to exactly match recorded state)
            next_state = {"states": states[i+1]}
            # Ensure no leftover frustum geoms before moving to next state
            if visualize:
                env.reset()
                env.env.reset_from_xml_string(base_xml)
            next_state_obs = env.reset_to(next_state)
            state_index = i + 1
            
            # Create next observation dictionary
            next_obs = {}
            # Add non-image observations if available
            if hasattr(env, "observe"):
                next_obs = env.observe()
            elif hasattr(env.env, "get_obs"):
                next_obs = env.env.get_obs()
            
            # Remove any default image from next_state_obs
            for key in list(next_obs.keys()):
                if key.endswith("_image"):
                    next_obs.pop(key)
            # Render images for each camera view at the new state
            for j, cam_id in enumerate(current_cam_indices):
                cam_pos, cam_rot = global_cam_poses[cam_id]
                if visualize:
                    if j > 0:
                        env.reset() 
                        env.env.reset_from_xml_string(base_xml)
                        env.reset_to({"states": states[state_index]})
                    mod_xml = add_visualization_to_xml(env, [(cam_pos, cam_rot)], camera_height, camera_width,
                                                       frustum_depth=frustum_depth, time_id=state_index)
                    qpos = env.env.sim.data.qpos.copy()
                    qvel = env.env.sim.data.qvel.copy()
                    env.reset()
                    env.env.reset_from_xml_string(mod_xml)
                    env.env.sim.data.qpos[:] = qpos[:env.env.sim.model.nq]
                    env.env.sim.data.qvel[:] = qvel[:env.env.sim.model.nv]
                    env.env.sim.forward()
                ref_cam_id = env.env.sim.model.camera_name2id(camera_names[0])
                env.env.sim.data.cam_xpos[ref_cam_id] = cam_pos
                env.env.sim.data.cam_xmat[ref_cam_id] = cam_rot.flatten()
                img = env.env.sim.render(camera_name=camera_names[0], width=camera_width, height=camera_height, depth=False)
                img = np.flipud(img)
                next_obs[f"cam_{cam_id}_image"] = img
                
            r = env.get_reward() if hasattr(env, "get_reward") else 0.0
            success = False
            if hasattr(env, "is_success"):
                success = env.is_success().get("task", False)
            if done_mode == 0:
                done = bool(success)
            elif done_mode == 1:
                done = bool(i == len(actions) - 1)  # done at last step only
            elif done_mode == 2:
                done = bool(i == len(actions) - 1 or success)
            else:
                done = False
            # Record this transition
            traj_obs_list.append(obs)
            traj_next_obs_list.append(next_obs)
            traj_rewards.append(r)
            traj_dones.append(int(done))
            # Prepare for next iteration
            obs = next_obs  # set current obs to this next_obs for the next action
        # Convert lists of dicts to dict of arrays for obs and next_obs
        traj_obs = TensorUtils.list_of_flat_dict_to_dict_of_list(traj_obs_list)
        traj_next_obs = TensorUtils.list_of_flat_dict_to_dict_of_list(traj_next_obs_list)
        # Create a group for this episode in output file and save data
        ep_name = f"demo_{idx}"
        ep_out = data_grp.create_group(ep_name)
        ep_out.create_dataset("actions", data=np.array(actions))
        ep_out.create_dataset("states", data=np.array(states))
        ep_out.create_dataset("rewards", data=np.array(traj_rewards))
        ep_out.create_dataset("dones", data=np.array(traj_dones))
        # Save observations and next_observations
        for key in traj_obs:
            ep_out.create_dataset(f"obs/{key}", data=np.array(traj_obs[key]))
            ep_out.create_dataset(f"next_obs/{key}", data=np.array(traj_next_obs[key]))
        # record which cameras this demo uses
        ep_out.attrs["cam_inds"] = json.dumps(current_cam_indices)
        ep_out.attrs["model_file"] = ep_group.attrs["model_file"]
        ep_out.attrs["num_samples"] = actions.shape[0]  # number of transitions
        # Update mask filters if present
        for mask_key in new_mask_entries:
            if ep.encode("utf-8") in input_file["mask"][mask_key]:
                new_mask_entries[mask_key] = np.append(new_mask_entries[mask_key], ep_name.encode("utf-8"))
        total_samples += actions.shape[0]

    # Write mask groups if any
    if "mask" in input_file:
        mask_grp = output_file.create_group("mask")
        for mask_key, arr in new_mask_entries.items():
            mask_grp.create_dataset(mask_key, data=arr)
    # Write global metadata
    data_grp.attrs["total"] = total_samples
    data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4)
    
    # Add metadata about all camera poses
    all_cam_info = []
    for cam_id, (cam_pos, cam_rot) in global_cam_poses.items():
        all_cam_info.append({
            "id": int(cam_id),
            "position": cam_pos.tolist(),
            "rotation": cam_rot.tolist()
        })
    data_grp.attrs["all_camera_poses"] = json.dumps(all_cam_info)
    # define camera index sets
    test_start = total_needed_cams - n_test_cams
    train_cam_inds = list(range(0, test_start))
    test_cam_inds = list(range(test_start, total_needed_cams))
    data_grp.attrs["train_cam_inds"] = json.dumps(train_cam_inds)
    data_grp.attrs["test_cam_inds"] = json.dumps(test_cam_inds)
    data_grp.attrs["num_total_cams"] = len(global_cam_poses)

    # ----------------------------------------------------------
    # Save train camera Plücker maps (no per‑demo duplication)
    # ----------------------------------------------------------
    train_grp = data_grp.create_group("train_plucker")
    for cam_id in train_cam_inds:
        train_grp.create_dataset(f"cam_{cam_id}_plucker", data=global_plk_maps[cam_id])

    # ----------------------------------------------------------
    # Save held‑out test camera Plücker maps and poses
    # ----------------------------------------------------------
    test_grp = data_grp.create_group("test_plucker")
    for cam_id in range(test_start, total_needed_cams):
        test_grp.create_dataset(f"cam_{cam_id}_plucker", data=global_plk_maps[cam_id])

    # Close files and environment
    input_file.close()
    output_file.close()
    env.env.close()
    print(f"Saved {len(demos)} demos to {output_path}")
    print(f"Included Plücker ray maps for {len(global_cam_poses)} unique views in the same file")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="/share/data/ripl/tianchong/vista/data/low_dim_v141.hdf5")
    parser.add_argument("--output_name", type=str, default="low_dim_v141_randcam_test.hdf5")
    parser.add_argument("--num_demos", type=int, default=3, help="Total number of demos to process (all if -1)")
    parser.add_argument("--num_views", type=int, default=3, help="Number of random camera views per demo")
    parser.add_argument("--k", type=int, default=1, help="Number of camera views to replace for each new demo")
    parser.add_argument("--camera_names", type=str, nargs='+', default=["agentview"], help="Camera name(s) in the environment to use for rendering")
    parser.add_argument("--camera_height", type=int, default=256, help="Height of rendered camera images")
    parser.add_argument("--camera_width", type=int, default=256, help="Width of rendered camera images")
    parser.add_argument("--done_mode", type=int, default=1, help="Done flag mode (0=success only, 1=horizon, 2=any of success or horizon)")
    parser.add_argument("--sampling_radius", type=float, default=0.5, help="Radius for random camera placement around target")
    parser.add_argument("--visualize", default=False, help="Enable visualization of camera frustums and rays in images")
    parser.add_argument("--frustum_depth", type=float, default=0.0625, help="Depth of frustum visualization (in meters)")
    args = parser.parse_args()
    dataset_states_to_obs_rand(
        dataset_path=args.dataset,
        output_name=args.output_name,
        num_demos=(None if args.num_demos is None or args.num_demos < 0 else args.num_demos),
        camera_names=args.camera_names,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        num_views=args.num_views,
        k=args.k,
        done_mode=args.done_mode,
        sampling_radius=args.sampling_radius,
        visualize=args.visualize,
        frustum_depth=args.frustum_depth
    )
