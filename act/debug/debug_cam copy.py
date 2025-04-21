import os
import cv2
import numpy as np
import robosuite as suite
import xml.etree.ElementTree as ET
import robosuite.utils.transform_utils as T

# Import the new sampling function
from ..cam_utils import sample_lookat_poses

# --- Helper Functions for Camera Geometry --- 
def get_camera_extrinsic_matrix(sim, camera_name):
    """Gets the corrected 4x4 extrinsic matrix for a camera."""
    cam_id = sim.model.camera_name2id(camera_name)
    camera_pos = sim.data.cam_xpos[cam_id].copy()
    camera_rot = sim.data.cam_xmat[cam_id].copy().reshape(3, 3)
    # Rotation to convert MuJoCo camera frame (X right, Y down, Z backward) 
    # to standard camera frame (X right, Y up, Z backward/out - depends on convention, using Z out)
    mujoco_to_standard_camera_rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    corrected_rot = camera_rot @ mujoco_to_standard_camera_rotation
    extrinsic_matrix = T.make_pose(camera_pos, corrected_rot)
    return extrinsic_matrix

def get_camera_params(sim, camera_name):
    """Gets camera position, corrected rotation matrix, fovy, and extrinsic matrix."""
    # This function is now less relevant as we sample poses, but keep it for fovy retrieval
    cam_id = sim.model.camera_name2id(camera_name)
    fovy = sim.model.cam_fovy[cam_id]
    # Original extrinsic calculation might not be needed unless debugging
    # extrinsic_matrix = get_camera_extrinsic_matrix(sim, camera_name)
    # cam_pos = extrinsic_matrix[:3, 3]
    # corrected_cam_rot = extrinsic_matrix[:3, :3]
    # return cam_pos, corrected_cam_rot, fovy, extrinsic_matrix
    return None, None, fovy, None # Return fovy, other values aren't used directly

def calculate_frustum_corners(cam_pos, corrected_cam_rot, fovy, width, height, depth=0.5):
    """Calculates the 4 corners of the camera frustum base using corrected rotation."""
    # Reverted to original definition
    aspect = width / height
    fovy_rad = fovy * np.pi / 180.0
    # Height and width of the far plane in standard camera coordinates
    h_at_depth = 2 * depth * np.tan(fovy_rad / 2.0) 
    w_at_depth = h_at_depth * aspect
    # Corners in standard camera frame (X right, Y up, Z forward -> +Z for far plane)
    corners_cam_standard = [
        np.array([ w_at_depth / 2,  h_at_depth / 2, depth]), # Top-right (+Z)
        np.array([-w_at_depth / 2,  h_at_depth / 2, depth]), # Top-left (+Z)
        np.array([-w_at_depth / 2, -h_at_depth / 2, depth]), # Bottom-left (+Z)
        np.array([ w_at_depth / 2, -h_at_depth / 2, depth]), # Bottom-right (+Z)
    ]
    # Transform corners to world frame using the provided camera rotation matrix
    # NOTE: Assumes corrected_cam_rot aligns standard camera frame (X right, Y up, Z out) with world frame
    corners_world = [cam_pos + corrected_cam_rot @ p_cam for p_cam in corners_cam_standard]
    return corners_world
# --- End Helper Functions ---

def main():
    # Create output directory
    output_dir = '/share/data/ripl/tianchong/vista/rendered'
    os.makedirs(output_dir, exist_ok=True)
    
    img_width, img_height = 256, 256
    reference_camera = "agentview" # Use agentview's fovy for calculations
    render_camera = "frontview"
    frustum_depth = 0.0625 # 1/4 previous depth -> shorter frustum edges
    edge_radius = 0.00075 # 1/4 previous size
    num_sampled_poses = 50
    cube_target_pos = np.array([0.0, 0.0, 0.85]) # Approximate cube position
    sampling_radius = 0.5 # How far from the cube to sample camera positions
    
    print(f"Creating environment...")
    
    # Create environment with both cameras initially (need agentview for fovy)
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=[render_camera, reference_camera],
        camera_heights=img_height,
        camera_widths=img_width,
    )
    
    # Reset the environment to get initial state and image
    obs = env.reset()
    initial_img = env.sim.render(
        camera_name=render_camera,
        width=img_width,
        height=img_height,
        depth=False
    )
    initial_img = np.flipud(initial_img)
    
    # Save the initial image (before frustums)
    cv2.imwrite(os.path.join(output_dir, 'before_vis.png'), cv2.cvtColor(initial_img.astype(np.uint8), cv2.COLOR_RGB2BGR))
    
    # --- Prepare XML Modifications --- 
    # 1. Sample Camera Poses
    print(f"Sampling {num_sampled_poses} camera poses looking at {cube_target_pos} with radius {sampling_radius}...")
    sampled_poses = sample_lookat_poses(cube_target_pos, sampling_radius, num_samples=num_sampled_poses)
    
    # 2. Get fovy from the reference camera model (assume constant for frustum calculations)
    try:
        ref_cam_id = env.sim.model.camera_name2id(reference_camera)
        fovy = env.sim.model.cam_fovy[ref_cam_id]
        print(f"Using fovy={fovy} from camera '{reference_camera}' for frustum calculations.")
    except ValueError as e:
        print(f"Error getting fovy for reference camera '{reference_camera}': {e}")
        env.close()
        return

    # 3. Get the current model XML
    xml_string = env.sim.model.get_xml()
    root = ET.fromstring(xml_string)
    worldbody = root.find(".//worldbody")
    if worldbody is None:
        print("Error: Could not find worldbody in XML!")
        env.close()
        return

    print(f"Adding {num_sampled_poses} frustum geoms to XML...")
    
    # Define colors
    origin_color_rgba = "1 0 0 1" # Red origin marker
    corner_color_rgba = "0 1 1 1" # Cyan corner marker
    edge_color_rgba = "1 1 0 1"   # Yellow edge marker
    
    # Helper for unique geom names
    pose_idx_str = lambda idx: f"pose{idx}"
    
    # Helper to format point to string
    def point_to_str(p):
        return f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}"
        
    # Helper to format fromto string
    def coords_to_str(p1, p2):
        return f"{point_to_str(p1)} {point_to_str(p2)}"

    # 4. Loop through sampled poses and add frustum geoms to XML
    for idx, (cam_pos, cam_rot) in enumerate(sampled_poses):
        # --- Coordinate Frame Correction (Reverting to state user indicated was correct) --- 
        R_y180 = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
        rot_for_frustum = cam_rot @ R_y180 # Reverted calculation
        # --- End Correction ---
        
        # Calculate frustum corners for this pose using the reverted rotation
        frustum_corners = calculate_frustum_corners(cam_pos, rot_for_frustum, fovy, img_width, img_height, depth=frustum_depth)
        p_str = pose_idx_str(idx)

        # Add Origin Sphere (RED) - Use original cam_pos
        geom_origin = ET.SubElement(worldbody, "geom")
        geom_origin.set("name", f"cam_origin_vis_{p_str}")
        geom_origin.set("type", "sphere")
        geom_origin.set("size", "0.005") # 1/4 previous size
        geom_origin.set("pos", point_to_str(cam_pos))
        geom_origin.set("rgba", origin_color_rgba)
        geom_origin.set("group", "1")
        geom_origin.set("contype", "0")
        geom_origin.set("conaffinity", "0")

        # Add Corner Spheres (CYAN)
        for i, corner_pos in enumerate(frustum_corners):
            geom_corner = ET.SubElement(worldbody, "geom")
            geom_corner.set("name", f"frustum_corner_vis_{p_str}_{i}")
            geom_corner.set("type", "sphere")
            geom_corner.set("size", "0.005") # 1/4 previous size
            geom_corner.set("pos", point_to_str(corner_pos))
            geom_corner.set("rgba", corner_color_rgba)
            geom_corner.set("group", "1")
            geom_corner.set("contype", "0")
            geom_corner.set("conaffinity", "0")

        # Add Cylinder Edges (YELLOW)
        # Edges from origin to corners
        for i, corner_pos in enumerate(frustum_corners):
            length = np.linalg.norm(corner_pos - cam_pos)
            if length < 1e-6: continue
            edge_geom = ET.SubElement(worldbody, "geom")
            edge_geom.set("name", f"frustum_edge_origin_{p_str}_{i}")
            edge_geom.set("type", "cylinder")
            edge_geom.set("fromto", coords_to_str(cam_pos, corner_pos))
            edge_geom.set("size", f"{edge_radius:.4f}")
            edge_geom.set("rgba", edge_color_rgba)
            edge_geom.set("group", "1")
            edge_geom.set("contype", "0")
            edge_geom.set("conaffinity", "0")

        # Edges connecting far plane corners
        for i in range(4):
            start_pos = frustum_corners[i]
            end_pos = frustum_corners[(i + 1) % 4]
            length = np.linalg.norm(end_pos - start_pos)
            if length < 1e-6: continue
            edge_geom = ET.SubElement(worldbody, "geom")
            edge_geom.set("name", f"frustum_edge_far_{p_str}_{i}")
            edge_geom.set("type", "cylinder")
            edge_geom.set("fromto", coords_to_str(start_pos, end_pos))
            edge_geom.set("size", f"{edge_radius:.4f}")
            edge_geom.set("rgba", edge_color_rgba)
            edge_geom.set("group", "1")
            edge_geom.set("contype", "0")
            edge_geom.set("conaffinity", "0")
        
    # --- End XML Modifications --- 

    # 5. Convert back to string and reset environment
    modified_xml = ET.tostring(root, encoding='unicode')
    # Optional: Save modified XML for debugging
    # with open(os.path.join(output_dir, "modified_scene.xml"), "w") as f:
    #     f.write(modified_xml)
    print("Resetting environment with XML containing 200 frustums...")
    current_qpos = env.sim.data.qpos.copy()
    current_qvel = env.sim.data.qvel.copy()
    
    try:
        env.reset_from_xml_string(modified_xml)
        # Attempt to restore state if needed (might not be crucial for just rendering)
        env.sim.data.qpos[:] = current_qpos[:env.sim.model.nq]
        env.sim.data.qvel[:] = current_qvel[:env.sim.model.nv]
        env.sim.forward()
        print("Environment reloaded with frustum visualizations.")
    except Exception as e:
        print(f"Error resetting environment from modified XML: {e}")
        env.close()
        return
    
    # 6. Render final image from frontview ONLY
    print(f"Rendering final image from {render_camera}...")
    final_img_frontview = env.sim.render(
        camera_name=render_camera,
        width=img_width,
        height=img_height,
        depth=False
    )
    final_img_frontview = np.flipud(final_img_frontview)
    # Save the frontview image
    cv2.imwrite(os.path.join(output_dir, 'frontview_frame.png'), cv2.cvtColor(final_img_frontview.astype(np.uint8), cv2.COLOR_RGB2BGR))
    print(f"  Saved frontview_frame.png (shows all {num_sampled_poses} frustums)")

    # 7. Render from each sampled pose
    print(f"Rendering {num_sampled_poses} images, one from each sampled pose...")
    try:
        ref_cam_id = env.sim.model.camera_name2id(reference_camera)
    except ValueError:
        print(f"Error: Could not find reference camera '{reference_camera}' in reloaded model to render agent views.")
        env.close()
        return
        
    for idx, (cam_pos, cam_rot) in enumerate(sampled_poses):
        # Update the reference camera's pose in the simulation data
        env.sim.data.cam_xpos[ref_cam_id] = cam_pos
        env.sim.data.cam_xmat[ref_cam_id] = cam_rot.flatten()
        # We might need env.sim.forward() here if camera pose affects other sim elements needed for rendering,
        # but often it's not strictly necessary just for camera rendering itself. Let's try without first.
        # env.sim.forward() 
        
        # Render from the updated camera pose
        img_agentview_pose = env.sim.render(
            camera_name=reference_camera,
            width=img_width,
            height=img_height,
            depth=False
        )
        img_agentview_pose = np.flipud(img_agentview_pose)
        
        # Save the image
        img_filename = f"agentview_pose_{idx}.png"
        cv2.imwrite(os.path.join(output_dir, img_filename), cv2.cvtColor(img_agentview_pose.astype(np.uint8), cv2.COLOR_RGB2BGR))

    print(f"  Saved {num_sampled_poses} agentview_pose_*.png images.")
    
    env.close()

if __name__ == "__main__":
    main()