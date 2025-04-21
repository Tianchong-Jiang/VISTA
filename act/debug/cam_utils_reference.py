# import os
# os.system("export PYTHONPATH=/home/tianchong/workspace/CamPoseManiskill/maniskill:$PYTHONPATH")

import numpy as np
import cv2
import gymnasium as gym
from mani_skill.envs.sapien_env import BaseEnv

# def get_default_cam(idx, total, ood, n):
#     return 'render_camera'

def get_default_cam(ep_idx, total_eps, distr, cam_args):
    return 'render_camera'

# def get_n_cam_6d(idx, total, ood, n):
#     cam_idx = idx * n // total
#     if ood == 'id':
#         return f'cam_{cam_idx}'
#     elif ood == 'ood':
#         return f'cam_{1000 + cam_idx}'
#     else:
#         raise ValueError(f"Unknown ood {ood}")

def get_n_cam_6d(ep_idx, total_eps, distr, cam_args):
    cam_idx = ep_idx * cam_args['n_cam'] // total_eps
    if distr == 'train':
        return f'cam_{cam_idx}'
    elif distr == 'test':
        return f'cam_{1000 + cam_idx}'
    else:
        raise ValueError(f"Unknown distribution {distr}")

# def get_n_cam_per_eps_6d(idx, total, ood, n):
#     cam_idx = np.random.randint(n)
#     if ood == 'id':
#         return f'cam_{cam_idx}'
#     elif ood == 'ood':
#         return f'cam_{1000 + cam_idx}'
#     else:
#         raise ValueError(f"Unknown ood {ood}")

def get_n_cam_per_eps_6d(ep_idx, total_eps, distr, cam_args):
    cam_idx = np.random.randint(cam_args['n_cam'])
    if distr == 'train':
        return f'cam_{cam_idx}'
    elif distr == 'test':
        return f'cam_{1000 + cam_idx}'
    else:
        raise ValueError(f"Unknown distribution {distr}")

# def get_stair_shaped(idx, total, ood, n):
#     if ood == 'id':
#         change = 1
#         start = idx * change
#         cam_idx = np.random.randint(start, start + 3)
#         assert cam_idx < 1000, f"we only have 1000 cameras for training"
#         return f'cam_{cam_idx}'
#     elif ood == 'ood':
#         cam_idx = np.random.randint(1000, 1100)
#         return f'cam_{cam_idx}'
#     else:
#         raise ValueError(f"Unknown ood {ood}")

def get_stair_shaped(ep_idx, total_eps, distr, cam_args):
    if distr == 'train':
        start = ep_idx * cam_args['change']
        cam_idx = np.random.randint(start, start + cam_args['cam_per_ep'])
        assert cam_idx < 1000, f"Sorry, we only have max of 1000 cameras for training"
        return f'cam_{cam_idx}'
    elif distr == 'test':
        cam_idx = np.random.randint(1000, 1100)
        return f'cam_{cam_idx}'
    else:
        raise ValueError(f"Unknown distribution {distr}")

# def get_n_6d_cam_per_eps_sep(idx, total, ood, n):
#     cam_idx = np.random.randint(n)
#     if ood == 'id':
#         return f'cam_{cam_idx}'
#     elif ood == 'ood':
#         return f'cam_{1100 + cam_idx}'
#     else:
#         raise ValueError(f"Unknown ood {ood}")


def test():
    env: BaseEnv = gym.make(
        'PullCube-v1',
        obs_mode=None,
        control_mode='pd_joint_pos',
        # render_mode="rgb_array",
        render_mode="human",
        sim_backend='cpu',
        render_backend='cpu',
        # enable_shadow=False,
    )
    env.reset()

    while True:
        env.step(np.zeros(8))
        env.render()

    # for i in range(100):
    #     image = env.unwrapped.render_rgb_array(f'cam_{i}')
    #     image = image.cpu().numpy()[0]
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #     cv2.imwrite(f'/lmn/maniskill/output/cam_{i}.png', image)

def test_plucker():
    from act.cam_embedding import PluckerEmbedder
    from mani_skill.utils import sapien_utils
    from mani_skill.utils.building import actors


    embedder = PluckerEmbedder(device='cpu')

    env: BaseEnv = gym.make(
        'LiftPegUpright-v1',
        obs_mode=None,
        control_mode='pd_joint_pos',
        # render_mode="rgb_array",
        render_mode="human",
        sim_backend='cpu',
        render_backend='cpu',
        # enable_shadow=False,
    )

    env.reset()

    cam_name = 'cam_0'
    for name, camera in env.unwrapped.scene.human_render_cameras.items():
        if name == cam_name:
            cam_config = camera.config.pose.raw_pose[0] # torch.Size([7])
            intrinsics = camera.get_params()["intrinsic_cv"]
            cam2world = camera.get_params()["cam2world_gl"]
            break

    # import pdb; pdb.set_trace()

    plucker_embedding = embedder(intrinsics, cam2world)["plucker"].permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
    origins = embedder(intrinsics, cam2world)["origins"].permute(0, 3, 1, 2)
    viewdirs = embedder(intrinsics, cam2world)["viewdirs"].permute(0, 3, 1, 2)

    breakpoint()

    # Define all four corners
    corners = np.array([[0, 0], [0, 255], [255, 0], [255, 255]])

    # Define four distinct colors for the spheres (in RGBA format)
    sphere_colors = [
        [1, 0, 0, 1],  # Red
        [0, 1, 0, 1],  # Green
        [0, 0, 1, 1],  # Blue
        [1, 1, 0, 1],  # Yellow
    ]

    # Loop over the corners with their corresponding color index
    for i, corner in enumerate(corners):
        origin = origins[0, :, corner[0], corner[1]]
        viewdir = viewdirs[0, :, corner[0], corner[1]]
        lookat = origin + viewdir * 0.2
        pose = sapien_utils.look_at(origin, lookat)

        # Build a two-color peg (this part remains unchanged)
        actors.build_twocolor_peg(
            env.unwrapped.scene,
            length=1,
            width=0.002,
            color_1=[0, 0, 1, 1],
            color_2=[0, 1, 0, 1],
            name=f"peg_{corner}",
            body_type="static",
            initial_pose=pose
        )

        # Build a sphere with a distinct color for each corner
        actors.build_sphere(
            env.unwrapped.scene,
            radius=0.02,
            color=sphere_colors[i],
            name=f"cam_{corner}",
            initial_pose=sapien_utils.look_at(lookat, lookat),
            body_type="static",
            add_collision=False
        )

    line = np.array([[0, 10], [15, 20], [30, 30], [45, 40], [60, 50], [75, 60], [90, 70], [105, 80]])
    for element in line:
        origin = origins[0, :, element[0], element[1]]
        viewdir = viewdirs[0, :, element[0], element[1]]
        lookat = origin + viewdir * 0.2
        pose = sapien_utils.look_at(origin, lookat)

        # Build a two-color peg (this part remains unchanged)
        actors.build_twocolor_peg(
            env.unwrapped.scene,
            length=1,
            width=0.002,
            color_1=[0, 0, 1, 1],
            color_2=[0, 1, 0, 1],
            name=f"peg_{element}",
            body_type="static",
            initial_pose=pose
        )

        # Build a sphere with a distinct color for each corner
        actors.build_sphere(
            env.unwrapped.scene,
            radius=0.02,
            color=[1, 1, 1, 1],
            name=f"cam_{element}",
            initial_pose=sapien_utils.look_at(lookat, lookat),
            body_type="static",
            add_collision=False
        )


    image = env.unwrapped.render_rgb_array(cam_name)

    # print color of image at corner
    print(f"color at corner {corners[0]}: {image[0, corners[0][0], corners[0][1]]}")
    print(f"color at corner {corners[1]}: {image[0, corners[1][0], corners[1][1]]}")


    while True:
        env.step(np.zeros(8))
        env.render()

if __name__ == '__main__':
    # test_plucker()
    test()