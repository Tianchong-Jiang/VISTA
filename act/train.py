import os
import sys
import resource
os.environ['WANDB_API_KEY'] = open('/share/data/ripl/tianchong/vista/.secret.env').read().split('=')[1].strip().strip('"')

# resource.setrlimit(resource.RLIMIT_NOFILE, (32768, 32768))

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robosuite.utils.camera_utils import CameraMover

import time
import torch
import numpy as np
import pickle
import argparse
from copy import deepcopy
import tqdm

from .utils import load_data
from .utils import compute_dict_mean, set_seed, detach_dict
from .utils import cleanup_ckpt, get_last_ckpt
from .utils import cosine_schedule_with_warmup, cosine_schedule
from .policy import ACTPolicy, ViTPolicy, DecOnlyPolicy
from .eval import Evaluator

import wandb

# Robomimic environment configuration
TASK_CONFIGS = {
    'lift': {
        'control_mode': 'ee_delta_pose', 
        'action_dim': 7,
        'data_path': '/share/data/ripl/tianchong/vista/data/low_dim_v141'
    },
    'can': {
        'control_mode': 'ee_delta_pose', 
        'action_dim': 7,
        'data_path': '/share/data/ripl/tianchong/vista/data/can_data'
    }
}



def main(args, ckpt=None):
    start_time = time.time()
    set_seed(1)

    print("Setting up Robomimic ENV...")
    env_id = args['task_name']
    
    # Get environment metadata from dataset
    dataset_path = TASK_CONFIGS[env_id]['data_path'] + '_obs.hdf5'  # Use the dataset with observations
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)
    
    # Use our custom function that works around version issues
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=["agentview"],
        camera_height=256, 
        camera_width=256, 
        reward_shaping=False,
        use_depth_obs=False,
        use_image_obs=False,
    )
    print("Setting up ENV... Done")

    # Use dataset with observations for training
    data_path = TASK_CONFIGS[env_id]['data_path'] + '_obs'

    # Updated load_data call with new signature (without environment)
    train_dataloader, val_dataloader, stats = load_data(
        dataset_path=data_path + '.hdf5',
        num_demos=args['num_episodes'] + 10,
        batch_size_train=args['batch_size'],
        batch_size_val=args['batch_size'],
        camera_names=args['camera_names'],
        transform=args['transform']
    )

    # save dataset stats
    if not os.path.isdir(args['ckpt_dir']):
        os.makedirs(args['ckpt_dir'])
    stats_path = os.path.join(args['ckpt_dir'], f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    set_seed(args['seed'])

    args.update({
        'enc_layers': 4,
        'dec_layers': 7,
        'nheads': 8,
        'obs_dim': 9,
        'action_dim': TASK_CONFIGS[env_id]['action_dim'],
        'lr_backbone': args['lr'],
        'num_queries': args['chunk_size'],
        'dropout': args['dropout'],
        'pre_norm': True,
        })

    if args['policy_class'] == 'ACT':
        policy = ACTPolicy(args)
    elif args['policy_class'] == 'ViT':
        policy = ViTPolicy(args)
    elif args['policy_class'] == 'Dec':
        policy = DecOnlyPolicy(args)
    else:
        raise NotImplementedError
    policy.cuda()
    optimizer = policy.configure_optimizers()
    warmup_steps = args['warmup_epochs'] * (args['num_episodes'] // args['batch_size'] + 1)
    total_steps = args['num_epochs'] * (args['num_episodes'] // args['batch_size'] + 1)
    scheduler = cosine_schedule(optimizer, total_steps=total_steps)
    
    if ckpt is not None:
        policy.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        print(f"Loaded ckpt from {ckpt['epoch']} epoch")

    # Create evaluator with the robomimic environment
    evaluator = Evaluator(env, stats, use_plucker=args['use_plucker'], chunk_size=args['chunk_size']//2, render=True, max_steps=200)

    min_val_loss = np.inf
    max_val_success = -np.inf
    best_ckpt_info = None
    step = 0 if ckpt is None else ckpt['step'] + 1
    epoch = 0 if ckpt is None else ckpt['epoch'] + 1

    pbar = tqdm.tqdm(total=args['num_epochs'] - epoch, desc="Training progress")
    last_epoch_time = time.time()
    while epoch < args['num_epochs']:
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)

            if epoch % args['eval_every'] == 0:
                for distr in ['train']:#, 'test']:
                    total_success = 0.0
                    for i in range(args['eval_nruns']):
                        video_dir = os.path.join(args['ckpt_dir'], 'videos')
                        os.makedirs(video_dir, exist_ok=True)
                        _, success, _ = evaluator.evaluate(
                            policy,
                            save_path=video_dir,  # Just provide the directory
                            video_prefix=f'{distr}_epoch_{epoch:04d}_run_{i:02d}',  # Use a meaningful prefix
                            camera_name=args['camera_names'][0]
                        )
                        total_success += success

                    success_rate = total_success / args['eval_nruns']
                    wandb.log({f'{distr}_success': success_rate}, step=epoch)
                    
                if max_val_success <= success_rate:
                    max_val_success = success_rate
                    best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))

            for k, v in epoch_summary.items():
                wandb.log({f'val_{k}': v.item()}, step=epoch)

        # training
        train_history = []
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            step += 1
            forward_dict = forward_pass(data, policy)
            loss = forward_dict['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.1)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history)

        for k, v in epoch_summary.items():
            wandb.log({f'train_{k}': v.item()}, step=epoch)
        wandb.log({'step': step}, step=epoch)
        wandb.log({'lr': scheduler.get_last_lr()[0]}, step=epoch)

        if epoch % args['eval_every'] == 0:
            ckpt_path = os.path.join(args['ckpt_dir'], f'epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'step': step,
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss,
                'wandb_id': wandb.run.id,
            }, ckpt_path)
            cleanup_ckpt(args['ckpt_dir']) # keep last 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 60 * 60 * 7:
                print(f"Elapsed time: {elapsed_time / 60 / 60} hours")
                return

        # Update progress bar with timing info
        current_time = time.time()
        sec_per_epoch = current_time - last_epoch_time
        last_epoch_time = current_time
        pbar.set_description(f"Epoch {epoch} done")
        pbar.update(1)
        pbar.set_postfix({"sec/epoch": f"{sec_per_epoch:.2f}"})
        epoch += 1
    
    pbar.close()
    wandb.finish(exit_code=0)


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad, cam_config = data
    image_data, qpos_data, action_data, is_pad, cam_config = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda(), cam_config.cuda()
    
    # Reshape the image tensor to have format [batch, num_views=1, channels, height, width]
    # Current shape: [batch, channels, height, width]
    image_data = image_data.unsqueeze(1)  # Add num_views dimension: [batch, 1, channels, height, width]
    
    # Extract just the robot state + gripper dimensions (first 9 dimensions)
    # The 32-dimensional vector includes environment state that we don't need
    qpos_data = qpos_data[:, :9]  # Use just the first 9 dimensions that correspond to robot + gripper
    
    return policy(qpos_data, image_data, action_data, is_pad, cam_config)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Train ACT policy for robomimic environments")
    parser.add_argument('--task_name', default='lift', choices=['lift', 'can'], 
                        help='Task to train on (lift, can)')
    parser.add_argument('--policy_class', default='ACT', choices=['ACT', 'ViT', 'Dec'], 
                        help='Policy architecture to use')
    parser.add_argument('--num_episodes', default=180, type=int, 
                        help='Number of episodes to use for training')
    parser.add_argument('--backbone', default='resnet18', choices=['resnet18', 'lin'], 
                        help='Backbone architecture')
    parser.add_argument('--batch_size', default=180, type=int, 
                        help='Batch size for training')
    parser.add_argument('--epochs', default=5000, type=int, 
                        help='Number of epochs to train')
    
    args = parser.parse_args()
    
    # Create the full configuration
    config = {
        'task_name': args.task_name,
        'camera_names': ['agentview'],
        'num_episodes': args.num_episodes,
        'policy_class': args.policy_class,
        'backbone': args.backbone,
        'batch_size': args.batch_size,
        'seed': 1,
        'num_epochs': args.epochs,
        'warmup_epochs': 100,
        'lr': 2e-5,
        'eval_nruns': 10,
        'eval_every': 100,
        'plucker_as_pe': False,
        'use_plucker': False, # when false, masked to 0 in data loader
        'use_proprio': False, # when false, masked to 0 in data loader
        'use_cam_pose': False, # when false, masked to 0 in data loader
        'transform': 'crop',
        'dropout': 0.1,
        'norm_type': 'layer',
        'kl_weight': 1,
        'chunk_size': 10,
        'hidden_dim': 512,
        'dim_feedforward': 2048,
        'weight_decay': 1e-4,
        'position_embedding': 'sine',
        'patch_size': 16,
        'dilation': False,
        'embed_method': 'lin',
        'activation': 'relu',
    }
    
    # Create a name for this run
    name = (
        f"{config['policy_class']}_{config['task_name']}_{config['num_episodes']}_eps"
        f"_robomimic_{config['backbone']}_180_demos_pred_10_chunk"
    )
    config['ckpt_dir'] = os.path.join("/share/data/ripl/tianchong/vista/checkpoints", name)
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(config['ckpt_dir'], exist_ok=True)
    
    # Check for existing checkpoint to resume
    ckpt_path = get_last_ckpt(config['ckpt_dir'])
    
    # Initialize wandb
    project = 'robomimic_training'
    
    if ckpt_path is not None:
        print(f"Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path)
        wandb_id = ckpt['wandb_id']
        wandb.init(project=project, id=wandb_id, resume='must')
        config = dict(wandb.config)  # Get config from wandb
        main(config, ckpt)
    else:
        print(f"Starting new training run: {name}")
        wandb.init(project=project, name=name, config=config)
        main(config)
