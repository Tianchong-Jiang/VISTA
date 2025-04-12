import os
import resource
resource.setrlimit(resource.RLIMIT_NOFILE, (32768, 32768))

import time
import torch
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import pickle
import argparse
from copy import deepcopy
from tqdm import tqdm
from functools import partial

from act.utils import load_data # data f unctions v  v
from act.utils import compute_dict_mean, set_seed, detach_dict # helper functions
from act.utils import TASK_CONFIGS
from act.utils import cleanup_ckpt, get_last_ckpt
from act.utils import cosine_schedule_with_warmup, cosine_schedule
from act.policy import ACTPolicy, ViTPolicy, DecOnlyPolicy
from act.eval import Evaluator

import gymnasium as gym
from mani_skill.envs.sapien_env import BaseEnv

from mani_skill.cam_utils import *
import wandb

def main(args, ckpt=None):
    start_time = time.time()
    set_seed(1)

    print("Setting up ENV...")
    env_id = args['task_name']
    env: BaseEnv = gym.make(
        env_id,
        obs_mode=None,
        control_mode=TASK_CONFIGS[env_id]['control_mode'],
        render_mode="rgb_array",
        sim_backend='gpu',
    )
    env.reset()
    print("Setting up ENV... Done")

    data_path = TASK_CONFIGS[env_id]['data_path']

    cam_args = {'change': args['change'],
                'cam_per_ep': args['cam_per_ep'],
                'n_cam': args['n_cam']}

    train_dataloader, val_dataloader, stats = load_data(env,
                                                        data_path + '.h5',
                                                        data_path + '.json',
                                                        args['num_episodes'] + 10,
                                                        args['batch_size'],
                                                        args['batch_size'],
                                                        partial(globals()[args['cam_config']], distr='train', cam_args=cam_args),
                                                        use_plucker=args['use_plucker'],
                                                        transform=args['transform'])


    # save dataset stats
    if not os.path.isdir(args['ckpt_dir']):
        os.makedirs(args['ckpt_dir'])
    stats_path = os.path.join(args['ckpt_dir'], f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    set_seed(args['seed'])

    args.update({'enc_layers': 4,
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
    # scheduler = cosine_schedule_with_warmup(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
    scheduler = cosine_schedule(optimizer, total_steps=total_steps)
    # scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    if ckpt is not None:
        policy.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        print(f"Loaded ckpt from {ckpt['epoch']} epoch")
    # wandb.watch(policy, log_freq=100)

    evaluator = Evaluator(env, stats, use_plucker=args['use_plucker'], chunk_size=args['chunk_size']//2)


    min_val_loss = np.inf
    max_val_reward = -np.inf
    best_ckpt_info = None
    step = 0 if ckpt is None else ckpt['step'] + 1
    epoch = 0 if ckpt is None else ckpt['epoch'] + 1

    while epoch < args['num_epochs']:
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)

            if epoch % args['eval_every'] == 0:
                for distr in ['train', 'test']:
                    total_reward = 0.0
                    total_success = 0.0
                    for i in range(args['eval_nruns']):
                        reward, success = evaluator.evaluate(policy,
                                                            save_path=os.path.join(args['ckpt_dir'], f'epoch_{epoch}_{i}.mp4'),
                                                            get_cam_fn=partial(globals()[args['cam_config']], distr=distr, cam_args=cam_args))
                        total_reward += reward
                        total_success += success

                    wandb.log({f'{distr}_reward': total_reward / args['eval_nruns']}, step=epoch)
                    wandb.log({f'{distr}_success': total_success / args['eval_nruns']}, step=epoch)
                if max_val_reward <= total_reward:
                    max_val_reward = total_reward
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

        print(f"Epoch {epoch} done")
        epoch += 1

    wandb.finish(exit_code=0)

    # ckpt_path = os.path.join(args['ckpt_dir'], f'policy_last.pth')
    # torch.save(policy, ckpt_path)

    # best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    # ckpt_path = os.path.join(args['ckpt_dir'], f'policy_best.pth')
    # policy.load_state_dict(best_state_dict)
    # torch.save(policy, ckpt_path)


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad, cam_config = data
    image_data, qpos_data, action_data, is_pad, cam_config = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda(), cam_config.cuda()
    return policy(qpos_data, image_data, action_data, is_pad, cam_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--line', '-l', default=0, type=int, help='line number of lmn')

    parser.add_argument('--num_episodes', default=800, type=int, help='num_episodes')
    parser.add_argument('--policy_class', default='ACT', type=str, help='policy_class, capitalize')
    parser.add_argument('--camera_names', default=['cam'], type=str, help='camera_names')
    parser.add_argument('--task_name', default='PushT-v1', type=str,
                        help='LiftPegUpright-v1, PickCube-v1, PokeCube-v1, PullCube-v1, PushCube-v1, PushT-v1, RollBall-v1, StackCube-v1, PullCubeTool-v1, PegInsertionSide-v1')
    parser.add_argument('--batch_size', default=64, type=int, help='batch_size')
    parser.add_argument('--seed', default=1, type=int, help='seed')
    parser.add_argument('--num_epochs', default=20000, type=int, help='num_epochs')
    parser.add_argument('--warmup_epochs', default=1000, type=int, help='warmup_epochs')
    parser.add_argument('--lr', default=2e-5, type=float, help='lr')
    parser.add_argument('--cam_config', default='get_default_cam', type=str,
                        help='get_default_cam, get_n_cam_6d, get_n_cam_per_eps_6d, get_n_cam_1d, get_n_cam_per_eps_1d,')
    parser.add_argument('--eval_nruns', default=10, type=int, help='eval_nruns')
    parser.add_argument('--use_proprio', default=False, type=bool, help='use_proprio')
    parser.add_argument('--use_cam_pose', default=False, type=bool, help='when false, masked to 0 in data loader')
    parser.add_argument('--use_plucker', default=True, type=bool, help='when false, masked to 0 in data loader')
    parser.add_argument('--transform', default='crop', type=str, help='transform: id, crop')
    parser.add_argument('--backbone', default='lin', type=str, help='backbone: resnet18, lin')
    parser.add_argument('--activation', default='relu', type=str, help='activation')
    parser.add_argument('--eval_every', default=500, type=int, help='eval_every')

    parser.add_argument('--ckpt_dir', default='/mount/scratch', type=str, help='ckpt_dir')

    # for ACT
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout')
    parser.add_argument('--norm_type', default='layer', type=str, help='norm_type')
    parser.add_argument('--kl_weight', default=1, type=int, help='KL Weight')
    parser.add_argument('--chunk_size', default=30, type=int, help='chunk_size')
    parser.add_argument('--hidden_dim', default=1024, type=int, help='hidden_dim')
    parser.add_argument('--dim_feedforward', default=4096, type=int, help='dim_feedforward')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight_decay')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # for encoder
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--patch_size', default=16, type=int, help='patch_size')
    parser.add_argument('--plucker_as_pe', default=True, type=bool, help='use_plucker_as_pos_embedding')
    parser.add_argument('--embed_method', default='lin', type=str, help='embedding_method')

    # for cam config
    parser.add_argument('--change', default=1, type=int, help='number of cam change per episode')
    parser.add_argument('--cam_per_ep', default=3, type=int, help='number of cameras per episode')
    parser.add_argument('--n_cam', default=1, type=int, help='number of cameras')

    parsed_args = vars(parser.parse_args())

    config_n_num = [
        # {'cam_config': 'get_default_cam', 'n_cam': 1, 'plucker_as_pe': False, 'backbone': 'resnet18'},
        # {'cam_config': 'get_n_cam_per_eps_6d', 'n_cam': 1000, 'plucker_as_pe': True, 'backbone': 'lin'},
        # {'cam_config': 'get_n_cam_per_eps_6d', 'n_cam': 1000, 'plucker_as_pe': True, 'backbone': 'lin'},
        # {'cam_config': 'get_default_cam', 'n_cam': 1, 'plucker_as_pe': True, 'backbone': 'lin'},
        # {'cam_config': 'get_default_cam', 'n_cam': 1, 'plucker_as_pe': True, 'backbone': 'lin', 'num_episodes': 100},
        # {'cam_config': 'get_default_cam', 'n_cam': 1, 'plucker_as_pe': True, 'backbone': 'lin', 'num_episodes': 200},
        # {'cam_config': 'get_default_cam', 'n_cam': 1, 'plucker_as_pe': True, 'backbone': 'lin', 'num_episodes': 500},
        # {'cam_config': 'get_stair_shaped', 'change': 1, 'cam_per_ep': 1},
        # {'cam_config': 'get_stair_shaped', 'change': 1, 'cam_per_ep': 2},
        # {'cam_config': 'get_stair_shaped', 'change': 1, 'cam_per_ep': 3},
        {'task_name': 'PokeCube-v1','cam_config': 'get_stair_shaped', 'change': 1, 'cam_per_ep': 3, 'plucker_as_pe': False, 'backbone': 'resnet18', 'num_episodes': 200},
        {'task_name': 'PullCube-v1','cam_config': 'get_stair_shaped', 'change': 1, 'cam_per_ep': 3, 'plucker_as_pe': False, 'backbone': 'resnet18', 'num_episodes': 200},
        {'task_name': 'PushCube-v1','cam_config': 'get_stair_shaped', 'change': 1, 'cam_per_ep': 3, 'plucker_as_pe': False, 'backbone': 'resnet18', 'num_episodes': 200},
        {'task_name': 'PokeCube-v1','cam_config': 'get_stair_shaped', 'change': 1, 'cam_per_ep': 3, 'plucker_as_pe': True, 'backbone': 'lin', 'num_episodes': 200},
        {'task_name': 'PullCube-v1','cam_config': 'get_stair_shaped', 'change': 1, 'cam_per_ep': 3, 'plucker_as_pe': True, 'backbone': 'lin', 'num_episodes': 200},
        {'task_name': 'PushCube-v1','cam_config': 'get_stair_shaped', 'change': 1, 'cam_per_ep': 3, 'plucker_as_pe': True, 'backbone': 'lin', 'num_episodes': 200},
        # {'task_name': 'PickCube-v1','cam_config': 'get_stair_shaped', 'change': 1, 'cam_per_ep': 3, 'plucker_as_pe': False, 'backbone': 'resnet18', 'num_episodes': 200},
    ]

    parsed_args.update(config_n_num[parsed_args['line']])

    name = (
        f"{parsed_args['policy_class']}_{parsed_args['task_name']}_{parsed_args['num_episodes']}_eps"
        f"_{parsed_args['cam_config']}_{parsed_args['n_cam']}_cams"
        f"_pe_{parsed_args['plucker_as_pe']}"
        f"_{parsed_args['cam_per_ep']}_cams_change_{parsed_args['change']}"
        f"_backbone_{parsed_args['backbone']}"
        "_crop"
        "_120by30"
        # "_threshold_0.01"
        # "_fov1"
        "_0331"
    )
    parsed_args['ckpt_dir'] = os.path.join("/mount/maniskill/", name)

    project = 'pose_invariance_maniskill_0327'

    # resume wandb run
    ckpt_path = get_last_ckpt(parsed_args['ckpt_dir'])
    if ckpt_path is not None:
        print(f"Resuming wandb run: {name}")
        ckpt = torch.load(ckpt_path)
        wandb_id = ckpt['wandb_id']
        wandb.init(project=project, id=wandb_id, resume='must')
        parsed_args = dict(wandb.config)
        main(parsed_args, ckpt)
    else:
        print(f"Starting new wandb run: {name}")
        wandb.init(project=project, name=name, config=parsed_args)
        main(parsed_args)

    # lmn run beehive --sweep 0 -d --contain -n 5 -- python -m act.train -l '$LMN_RUN_SWEEP_IDX'
