import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import math
import dill
import wandb.sdk.data_types.video as wv
import hydra
from omegaconf import OmegaConf

import sys
sys.path.append('/home/bing4090/yixuan_old_branch/general_dp/sapien_env')
sys.path.append('/home/bing4090/yixuan_old_branch/general_dp/general_dp')

from sapien_env.rl_env.mug_collect_env import BaseRLEnv
from sapien_env.sim_env.constructor import add_default_scene_light
from gendp.gym_util.async_vector_env import AsyncVectorEnv
from gendp.gym_util.sync_vector_env import SyncVectorEnv
from gendp.gym_util.multistep_wrapper import MultiStepWrapper
from gendp.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from gendp.model.common.rotation_transformer import RotationTransformer
from gendp.policy.base_image_policy import BaseImagePolicy
from gendp.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from gendp.common.pytorch_util import dict_apply
from gendp.dataset.base_dataset import BaseImageDataset
from gendp.env_runner.base_image_runner import BaseImageRunner
from gendp.env.sapien_env.sapien_env_wrapper import SapienEnvWrapper
from d3fields.fusion import Fusion

class SapienImageRunner(BaseImageRunner):

    def __init__(self, 
            output_dir,
            dataset_dir,
            shape_meta:dict,
            n_train=10,
            n_train_vis=3,
            train_start_idx=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            render_obs_keys=['right_bottom_view'],
            policy_keys=None,
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            train_obj_ls=None,
            test_obj_ls=None,
            repetitive=False,
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test
        steps_per_render = 1

        env_cfg_path = os.path.join(dataset_dir, 'config.yaml')
        env_cfg = OmegaConf.load(env_cfg_path)

        rotation_transformer = RotationTransformer('euler_angles', 'rotation_6d', from_convention='xyz')

        fusion = None
        if 'd3fields' in shape_meta.obs:
            num_cam = len(shape_meta.obs['d3fields']['info']['view_keys'])
            fusion = Fusion(num_cam=num_cam, device='cuda', dtype=torch.float16)

        def env_fn(inst_name = None):
            if inst_name is not None:
                env_cfg.manip_obj = inst_name
            env : BaseRLEnv = hydra.utils.instantiate(env_cfg)
            add_default_scene_light(env.scene, env.renderer)
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    SapienEnvWrapper(
                        env=env,
                        shape_meta=shape_meta,
                        init_states=None,
                        render_obs_keys=render_obs_keys,
                        fusion=fusion,
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='bgr24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = []
        if train_obj_ls is not None:
            for obj in train_obj_ls:
                for _ in range(n_train):
                    env_fns.append(lambda obj=obj: env_fn(obj))
        if test_obj_ls is not None:
            for obj in test_obj_ls:
                for _ in range(n_test):
                    env_fns.append(lambda obj=obj: env_fn(obj))
        if (train_obj_ls is None) and (test_obj_ls is None):
            env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # train
        if train_obj_ls is not None:
            n_train_obj = len(train_obj_ls)
        else:
            n_train_obj = 1
        for j in range(n_train_obj):
            for i in range(n_train):
                if not repetitive:
                    seed = train_start_idx + i
                    dataset_path = os.path.join(dataset_dir, f'episode_{i}.hdf5')
                else:
                    seed = train_start_idx
                    dataset_path = os.path.join(dataset_dir, f'episode_0.hdf5')
                enable_render = i < n_train_vis
                with h5py.File(dataset_path, 'r') as f:
                    init_states = f['info']['init_poses'][()]

                    def init_fn(env, init_states=init_states, 
                        enable_render=enable_render):
                        # setup rendering
                        # video_wrapper
                        assert isinstance(env.env, VideoRecordingWrapper)
                        env.env.video_recoder.stop()
                        env.env.file_path = None
                        if enable_render:
                            filename = pathlib.Path(output_dir).joinpath(
                                'media', wv.util.generate_id() + ".mp4")
                            filename.parent.mkdir(parents=False, exist_ok=True)
                            filename = str(filename)
                            env.env.file_path = filename

                        # switch to init_state reset
                        assert isinstance(env.env.env, SapienEnvWrapper)
                        env.env.env.init_states = init_states
                        env.seed(seed)
                        env.env.env.reset()

                    env_seeds.append(seed)
                    env_prefixs.append('train/')
                    env_init_fn_dills.append(dill.dumps(init_fn))
        
        # test
        if test_obj_ls is not None:
            n_test_obj = len(test_obj_ls)
        else:
            n_test_obj = 1
        for j in range(n_test_obj):
            for i in range(n_test):
                if not repetitive:
                    seed = test_start_seed + i
                else:
                    seed = test_start_seed
                enable_render = i < n_test_vis

                def init_fn(env, seed=seed, 
                    enable_render=enable_render):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    # switch to seed reset
                    assert isinstance(env.env.env, SapienEnvWrapper)
                    env.env.env.init_state = None
                    env.seed(seed)
                    env.env.env.reset()

                env_seeds.append(seed)
                env_prefixs.append('test/')
                env_init_fn_dills.append(dill.dumps(init_fn))

        # env = AsyncVectorEnv(env_fns)
        env = SyncVectorEnv(env_fns)


        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec
        self.policy_keys = policy_keys

    def run(self, policy: BaseImagePolicy, vis_3d=False):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        
        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            
            done = False
            action_ls = list()
            while not done:                
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    # filter out policy keys
                    if self.policy_keys is not None:
                        obs_dict = dict((k, obs_dict[k]) for k in self.policy_keys)
                    action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")
                
                # step env
                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action) # (n_envs, n_steps, action_dim)

                action_ls.append(env_action)
                obs, reward, done, info = env.step(env_action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
        # clear out video buffer
        _ = env.reset()
        
        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video
        
        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value
        
        if vis_3d:
            action_ls = np.concatenate(action_ls, axis=1) # (n_envs, n_max_steps, action_dim)
            action_pos = action_ls[...,:3]
            import matplotlib.pyplot as plt
            ax = plt.figure().add_subplot(projection='3d')
            for i in range(n_envs):
                ax.plot(action_pos[i,:,0], action_pos[i,:,1], action_pos[i,:,2], label=f'traj {i}')
            ax.legend()
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_aspect('equal')
            plt.show()

        return log_data

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction

def test():
    cfg_path = '/home/bing4090/yixuan_old_branch/general_dp/general_dp/config/hang_mug_sim/no_seg_no_dino_N_4000.yaml'
    cfg = OmegaConf.load(cfg_path)
    os.system('mkdir -p temp')
    
    env_runner : BaseImageRunner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir='temp')
    
    policy: DiffusionUnetHybridImagePolicy = hydra.utils.instantiate(cfg.policy)
    
    # configure dataset
    dataset : BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
    normalizer = dataset.get_normalizer()

    policy.set_normalizer(normalizer)
    
    env_runner.run(policy)

if __name__ == '__main__':
    test()
    