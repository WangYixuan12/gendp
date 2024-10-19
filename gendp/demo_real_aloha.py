"""
Usage:
(robodiff)$ python demo_real_robot.py -o <demo_save_dir> --robot_ip <ip_of_ur5>

Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start recording.
Press "S" to stop recording.
Press "Q" to exit program.
Press "Backspace" to delete the previously recorded episode.
"""

# %%
import os
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import transforms3d

from gendp.real_world.real_aloha_env import RealAlohaEnv
from gendp.common.precise_sleep import precise_wait
from gendp.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from gendp.common.aloha_utils import (
    torque_on, torque_off, move_arms, move_grippers, get_arm_gripper_positions,
    START_ARM_POSE, START_EE_POSE, MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE, DT, MASTER2PUPPET_JOINT_FN
)
from gendp.common.kinematics_utils import KinHelper
from gendp.real_world.aloha_master import AlohaMaster
from gendp.real_world.aloha_bimanual_master import AlohaBimanualMaster
from gendp.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer

@click.command()
@click.option('--output_dir', '-o', required=True, multiple=True, help="Directory to save demonstration dataset.")
@click.option('--robot_sides', '-r', default=['right'], multiple=True, help="Which robot to control.")
@click.option('--robot_init_joints', '-j', default=None, multiple=True, help="Initial robot joint configuration.")
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
def main(output_dir, robot_sides, robot_init_joints, vis_camera_idx, frequency, command_latency):
    dt = 1/frequency
    for output in output_dir:
        os.system(f'mkdir -p {output}')
    kin_helper = KinHelper(robot_name='trossen_vx300s')
    if len(robot_init_joints) == 0:
        robot_init_joints = None
    else:
        robot_init_joints = np.array(robot_init_joints, dtype=np.float32)
    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
            RealAlohaEnv(
                output_dir=output_dir[0],
                robot_sides=robot_sides,
                ctrl_mode='joint',
                # recording resolution
                obs_image_resolution=(640,480),
                frequency=frequency,
                enable_multi_cam_vis=True,
                record_raw_video=True,
                video_capture_fps=30,
                video_capture_resolution=(640,480),
                # number of threads per camera view for video recording (H.264)
                thread_per_video=3,
                # video recording quality, lower is better (but slower).
                video_crf=21,
                shm_manager=shm_manager,
                init_qpos=robot_init_joints,
            ) as env, \
            AlohaMaster(shm_manager=shm_manager, robot_side=robot_sides[0]) if len(robot_sides) == 1 else \
                AlohaBimanualMaster(shm_manager=shm_manager, robot_sides=robot_sides) as master_bot:
            cv2.setNumThreads(1)

            # realsense exposure
            # env.realsense.set_exposure(exposure=120, gain=0)
            # realsense white balance
            # env.realsense.set_white_balance(white_balance=5900)

            time.sleep(1.0)
            print('Ready!')
            state = env.get_robot_state()
            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False
            while not stop:
                # calculate timing
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                # pump obs
                obs = env.get_obs()

                # handle key presses
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        # Exit program
                        stop = True
                    elif key_stroke == KeyCode(char='c'):
                        # Start recording
                        env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time(), curr_outdir=output_dir[0])
                        key_counter.clear()
                        is_recording = True
                        print('Recording!')
                    elif key_stroke == KeyCode(char='s'):
                        # Stop recording
                        env.end_episode(curr_outdir = output_dir[stage], incr_epi=True)
                        key_counter.clear()
                        is_recording = False
                        print('Stopped.')
                    elif key_stroke == Key.space:
                        # Save episode for current stage and start next stage
                        env.end_episode(curr_outdir = output_dir[stage], incr_epi=False)
                        is_recording = False
                        env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time(), curr_outdir=output_dir[stage+1])
                        is_recording = True
                    elif key_stroke == Key.backspace:
                        # Delete the most recent recorded episode
                        if click.confirm('Are you sure to drop an episode?'):
                            env.drop_episode()
                            key_counter.clear()
                            is_recording = False
                        # delete
                stage = key_counter[Key.space]
                if stage >= len(output_dir):
                    print('exceeds max stage')
                    env.drop_episode()
                    key_counter.clear()
                    is_recording = False

                # visualize
                vis_img = obs[f'camera_{vis_camera_idx}_color'][-1,:,:,::-1].copy()
                episode_id = env.episode_id
                text = f'Episode: {episode_id}, Stage: {stage}'
                if is_recording:
                    text += ', Recording!'
                cv2.putText(
                    vis_img,
                    text,
                    (10,30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    thickness=2,
                    color=(255,255,255)
                )

                cv2.imshow('default', vis_img)
                cv2.pollKey()

                precise_wait(t_sample)
                
                state = master_bot.get_motion_state()
                if len(robot_sides) == 1:
                    target_state = state['joint_pos'].copy()
                    target_state[6] = MASTER2PUPPET_JOINT_FN(state['joint_pos'][6])
                    curr_master_joint_pos = state['joint_pos'] # 6dof + gripper
                    fk_joint_pos = np.zeros(8)
                    fk_joint_pos[0:6] = curr_master_joint_pos[:6]
                    eef_pose = kin_helper.compute_fk_sapien_links(fk_joint_pos,
                                                                  [kin_helper.sapien_eef_idx])[0] # (4, 4)
                    eef_actions = np.concatenate([eef_pose[:3,3],
                                                transforms3d.euler.mat2euler(eef_pose[:3,:3], axes='sxyz'),
                                                np.array([MASTER2PUPPET_JOINT_FN(state['joint_pos'][-1])])])
                elif len(robot_sides) == 2:
                    target_state = state['joint_pos'].copy()
                    for rob_i in range(len(robot_sides)):
                        target_state[7*rob_i + 6] = MASTER2PUPPET_JOINT_FN(state['joint_pos'][7*rob_i + 6])
                    curr_master_joint_pos = state['joint_pos'] # 6dof + gripper
                    eef_actions = np.zeros(7 * len(robot_sides))
                    for rob_i in range(len(robot_sides)):
                        fk_joint_pos = np.zeros(8)
                        fk_joint_pos[0:6] = curr_master_joint_pos[rob_i*7:rob_i*7+6]
                        curr_puppet_gripper_joint = MASTER2PUPPET_JOINT_FN(curr_master_joint_pos[rob_i*7+6])
                    
                        target_puppet_ee_pose = kin_helper.compute_fk_sapien_links(fk_joint_pos,
                                                                                [kin_helper.sapien_eef_idx])[0] # (4, 4)
                        target_puppet_ee_pose = np.linalg.inv(env.puppet_bot.base_pose_in_world[0]) @ env.puppet_bot.base_pose_in_world[rob_i] @ target_puppet_ee_pose

                        eef_actions[rob_i*7:(rob_i+1)*7] = np.concatenate([target_puppet_ee_pose[:3,3],
                                                    transforms3d.euler.mat2euler(target_puppet_ee_pose[:3,:3], axes='sxyz'),
                                                    np.array([curr_puppet_gripper_joint])])

                # robot_base_in_world = env.puppet_bot.base_pose_in_world
                # eef_actions_in_world_mat = robot_base_in_world @ eef_pose
                # eef_actions_in_world = np.concatenate([eef_actions_in_world_mat[:3,3],
                #                                        transforms3d.euler.mat2euler(eef_actions_in_world_mat[:3,:3], axes='sxyz'),
                #                                        eef_actions[-1:]])

                # execute teleop command
                env.exec_actions(
                    joint_actions=[target_state],
                    eef_actions=[eef_actions],
                    timestamps=[t_command_target-time.monotonic()+time.time()])
                precise_wait(t_cycle_end)
                iter_idx += 1

# %%
if __name__ == '__main__':
    main()
