import os
import time
import warnings
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import scipy.interpolate as si
import scipy.spatial.transform as st
import transforms3d
import numpy as np
import sapien.core as sapien
from interbotix_xs_modules.arm import InterbotixManipulatorXS

from interbotix_xs_msgs.msg import JointSingleCommand
from gendp.common.aloha_utils import (
    torque_on, torque_off, move_arms, move_grippers, get_arm_gripper_positions, prep_puppet_robot,
    START_ARM_POSE, START_EE_POSE, MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE, DT, MASTER2PUPPET_JOINT_FN,
    PUPPET_GRIPPER_JOINT_NORMALIZE_FN, PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
)
from gendp.common.kinematics_utils import KinHelper
from gendp.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from gendp.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from gendp.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from gendp.real_world.aloha_master import AlohaMaster

class Command(enum.Enum):
    STOP = 0
    JOINT = 1 # joint space
    EE = 2 # end effector

class AlohaPuppet(mp.Process):
    """
    To ensure sending command to the robot with predictable latency
    this controller need its separate process (due to python GIL)
    """


    def __init__(self,
            shm_manager: SharedMemoryManager,
            frequency=125,
            launch_timeout=3,
            verbose=False,
            get_max_k=128,
            robot_side='right',
            extrinsics_dir=os.path.join(os.path.dirname(__file__), 'aloha_extrinsics'),
            init_qpos=None,
            ):
        """
        frequency: 100 for aloha
        """
        # verify
        assert 0 < frequency <= 500

        super().__init__(name=f"AlohaPuppet_{robot_side}")
        self.frequency = frequency
        self.launch_timeout = launch_timeout
        self.verbose = verbose
        self.robot_side = robot_side
        self.puppet_bot = None
        self.init_qpos = init_qpos
        os.system(f'mkdir -p {extrinsics_dir}')
        if extrinsics_dir is None:
            self.base_pose_in_world = np.eye(4)
            warnings.warn("extrinsics_dir is None, using identity matrix as base pose in world")
        else:
            extrinsics_path = os.path.join(extrinsics_dir, f'{robot_side}_base_pose_in_world.npy')
            if not os.path.exists(extrinsics_path):
                self.base_pose_in_world = np.eye(4)
                warnings.warn(f"extrinsics_path {extrinsics_path} does not exist, using identity matrix as base pose in world")
            else:
                self.base_pose_in_world = np.load(extrinsics_path)

        # build input queue
        example = {
            'cmd': Command.JOINT.value,
            'target_joint_pos': np.empty(shape=(7,), dtype=np.float32), # 6dof + gripper
            'target_ee_pose': np.empty(shape=(7,), dtype=np.float32), # (xyz, euler_xyz, gripper)
            'target_time': time.time(),
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )
            

        # build ring buffer
        example = {
            'curr_joint_pos': np.empty(shape=(7,), dtype=np.float32), # 6dof + gripper
            'curr_full_joint_pos': np.empty(shape=(8,), dtype=np.float32), # 6dof + 2 gripper
            'curr_ee_pose': np.empty(shape=(7,), dtype=np.float32), # (xyz, euler_xyz, gripper)
            'robot_base_pose_in_world': np.empty(shape=(4,4), dtype=np.float32),
            # 'target_joint_pos': np.empty(shape=(7,), dtype=np.float32), # 6dof + gripper
            # 'target_ee_pose': np.empty(shape=(7,), dtype=np.float32), # (xyz, euler_xyz, gripper)
            # 'left_finger_pose': np.empty(shape=(4,4), dtype=np.float32),
            # 'right_finger_pose': np.empty(shape=(4,4), dtype=np.float32),
            'robot_receive_timestamp': time.time()
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer

        # build teleop_robot
        self.teleop_robot = KinHelper(robot_name='trossen_vx300s')
        self.last_qpos = None
    
    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"Puppet aloha process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait()
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    # ========= command methods ============
    def set_target_joint_pos(self, target_joint_pos, target_time):
        message = {
            'cmd': Command.JOINT.value,
            'target_joint_pos': target_joint_pos,
            'target_ee_pose': np.zeros((7,), dtype=np.float64), # 'target_ee_pose' is not used in this case, so we just put a dummy value here
            'target_time': target_time,
        }
        self.input_queue.put(message)
    
    def set_target_ee_pose(self, target_ee_pose, target_time, initial_qpos=None):
        # initial_qpos: (m ,1) numpy array
        if initial_qpos is None:
            if self.last_qpos is None:
                initial_qpos = START_ARM_POSE[:6]
                initial_qpos = np.concatenate([initial_qpos, np.zeros(2)])
                self.last_qpos = initial_qpos
            else:
                initial_qpos = self.last_qpos
        # target_puppet_joint_pos = self.teleop_robot.compute_ik(initial_qpos=initial_qpos, cartesian=target_ee_pose[:6])
        target_puppet_joint_pos = self.teleop_robot.compute_ik_sapien(initial_qpos=initial_qpos, cartesian=target_ee_pose[:6])
        
        # # verify fk res
        # fk_res_mat = self.teleop_robot.compute_fk_links(target_puppet_joint_pos, [self.teleop_robot.eef_link_idx])[0]
        # fk_res = np.concatenate([fk_res_mat[:3, 3], transforms3d.euler.mat2euler(fk_res_mat[:3, :3])])
        # assert np.allclose(fk_res, target_ee_pose[:6], atol=1e-2), f"fk_res: {fk_res}, target_ee_pose: {target_ee_pose}"

        self.last_qpos = target_puppet_joint_pos
        gripper = target_ee_pose[-1]
        target_puppet_joint_pos = target_puppet_joint_pos[:7]
        target_puppet_joint_pos[-1] = gripper
        print(f"set target puppet joint pos: {target_puppet_joint_pos}")
        message = {
            'cmd': Command.JOINT.value,
            'target_joint_pos': target_puppet_joint_pos, # 'target_joint_pos' is not used in this case, so we just put a dummy value here
            'target_ee_pose': np.zeros((7,), dtype=np.float64), # 'target_ee_pose' is not used in this case, so we just put a dummy value here
            'target_time': target_time,
        }
        self.input_queue.put(message)

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    
    def save_state(self):
        # update robot state
        state = dict()
        
        # get current joint pos
        curr_joint_states = self.puppet_bot.dxl.robot_get_joint_states()
        # curr_joint_pos = np.array(curr_joint_states.position[:6] + curr_joint_states.position[7:8])
        curr_joint_pos = np.array(curr_joint_states.position[:7])
        
        # get current ee pose
        curr_ee_pose_mat = self.puppet_bot.arm.get_ee_pose()
        # curr_ee_pose_in_world_mat = self.base_pose_in_world @ curr_ee_pose_mat
        # curr_ee_pose = np.concatenate([curr_ee_pose_in_world_mat[:3, 3],
        #                                 np.array(transforms3d.euler.mat2euler(curr_ee_pose_in_world_mat[:3, :3])),
        #                                 [curr_joint_pos[-1]]])
        curr_ee_pose = np.concatenate([curr_ee_pose_mat[:3, 3],
                                       np.array(transforms3d.euler.mat2euler(curr_ee_pose_mat[:3, :3])),
                                       [curr_joint_pos[-1]]])
        
        # get current finger pose
        full_joint_qpos = np.array(curr_joint_states.position[:6] + curr_joint_states.position[7:])
        # finger_poses_sapien = teleop_robot.compute_fk_links(joint_qpos, [11, 12]) # finger links
        # finger_poses = np.stack([pose.to_transformation_matrix() for pose in finger_poses_sapien]) # (2, 4, 4)
        # finger_poses_in_world = np.stack([self.base_pose_in_world @ pose for pose in finger_poses]) # (2, 4, 4)
        
        # get target joint pos
        target_joint_pos = self.puppet_bot.arm.get_joint_commands()
        target_gripper_cmd = self.puppet_bot.gripper.gripper_command.cmd
        target_joint_pos.append(target_gripper_cmd)
        target_joint_pos = np.array(target_joint_pos)
        
        # get target ee pose
        target_ee_pose_mat = self.puppet_bot.arm.get_ee_pose_command()
        if (target_ee_pose_mat == None).any():
            START_EE_POSE_MAT = np.eye(4)
            START_EE_POSE_MAT[:3, 3] = START_EE_POSE[:3]
            START_EE_POSE_MAT[:3, :3] = transforms3d.quaternions.quat2mat(START_EE_POSE[3:7])
            START_EE_POSE_IN_WORLD_MAT = self.base_pose_in_world @ START_EE_POSE_MAT
            target_ee_pose = np.concatenate([START_EE_POSE_IN_WORLD_MAT[:3, 3],
                                            transforms3d.euler.mat2euler(START_EE_POSE_IN_WORLD_MAT[:3, :3]),
                                            [target_joint_pos[-1]]])
        else:
            target_ee_pose_in_world_mat = self.base_pose_in_world @ target_ee_pose_mat
            target_ee_pose = np.concatenate([target_ee_pose_in_world_mat[:3, 3],
                                                transforms3d.euler.mat2euler(target_ee_pose_in_world_mat[:3, :3]),
                                                [target_joint_pos[-1]]])
        state = {
            'curr_joint_pos': curr_joint_pos,
            'curr_full_joint_pos': full_joint_qpos,
            'curr_ee_pose': curr_ee_pose,
            'robot_base_pose_in_world': self.base_pose_in_world,
            # 'target_joint_pos': target_joint_pos,
            # 'target_ee_pose': target_ee_pose,
            # 'left_finger_pose': finger_poses_in_world[0],
            # 'right_finger_pose': finger_poses_in_world[1],
            'robot_receive_timestamp': time.time()
        }
        # if self.verbose:
        #     print(f"-------------------")
        #     print(f"target joint pos: {target_joint_pos}")
        #     print(f"target ee pose: {target_ee_pose}")
        #     print(f"curr joint pos: {curr_joint_pos}")
        #     print(f"curr ee pose: {curr_ee_pose}")
        #     print(f"-------------------")
        #     print()
        self.ring_buffer.put(state)
        return state

    # ========= main loop in process ============
    def run(self):
        try:
            # set up robot
            self.puppet_bot = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_{self.robot_side}', init_node=True)
            prep_puppet_robot(self.puppet_bot, self.init_qpos)
            self.last_joint_pos = None
            self.last_eef_pose = None
            # self.max_joint_vels = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) # rad/s
            self.max_joint_vels = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) # rad/s
            # self.max_joint_vels = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0]) # rad/s
            
            self.ready_event.set()

            gripper_command = JointSingleCommand(name="gripper")
            # main loop
            dt = 1. / self.frequency
            
            keep_running = True
            while keep_running:
                # self.test_inter_joint_pos_fn() # MUST BE COMMENTED OUT
                # exit(0)

                # start control iteration
                t_start = time.perf_counter()
                
                # send command to robot
                t_now = time.monotonic()
                
                # fetch single command from queue
                if self.input_queue.empty():
                    self.save_state()
                    continue
                command = self.input_queue.get()
                cmd = command['cmd']
                if cmd == Command.STOP.value:
                    keep_running = False
                    # stop immediately, ignore later commands
                    break
                elif cmd == Command.JOINT.value:
                    ### DEPRECATED ###
                    # target_joint_pos = command['target_joint_pos'][:6]
                    # gripper_command.cmd = command['target_joint_pos'][-1]
                    # # curr_joint_pos = self.puppet_bot.dxl.robot_get_joint_states().position[:7]
                    # # interpolate_joint_pos = curr_joint_pos[:6] + np.clip((target_joint_pos[:6] - curr_joint_pos[:6]), -max_joint_vel * dt, max_joint_vel * dt)
                    # # if self.verbose:
                    # #     print(f"target joint pos: {interpolate_joint_pos}")
                    # # if not self.puppet_bot.arm.set_joint_positions(interpolate_joint_pos, blocking=False):
                    # #     raise RuntimeError("joint command failed")
                    # if self.last_joint_pos is None:
                    #     self.last_joint_pos = curr_joint_pos[:6]
                    # # delta_joint_pos = np.clip(target_joint_pos - self.last_joint_pos, -self.max_joint_vels * dt, self.max_joint_vels * dt)
                    # delta_joint_pos = 0.3 * (target_joint_pos - self.last_joint_pos)
                    # self.last_joint_pos = self.last_joint_pos + delta_joint_pos
                    # if not self.puppet_bot.arm.set_joint_positions(self.last_joint_pos, blocking=False):
                    #     raise RuntimeError("joint command failed")
                    # self.puppet_bot.gripper.core.pub_single.publish(gripper_command)
                    
                    # handle time
                    target_time = float(command['target_time'])
                    # translate global time to monotonic time
                    target_time = time.monotonic() - time.time() + target_time

                    print("execute joint command at target time: ", target_time)
                    if time.monotonic() >= target_time:
                        print("target time is already passed, skip")
                    while time.monotonic() < target_time:
                        ### this is the loop where frequency is expected to be self.frequency
                        step_start_time = time.perf_counter()

                        # save state
                        state = self.save_state()
                        curr_joint_pos = state['curr_joint_pos']
                        curr_ee_pose = state['curr_ee_pose']

                        # update last joint pos
                        if self.last_joint_pos is None:
                            self.last_joint_pos = curr_joint_pos
                        
                        # # obtain cliped target joint pos
                        # time_diff = target_time - time.monotonic()
                        # max_joint_vel = 1.0 # rad/s
                        # target_joint_pos = command['target_joint_pos'][:6]
                        # target_joint_pos = np.clip(target_joint_pos, curr_joint_pos[:6] - max_joint_vel * time_diff, curr_joint_pos[:6] + max_joint_vel * time_diff)
                        # target_joint_pos = np.concatenate([target_joint_pos, np.array([command['target_joint_pos'][-1]])])

                        alpha = 0.3
                        inter_joint_pos = self.last_joint_pos * (1 - alpha) + command['target_joint_pos'] * alpha
                        # inter_joint_pos = self.last_joint_pos * (1 - alpha) + target_joint_pos * alpha
                        
                        # curr_joint_pos = np.array(curr_joint_pos)[:6]
                        # clipped_joint_pos = np.clip(inter_joint_pos[:6], curr_joint_pos - self.max_joint_vels * dt, curr_joint_pos + self.max_joint_vels * dt)
                        # self.puppet_bot.arm.set_joint_positions(clipped_joint_pos, blocking=False)

                        # print(f"set joint pos: {inter_joint_pos[:6]}")
                        self.puppet_bot.arm.set_joint_positions(inter_joint_pos[:6], blocking=False)

                        gripper_command.cmd = inter_joint_pos[-1]
                        self.puppet_bot.gripper.core.pub_single.publish(gripper_command)

                        # measure frequency
                        time.sleep(max(0, dt - (time.perf_counter() - step_start_time)))
                        frequency = 1/(time.perf_counter() - step_start_time)
                        if frequency < self.frequency - 10:
                            warnings.warn(f"Puppet aloha Actual frequency {frequency} Hz is much smaller than desired frequency {self.frequency} Hz")
                        
                        self.last_joint_pos = inter_joint_pos

                    # regulate frequency
                    time.sleep(max(0, dt - (time.perf_counter() - t_start)))
                    frequency = 1/(time.perf_counter() - t_start)

                elif cmd == Command.EE.value:
                    raise RuntimeError("EE command is not supported anymore")
                    ### DEPRECATED ###
                    # target_ee_pose = command['target_ee_pose'][:-1]
                    # target_ee_quat = transforms3d.euler.euler2quat(target_ee_pose[3], target_ee_pose[4], target_ee_pose[5], axes='sxyz')
                    # gripper_command.cmd = command['target_ee_pose'][-1]
                    
                    # target_ee_pose_sapien = sapien.Pose(target_ee_pose[:3], target_ee_quat)
                    # target_puppet_joint_pos = self.teleop_robot.ik_vx300s_sapien_pose(initial_qpos=np.concatenate([curr_joint_pos[:6], np.zeros(2)])[:,None],
                    #                                                             pose=target_ee_pose_sapien,)
                    # self.puppet_bot.arm.set_joint_positions(target_puppet_joint_pos, blocking=False)
                    
                    # self.puppet_bot.gripper.core.pub_single.publish(gripper_command)

                    # handle time
                    target_time = float(command['target_time'])
                    # translate global time to monotonic time
                    target_time = time.monotonic() - time.time() + target_time
                    target_ee_pose = command['target_ee_pose']

                    # execute actions until target_time
                    while time.monotonic() < target_time:
                        ### this is the loop where frequency is expected to be self.frequency
                        step_start_time = time.perf_counter()

                        # save state
                        state = self.save_state()
                        curr_joint_pos = state['curr_joint_pos']
                        curr_ee_pose = state['curr_ee_pose']

                        # update last ee pose
                        if self.last_eef_pose is None:
                            self.last_eef_pose = curr_ee_pose

                        key_ee_rot = np.stack([curr_ee_pose[3:6], target_ee_pose[3:6]])
                        key_ee_rot = st.Rotation.from_euler('xyz', key_ee_rot)

                        alpha = 0.3
                        inter_ee_pose = np.zeros((7,), dtype=self.last_eef_pose.dtype)
                        inter_ee_pose[:3] = self.last_eef_pose[:3] * (1 - alpha) + target_ee_pose[:3] * alpha
                        slerp = st.Slerp([0, 1], key_ee_rot)
                        inter_ee_pose[3:6] = slerp([alpha]).as_euler('xyz')[0]
                        inter_ee_pose[6] = self.last_eef_pose[6] * (1 - alpha) + target_ee_pose[6] * alpha

                        # computer target joint pos
                        inter_ee_pose_sapien = sapien.Pose(inter_ee_pose[:3], transforms3d.euler.euler2quat(*inter_ee_pose[3:6], axes='sxyz'))
                        target_puppet_joint_pos = self.teleop_robot.ik_vx300s_sapien_pose(initial_qpos=np.concatenate([curr_joint_pos[:6], np.zeros(2)])[:,None],
                                                                                        pose=inter_ee_pose_sapien,)
                        
                        # publish command
                        self.puppet_bot.arm.set_joint_positions(target_puppet_joint_pos, blocking=False)

                        gripper_command.cmd = inter_ee_pose[-1]
                        self.puppet_bot.gripper.core.pub_single.publish(gripper_command)

                        # measure frequency
                        time.sleep(max(0, dt - (time.perf_counter() - step_start_time)))
                        frequency = 1/(time.perf_counter() - step_start_time)
                        if frequency < self.frequency - 10:
                            warnings.warn(f"Puppet aloha Actual frequency {frequency} Hz is much smaller than desired frequency {self.frequency} Hz")

                        self.last_eef_pose = inter_ee_pose
                else:
                    keep_running = False
                    break
            
                # regulate frequency
                time.sleep(max(0, dt - (time.perf_counter() - t_start)))
                frequency = 1/(time.perf_counter() - t_start)

        finally:
            self.ready_event.set()

def test_joint_teleop():
    shm_manager = SharedMemoryManager()
    shm_manager.start()
    # order is important
    frequency = 50
    puppet_robot = AlohaPuppet(shm_manager=shm_manager, robot_side='right', frequency=frequency, verbose=True)
    puppet_robot.start()
    master_robot = AlohaMaster(shm_manager=shm_manager, robot_side='right', frequency=frequency)
    master_robot.start()
    while True:
        dt = 0.1
        start_time = time.monotonic()
        state = master_robot.get_motion_state()
        target_state = np.concatenate([state['joint_pos'][:6], np.array([MASTER2PUPPET_JOINT_FN(state['joint_pos'][-1])])])
        target_time = time.time() + 0.1
        puppet_robot.set_target_joint_pos(target_state, target_time=target_time)
        time.sleep(max(0, dt - (time.monotonic() - start_time)))

def test_ee_teleop():
    kin_helper = KinHelper(robot_name='trossen_vx300s')
    shm_manager = SharedMemoryManager()
    shm_manager.start()
    # order is important
    frequency = 50
    puppet_robot = AlohaPuppet(shm_manager=shm_manager, robot_side='right', frequency=frequency, verbose=True)
    puppet_robot.start()
    master_robot = AlohaMaster(shm_manager=shm_manager, robot_side='right', frequency=frequency)
    master_robot.start()
    while True:
        main_loop_start_time = time.perf_counter()
        dt = 0.1
        start_time = time.monotonic()
        
        curr_master_states = master_robot.get_motion_state()
        curr_master_joint_pos = curr_master_states['joint_pos'] # 6dof + gripper

        curr_puppet_gripper_joint = MASTER2PUPPET_JOINT_FN(curr_master_joint_pos[-1])
        curr_puppet_gripper_pos = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(PUPPET_GRIPPER_JOINT_NORMALIZE_FN(curr_puppet_gripper_joint))
        
        target_puppet_ee_pose = kin_helper.compute_fk_links(np.concatenate([curr_master_joint_pos[:6], np.array([curr_puppet_gripper_pos, -curr_puppet_gripper_pos])])[:,None],
                                                            [kin_helper.eef_link_idx]) # (1, 4, 4)
        
        target_state = np.concatenate([target_puppet_ee_pose.p,
                                       transforms3d.euler.quat2euler(target_puppet_ee_pose.q, axes='sxyz'),
                                       np.array([curr_puppet_gripper_joint])])
        target_time = time.time() + 0.1
        puppet_robot.set_target_ee_pose(target_state, target_time=target_time, initial_qpos=np.concatenate([curr_master_joint_pos[:6], np.zeros(2)])[:,None])
        
        time.sleep(max(0, dt - (time.monotonic() - start_time)))
        frequency = 1/(time.perf_counter() - main_loop_start_time)
        print(f"main loop frequency: {frequency}")

def test_joint_inter():
    shm_manager = SharedMemoryManager()
    shm_manager.start()
    # order is important
    frequency = 50
    puppet_robot = AlohaPuppet(shm_manager=shm_manager, robot_side='right', frequency=frequency, verbose=False)
    puppet_robot.start()

if __name__ == '__main__':
    test_joint_teleop()
    # test_ee_teleop()
    # test_joint_inter()
