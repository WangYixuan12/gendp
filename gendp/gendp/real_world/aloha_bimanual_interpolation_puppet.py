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
    PUPPET_GRIPPER_JOINT_NORMALIZE_FN, PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN, PUPPET_JOINT2POS, PUPPET_POS2JOINT
)
from gendp.common.kinematics_utils import KinHelper
from gendp.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from gendp.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from gendp.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from gendp.common.linear_interpolator import LinearInterpolator
from gendp.real_world.aloha_bimanual_master import AlohaBimanualMaster

class Command(enum.Enum):
    STOP = 0
    JOINT = 1 # joint space
    EE = 2 # end effector
    SCHEDULE_EE = 3 # schedule end effector
    SCHEDULE_JOINT = 4 # schedule joint

class AlohaBimanualInterpPuppet(mp.Process):
    """
    To ensure sending command to the robot with predictable latency
    this controller need its separate process (due to python GIL)
    """


    def __init__(self,
            shm_manager: SharedMemoryManager,
            frequency=50,
            launch_timeout=3,
            verbose=False,
            get_max_k=128,
            robot_sides=['right'],
            extrinsics_dir=os.path.join(os.path.dirname(__file__), 'aloha_extrinsics'),
            init_qpos=None,
            max_pos_speed=0.25,
            max_rot_speed=1.0,
            ctrl_mode='eef',
            ):
        """
        frequency: 100 for aloha
        """
        # verify
        assert 0 < frequency <= 500

        side_str = '_'.join(robot_sides)
        super().__init__(name=f"AlohaPuppet_{side_str}")

        # check parameters
        assert ctrl_mode in ['eef', 'joint'], "ctrl_mode must be either 'eef' or 'joint'"

        # copy parameters
        self.frequency = frequency
        self.launch_timeout = launch_timeout
        self.verbose = verbose
        self.robot_sides = robot_sides
        self.num_bot = len(robot_sides)
        self.puppet_bots = None
        self.init_qpos = init_qpos
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.ctrl_mode = ctrl_mode

        os.system(f'mkdir -p {extrinsics_dir}')
        if extrinsics_dir is None:
            self.base_pose_in_world = np.tile(np.eye(4)[None], (len(robot_sides), 1, 1))
            warnings.warn("extrinsics_dir is None, using identity matrix as base pose in world")
        else:
            extrinsics_paths = [os.path.join(extrinsics_dir, f'{robot_side}_base_pose_in_world.npy') for robot_side in robot_sides]
            self.base_pose_in_world = np.tile(np.eye(4)[None], (len(robot_sides), 1, 1))
            for bot_i, extrinsics_path in enumerate(extrinsics_paths):
                if not os.path.exists(extrinsics_path):
                    self.base_pose_in_world[bot_i] = np.eye(4)
                    warnings.warn(f"extrinsics_path {extrinsics_path} does not exist, using identity matrix as base pose in world")
                else:
                    self.base_pose_in_world[bot_i] = np.load(extrinsics_path)

        # build input queue
        example = {
            'cmd': Command.JOINT.value,
            'target_joint_pos': np.empty(shape=(7 * self.num_bot,), dtype=np.float32), # 6dof + gripper
            'target_ee_pose': np.empty(shape=(7 * self.num_bot,), dtype=np.float32), # (xyz, euler_xyz, gripper)
            'target_time': time.time(),
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )
            

        # build ring buffer
        example = {
            'curr_joint_pos': np.empty(shape=(7 * self.num_bot,), dtype=np.float32), # 6dof + gripper
            'curr_full_joint_pos': np.empty(shape=(8 * self.num_bot,), dtype=np.float32), # 6dof + 2 gripper
            'curr_ee_pose': np.empty(shape=(7 * self.num_bot,), dtype=np.float32), # (xyz, euler_xyz, gripper)
            'robot_base_pose_in_world': np.empty(shape=(self.num_bot, 4,4), dtype=np.float32),
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
        if init_qpos is None:
            self.last_qpos = None
        else:
            self.last_qpos = np.zeros((8 * self.num_bot,), dtype=np.float32)
            for bot_i in range(self.num_bot):
                self.last_qpos[bot_i*8:bot_i*8+6] = init_qpos[bot_i*7:bot_i*7+6]
    
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
        assert self.ctrl_mode == 'joint', "This function is only supported in joint mode"
        message = {
            'cmd': Command.JOINT.value,
            'target_joint_pos': target_joint_pos,
            'target_ee_pose': np.zeros((7 * self.num_bot,), dtype=np.float64), # 'target_ee_pose' is not used in this case, so we just put a dummy value here
            'target_time': target_time,
        }
        self.input_queue.put(message)
    
    def set_target_ee_pose(self, target_ee_pose, target_time, initial_qpos=None):
        assert self.ctrl_mode == 'eef', "This function is only supported in eef mode"
        # initial_qpos: (m ,1) numpy array
        if initial_qpos is None:
            if self.last_qpos is None:
                initial_qpos = START_ARM_POSE[:6]
                initial_qpos = np.concatenate([initial_qpos, np.zeros(2)])
                self.last_qpos = np.concatenate([initial_qpos] * self.num_bot)
            initial_qpos = self.last_qpos
        # target_puppet_joint_pos = self.teleop_robot.compute_ik(initial_qpos=initial_qpos, cartesian=target_ee_pose[:6])
        target_puppet_joint_pos = np.zeros((7 * self.num_bot,), dtype=np.float64)
        for bot_i in range(self.num_bot):
            pos = target_ee_pose[bot_i*7:bot_i*7+3]
            euler = target_ee_pose[bot_i*7+3:bot_i*7+6]
            ee_mat = np.eye(4)
            ee_mat[:3, 3] = pos
            ee_mat[:3, :3] = transforms3d.euler.euler2mat(*euler, axes='sxyz')

            ee_mat_in_world = self.base_pose_in_world[0] @ ee_mat
            ee_mat_in_bot_i = np.linalg.inv(self.base_pose_in_world[bot_i]) @ ee_mat_in_world

            pos_tf = ee_mat_in_bot_i[:3, 3]
            euler_tf = transforms3d.euler.mat2euler(ee_mat_in_bot_i[:3, :3], axes='sxyz')
            cart_tf = np.concatenate([pos_tf, euler_tf])

            target_puppet_joint_pos[bot_i*7:bot_i*7+6] = self.teleop_robot.compute_ik_sapien(initial_qpos=initial_qpos[bot_i*8:(bot_i+1)*8], cartesian=cart_tf)[:6]
            target_puppet_joint_pos[bot_i*7+6] = target_ee_pose[bot_i*7+6]
        self.last_qpos = np.zeros((8 * self.num_bot,), dtype=np.float32)
        for bot_i in range(self.num_bot):
            self.last_qpos[bot_i*8:bot_i*8+6] = target_puppet_joint_pos[bot_i*7:bot_i*7+6]
        
        # # verify fk res
        # fk_res_mat = self.teleop_robot.compute_fk_links(target_puppet_joint_pos, [self.teleop_robot.eef_link_idx])[0]
        # fk_res = np.concatenate([fk_res_mat[:3, 3], transforms3d.euler.mat2euler(fk_res_mat[:3, :3])])
        # assert np.allclose(fk_res, target_ee_pose[:6], atol=1e-2), f"fk_res: {fk_res}, target_ee_pose: {target_ee_pose}"

        print(f"set target puppet joint pos: {target_puppet_joint_pos}")
        message = {
            'cmd': Command.JOINT.value,
            'target_joint_pos': target_puppet_joint_pos, # 'target_joint_pos' is not used in this case, so we just put a dummy value here
            'target_ee_pose': np.zeros((7 * self.num_bot,), dtype=np.float64), # 'target_ee_pose' is not used in this case, so we just put a dummy value here
            'target_time': target_time,
        }
        self.input_queue.put(message)

    def schedule_target_ee_pose(self, target_ee_pose, target_time, initial_qpos=None):
        assert self.ctrl_mode == 'eef', "This function is only supported in eef mode"
        target_ee_pose_ls = []
        for bot_i in range(self.num_bot):
            pos = target_ee_pose[bot_i*7:bot_i*7+3]
            euler = target_ee_pose[bot_i*7+3:bot_i*7+6]
            rotvec = st.Rotation.from_euler('xyz', euler).as_rotvec()
            target_ee_pose_ls.append(np.concatenate([pos, rotvec, [target_ee_pose[bot_i*7+6]]]))
        target_ee_pose = np.concatenate(target_ee_pose_ls) # (7 * num_bot, )
        message = {
            'cmd': Command.SCHEDULE_EE.value,
            'target_ee_pose': target_ee_pose,
            'target_time': target_time,
        }
        self.input_queue.put(message)
    
    def schedule_target_joint_pos(self, target_joint_pos, target_time):
        assert self.ctrl_mode == 'joint', "This function is only supported in joint mode"
        message = {
            'cmd': Command.SCHEDULE_JOINT.value,
            'target_joint_pos': target_joint_pos,
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
        curr_joint_pos_ls = []
        curr_ee_pose_ls = []
        full_joint_qpos_ls = []
        for bot_i in range(self.num_bot):
            # get current joint pos
            curr_joint_states = self.puppet_bots[bot_i].dxl.robot_get_joint_states()
            curr_joint_pos = np.array(curr_joint_states.position[:7])
            curr_joint_pos_ls.append(curr_joint_pos)

            # get current ee pose
            curr_ee_pose_mat = self.puppet_bots[bot_i].arm.get_ee_pose()
            curr_ee_pose_mat = self.base_pose_in_world[bot_i] @ curr_ee_pose_mat
            curr_ee_pose_mat = np.linalg.inv(self.base_pose_in_world[0]) @ curr_ee_pose_mat
            curr_ee_pose = np.concatenate([curr_ee_pose_mat[:3, 3],
                                           np.array(transforms3d.euler.mat2euler(curr_ee_pose_mat[:3, :3])),
                                           [curr_joint_pos[-1]]])
            curr_ee_pose_ls.append(curr_ee_pose)
        
            # get current full joint pos (for forward kinematics)
            full_joint_qpos = np.array(curr_joint_states.position[:6] + curr_joint_states.position[7:])
            full_joint_qpos_ls.append(full_joint_qpos)
        
        curr_joint_pos = np.concatenate(curr_joint_pos_ls)
        full_joint_qpos = np.concatenate(full_joint_qpos_ls)
        curr_ee_pose = np.concatenate(curr_ee_pose_ls)

        state = {
            'curr_joint_pos': curr_joint_pos,
            'curr_full_joint_pos': full_joint_qpos,
            'curr_ee_pose': curr_ee_pose,
            'robot_base_pose_in_world': self.base_pose_in_world,
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
        # set up robot
        self.puppet_bots = []
        for bot_i, side in enumerate(self.robot_sides):
            if bot_i == 0:
                init_node = True
            else:
                init_node = False
            self.puppet_bots.append(InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_{side}', init_node=init_node))
            prep_puppet_robot(self.puppet_bots[-1], self.init_qpos)
        self.last_joint_pos = None
        self.last_eef_pose = None
        # self.max_joint_vels = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) # rad/s
        self.max_joint_vels = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0] * self.num_bot) # rad/s
        # self.max_joint_vels = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0]) # rad/s
        
        self.ready_event.set()

        gripper_command = JointSingleCommand(name="gripper")
        # main loop
        dt = 1. / self.frequency
        # use monotonic time to make sure the control loop never go backward
        curr_t = time.monotonic()
        last_waypoint_time = curr_t
        cmd_interp_ls = []
        gripper_interp_ls = []
        for bot_i in range(self.num_bot):
            curr_joint_states = self.puppet_bots[bot_i].dxl.robot_get_joint_states()
            curr_joint_pos = np.array(curr_joint_states.position[:6] + curr_joint_states.position[7:])
            curr_pose = self.teleop_robot.compute_fk_sapien_links(curr_joint_pos, [self.teleop_robot.sapien_eef_idx])[0]
            
            curr_pose = np.linalg.inv(self.base_pose_in_world[0]) @ self.base_pose_in_world[bot_i] @ curr_pose
            
            curr_pose_tf = np.zeros(6)
            curr_pose_tf[:3] = curr_pose[:3, 3]
            curr_pose_tf[3:] = st.Rotation.from_matrix(curr_pose[:3, :3]).as_rotvec()
            if self.ctrl_mode == 'eef':
                cmd_interp_ls.append(PoseTrajectoryInterpolator(
                    times=[curr_t],
                    poses=[curr_pose_tf]
                ))
            elif self.ctrl_mode == 'joint':
                cmd_interp_ls.append(LinearInterpolator(
                    times=[curr_t],
                    cmds=[curr_joint_pos[:6]]
                ))
            curr_gripper_pos = np.array(curr_joint_states.position[6:7])
            gripper_interp = LinearInterpolator(
                times=[curr_t],
                cmds=[curr_gripper_pos]
            )
            gripper_interp_ls.append(gripper_interp)
        
        # initial_qpos: (m ,1) numpy array
        if self.last_qpos is None:
            initial_qpos = START_ARM_POSE[:6]
            initial_qpos = np.concatenate([initial_qpos, np.zeros(2)])
            self.last_qpos = np.concatenate([initial_qpos] * self.num_bot)

        keep_running = True
        while keep_running:
            # self.test_inter_joint_pos_fn() # MUST BE COMMENTED OUT
            # exit(0)

            # start control iteration
            t_start = time.perf_counter()
            
            # send command to robot
            t_now = time.monotonic()
            for bot_i in range(self.num_bot):
                if self.ctrl_mode == 'eef':
                    # read from interpolation
                    pose_command = cmd_interp_ls[bot_i](t_now)
                    gripper_pos = gripper_interp_ls[bot_i](t_now)
                    
                    # convert to eef pose in world in the format of matrix
                    pos = pose_command[:3]
                    rotvec = pose_command[3:]
                    rotmat = st.Rotation.from_rotvec(rotvec).as_matrix()
                    pose_mat = np.eye(4)
                    pose_mat[:3, :3] = rotmat
                    pose_mat[:3, 3] = pos
                    ee_mat_in_world = self.base_pose_in_world[0] @ pose_mat
                    ee_mat_in_bot_i = np.linalg.inv(self.base_pose_in_world[bot_i]) @ ee_mat_in_world

                    # inverse kinematics
                    joint_pos = self.teleop_robot.compute_ik_sapien(initial_qpos=self.last_qpos[bot_i*8:(bot_i+1)*8], cartesian=ee_mat_in_bot_i, pose_fmt='mat')[:6]
                elif self.ctrl_mode == 'joint':
                    joint_pos = cmd_interp_ls[bot_i](t_now)
                    gripper_pos = gripper_interp_ls[bot_i](t_now)
                
                # send command
                self.puppet_bots[bot_i].arm.set_joint_positions(joint_pos, blocking=False)
                gripper_command.cmd = gripper_pos
                self.puppet_bots[bot_i].gripper.core.pub_single.publish(gripper_command)

                # update last_qpos
                self.last_qpos[bot_i*8:bot_i*8+6] = joint_pos
                self.last_qpos[bot_i*8+6] = gripper_pos
                self.last_qpos[bot_i*8+7] = -gripper_pos
            
            # save robot state
            self.save_state()

            # fetch command from queue
            try:
                commands = self.input_queue.get_all()
                n_cmd = len(commands['cmd'])
            except Empty:
                n_cmd = 0

            for i in range(n_cmd):
                command = dict()
                for key, value in commands.items():
                    command[key] = value[i]
                cmd = command['cmd']
                if cmd == Command.STOP.value:
                    keep_running = False
                    # stop immediately, ignore later commands
                    break
                elif cmd == Command.JOINT.value:
                    raise ValueError("This function is not supported in this class")
                elif cmd == Command.EE.value:
                    raise ValueError("This function is not supported in this class")
                elif cmd == Command.SCHEDULE_EE.value:
                    target_ee_pose = command['target_ee_pose'] # (xyz, rotvec, gripper)
                    target_time = command['target_time']
                    # translate global time to monotonic time
                    target_time = time.monotonic() - time.time() + target_time
                    curr_time = t_now + dt
                    for bot_i in range(self.num_bot):
                        cmd_interp_ls[bot_i] = cmd_interp_ls[bot_i].schedule_waypoint(
                            pose=target_ee_pose[bot_i*7:bot_i*7+6],
                            time=target_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        gripper_interp_ls[bot_i] = gripper_interp_ls[bot_i].schedule_waypoint(
                            cmd=target_ee_pose[bot_i*7+6:bot_i*7+7],
                            time=target_time,
                            max_cmd_speed=5.0,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                elif cmd == Command.SCHEDULE_JOINT.value:
                    target_joint_pos = command['target_joint_pos']
                    target_time = command['target_time']
                    # translate global time to monotonic time
                    target_time = time.monotonic() - time.time() + target_time
                    curr_time = t_now + dt
                    for bot_i in range(self.num_bot):
                        cmd_interp_ls[bot_i] = cmd_interp_ls[bot_i].schedule_waypoint(
                            cmd=target_joint_pos[bot_i*7:bot_i*7+6],
                            time=target_time,
                            # max_cmd_speed=1.0,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        gripper_interp_ls[bot_i] = gripper_interp_ls[bot_i].schedule_waypoint(
                            cmd=target_joint_pos[bot_i*7+6:bot_i*7+7],
                            time=target_time,
                            # max_cmd_speed=0.5,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                else:
                    keep_running = False
                    break
        
            # regulate frequency
            time.sleep(max(0, dt - (time.perf_counter() - t_start)))
            frequency = 1/(time.perf_counter() - t_start)
            
            # print(f"puppet frequency: {frequency}")


def test_joint_teleop(robot_sides):
    shm_manager = SharedMemoryManager()
    shm_manager.start()
    # order is important
    frequency = 50
    puppet_robot = AlohaBimanualInterpPuppet(shm_manager=shm_manager, robot_sides=robot_sides, frequency=frequency, verbose=True, ctrl_mode='joint')
    puppet_robot.start()
    master_robot = AlohaBimanualMaster(shm_manager=shm_manager, robot_sides=robot_sides, frequency=frequency)
    master_robot.start()
    while True:
        dt = 0.1
        start_time = time.monotonic()
        state = master_robot.get_motion_state()
        target_state = state['joint_pos'].copy()
        for rob_i in range(len(robot_sides)):
            target_state[7*rob_i + 6] = MASTER2PUPPET_JOINT_FN(state['joint_pos'][7*rob_i + 6])
        target_time = time.time() + 0.1
        puppet_robot.schedule_target_joint_pos(target_state, target_time=target_time)
        time.sleep(max(0, dt - (time.monotonic() - start_time)))

def test_ee_teleop(robot_sides=['right', 'left']):
    kin_helper = KinHelper(robot_name='trossen_vx300s')
    shm_manager = SharedMemoryManager()
    shm_manager.start()
    # order is important
    frequency = 50
    puppet_robot = AlohaBimanualInterpPuppet(shm_manager=shm_manager, robot_sides=robot_sides, frequency=frequency, verbose=True)
    puppet_robot.start()
    master_robot = AlohaBimanualMaster(shm_manager=shm_manager, robot_sides=robot_sides, frequency=frequency)
    master_robot.start()
    while True:
        main_loop_start_time = time.perf_counter()
        dt = 0.1
        start_time = time.monotonic()
        
        curr_master_states = master_robot.get_motion_state()
        curr_master_joint_pos = curr_master_states['joint_pos'] # 6dof + gripper
        target_state = np.zeros(7 * len(robot_sides))
        for rob_i in range(len(robot_sides)):
            fk_joint_pos = np.zeros(8)
            fk_joint_pos[0:6] = curr_master_joint_pos[rob_i*7:rob_i*7+6]
            curr_puppet_gripper_joint = MASTER2PUPPET_JOINT_FN(curr_master_joint_pos[rob_i*7+6])
        
            target_puppet_ee_pose = kin_helper.compute_fk_sapien_links(fk_joint_pos,
                                                                       [kin_helper.sapien_eef_idx])[0] # (1, 4, 4)
            target_puppet_ee_pose = np.linalg.inv(puppet_robot.base_pose_in_world[0]) @ puppet_robot.base_pose_in_world[rob_i] @ target_puppet_ee_pose

            target_state[rob_i*7:(rob_i+1)*7] = np.concatenate([target_puppet_ee_pose[:3,3], # + np.random.random(3) * 0.01,
                                           transforms3d.euler.mat2euler(target_puppet_ee_pose[:3,:3], axes='sxyz'), # + np.random.random(3) * 0.01,
                                           np.array([curr_puppet_gripper_joint])])
        target_time = time.time() + 0.1
        puppet_robot.schedule_target_ee_pose(target_state, target_time=target_time)
        
        time.sleep(max(0, dt - (time.monotonic() - start_time)))
        frequency = 1/(time.perf_counter() - main_loop_start_time)
        print(f"main loop frequency: {frequency}")

if __name__ == '__main__':
    # test_joint_teleop(['right', 'left'])
    test_ee_teleop(['right'])
