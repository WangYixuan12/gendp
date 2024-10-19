import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import numpy as np
import time
import transforms3d

from interbotix_xs_modules.arm import InterbotixManipulatorXS

from gendp.common.aloha_utils import (
    torque_on, torque_off, move_arms, move_grippers, get_arm_gripper_positions,
    START_ARM_POSE, START_EE_POSE, MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE, DT
)
from gendp.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer

def prep_robots(master_bot):
    # reboot gripper motors, and set operating modes for all motors
    master_bot.dxl.robot_set_operating_modes("group", "arm", "position")
    master_bot.dxl.robot_set_operating_modes("single", "gripper", "position")
    # puppet_bot.dxl.robot_set_motor_registers("single", "gripper", 'current_limit', 1000) # TODO(tonyzhaozh) figure out how to set this limit
    torque_on(master_bot)

    # move arms to starting position
    start_arm_qpos = START_ARM_POSE[:6]
    move_arms([master_bot], [start_arm_qpos], move_time=1)
    # move grippers to starting position
    move_grippers([master_bot], [MASTER_GRIPPER_JOINT_MID], move_time=0.5)


def press_to_start(master_bot):
    # press gripper to start data collection
    # disable torque for only gripper joint of master robot to allow user movement
    master_bot.dxl.robot_torque_enable("single", "gripper", False)
    print(f'Close the gripper to start')
    close_thresh = -0.3
    pressed = False
    while not pressed:
        gripper_pos = get_arm_gripper_positions(master_bot)
        if gripper_pos < close_thresh:
            pressed = True
        time.sleep(DT/10)
    torque_off(master_bot)
    print(f'Started!')

class AlohaMaster(mp.Process):
    def __init__(self, 
            shm_manager, 
            get_max_k=30, 
            frequency=200,
            robot_side='right',
            verbose=False
            ):
        """
        Read from master robot
        """
        super().__init__()

        # copied variables
        self.frequency = frequency
        self.robot_side = robot_side
        self.verbose = verbose

        example = {
            'joint_pos': np.array(START_ARM_POSE[:7]),
            'ee_pose': np.array(START_EE_POSE).astype(np.float32), # (xyz, quat, gripper)
            'receive_timestamp': time.time()
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager, 
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        # shared variables
        self.ready_event = mp.Event()
        self.stop_event = mp.Event()
        self.ring_buffer = ring_buffer
        
    # ======= get state APIs ==========
    def get_motion_state(self):
        state = self.ring_buffer.get()
        return state
    
    #========== start stop API ===========

    def start(self, wait=True):
        super().start()
        if wait:
            self.ready_event.wait()
    
    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.join()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= main loop ==========
    def run(self):
        try:
            # set up robot
            self.master_bot = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper", robot_name=f'master_{self.robot_side}', init_node=True)
            prep_robots(self.master_bot)
            press_to_start(self.master_bot)
            
            # send one message immediately so client can start reading
            self.ring_buffer.put({
                'joint_pos': np.array(START_ARM_POSE[:7]),
                'ee_pose': np.array(START_EE_POSE).astype(np.float32),
                'receive_timestamp': time.time()
            })
            self.ready_event.set()

            while not self.stop_event.is_set():
                t_start = time.perf_counter()
                joint_pos = self.master_bot.dxl.robot_get_joint_states().position[:7]
                ee_pose_mat = self.master_bot.arm.get_ee_pose()
                ee_pose_quat = np.array(transforms3d.quaternions.mat2quat(ee_pose_mat[:3,:3]).tolist())
                ee_pose = np.concatenate([ee_pose_mat[:3,3], ee_pose_quat, [joint_pos[6]]])
                receive_timestamp = time.time()
                self.ring_buffer.put({
                    'joint_pos': joint_pos[:7],
                    'ee_pose': ee_pose, # (xyz, quat, gripper)
                    'receive_timestamp': receive_timestamp
                })
                time.sleep(1/self.frequency)
                if self.verbose:
                    print(f'AlohaMaster: {1/(time.perf_counter() - t_start)} Hz')
        finally:
            self.ready_event.set()

def test():
    with SharedMemoryManager() as shm_manager:
        with AlohaMaster(shm_manager=shm_manager,
                         get_max_k=10,
                         frequency=100,
                         robot_side='right') as master:
            for _ in range(1000):
                state = master.get_motion_state()
                print('receive_timestamp: ', state['receive_timestamp'])
                print('joint_pos: ', state['joint_pos'])
                print('ee_pose: ', state['ee_pose'])
                time.sleep(0.1)

if __name__ == '__main__':
    test()
