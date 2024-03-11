import time
import sys
import IPython
e = IPython.embed

import numpy as np
import transforms3d

from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
from gild.common.aloha_utils import (
    torque_on, torque_off, move_arms, move_grippers, get_arm_gripper_positions,
    START_ARM_POSE, MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE, DT, MASTER2PUPPET_JOINT_FN
)

def prep_robots(master_bot, puppet_bot):
    # reboot gripper motors, and set operating modes for all motors
    puppet_bot.dxl.robot_reboot_motors("single", "gripper", True)
    puppet_bot.dxl.robot_set_operating_modes("group", "arm", "position")
    puppet_bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    master_bot.dxl.robot_set_operating_modes("group", "arm", "position")
    master_bot.dxl.robot_set_operating_modes("single", "gripper", "position")
    # puppet_bot.dxl.robot_set_motor_registers("single", "gripper", 'current_limit', 1000) # TODO(tonyzhaozh) figure out how to set this limit
    torque_on(puppet_bot)
    torque_on(master_bot)

    # move arms to starting position
    start_arm_qpos = START_ARM_POSE[:6]
    move_arms([master_bot, puppet_bot], [start_arm_qpos] * 2, move_time=1)
    # move grippers to starting position
    move_grippers([master_bot, puppet_bot], [MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE], move_time=0.5)


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

def teleop(robot_side):
    """ A standalone function for experimenting with teleoperation. No data recording. """
    puppet_bot = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_{robot_side}', init_node=True)
    master_bot = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper", robot_name=f'master_{robot_side}', init_node=False)

    prep_robots(master_bot, puppet_bot)
    press_to_start(master_bot)

    ### Teleoperation loop
    gripper_command = JointSingleCommand(name="gripper")
    while True:
        start_time = time.monotonic()
        
        # sync joint positions
        master_state_joints = master_bot.dxl.joint_states.position[:6]
        puppet_bot.arm.set_joint_positions(master_state_joints, blocking=False)
        
        # # sapien impl
        # curr_puppet_joint_pos = np.array(puppet_bot.dxl.joint_states.position[:6])
        # master_joint_pos = np.array(master_bot.dxl.joint_states.position[:6])
        # target_puppet_ee_pose = teleop_robot.compute_fk(np.concatenate([master_joint_pos, np.zeros(2)])[:,None])
        # target_puppet_joint_pos = teleop_robot.ik_vx300s_sapien_pose(initial_qpos=np.concatenate([curr_puppet_joint_pos, np.zeros(2)])[:,None],
        #                                                  pose=target_puppet_ee_pose,)
        # puppet_bot.arm.set_joint_positions(target_puppet_joint_pos, blocking=False)
        
        # sync gripper positions
        master_gripper_joint = master_bot.dxl.joint_states.position[6]
        puppet_gripper_joint_target = MASTER2PUPPET_JOINT_FN(master_gripper_joint)
        gripper_command.cmd = puppet_gripper_joint_target
        puppet_bot.gripper.core.pub_single.publish(gripper_command)
        # sleep DT
        time.sleep(max(0, DT - (time.monotonic() - start_time)))
        
        end_time = time.monotonic()
        print('control freq: ', 1/(end_time - start_time))

def low_freq_teleop(robot_side):
    """ A standalone function for experimenting with teleoperation. No data recording. """
    puppet_bot = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_{robot_side}', init_node=True)
    master_bot = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper", robot_name=f'master_{robot_side}', init_node=False)

    prep_robots(master_bot, puppet_bot)
    press_to_start(master_bot)

    ### Teleoperation loop
    gripper_command = JointSingleCommand(name="gripper")
    DT = 0.1
    puppet_DT = 0.02
    max_joint_vel = 0.4 # rad/s
    last_master_state_joints = None
    while True:
        start_time = time.monotonic()
        
        # sync joint positions
        master_state_joints = np.array(master_bot.dxl.joint_states.position[:6])
        
        last_master_state_joints = master_state_joints if last_master_state_joints is None else last_master_state_joints
        curr_puppet_state_joints = last_master_state_joints.copy()
        for i in range(int(DT/puppet_DT)):
            start_time = time.monotonic()
            delta_puppet_state_joints = np.clip(master_state_joints - curr_puppet_state_joints, -max_joint_vel*puppet_DT, max_joint_vel*puppet_DT)
            curr_puppet_state_joints = curr_puppet_state_joints + delta_puppet_state_joints
            puppet_bot.arm.set_joint_positions(curr_puppet_state_joints, blocking=False)
            # puppet_bot.arm.set_joint_positions(inter_state_joints_ls[i], blocking=False)
            
            # sync gripper positions
            master_gripper_joint = master_bot.dxl.joint_states.position[6]
            puppet_gripper_joint_target = MASTER2PUPPET_JOINT_FN(master_gripper_joint)
            gripper_command.cmd = puppet_gripper_joint_target
            puppet_bot.gripper.core.pub_single.publish(gripper_command)
            # sleep DT
            time.sleep(max(0, puppet_DT - (time.monotonic() - start_time)))
            
            end_time = time.monotonic()
            print('control freq: ', 1/(end_time - start_time))
        
        last_master_state_joints = curr_puppet_state_joints.copy()
        last_master_state_joints = master_state_joints.copy()

if __name__=='__main__':
    side = 'right'
    teleop(side)
    # low_freq_teleop(side)