import argparse
import time

import numpy as np
from gendp.common.aloha_utils import (
    DT,
    MASTER2PUPPET_JOINT_FN,
    MASTER_GRIPPER_JOINT_MID,
    MASTER_GRIPPER_JOINT_CLOSE,
    PUPPET_GRIPPER_JOINT_CLOSE,
    START_ARM_POSE,
    get_arm_gripper_positions,
    move_arms,
    move_grippers,
    torque_off,
    torque_on,
)
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand


def prep_robots(master_bot, puppet_bot):
    # reboot gripper motors, and set operating modes for all motors
    puppet_bot.dxl.robot_reboot_motors("single", "gripper", True)
    puppet_bot.dxl.robot_set_operating_modes("group", "arm", "position")
    puppet_bot.dxl.robot_set_operating_modes(
        "single", "gripper", "current_based_position"
    )
    master_bot.dxl.robot_set_operating_modes("group", "arm", "position")
    master_bot.dxl.robot_set_operating_modes("single", "gripper", "position")
    # puppet_bot.dxl.robot_set_motor_registers("single", "gripper", 'current_limit', 1000) # TODO(tonyzhaozh) figure out how to set this limit
    torque_on(puppet_bot)
    torque_on(master_bot)

    # move arms to starting position
    start_arm_qpos = START_ARM_POSE[:6]
    move_arms([master_bot, puppet_bot], [start_arm_qpos] * 2, move_time=1)
    # move grippers to starting position
    move_grippers(
        [master_bot, puppet_bot],
        [MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE],
        move_time=0.5,
    )


def press_to_start(master_bot):
    # press gripper to start data collection
    # disable torque for only gripper joint of master robot to allow user movement
    master_bot.dxl.robot_torque_enable("single", "gripper", False)
    print("Close the gripper to start")
    close_thresh = (MASTER_GRIPPER_JOINT_MID + MASTER_GRIPPER_JOINT_CLOSE) / 2.0
    pressed = False
    while not pressed:
        gripper_pos = get_arm_gripper_positions(master_bot)
        if gripper_pos < close_thresh:
            pressed = True
        time.sleep(DT / 10)
    torque_off(master_bot)
    print("Started!")


def teleop(robot_sides):
    """ A standalone function for experimenting with teleoperation. No data recording. """
    num_bot = len(robot_sides)
    puppet_bots = [InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_{robot_sides[0]}', init_node=True)]
    puppet_bots += [InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_{robot_side}', init_node=False) for robot_side in robot_sides[1:]]
    master_bots = [InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper", robot_name=f'master_{robot_side}', init_node=False) for robot_side in robot_sides]

    for i in range(num_bot):
        prep_robots(master_bots[i], puppet_bots[i])
        press_to_start(master_bots[i])

    ### Teleoperation loop
    gripper_command = JointSingleCommand(name="gripper")
    while True:
        start_time = time.monotonic()
        
        for i in range(num_bot):
            # sync joint positions
            master_state_joints = master_bots[i].dxl.joint_states.position[:6]
            puppet_bots[i].arm.set_joint_positions(master_state_joints, blocking=False)
            
            # sync gripper positions
            master_gripper_joint = master_bots[i].dxl.joint_states.position[6]
            puppet_gripper_joint_target = MASTER2PUPPET_JOINT_FN(master_gripper_joint)
            gripper_command.cmd = puppet_gripper_joint_target
            puppet_bots[i].gripper.core.pub_single.publish(gripper_command)
        
        # sleep DT
        time.sleep(max(0, DT - (time.monotonic() - start_time)))
        
        end_time = time.monotonic()
        print('control freq: ', 1/(end_time - start_time))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--right', action='store_true', help='Teleoperate right robot')
    parser.add_argument('--left', action='store_true', help='Teleoperate left robot')
    args = parser.parse_args()
    
    sides = []
    if args.right:
        sides.append('right')
    if args.left:
        sides.append('left')
    if not args.right and not args.left:
        raise ValueError('At least one robot side should be teleoperated')
    teleop(sides)
