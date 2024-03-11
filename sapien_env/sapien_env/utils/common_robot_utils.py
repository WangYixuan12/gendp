from pathlib import Path
from typing import NamedTuple, List, Dict

import numpy as np
import sapien.core as sapien


class FreeRobotInfo(NamedTuple):
    path: str
    dof: int
    palm_name: str


class ArmRobotInfo(NamedTuple):
    path: str
    arm_dof: int
    hand_dof: int
    palm_name: str
    arm_init_qpos: List[float]
    root_offset: List[float] = [0.0, 0.0, 0.0]



def generate_free_robot_hand_info() -> Dict[str, FreeRobotInfo]:
    shadow_hand_free_info = FreeRobotInfo(path="robot/shadow_hand_description/shadowhand_free.urdf", dof=28,
                                          palm_name="palm_center")
    adroit_hand_free_info = FreeRobotInfo(path="robot/adroit_hand_free.urdf", dof=28, palm_name="palm")
    allegro_hand_free_info = FreeRobotInfo(path="robot/allegro_hand_description/allegro_hand_free.urdf", dof=22,
                                           palm_name="palm_center")
    svh_hand_free_info = FreeRobotInfo(path="robot/svh_hand_right.urdf", dof=26, palm_name="right_hand_e1")
    mano_hand_free_info = FreeRobotInfo(path="robot/mano_hand_free.urdf", dof=51, palm_name="palm")
    panda_hand_free_info = FreeRobotInfo(path="robot/panda_hand_free.urdf", dof=8, palm_name="panda_hand")

    info_dict = dict(shadow_hand_free=shadow_hand_free_info,
                     allegro_hand_free=allegro_hand_free_info,
                     svh_hand_free=svh_hand_free_info,
                     mano_hand_free=mano_hand_free_info,
                     panda_hand_free=panda_hand_free_info,
                     adroit_hand_free=adroit_hand_free_info)
    return info_dict


def generate_arm_robot_hand_info() -> Dict[str, ArmRobotInfo]:
    xarm_path = Path("robot/xarm6_description/")
    shadow_hand_xarm6 = ArmRobotInfo(path=str(xarm_path / "xarm6_shadow.urdf"), hand_dof=22, arm_dof=6,
                                     palm_name="palm_center", arm_init_qpos=[0, 0, 0, 0, -np.pi / 2, 0])
    allegro_hand_xarm6 = ArmRobotInfo(path=str(xarm_path / "xarm6_allegro.urdf"), hand_dof=16, arm_dof=6,
                                      palm_name="palm_center", arm_init_qpos=[0, 0, 0, 0, -np.pi / 2, 0],
                                      root_offset=[-0.0244, 0, 0])
    allegro_hand_xarm6_wrist_mounted_face_down = ArmRobotInfo(
        path="robot/xarm6_description/xarm6_allegro_wrist_mounted_rotate.urdf",
        hand_dof=16, arm_dof=6, palm_name="palm_center", arm_init_qpos=[0, 0, 0, 0, 0, -np.pi / 2],
        root_offset=[0.00, 0, 0])
    allegro_hand_xarm6_wrist_mounted_face_front = ArmRobotInfo(
        path="robot/xarm6_description/xarm6_allegro_wrist_mounted_rotate.urdf",
        hand_dof=16, arm_dof=6, palm_name="palm_center", arm_init_qpos=[0, 0, 0, np.pi, np.pi / 2, np.pi],
        root_offset=[0.00, 0, 0])
    allegro_hand_digit_xarm6_wrist_mounted_face_front = ArmRobotInfo(
        path="robot/xarm6_description/xarm6_allegro_digit_wrist_mounted_rotate.urdf",
        hand_dof=16, arm_dof=6, palm_name="palm_center", arm_init_qpos=[0, 0, 0, np.pi, np.pi / 2, np.pi],
        root_offset=[0.00, 0, 0])
    xarm6 = ArmRobotInfo(path=str(xarm_path / "xarm6.urdf"), hand_dof=0, arm_dof=6, palm_name="link6", arm_init_qpos=[0, 0, 0, 0, -np.pi / 2, 0])
    xarm6_with_gripper = ArmRobotInfo(path=str(xarm_path / "xarm6_with_gripper.urdf"), hand_dof=0, arm_dof=6, palm_name="link6", arm_init_qpos=[0, 0, 0, 0, -np.pi / 2, 0])
    info_dict = dict(
        xarm6=xarm6,
        xarm6_with_gripper=xarm6_with_gripper,
        shadow_hand_xarm6=shadow_hand_xarm6,
        allegro_hand_xarm6=allegro_hand_xarm6,
        allegro_hand_xarm6_wrist_mounted_face_down=allegro_hand_xarm6_wrist_mounted_face_down,
        allegro_hand_xarm6_wrist_mounted_face_front=allegro_hand_xarm6_wrist_mounted_face_front,
        allegro_hand_digit_xarm6_wrist_mounted_face_front=allegro_hand_digit_xarm6_wrist_mounted_face_front,
    )
    return info_dict


def generate_trossen_info() -> Dict[str, ArmRobotInfo]:
    trossen_path = Path("robot/trossen_description/")
    trossen_vx300 = ArmRobotInfo(path=str(trossen_path / "vx300.urdf"),  hand_dof = 2, arm_dof=5, palm_name="vx300/ee_arm_link", arm_init_qpos=[0, -0.8, 1, 1, 0])
    trossen_vx300_tactile = ArmRobotInfo(path=str(trossen_path / "vx300_tactile.urdf"),  hand_dof = 2, arm_dof=5, palm_name="vx300/ee_arm_link", arm_init_qpos=[0, -0.8, 1, 1, 0])
    trossen_vx300_tactile_map = ArmRobotInfo(path=str(trossen_path / "vx300_tactile_map_8.urdf"),  hand_dof = 2, arm_dof=5, palm_name="vx300/ee_arm_link", arm_init_qpos=[0, -0.8, 1, 1, 0])
    trossen_vx300_tactile_map_4x4 = ArmRobotInfo(path=str(trossen_path / "vx300_tactile_map_4x4.urdf"),  hand_dof = 2, arm_dof=5, palm_name="vx300/ee_arm_link", arm_init_qpos=[0, -0.8, 1, 1, 0])
    trossen_vx300_tactile_map_4x4_thin = ArmRobotInfo(path=str(trossen_path / "vx300_tactile_map_4x4_thin.urdf"),  hand_dof = 2, arm_dof=5, palm_name="vx300/ee_arm_link", arm_init_qpos=[0, -0.8, 1, 1, 0])
    trossen_vx300s = ArmRobotInfo(path=str(trossen_path / "vx300s.urdf"),  hand_dof = 2, arm_dof=6, palm_name="vx300s/ee_arm_link", arm_init_qpos=[0, -0.8, 1.3, 0, 0,0])
    trossen_vx300s_tactile_thin = ArmRobotInfo(path=str(trossen_path / "vx300s_tactile_thin_fix.urdf"),  hand_dof = 2, arm_dof=6, palm_name="vx300s/ee_arm_link", arm_init_qpos=[0, -0.8, 0.9, 0, 1.4,0])
    info_dict = dict(
        trossen_vx300=trossen_vx300,
        trossen_vx300_tactile=trossen_vx300_tactile,
        trossen_vx300_tactile_map =trossen_vx300_tactile_map,
        trossen_vx300_tactile_map_4x4=trossen_vx300_tactile_map_4x4, 
        trossen_vx300_tactile_map_4x4_thin =trossen_vx300_tactile_map_4x4_thin,
        trossen_vx300s =trossen_vx300s,
        trossen_vx300s_tactile_thin=trossen_vx300s_tactile_thin
    )
    return info_dict

def generate_panda_info() -> Dict[str, ArmRobotInfo]:
    panda = ArmRobotInfo(
        path="robot/panda/panda.urdf",
        # hand_dof=2, arm_dof=7, palm_name="panda_hand", arm_init_qpos=[0.0, 0.0, 0.4, np.pi, 0.0, np.pi / 2, 0.9],
        hand_dof=2, arm_dof=7, palm_name="panda_hand", arm_init_qpos=[-2.214023, 0.17274654, 2.238009, -2.2748125, -0.16332519, 2.1609645, 0.9082864, 0.04, 0.04],
        root_offset=[0.00, 0, 0])
    info_dict = dict(
        panda=panda
    )
    return info_dict

def generate_retargeting_link_names(robot_name):
    if "shadow_hand" in robot_name or "adroit_hand" in robot_name:
        link_names = ["palm", "thtip", "fftip", "mftip", "rftip", "lftip"]
        link_names += ["thmiddle", "ffmiddle", "mfmiddle", "rfmiddle", "lfmiddle"]
        link_hand_indices = [0, 4, 8, 12, 16, 20] + [2, 6, 10, 14, 18]
    elif "allegro_hand" in robot_name:
        link_names = ["palm", "link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_14.0", "link_2.0",
                      "link_6.0", "link_10.0"]
        link_hand_indices = [0, 4, 8, 12, 16] + [2, 6, 10, 14]
    else:
        raise NotImplementedError
    return link_names, link_hand_indices


def wrap_link_hand_indices(link_hand_indices, method="tip_middle"):
    if method == "tip_middle":
        mapping = {i * 4: i for i in range(6)}  # tip
        mapping.update({i * 4 + 2: i + 5 for i in range(5)})  # middle
        result = [mapping[i] for i in link_hand_indices]
        if 0 in result:
            del result[0]
    else:
        raise NotImplementedError
    return result



def load_robot(scene: sapien.Scene, robot_name, disable_self_collision=True) -> sapien.Articulation:
    loader = scene.create_urdf_loader()
    current_dir = Path(__file__).parent
    package_dir = (current_dir.parent / "assets").resolve()
    if "free" in robot_name:
        info = generate_free_robot_hand_info()[robot_name]
        config = {}
    elif "xarm" in robot_name:
        info = generate_arm_robot_hand_info()[robot_name]
        config = {}
    elif "trossen" in robot_name:
        info = generate_trossen_info()[robot_name]
        config = {}
    elif "panda" in robot_name:
        info = generate_panda_info()[robot_name]
        config = {}
        config = {
            "link": {
                "panda_leftfinger": {
                    "material": scene.create_physical_material(
                        **dict(
                            static_friction=1000, dynamic_friction=1000, restitution=0
                        )
                    )
                },
                "panda_rightfinger": {
                    "material": scene.create_physical_material(
                        **dict(
                            static_friction=1000, dynamic_friction=1000, restitution=0
                        )
                    )
                },
            }
        }
    robot_file = info.path
    filename = str(package_dir / robot_file)
    robot_builder = loader.load_file_as_articulation_builder(filename, config=config)
    if disable_self_collision:
        for link_builder in robot_builder.get_link_builders():
            link_builder.set_collision_groups(1, 1, 17, 0)
    else:
        if "allegro" in robot_name:
            for link_builder in robot_builder.get_link_builders():
                if link_builder.get_name() in ["link_9.0", "link_5.0", "link_1.0", "link_13.0", "base_link"]:
                    link_builder.set_collision_groups(1, 1, 17, 0)
    robot = robot_builder.build(fix_root_link=True)
    robot.set_name(robot_name)

    robot_arm_control_params = np.array([200000, 40000, 500])
    root_translation_control_params = np.array([0, 20000, 20000])
    root_rotation_control_params = np.array([0, 5000, 5000])
    finger_control_params = np.array([200, 60, 10])


    if "free" in robot_name:
        for joint in robot.get_active_joints():
            name = joint.get_name()
            if "x_joint" in name or "y_joint" in name or "z_joint" in name:
                joint.set_drive_property(*(1 * root_translation_control_params), mode="force")
            elif "x_rotation_joint" in name or "y_rotation_joint" in name or "z_rotation_joint" in name:
                joint.set_drive_property(*(1 * root_rotation_control_params), mode="force")
            else:
                joint.set_drive_property(*(1 * finger_control_params), mode="force")
    elif "xarm" in robot_name:
        arm_joint_names = [f"joint{i}" for i in range(1, 8)]
        for joint in robot.get_active_joints():
            name = joint.get_name()
            if name in arm_joint_names:
                joint.set_drive_property(*(1 * robot_arm_control_params), mode="force")
            else:
                joint.set_drive_property(*(1 * finger_control_params), mode="force")

    elif "panda" in robot_name:
        arm_joint_names = [f"joint{i}" for i in range(1, 8)]
        print("debug",arm_joint_names)
        for joint in robot.get_active_joints():
            name = joint.get_name()
            if name in arm_joint_names:
                joint.set_drive_property(*(1 * robot_arm_control_params), mode="force")
            else:
                joint.set_drive_property(*(3 * finger_control_params), mode="force")

    elif "trossen" in robot_name:
        # arm_joint_names = [f"joint{i}" for i in range(0, 6)]
        # print(robot.get_active_joints())
        # print("debug",arm_joint_names)
        for joint in robot.get_active_joints():
            name = joint.get_name()
            # if "left_finger" in name or "right_finger" in name:
            #     joint.set_drive_property(*10*(np.array([100,8,5])), mode="force")
            # else:
            #     joint.set_drive_property(*8*(np.array([100,10,5])), mode="force")
            # if "left_finger" in name or "right_finger" in name:
            #     joint.set_drive_property(*10*(np.array([100,10,5])), mode="force")
            # else:
            #     joint.set_drive_property(*10*(np.array([100,10,5])), mode="force")
            if "left_finger" in name or "right_finger" in name:
                joint.set_drive_property(*50*(np.array([100,10,5])), mode="force")
            else:
                joint.set_drive_property(*30*(np.array([100,10,5])), mode="force")
    else:
        raise NotImplementedError

    mat = scene.engine.create_physical_material(1.5, 1, 0.01)
    for link in robot.get_links():
        for geom in link.get_collision_shapes():
            geom.min_patch_radius = 0.02
            geom.patch_radius = 0.04
            geom.set_physical_material(mat)

    return robot


def modify_robot_visual(robot: sapien.Articulation):
    robot_name = robot.get_name()
    if "mano" in robot_name:
        return robot
    arm_link_names = [f"link{i}" for i in range(1, 8)] + ["link_base"]
    for link in robot.get_links():
        if link.get_name() in arm_link_names:
            pass
        else:
            for geom in link.get_visual_bodies():
                for shape in geom.get_render_shapes():
                    mat_viz = shape.material
                    mat_viz.set_specular(0.07)
                    mat_viz.set_metallic(0.3)
                    mat_viz.set_roughness(0.2)
                    if 'adroit' in robot_name:
                        mat_viz.set_specular(0.02)
                        mat_viz.set_metallic(0.1)
                        mat_viz.set_base_color(np.power(np.array([0.9, 0.7, 0.5, 1]), 1.5))
                    elif 'allegro' in robot_name:
                        if "tip" not in link.get_name():
                            mat_viz.set_specular(0.8)
                            mat_viz.set_base_color(np.array([0.1, 0.1, 0.1, 1]))
                        else:
                            mat_viz.set_base_color(np.array([0.9, 0.9, 0.9, 1]))
                    elif 'svh' in robot_name:
                        link_names = ["right_hand_c", "right_hand_t", "right_hand_s", "right_hand_r", "right_hand_q",
                                      "right_hand_e1"]
                        if link.get_name() not in link_names:
                            mat_viz.set_specular(0.02)
                            mat_viz.set_metallic(0.1)
                    else:
                        pass
    return robot


class LPFilter:
    def __init__(self, control_freq, cutoff_freq):
        dt = 1 / control_freq
        wc = cutoff_freq * 2 * np.pi
        y_cos = 1 - np.cos(wc * dt)
        self.alpha = -y_cos + np.sqrt(y_cos ** 2 + 2 * y_cos)
        self.y = 0
        self.is_init = False

    def next(self, x):
        self.y = self.y + self.alpha * (x - self.y)
        return self.y.copy()

    def init(self, y):
        self.y = y.copy()
        self.is_init = True


class PIDController:
    def __init__(self, kp, ki, kd, dt, output_range):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.output_range = output_range
        self._prev_err = None
        self._cum_err = 0

    def reset(self):
        self._prev_err = None
        self._cum_err = 0

    def control(self, err):
        if self._prev_err is None:
            self._prev_err = err

        value = (
                self.kp * err
                + self.kd * (err - self._prev_err) / self.dt
                + self.ki * self._cum_err
        )

        self._prev_err = err
        self._cum_err += self.dt * err

        return np.clip(value, self.output_range[0], self.output_range[1])
