import numpy as np
import sapien.core as sapien
import transforms3d.euler
from transforms3d.quaternions import qmult

from sapien_env.sim_env.base import BaseSimulationEnv
from sapien_env.utils.yx_object_utils import load_open_box, load_yx_obj
from sapien_env.sim_env.constructor import add_default_scene_light
from sapien_env.gui.gui_base import GUIBase, DEFAULT_TABLE_TOP_CAMERAS, YX_TABLE_TOP_CAMERAS


class HangMugEnv(BaseSimulationEnv):
    def __init__(self, use_gui=True, frame_skip=5, object_scale=1, randomness_scale=1, friction=0.3, seed=None,
                 use_ray_tracing=True, manip_obj="nescafe_mug",**renderer_kwargs):
        super().__init__(use_gui=use_gui, frame_skip=frame_skip, use_ray_tracing=use_ray_tracing, **renderer_kwargs)

        # Construct scene
        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(config=scene_config)
        self.scene.set_timestep(0.004)
        self.friction = friction
        self.object_scale = object_scale
        self.randomness_scale = randomness_scale

        # Load table
        self.table = self.create_table(table_height=0.6, table_half_size=[0.35, 0.7, 0.025])

        # Load object
        self.mug_tree = load_yx_obj(self.scene, "headless_mug_tree", collision_shape='multiple', is_static=True, density=1000)
        self.manip_obj_name = manip_obj
        self.manipulated_object = load_yx_obj(self.scene, manip_obj, collision_shape='multiple', density=10)
        self.original_object_pos = np.zeros(3)
        
        # set up workspace boundary
        self.wkspc_half_w = 0.18
        self.wkspc_half_l = 0.18

    def reset_env(self):
        mug_tree_q = np.array(transforms3d.euler.euler2quat(np.pi/2., 0, -0.5, axes='sxyz').tolist())
        self.mug_tree_pose = sapien.Pose(np.array([0., 0.1, 0.02]), mug_tree_q)
        self.mug_tree.set_pose(self.mug_tree_pose)
        pose = self.generate_random_init_pose(self.randomness_scale)
        self.manipulated_object.set_pose(pose)
        self.original_object_pos = pose.p

    def generate_random_init_pose(self, randomness_scale):
        # pos = self.np_random.uniform(low=-0.1, high=0.1, size=2) * randomness_scale
        # pos = np.array([self.np_random.uniform(low=-0.2, high=-0.05),
        #                 self.np_random.uniform(low=-0.2, high=0.2)]) * randomness_scale
        
        # select pos that is within workspace and not too close to the box
        dist_thresh = 0.1
        # while True:
        if self.manip_obj_name == 'nescafe_mug':
            pos = np.array([self.np_random.uniform(0.08, 0.15),
                            self.np_random.uniform(-0.15, -0.08)])
            random_z_rotate = self.np_random.uniform(0.75 * np.pi, 1.25 * np.pi)
            position = np.array([pos[0], pos[1], 0.04])
        elif self.manip_obj_name == 'white_mug':
            pos = np.array([self.np_random.uniform(0.08, 0.15),
                            self.np_random.uniform(-0.12, -0.08)])
            random_z_rotate = self.np_random.uniform(0.75 * np.pi, 1.25 * np.pi)
            position = np.array([pos[0], pos[1], 0.1])
        # elif self.manip_obj_name == 'beer_mug':
        #     pos = np.array([self.np_random.uniform(0.1, 0.15),
        #                     self.np_random.uniform(-0.15, -0.1)])
        #     random_z_rotate = self.np_random.uniform(0.75 * np.pi, 1.25 * np.pi)
            position = np.array([pos[0], pos[1], 0.1])
        elif self.manip_obj_name == 'aluminum_mug':
            pos = np.array([self.np_random.uniform(0.08, 0.15),
                            self.np_random.uniform(-0.12, -0.08)])
            random_z_rotate = self.np_random.uniform(0.5 * np.pi, 1.0 * np.pi)
            position = np.array([pos[0], pos[1], 0.1])
        elif self.manip_obj_name == 'black_mug':
            pos = np.array([self.np_random.uniform(0.08, 0.15),
                            self.np_random.uniform(-0.15, -0.08)])
            random_z_rotate = self.np_random.uniform(0.25 * np.pi, 0.75 * np.pi)
            position = np.array([pos[0], pos[1], 0.1])
        elif self.manip_obj_name == 'blue_mug':
            pos = np.array([self.np_random.uniform(0.08, 0.15),
                            self.np_random.uniform(-0.15, -0.08)])
            random_z_rotate = self.np_random.uniform(0.75 * np.pi, 1.25 * np.pi)
            position = np.array([pos[0], pos[1], 0.1])
        elif self.manip_obj_name == 'kor_mug':
            pos = np.array([self.np_random.uniform(0.08, 0.15),
                            self.np_random.uniform(-0.15, -0.08)])
            random_z_rotate = self.np_random.uniform(0.75 * np.pi, 1.25 * np.pi)
            position = np.array([pos[0], pos[1], 0.1])
        elif self.manip_obj_name == 'low_poly_mug':
            pos = np.array([self.np_random.uniform(0.08, 0.15),
                            self.np_random.uniform(-0.15, -0.08)])
            random_z_rotate = self.np_random.uniform(0.75 * np.pi, 1.25 * np.pi)
            position = np.array([pos[0], pos[1], 0.1])
        elif self.manip_obj_name == 'blender_mug':
            pos = np.array([self.np_random.uniform(0.08, 0.15),
                            self.np_random.uniform(-0.15, -0.08)])
            random_z_rotate = self.np_random.uniform(0.75 * np.pi, 1.25 * np.pi)
            position = np.array([pos[0], pos[1], 0.1])
        elif self.manip_obj_name == 'mug_2':
            pos = np.array([self.np_random.uniform(0.08, 0.15),
                            self.np_random.uniform(-0.15, -0.08)])
            random_z_rotate = self.np_random.uniform(0.25 * np.pi, 0.75 * np.pi)
            position = np.array([pos[0], pos[1], 0.1])
        else:
            raise RuntimeError('Unknown object name')
        
        orientation = transforms3d.euler.euler2quat(np.pi/2, 0, random_z_rotate)
        pose = sapien.Pose(position, orientation)
        return pose
    
    def get_init_poses(self):
        init_poses = np.stack([self.manipulated_object.get_pose().to_transformation_matrix(),
                               self.mug_tree.get_pose().to_transformation_matrix()])
        return init_poses


def env_test():
    from sapien.utils import Viewer
    from constructor import add_default_scene_light
    env = HangMugEnv(use_ray_tracing=False, manip_obj='mug_2')
    env.reset_env()
    viewer = Viewer(env.renderer)
    viewer.set_scene(env.scene)
    viewer.set_camera_xyz(x=0, y=0.5, z = 0.5)
    viewer.set_camera_rpy(r=0, p=-0.5, y=np.pi/2)
    viewer.set_fovy(2.0)
    add_default_scene_light(env.scene, env.renderer)
    env.viewer = viewer

    while not viewer.closed:
        env.simple_step()
        env.render()

def get_init_pic():
    # obj_name = 'nescafe_mug'
    # obj_name = 'beer_mug'
    # obj_name = 'aluminum_mug'
    # obj_name = 'black_mug'
    # obj_name = 'white_mug'
    obj_name = 'blue_mug'
    # obj_name = 'kor_mug'
    # obj_name = 'low_poly_mug'
    output_dir = f'/home/yixuan/bdai/general_dp/d3fields_dev/d3fields/data/sapien/mug/{obj_name}'
    os.system(f'mkdir -p {output_dir}')
    env = HangMugEnv(manip_obj=obj_name, use_ray_tracing=False, seed=0)
    env.reset_env()
    
    # Setup viewer and camera
    add_default_scene_light(env.scene, env.renderer)
    gui = GUIBase(env.scene, env.renderer,headless=False)
    for name, params in YX_TABLE_TOP_CAMERAS.items():
        if 'rotation' in params:
            gui.create_camera_from_pos_rot(**params)
        else:
            gui.create_camera(**params)
    if not gui.headless:
        gui.viewer.set_camera_rpy(r=0, p=-0.5, y=np.pi/2)
        gui.viewer.set_camera_xyz(x=0, y=0.5, z=0.5)
    scene = env.scene
    scene.step()
    for i in range(10000):
        env.simple_step()
        rgbs = gui.render()
        print('current camera position: ', gui.viewer.window.get_camera_position())
        print('current camera rotation: ', qmult(gui.viewer.window.get_camera_rotation(), [0.5, -0.5, 0.5, 0.5]))
        print()
    for i, rgb in enumerate(rgbs):
        import cv2
        cv2.imwrite(f'{output_dir}/{i}.png', rgb)

if __name__ == '__main__':
    env_test()
    # get_init_pic()
