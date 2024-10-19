import numpy as np
import sapien.core as sapien
import transforms3d.euler

from sapien_env.sim_env.base import BaseSimulationEnv
from sapien_env.utils.yx_object_utils import load_open_box, load_yx_obj


class MugCollectEnv(BaseSimulationEnv):
    def __init__(self, use_gui=True, frame_skip=5, object_scale=1, randomness_scale=1, friction=0.3, seed=None,
                 use_ray_tracing=True, manip_obj="cola", randomness_level='full', **renderer_kwargs):
        super().__init__(use_gui=use_gui, frame_skip=frame_skip, use_ray_tracing=use_ray_tracing, **renderer_kwargs)

        # Construct scene
        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(config=scene_config)
        self.scene.set_timestep(0.004)
        self.friction = friction
        self.object_scale = object_scale
        self.randomness_scale = randomness_scale
        self.randomness_level = randomness_level

        # Load table
        self.table = self.create_table(table_height=0.6, table_half_size=[0.35, 0.7, 0.025])

        # Load object
        self.manip_obj_name = manip_obj
        self.manipulated_object = load_yx_obj(self.scene, manip_obj, density=1000)
        self.original_object_pos = np.zeros(3)
        
        # set up workspace boundary
        self.wkspc_half_w = 0.18
        self.wkspc_half_l = 0.18
        
        self.box_ls = None

    def reset_env(self):
        if self.randomness_level == 'full':
            if self.box_ls is None:
                # Load box
                self.box_pos = np.array([0.0, 0.0, 0.0])
                self.box_ls = load_open_box(self.scene, self.renderer, half_l=0.06, half_w=0.06, h=0.02, floor_width=0.005, origin=self.box_pos,)
                self.box_pos = np.array([self.np_random.uniform(-self.wkspc_half_w, self.wkspc_half_w),
                                        self.np_random.uniform(-self.wkspc_half_l, self.wkspc_half_l),
                                        0.0])
                self.box_ori = np.array([1, 0, 0, 0])
                self.box_pose = sapien.Pose(self.box_pos, self.box_ori)
                self.box_ls.set_pose(self.box_pose)
        elif self.randomness_level == 'half':
            if self.box_ls is None:
                # Load box
                self.box_pos = np.array([0.0, 0.0, 0.0])
                self.box_ls = load_open_box(self.scene, self.renderer, half_l=0.06, half_w=0.06, h=0.02, floor_width=0.005, origin=self.box_pos,)
                self.box_pos = np.array([self.np_random.uniform(0.05, 0.10),
                                        self.np_random.uniform(-0.05, 0.05),
                                        0.0])
                self.box_ori = np.array([1, 0, 0, 0])
                self.box_pose = sapien.Pose(self.box_pos, self.box_ori)
                self.box_ls.set_pose(self.box_pose)
        
        pose = self.generate_random_init_pose(self.randomness_scale)
        self.manipulated_object.set_pose(pose)
        self.original_object_pos = pose.p

    def generate_random_init_pose(self, randomness_scale):
        # select pos that is within workspace and not too close to the box
        if self.randomness_level == 'full':
            dist_thresh = 0.1
            while True:
                pos = np.array([self.np_random.uniform(-self.wkspc_half_w, self.wkspc_half_w),
                                self.np_random.uniform(-self.wkspc_half_l, self.wkspc_half_l)])
                dist = np.linalg.norm(pos - self.box_pos[:2])
                if dist > dist_thresh:
                    break
        elif self.randomness_level == 'half':
            pos = np.array([self.np_random.uniform(low=-self.wkspc_half_w, high=-0.05),
                            self.np_random.uniform(low=-self.wkspc_half_l, high=self.wkspc_half_l)])
        
        random_z_rotate = self.np_random.uniform(0, np.pi)
        orientation = transforms3d.euler.euler2quat(np.pi/2, 0, random_z_rotate)
        position = np.array([pos[0], pos[1], 0.01])
        pose = sapien.Pose(position, orientation)
        return pose

    def get_init_poses(self):
        init_poses = np.stack([self.manipulated_object.get_pose().to_transformation_matrix(),
                               self.box_ls.get_pose().to_transformation_matrix()])
        return init_poses

def env_test():
    from sapien.utils import Viewer
    from constructor import add_default_scene_light
    env = MugCollectEnv(use_ray_tracing=False)
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


if __name__ == '__main__':
    env_test()
