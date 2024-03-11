import time
import sys
sys.path.append('/home/yixuan/general_dp')
import os

from tqdm import tqdm
import h5py

from gild.common.data_utils import load_dict_from_hdf5

def load_hdf5(dataset_path):
    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        action = root['/action'][()]
        #TODO image
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            if 'color' in cam_name:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]
    return qpos, qvel, action, image_dict

num_episode = 10

dic_ls = []
start = time.time()
for i in tqdm(range(num_episode)):
    file_name = f'/home/yixuan/sapien_env/data/teleop_data/pick_place_soda/episode_{i}.hdf5'
    dic, _ = load_dict_from_hdf5(file_name)
    dic_ls.append(dic)
end = time.time()
print(f'load_dict_from_hdf5: {end-start}')
