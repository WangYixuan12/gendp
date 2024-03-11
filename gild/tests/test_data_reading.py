import time
import sys
sys.path.append('/home/yixuan/general_dp')
import os

from tqdm import tqdm
import h5py
import numpy as np

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

epi_s = 8
num_episode = 1

dic_ls = []
start = time.time()
for i in tqdm(range(epi_s, epi_s+num_episode)):
    file_name = f'/media/yixuan_2T/diffusion_policy/data/sapien_env/teleop_data/pick_place_soda/2023-10-19-21-37-12-135326/episode_{i}.hdf5'
    dic, _ = load_dict_from_hdf5(file_name)
    print(dic['observations']['images']['front_view_color'][0])
    dic_ls.append(dic)
end = time.time()
print(f'load_dict_from_hdf5: {end-start}')
