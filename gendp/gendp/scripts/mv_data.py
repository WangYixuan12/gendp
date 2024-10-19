import os
from tqdm import tqdm

src_dir = '/data/yixuan22/diffusion_policy/data/sapien_env/teleop_data/pick_place_soda/small_rand_white_pepsi_demo_100'
tgt_dir = '/data/yixuan22/diffusion_policy/data/sapien_env/teleop_data/pick_place_soda/small_rand_mixed_demo_400'

idx_s = 0
idx_e = 100
idx_offset = 300

os.system(f'mkdir -p {tgt_dir}')
os.system(f'cp {src_dir}/config.yaml {tgt_dir}/config.yaml')
os.system(f'cp -r {src_dir}/sapien_env {tgt_dir}')

for i in tqdm(range(idx_s, idx_e)):
    os.system(f'cp {src_dir}/episode_{i}.hdf5 {tgt_dir}/episode_{i+idx_offset}.hdf5')
