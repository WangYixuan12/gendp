import os
from tqdm import tqdm

src_dir = '/media/yixuan_2T/diffusion_policy/data/sapien_env/teleop_data/pick_place_soda/yx_demo_500'
tgt_dir = '/media/yixuan_2T/diffusion_policy/data/sapien_env/teleop_data/pick_place_soda/xy_demo_500_yx_demo_500'

for i in tqdm(range(350, 500)):
    src_fn = f'{src_dir}/episode_{i}.hdf5'
    tgt_fn = f'{tgt_dir}/episode_{i+500}.hdf5'
    CMD = f'cp {src_fn} {tgt_fn}'
    os.system(CMD)
