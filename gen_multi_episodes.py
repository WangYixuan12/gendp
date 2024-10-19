import os
from pathlib import Path

from tqdm import tqdm
from sapien_env.utils.misc_utils import get_current_YYYY_MM_DD_hh_mm_ss_ms

### hyper parameter specification
mode = 'straight'
# obj = 'cola'
# obj = 'diet_soda'
# obj = 'drpepper'
# obj = 'white_pepsi'
# obj = 'nescafe_mug'
# obj = 'kor_mug'
# obj = 'low_poly_mug'
# obj = 'blue_mug'
# obj = 'black_mug'
# obj = 'white_mug'
# obj = 'aluminum_mug'
# obj = 'pencil'
obj = 'pencil_2'
# task_name = 'hang_mug'
task_name = 'pen_insertion'
# task_name = 'mug_collect'
dataset_name = f'{obj}_demo_100'
headless = False
s_idx = 50
e_idx = 100
data_root = '/home/yixuan/gendp'
dataset_dir = f"{data_root}/data/sapien_demo/{dataset_name}"
os.system(f'mkdir -p {dataset_dir}')


### copy current repo
save_repo_path = f'sapien_env'
save_repo_dir = os.path.join(dataset_dir, save_repo_path)
os.system(f'mkdir -p {save_repo_dir}')

curr_repo_dir = Path(__file__).parent / 'sapien_env' / 'sapien_env'
ignore_list = ['.git', '__pycache__', 'data']
for sub_dir in os.listdir(curr_repo_dir):
    if sub_dir not in ignore_list:
        os.system(f'cp -r {os.path.join(curr_repo_dir, sub_dir)} {save_repo_dir}')

### generate episodes
seed_range = range(s_idx, e_idx)
for i in tqdm(seed_range):
    if headless:
        PY_CMD = f'python gen_single_episode.py {i} {dataset_dir} {task_name} --headless --obj_name {obj} --mode {mode}'
    else:
        PY_CMD = f'python gen_single_episode.py {i} {dataset_dir} {task_name} --obj_name {obj} --mode {mode}'
    os.system(PY_CMD)
