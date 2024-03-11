import numpy as np
import time
import h5py

s = time.time()
for i in range(2 * 64):
    np.load('temp.npy')
print(time.time() - s)

data_path = '/data/yixuan22/diffusion_policy/data/sapien_env/teleop_data/pick_place_soda/small_rand_mixed_demo_400/feats'
s = time.time()
for i in range(64):
    epi = i % 20
    hdf5_path = f'{data_path}/episode_{epi}.hdf5'
    with h5py.File(hdf5_path, 'r') as f:
        feats = f['feats'][0:2]
print(time.time() - s)
