import numpy as np
import matplotlib.pyplot as plt

feat_map_1 = np.load('temp/feat_map_0_pepsi.npy')
feat_map_2 = np.load('temp/feat_map_0_cola.npy')

feat_map_diff = np.linalg.norm(feat_map_1 - feat_map_2, axis=-1)

plt.imshow(feat_map_diff)
plt.show()
