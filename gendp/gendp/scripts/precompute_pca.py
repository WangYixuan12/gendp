import torch
import torch.nn.functional as F
import torchvision.transforms as T
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import sys
sys.path.append(os.getcwd())

data_root = '/media/yixuan_2T/diffusion_policy/data/pca_imgs'

# patch_h = 224 // 14
# patch_w = 224 // 14
patch_h = 75
patch_w = 75
# feat_dim = 384 # vits14
# feat_dim = 768 # vitb14
feat_dim = 1024 # vitl14
# feat_dim = 1536 # vitg14

obj_type = 'can'

device = 'cuda'

transform = T.Compose([
    # T.GaussianBlur(9, sigma=(0.1, 2.0)),
    T.Resize((patch_h * 14, patch_w * 14)),
    T.CenterCrop((patch_h * 14, patch_w * 14)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
# model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)
# model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').to(device)

print(model)

# root_path = '/media/yixuan_2T/dynamic_repr/NeuRay/data/don/single_object/caterpillar/2018-04-10-16-02-59/processed/images'

# img_path = os.path.join(root_path, '000000_rgb.png')
features = torch.zeros(4, patch_h * patch_w, feat_dim).to(device)
imgs_tensor = torch.zeros(4, 3, patch_h * 14, patch_w * 14).to(device)
for i in range(4):
    img_path = os.path.join(data_root, obj_type, f'{i}.png')
    img = cv2.imread(img_path)
    img = cv2.resize(img, (14 * patch_w, 14 * patch_h))
    # img = Image.open(img_path).convert('RGB')
    img = Image.fromarray(img)
    imgs_tensor[i] = transform(img)[:3]
with torch.no_grad():
    features_dict = model.forward_features(imgs_tensor)
    features = features_dict['x_norm_patchtokens']

# fg_seg_features = torch.zeros(1, patch_h * patch_w, feat_dim).to(device)
# fg_seg_imgs_tensor = torch.zeros(1, 3, patch_h * 14, patch_w * 14).to(device)
# img_path = f'dino_test_imgs/fly_3.png'
# img = Image.open(img_path).convert('RGB')
# fg_seg_imgs_tensor[0] = transform(img)[:3]

# with torch.no_grad():
#     features_dict = model.forward_features(fg_seg_imgs_tensor)
#     fg_seg_features = features_dict['x_norm_patchtokens']

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, normalize

# features = features.reshape(1, 30, 40, 384).permute(0, 3, 1, 2)
# features = F.upsample_bilinear(features, size=(480, 640))
# features = features.squeeze(0).permute(1,2,0).reshape(480 * 640, 384)

# features = features.reshape(4 * patch_h * patch_w, feat_dim).detach().cpu().numpy()
features = features.reshape(4 * patch_h * patch_w, feat_dim).detach().cpu().numpy()

# normalize the features before pca
# features = features - features.mean(axis=1, keepdims=True)
# features = features / features.std(axis=1, keepdims=True)
# features = features / features.std(axis=0, keepdims=True)
# features = features / features.max(axis=1, keepdims=True)
# features = features / features.max(axis=0, keepdims=True)
# features = normalize(features, axis=1, norm='max')
# features = scale(features, axis=0)

fg_seg = PCA(n_components=3)
fg_seg.fit(features)
pca_features = fg_seg.transform(features)

# pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
# pca_features = pca_features * 255

plt.subplot(1, 3, 1)
plt.hist(pca_features[:, 0])
plt.subplot(1, 3, 2)
plt.hist(pca_features[:, 1])
plt.subplot(1, 3, 3)
plt.hist(pca_features[:, 2])
plt.show()
plt.close()

# # plot the first pca component
# for c_i in range(3):
#     pca_features[:, c_i] = (pca_features[:, c_i] - pca_features[:, c_i].min()) / (pca_features[:, c_i].max() - pca_features[:, c_i].min())
#     for i in range(4):
#         plt.subplot(2, 2, i+1)
#         plt.imshow(pca_features[i * patch_h * patch_w: (i+1) * patch_h * patch_w, c_i].reshape(patch_h, patch_w))
#     plt.show()
#     plt.close()

# for i in range(1):
#     plt.subplot(1, 1, i+1)
#     plt.imshow(pca_features[i * patch_h * patch_w: (i+1) * patch_h * patch_w, 0].reshape(patch_h, patch_w))
# plt.show()
# plt.close()

pca_features_bg = pca_features[:, 0] < 5
pca_features_fg = ~pca_features_bg

# plot the pca_feature_bg
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(pca_features_bg[i * patch_h * patch_w: (i+1) * patch_h * patch_w].reshape(patch_h, patch_w))
plt.show()

pca = PCA(n_components=3)
fg_features = features[pca_features_fg]
num_fg_list = []
for i in range(4):
    num_fg_list.append(pca_features_fg[i * patch_h * patch_w: (i+1) * patch_h * patch_w].sum())
# fg_features = normalize(fg_features, axis=1, norm='max')
# fg_features = fg_features - fg_features.mean(axis=1, keepdims=True)
# features = features / features.std(axis=1, keepdims=True)
# fg_features = fg_features / fg_features.max(axis=1, keepdims=True)
pca.fit(fg_features)
pca_features_rem = pca.transform(fg_features)

# save pca model
import pickle
with open(f'pca_model/{obj_type}.pkl', 'wb') as f:
    pickle.dump(pca, f)

# pca_features_rem = (pca_features_rem - pca_features_rem.min()) / (pca_features_rem.max() - pca_features_rem.min())
for i in range(3):
    # pca_features_rem[:, i] = (pca_features_rem[:, i] - pca_features_rem[:, i].min()) / (pca_features_rem[:, i].max() - pca_features_rem[:, i].min())
    start_idx = 0
    end_idx = num_fg_list[0]
    for j in range(4):
        print('min: ', pca_features_rem[start_idx:end_idx, i].min())
        print('max: ', pca_features_rem[start_idx:end_idx, i].max())
        pca_features_rem[start_idx:end_idx, i] = \
            (pca_features_rem[start_idx:end_idx, i] - pca_features_rem[start_idx:end_idx, i].min()) /\
                (pca_features_rem[start_idx:end_idx, i].max() - pca_features_rem[start_idx:end_idx, i].min())
        start_idx = end_idx
        end_idx += num_fg_list[j+1] if j < 3 else 0
    # transform using mean and std
    # pca_features_rem[:, i] = (pca_features_rem[:, i] - pca_features_rem[:, i].mean()) / (0.5 * pca_features_rem[:, i].std() ** 2) + 0.5

pca_features_rgb = pca_features.copy()
pca_features_rgb[pca_features_bg] = 0
pca_features_rgb[pca_features_fg] = pca_features_rem

pca_features_rgb = pca_features_rgb.reshape(4, patch_h, patch_w, 3)
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(pca_features_rgb[i][..., ::-1])
    # plt.imshow(pca_features_rgb[i][..., 2])
# plt.savefig('features.png')
plt.show()
plt.close()
