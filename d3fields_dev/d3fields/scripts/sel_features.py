import torch
import torch.nn.functional as F
import torchvision.transforms as T
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from PIL import Image

import sys
sys.path.append(os.getcwd())

from d3fields.utils.draw_utils import draw_keypoints

patch_h = 40
patch_w = 40
# feat_dim = 384 # vits14
# feat_dim = 768 # vitb14
feat_dim = 1024 # vitl14
# feat_dim = 1536 # vitg14

root_dir = '/home/yixuan/bdai/general_dp/d3fields_dev/d3fields'
obj_type = 'spoon'
os.system(f'mkdir -p {root_dir}/sel_feats/{obj_type}')

device = 'cuda'

img_paths = [f'{root_dir}/data/wild/{obj_type}/0.jpg',
             f'{root_dir}/data/wild/{obj_type}/1.jpg',
             f'{root_dir}/data/wild/{obj_type}/2.jpg',
             f'{root_dir}/data/wild/{obj_type}/3.jpg',]
# img_paths = [f'{root_dir}/data/sapien/{obj_type}/0.png',
#              f'{root_dir}/data/sapien/{obj_type}/1.png',
#              f'{root_dir}/data/sapien/{obj_type}/2.png',
#              f'{root_dir}/data/sapien/{obj_type}/3.png',]

transform = T.Compose([
    T.Resize((patch_h * 14, patch_w * 14)),
    T.CenterCrop((patch_h * 14, patch_w * 14)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
# model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)
# model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').to(device)

features = torch.zeros(4, patch_h * patch_w, feat_dim).to(device)
imgs = np.zeros((4, patch_h * 14, patch_w * 14, 3))
imgs_tensor = torch.zeros(4, 3, patch_h * 14, patch_w * 14).to(device)
for i in range(4):
    img_path = img_paths[i]
    img = cv2.imread(img_path)
    img = cv2.resize(img, (14 * patch_w, 14 * patch_h))
    imgs[i] = img
    # img = Image.open(img_path).convert('RGB')
    img = Image.fromarray(img)
    imgs_tensor[i] = transform(img)[:3]
with torch.no_grad():
    features_dict = model.forward_features(imgs_tensor)
    features = features_dict['x_norm_patchtokens']
    features = features.reshape(4, patch_h, patch_w, feat_dim)
imgs = imgs.astype(np.uint8)

# # quick PCA visualization
# from sklearn.decomposition import PCA
# pca = PCA(n_components=3)
# pca_features = pca.fit_transform(features.detach().cpu().numpy().reshape(-1, feat_dim)).reshape(4, patch_h, patch_w, 3)
# for i in range(4):
#     pca_feat = pca_features[i]
#     pca_feat = (pca_feat - pca_feat.min()) / (pca_feat.max() - pca_feat.min())
#     pca_feat = (pca_feat * 255).astype(np.uint8)
#     cv2.imshow(f'pca_feat{i}', pca_feat)

sel_features = []

def sel_feats(event, x, y, flags, param):
    curr_img = param['curr_img']
    idx = param['idx']
    sel_x = param['sel_x']
    sel_y = param['sel_y']
    if sel_x >= 0 and sel_y >= 0:
        curr_img = draw_keypoints(curr_img, np.array([[sel_x, sel_y]]), colors=[(255, 0, 0)], radius=15)
    if event == cv2.EVENT_LBUTTONDOWN:
        x_idx = x // 14
        y_idx = y // 14
        sel_feature = features[idx, y_idx, x_idx]
        param['curr_feats'][idx] = sel_feature.detach().cpu().numpy()
        param['sel_x'] = x
        param['sel_y'] = y
        print(f'feature {len(sel_features)} selected')
    elif event == cv2.EVENT_MOUSEMOVE:
        curr_img = draw_keypoints(curr_img, np.array([[x, y]]), colors=[(0, 255, 0)], radius=15)
        curr_feature = features[idx, y // 14, x // 14]
        sim_map = np.zeros((4, patch_h, patch_w))
        for i in range(4):
            sim_map[i] = F.cosine_similarity(curr_feature[None], features[i].reshape(-1, feat_dim), dim=-1).detach().cpu().numpy().reshape(patch_h, patch_w)
        sim_map_vis = np.zeros((4, patch_h, patch_w, 3))
        cmap = cm.get_cmap('viridis')
        sim_map_vis = cmap(sim_map)[:, :, :, :3]
        sim_map_vis = (sim_map_vis * 255).astype(np.uint8)[..., ::-1]
        for i in range(4):
            cv2.imshow(f'sim_map{i}', sim_map_vis[i])
    cv2.imshow(f'img', curr_img)

while True:
    param = {'curr_feats': [None] * 4,
             'curr_img': None,
             'idx': -1,
             'sel_x': -1,
             'sel_y': -1}
    end = False
    curr_feat_imgs = []
    for i in range(4):
        img = imgs[i]
        param['curr_img'] = img
        param['idx'] = i
        param['sel_x'] = -1
        param['sel_y'] = -1
        cv2.imshow(f'img', img)
        if i == 0:
            cv2.setMouseCallback(f'img', sel_feats, param)
        
        key = cv2.waitKey(0)
        curr_img_kypts = draw_keypoints(img, np.array([[param['sel_x'], param['sel_y']]]), colors=[(255, 0, 0)], radius=15)
        curr_feat_imgs.append(curr_img_kypts)
        if key == ord('q'):
            end = True
            break
        elif key == ord('n'):
            continue
    aggr_feat_img = np.zeros((patch_h * 2 * 14, patch_w * 2 * 14, 3))
    for i in range(2):
        for j in range(2):
            aggr_feat_img[i * patch_h * 14:(i + 1) * patch_h * 14, j * patch_w * 14:(j + 1) * patch_w * 14] = curr_feat_imgs[i * 2 + j]
    cv2.imwrite(f'{root_dir}/sel_feats/{obj_type}/{len(sel_features)}.png', aggr_feat_img)
    
    curr_feat_avg = np.stack(param['curr_feats']).mean(axis=0)
    sel_features.append(curr_feat_avg)
    
    if end:
        break

sel_features = np.stack(sel_features) # (n, feat_dim)
np.save(f'{root_dir}/sel_feats/{obj_type}.npy', sel_features)
