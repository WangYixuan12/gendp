# %% [markdown]
# # PointNet

# %% [markdown]
# This is an implementation of [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593) using PyTorch.
# 

# %% [markdown]
# ## Getting started

# %% [markdown]
# Don't forget to turn on GPU if you want to start training directly.
# 
# 
# **Runtime** -> **Change runtime type**-> **Hardware accelerator**
# 
# 

# %%
import numpy as np
import math
import random
import os
import torch
import scipy.spatial.distance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

import open3d as o3d
import cv2

# %%
random.seed = 42


# %%
path = Path("/home/yixuan/bdai/general_dp/general_dp/tests/ModelNet10")

# %%
folders = [dir for dir in sorted(os.listdir(path)) if os.path.isdir(path/dir)]
classes = {folder: i for i, folder in enumerate(folders)};
classes

# %% [markdown]
# This dataset consists of **.off** files that contain meshes represented by *vertices* and *triangular faces*.
# 
# We will need a function to read this type of files:

# %%
def read_off(file):
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces

# %%
with open(path/"bed/train/bed_0001.off", 'r') as f:
  verts, faces = read_off(f)

# %%
i,j,k = np.array(faces).T
x,y,z = np.array(verts).T

# %%
len(x)

# %% [markdown]
# Don't be scared of this function. It's just to display animated rotation of meshes and point clouds.

# %%
def visualize_rotate(data):
    x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
    frames=[]

    def rotate_z(x, y, z, theta):
        w = x+1j*y
        return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z

    for t in np.arange(0, 10.26, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))
    fig = go.Figure(data=data,
                    layout=go.Layout(
                        updatemenus=[dict(type='buttons',
                                    showactive=False,
                                    y=1,
                                    x=0.8,
                                    xanchor='left',
                                    yanchor='bottom',
                                    pad=dict(t=45, r=10),
                                    buttons=[dict(label='Play',
                                                    method='animate',
                                                    args=[None, dict(frame=dict(duration=50, redraw=True),
                                                                    transition=dict(duration=0),
                                                                    fromcurrent=True,
                                                                    mode='immediate'
                                                                    )]
                                                    )
                                            ]
                                    )
                                ]
                    ),
                    frames=frames
            )

    return fig

# %%
# visualize_rotate([go.Mesh3d(x=x, y=y, z=z, color='lightpink', opacity=0.50, i=i,j=j,k=k)]).show()

# %% [markdown]
# This mesh definitely looks like a bed.

# %%
# visualize_rotate([go.Scatter3d(x=x, y=y, z=z,
#                                    mode='markers')]).show()

# %% [markdown]
# Unfortunately, that's not the case for its vertices. It would be difficult for PointNet to classify point clouds like this one.

# %% [markdown]
# First things first, let's write a function to accurately visualize point clouds so we could see vertices better.

# %%
def pcshow(xs,ys,zs):
    data=[go.Scatter3d(x=xs, y=ys, z=zs,
                                   mode='markers')]
    fig = visualize_rotate(data)
    fig.update_traces(marker=dict(size=2,
                      line=dict(width=2,
                      color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.show()


# %%
# pcshow(x,y,z)

# %% [markdown]
# ## Transforms

# %% [markdown]
# As we want it to look more like a real bed, let's write a function to sample points on the surface uniformly.

# %% [markdown]
#  ### Sample points

# %%
class PointSampler(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * ( side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0)**0.5

    def sample_point(self, pt1, pt2, pt3):
        # barycentric coordinates on a triangle
        # https://mathworld.wolfram.com/BarycentricCoordinates.html
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t-s)*pt2[i] + (1-t)*pt3[i]
        return (f(0), f(1), f(2))


    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))

        for i in range(len(areas)):
            areas[i] = (self.triangle_area(verts[faces[i][0]],
                                           verts[faces[i][1]],
                                           verts[faces[i][2]]))

        sampled_faces = (random.choices(faces,
                                      weights=areas,
                                      cum_weights=None,
                                      k=self.output_size))

        sampled_points = np.zeros((self.output_size, 3))

        for i in range(len(sampled_faces)):
            sampled_points[i] = (self.sample_point(verts[sampled_faces[i][0]],
                                                   verts[sampled_faces[i][1]],
                                                   verts[sampled_faces[i][2]]))

        return sampled_points


# %%
pointcloud = PointSampler(3000)((verts, faces))

# %%
# pcshow(*pointcloud.T)

# %% [markdown]
# This pointcloud looks much more like a bed!

# %% [markdown]
# ### Normalize

# %% [markdown]
# Unit sphere

# %%
class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return  norm_pointcloud

# %%
norm_pointcloud = Normalize()(pointcloud)

# %%
# pcshow(*norm_pointcloud.T)

# %% [markdown]
# Notice that axis limits have changed.

# %% [markdown]
# ### Augmentations

# %% [markdown]
# Let's add *random rotation* of the whole pointcloud and random noise to its points.

# %%
class RandRotation_z(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),    0],
                               [ math.sin(theta),  math.cos(theta),    0],
                               [0,                             0,      1]])

        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return  rot_pointcloud

class RandomNoise(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        noise = np.random.normal(0, 0.02, (pointcloud.shape))

        noisy_pointcloud = pointcloud + noise
        return  noisy_pointcloud

# %%
rot_pointcloud = RandRotation_z()(norm_pointcloud)
noisy_rot_pointcloud = RandomNoise()(rot_pointcloud)

# %%
# pcshow(*noisy_rot_pointcloud.T)

# %% [markdown]
# ### ToTensor

# %%
class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        return torch.from_numpy(pointcloud)

# %%
ToTensor()(noisy_rot_pointcloud)

# %%
def default_transforms():
    return transforms.Compose([
                                PointSampler(1024),
                                Normalize(),
                                ToTensor()
                              ])

# %% [markdown]
# ## Dataset

# %% [markdown]
# Now we can create a [custom PyTorch Dataset](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)

# %%
class PointCloudData(Dataset):
    def __init__(self, root_dir, valid=False, folder="train", transform=default_transforms()):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform if not valid else default_transforms()
        self.valid = valid
        self.files = []
        for category in self.classes.keys():
            new_dir = root_dir/Path(category)/folder
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    sample = {}
                    sample['pcd_path'] = new_dir/file
                    sample['category'] = category
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __preproc__(self, file):
        verts, faces = read_off(file)
        if self.transforms:
            pointcloud = self.transforms((verts, faces))
        return pointcloud

    def __getitem__(self, idx):
        pcd_path = self.files[idx]['pcd_path']
        category = self.files[idx]['category']
        with open(pcd_path, 'r') as f:
            pointcloud = self.__preproc__(f)
        return {'pointcloud': pointcloud,
                'category': self.classes[category]}

# %% [markdown]
# Transforms for training. 1024 points per cloud as in the paper!

# %%
train_transforms = transforms.Compose([
                    PointSampler(1024),
                    Normalize(),
                    RandRotation_z(),
                    RandomNoise(),
                    ToTensor()
                    ])

# %%
train_ds = PointCloudData(path, transform=train_transforms)
valid_ds = PointCloudData(path, valid=True, folder='test', transform=train_transforms)

# %%
inv_classes = {i: cat for cat, i in train_ds.classes.items()};
inv_classes

# %%
print('Train dataset size: ', len(train_ds))
print('Valid dataset size: ', len(valid_ds))
print('Number of classes: ', len(train_ds.classes))
print('Sample pointcloud shape: ', train_ds[0]['pointcloud'].size())
print('Class: ', inv_classes[train_ds[0]['category']])

# %%
train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset=valid_ds, batch_size=64)

# %% [markdown]
# ## Model

# %%
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Tnet(nn.Module):
   def __init__(self, k=3):
      super().__init__()
      self.k=k
      self.conv1 = nn.Conv1d(k,64,1)
      self.conv2 = nn.Conv1d(64,128,1)
      self.conv3 = nn.Conv1d(128,1024,1)
      self.fc1 = nn.Linear(1024,512)
      self.fc2 = nn.Linear(512,256)
      self.fc3 = nn.Linear(256,k*k)

      self.bn1 = nn.BatchNorm1d(64)
      self.bn2 = nn.BatchNorm1d(128)
      self.bn3 = nn.BatchNorm1d(1024)
      self.bn4 = nn.BatchNorm1d(512)
      self.bn5 = nn.BatchNorm1d(256)


   def forward(self, input):
      # input.shape == (bs,n,3)
      bs = input.size(0)
      xb = F.relu(self.bn1(self.conv1(input)))
      xb = F.relu(self.bn2(self.conv2(xb)))
      xb = F.relu(self.bn3(self.conv3(xb)))
      pool = nn.MaxPool1d(xb.size(-1))(xb)
      flat = nn.Flatten(1)(pool)
      xb = F.relu(self.bn4(self.fc1(flat)))
      xb = F.relu(self.bn5(self.fc2(xb)))

      #initialize as identity
      init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1)
      if xb.is_cuda:
        init=init.cuda()
      matrix = self.fc3(xb).view(-1,self.k,self.k) + init
      return matrix


class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)
        self.conv1 = nn.Conv1d(3,64,1)

        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)


        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, input):
        matrix3x3 = self.input_transform(input)
        # batch matrix multiplication
        xb = torch.bmm(torch.transpose(input,1,2), matrix3x3).transpose(1,2)

        xb = F.relu(self.bn1(self.conv1(xb)))

        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb,1,2), matrix64x64).transpose(1,2)

        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = self.bn3(self.conv3(xb))
        xb = nn.MaxPool1d(xb.size(-1))(xb)
        output = nn.Flatten(1)(xb)
        return output, matrix3x3, matrix64x64
    
    def forward_wo_pool(self, input):
        matrix3x3 = self.input_transform(input)
        # batch matrix multiplication
        xb = torch.bmm(torch.transpose(input,1,2), matrix3x3).transpose(1,2)

        xb = F.relu(self.bn1(self.conv1(xb)))

        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb,1,2), matrix64x64).transpose(1,2)

        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = self.bn3(self.conv3(xb))
        output = xb
        return output, matrix3x3, matrix64x64

class PointNet(nn.Module):
    def __init__(self, classes = 10):
        super().__init__()
        self.transform = Transform()
        # self.fc1 = nn.Linear(1024, 512)
        # self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, classes)
        self.fc4 = nn.Linear(1024, classes)


        # self.bn1 = nn.BatchNorm1d(512)
        # self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        xb, matrix3x3, matrix64x64 = self.transform(input)
        # xb = F.relu(self.bn1(self.fc1(xb)))
        # xb = F.relu(self.bn2(self.dropout(self.fc2(xb))))
        # output = self.fc3(xb)
        output = self.fc4(xb)
        return self.logsoftmax(output), matrix3x3, matrix64x64

# %%
def pointnetloss(outputs, labels, m3x3, m64x64, alpha = 0.0001):
    criterion = torch.nn.NLLLoss()
    bs=outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs,1,1)
    if outputs.is_cuda:
        id3x3=id3x3.cuda()
        id64x64=id64x64.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    diff64x64 = id64x64-torch.bmm(m64x64,m64x64.transpose(1,2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)+torch.norm(diff64x64)) / float(bs)

# %% [markdown]
# ## Training loop

# %% [markdown]
# You can find a pretrained model [here](https://drive.google.com/open?id=1nDG0maaqoTkRkVsOLtUAR9X3kn__LMSL)

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# %%
pointnet = PointNet()
pointnet.to(device);


# %% [markdown]
# ## Test

# %%
from sklearn.metrics import confusion_matrix

# %%
pointnet = PointNet()
pointnet.load_state_dict(torch.load('/home/yixuan/bdai/general_dp/general_dp/tests/save_13.pth'))
pointnet.eval()

# %%
all_preds = []
all_labels = []
with torch.no_grad():
    for batch_i, data in enumerate(valid_loader):
        print('Batch [%4d / %4d]' % (batch_i+1, len(valid_loader)))

        inputs, labels = data['pointcloud'].float(), data['category']
        outputs, __, __ = pointnet(inputs.transpose(1,2))
        
        _, preds = torch.max(outputs.data, 1)
        all_preds += list(preds.numpy())
        all_labels += list(labels.numpy())
        
        # get perpoint features
        feat_per_pt = pointnet.transform.forward_wo_pool(inputs.transpose(1,2))[0] # B, 1024, N
        last_w = pointnet.fc4.weight.data # 10, 1024
        feat_per_pt_batch_prod = last_w.unsqueeze(0).unsqueeze(-1) * feat_per_pt.unsqueeze(1) # B, 10, 1024, N
        feat_per_pt_batch_prod = feat_per_pt_batch_prod.sum(2) # B, 10, N
        norm_max = feat_per_pt_batch_prod[0].max().item()
        norm_min = feat_per_pt_batch_prod[0].min().item()
        print('pred label:', inv_classes[preds[0].item()])
        print('pred score:', torch.exp(outputs[0])[preds[0]].item())
        print('true label:', inv_classes[labels[0].item()])
        
        # construct visualizer
        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window()
        
        vis_pcd = o3d.geometry.PointCloud()
        
        for i in range(last_w.shape[0]):
            # feat_per_pt_curr_cls = last_w[i:i+1].unsqueeze(-1) * feat_per_pt # B, 1024, N
            # feat_per_pt_curr_cls = feat_per_pt_batch_prod[i]
            # feat_per_pt_curr_cls = feat_per_pt_curr_cls.sum(1) # B, N
            feat_per_pt_curr_cls = feat_per_pt_batch_prod[0, i] # N

            # visualize
            feat_per_pt_curr_cls = feat_per_pt_curr_cls.cpu().numpy() # N
            raw_pcd = inputs[0].cpu().numpy() # N, 3
            from d3fields.utils.draw_utils import np2o3d
            from matplotlib import cm
            cmap = cm.get_cmap('viridis')
            feat_per_pt_curr_cls_norm = (feat_per_pt_curr_cls - norm_min) / (norm_max - norm_min)
            feat_per_pt_curr_cls_color = cmap(feat_per_pt_curr_cls_norm)[:,:3]
            pcd = np2o3d(raw_pcd, feat_per_pt_curr_cls_color)
            print('current class:', inv_classes[i])
            print('current score:', torch.exp(outputs[0][i]).item())
            
            vis_pcd.points = pcd.points
            vis_pcd.colors = pcd.colors
            if i == 0:
                visualizer.add_geometry(vis_pcd)
            
            visualizer.update_geometry(vis_pcd)
            visualizer.poll_events()
            visualizer.update_renderer()
            if i == 0:
                visualizer.run()
            os.system(f"mkdir -p save/batch_{batch_i}")
            visualizer.capture_screen_image(f"save/batch_{batch_i}/{i}.png")
            img = cv2.imread(f"save/batch_{batch_i}/{i}.png")
            # add texts to the right bottom as printed
            font = cv2.FONT_HERSHEY_SIMPLEX
            img_h, img_w, _ = img.shape
            ori_w = img_w - 400
            ori_h = img_h - 200
            img = cv2.putText(img, f"true label: {inv_classes[labels[0].item()]}", (ori_w, ori_h), font, 1.0, (255, 0, 0), 3)
            img = cv2.putText(img, f"curr label: {inv_classes[i]}", (ori_w, ori_h + 50), font, 1.0, (255, 0, 0), 3)
            img = cv2.putText(img, f"curr score: {torch.exp(outputs[0][i]).item():.3f}", (ori_w, ori_h + 100), font, 1.0, (255, 0, 0), 3)
            cv2.imwrite(f"save/batch_{batch_i}/{i}.png", img)
            
        visualizer.destroy_window()
        # make a video
        os.system(f"ffmpeg -framerate 1 -i save/batch_{batch_i}/%d.png save/batch_{batch_i}.mp4")
        print()


# %%
cm = confusion_matrix(all_labels, all_preds)
cm

# %%
import itertools
import numpy as np
import matplotlib.pyplot as plt

# function from https://deeplizard.com/learn/video/0LhiS6yu2qQ
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# %%
plt.figure(figsize=(8,8))
plot_confusion_matrix(cm, list(classes.keys()), normalize=True)

# %%
plt.figure(figsize=(8,8))
plot_confusion_matrix(cm, list(classes.keys()), normalize=False)


