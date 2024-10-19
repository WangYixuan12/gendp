import time
import os
import copy
from enum import Enum
from typing import Dict, List, Optional, Union

from d3fields.utils.my_utils import depth2fgpcd, np2o3d, voxel_downsample
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import colormaps
import open3d as o3d

from pytorch3d.ops import ball_query

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go


class ImgEncoding(Enum):
    RGB_UINT8 = "rgb_uint8"
    BGR_UINT8 = "bgr_uint8"
    DEPTH_UINT16 = "depth_uint16"
    DEPTH_FLOAT = "depth_float"


class ExtriConvention(Enum):
    CAM_IN_WORLD = "cam_in_world"  # camera pose in world coord
    WORLD_IN_CAM = "world_in_cam"  # world pose in camera coord


def newline(p1, p2):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if p2[0] == p1[0]:
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    l = mlines.Line2D([xmin,xmax], [ymin,ymax])
    ax.add_line(l)
    return l

def draw_correspondence(img0, img1, kps0, kps1, matches=None, colors=None, max_draw_line_num=None, kps_color=(0,0,255),vert=False):
    if len(img0.shape)==2:
        img0=np.repeat(img0[:,:,None],3,2)
    if len(img1.shape)==2:
        img1=np.repeat(img1[:,:,None],3,2)

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    if matches is None:
        assert(kps0.shape[0]==kps1.shape[0])
        matches=np.repeat(np.arange(kps0.shape[0])[:,None],2,1)

    if vert:
        w = max(w0, w1)
        h = h0 + h1
        out_img = np.zeros([h, w, 3], np.uint8)
        out_img[:h0, :w0] = img0
        out_img[h0:, :w1] = img1
    else:
        h = max(h0, h1)
        w = w0 + w1
        out_img = np.zeros([h, w, 3], np.uint8)
        out_img[:h0, :w0] = img0
        out_img[:h1, w0:] = img1

    for pt in kps0:
        pt = np.round(pt).astype(np.int32)
        cv2.circle(out_img, tuple(pt), 1, kps_color, -1)

    for pt in kps1:
        pt = np.round(pt).astype(np.int32)
        pt = pt.copy()
        if vert:
            pt[1] += h0
        else:
            pt[0] += w0
        cv2.circle(out_img, tuple(pt), 1, kps_color, -1)

    if max_draw_line_num is not None and matches.shape[0]>max_draw_line_num:
        np.random.seed(6033)
        idxs=np.arange(matches.shape[0])
        np.random.shuffle(idxs)
        idxs=idxs[:max_draw_line_num]
        matches= matches[idxs]

        if colors is not None and (type(colors)==list or type(colors)==np.ndarray):
            colors=np.asarray(colors)
            colors= colors[idxs]

    for mi,m in enumerate(matches):
        pt = np.round(kps0[m[0]]).astype(np.int32)
        pr_pt = np.round(kps1[m[1]]).astype(np.int32)
        if vert:
            pr_pt[1] += h0
        else:
            pr_pt[0] += w0
        if colors is None:
            cv2.line(out_img, tuple(pt), tuple(pr_pt), (0, 255, 0), 1)
        elif type(colors)==list or type(colors)==np.ndarray:
            color=(int(c) for c in colors[mi])
            cv2.line(out_img, tuple(pt), tuple(pr_pt), tuple(color), 1)
        else:
            color=(int(c) for c in colors)
            cv2.line(out_img, tuple(pt), tuple(pr_pt), tuple(color), 1)

    return out_img

def draw_keypoints(img,kps,colors=None,radius=2,thickness=-1):
    out_img=img.copy()
    for pi, pt in enumerate(kps):
        pt = np.round(pt).astype(np.int32)
        if colors is not None:
            color=[int(c) for c in colors[pi]]
            cv2.circle(out_img, tuple(pt), radius, color, thickness, cv2.FILLED)
        else:
            cv2.circle(out_img, tuple(pt), radius, (0,255,0), thickness)
    return out_img

def draw_epipolar_line(F, img0, img1, pt0, color):
    h1,w1=img1.shape[:2]
    hpt = np.asarray([pt0[0], pt0[1], 1], dtype=np.float32)[:, None]
    l = F @ hpt
    l = l[:, 0]
    a, b, c = l[0], l[1], l[2]
    pt1 = np.asarray([0, -c / b]).astype(np.int32)
    pt2 = np.asarray([w1, (-a * w1 - c) / b]).astype(np.int32)

    img0 = cv2.circle(img0, tuple(pt0.astype(np.int32)), 5, color, 2)
    img1 = cv2.line(img1, tuple(pt1), tuple(pt2), color, 2)
    return img0, img1

def draw_epipolar_lines(F, img0, img1,num=20):
    img0,img1=img0.copy(),img1.copy()
    h0, w0, _ = img0.shape
    h1, w1, _ = img1.shape

    for k in range(num):
        color = np.random.randint(0, 255, [3], dtype=np.int32)
        color = [int(c) for c in color]
        pt = np.random.uniform(0, 1, 2)
        pt[0] *= w0
        pt[1] *= h0
        pt = pt.astype(np.int32)
        img0, img1 = draw_epipolar_line(F, img0, img1, pt, color)

    return img0, img1

def scale_float_image(image):
    max_val, min_val = np.max(image), np.min(image)
    image = (image - min_val) / (max_val - min_val) * 255
    return image.astype(np.uint8)

def concat_images(img0,img1,vert=False):
    if not vert:
        h0,h1=img0.shape[0],img1.shape[0],
        if h0<h1: img0=cv2.copyMakeBorder(img0,0,h1-h0,0,0,borderType=cv2.BORDER_CONSTANT,value=0)
        if h1<h0: img1=cv2.copyMakeBorder(img1,0,h0-h1,0,0,borderType=cv2.BORDER_CONSTANT,value=0)
        img = np.concatenate([img0, img1], axis=1)
    else:
        w0,w1=img0.shape[1],img1.shape[1]
        if w0<w1: img0=cv2.copyMakeBorder(img0,0,0,0,w1-w0,borderType=cv2.BORDER_CONSTANT,value=0)
        if w1<w0: img1=cv2.copyMakeBorder(img1,0,0,0,w0-w1,borderType=cv2.BORDER_CONSTANT,value=0)
        img = np.concatenate([img0, img1], axis=0)

    return img


def concat_images_list(*args,vert=False):
    if len(args)==1: return args[0]
    img_out=args[0]
    for img in args[1:]:
        img_out=concat_images(img_out,img,vert)
    return img_out


def get_colors_gt_pr(gt,pr=None):
    if pr is None:
        pr=np.ones_like(gt)
    colors=np.zeros([gt.shape[0],3],np.uint8)
    colors[gt & pr]=np.asarray([0,255,0])[None,:]     # tp
    colors[ (~gt) & pr]=np.asarray([255,0,0])[None,:] # fp
    colors[ gt & (~pr)]=np.asarray([0,0,255])[None,:] # fn
    return colors


def draw_hist(fn,vals,bins=100,hist_range=None,names=None):
    if type(vals)==list:
        val_num=len(vals)
        if hist_range is None:
            hist_range = (np.min(vals),np.max(vals))
        if names is None:
            names=[str(k) for k in range(val_num)]
        for k in range(val_num):
            plt.hist(vals[k], bins=bins, range=hist_range, alpha=0.5, label=names[k])
        plt.legend()
    else:
        if hist_range is None:
            hist_range = (np.min(vals),np.max(vals))
        plt.hist(vals,bins=bins,range=hist_range)

    plt.savefig(fn)
    plt.close()

def draw_pr_curve(fn,gt_sort):
    pos_num_all=np.sum(gt_sort)
    pos_nums=np.cumsum(gt_sort)
    sample_nums=np.arange(gt_sort.shape[0])+1
    precisions=pos_nums.astype(np.float64)/sample_nums
    recalls=pos_nums/pos_num_all

    precisions=precisions[np.arange(0,gt_sort.shape[0],gt_sort.shape[0]//40)]
    recalls=recalls[np.arange(0,gt_sort.shape[0],gt_sort.shape[0]//40)]
    plt.plot(recalls,precisions,'r-')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.savefig(fn)
    plt.close()

def draw_features_distribution(fn,feats,colors,ds_type='pca'):
    n,d=feats.shape
    if d>2:
        if ds_type=='pca':
            pca=PCA(2)
            feats=pca.fit_transform(feats)
        elif ds_type=='tsne':
            tsne=TSNE(2)
            feats=tsne.fit_transform(feats)
        elif ds_type=='pca-tsne':
            if d>50:
                tsne=PCA(50)
                feats=tsne.fit_transform(feats)
            tsne=TSNE(2,100.0)
            feats=tsne.fit_transform(feats)
        else:
            raise NotImplementedError

    colors=[np.array([c[0],c[1],c[2]],np.float64)/255.0 for c in colors]
    feats_min=np.min(feats,0,keepdims=True)
    feats_max=np.max(feats,0,keepdims=True)
    feats=(feats-(feats_min+feats_max)/2)*10/(feats_max-feats_min)
    plt.scatter(feats[:,0],feats[:,1],s=0.5,c=colors)
    plt.savefig(fn)
    plt.close()
    return feats

def draw_points(img,points):
    pts=np.round(points).astype(np.int32)
    h,w,_=img.shape
    pts[:,0]=np.clip(pts[:,0],a_min=0,a_max=w-1)
    pts[:,1]=np.clip(pts[:,1],a_min=0,a_max=h-1)
    img=img.copy()
    img[pts[:,1],pts[:,0]]=255
    # img[pts[:,1],pts[:,0]]+=np.asarray([127,0,0],np.uint8)[None,:]
    return img

def draw_bbox(img,bbox,color=None):
    if color is not None:
        color=[int(c) for c in color]
    else:
        color=(0,255,0)
    img=cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),color)
    return img

def output_points(fn,pts,colors=None):
    with open(fn, 'w') as f:
        for pi, pt in enumerate(pts):
            f.write(f'{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} ')
            if colors is not None:
                f.write(f'{int(colors[pi,0])} {int(colors[pi,1])} {int(colors[pi,2])}')
            f.write('\n')

def compute_axis_points(pose):
    R=pose[:,:3] # 3,3
    t=pose[:,3:] # 3,1
    pts = np.concatenate([np.identity(3),np.zeros([3,1])],1) # 3,4
    pts = R.T @ (pts - t)
    colors = np.asarray([[255,0,0],[0,255,0,],[0,0,255],[0,0,0]],np.uint8)
    return pts.T, colors

def aggr_point_cloud(database, sel_time=None, downsample=True):
    img_ids = database.get_img_ids()
    start = 0
    end = len(img_ids)
    # end = 2
    step = 1
    # visualize scaled COLMAP poses
    COLMAP_geometries = []
    base_COLMAP_pose = database.get_base_pose()
    base_COLMAP_pose = np.concatenate([base_COLMAP_pose, np.array([[0, 0, 0, 1]])], axis=0)
    for i in range(start, end, step):
        id = img_ids[i]
        depth = database.get_depth(id, sel_time=sel_time)
        color = database.get_image(id, sel_time=sel_time) / 255.
        K = database.get_K(id)
        cam_param = [K[0,0], K[1,1], K[0,2], K[1,2]] # fx, fy, cx, cy
        mask = database.get_mask(id, sel_time=sel_time) & (depth > 0)
        pcd = depth2fgpcd(depth, mask, cam_param)
        pose = database.get_pose(id)
        pose = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)
        pose = base_COLMAP_pose @ np.linalg.inv(pose)
        # print('COLMAP pose:')
        # print(pose)
        # pose[:3, 3] = pose[:3, 3] / 82.5
        trans_pcd = pose @ np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0)
        trans_pcd = trans_pcd[:3, :].T
        pcd_o3d = np2o3d(trans_pcd, color[mask])
        radius = 0.01
        # downsample
        if downsample:
            pcd_o3d = pcd_o3d.voxel_down_sample(radius)
        COLMAP_geometries.append(pcd_o3d)
    aggr_geometry = o3d.geometry.PointCloud()
    for pcd in COLMAP_geometries:
        aggr_geometry += pcd
    return aggr_geometry

def color_seg(colors, color_name):
    """color segmentation

    Args:
        colors (np.ndaaray): (N, H, W, 3) RGB images in float32 ranging from 0 to 1
        color_name (string): color name of supported colors
    """
    color_name_to_seg_dict = {
        'yellow': {
            'h_min': 0.0,
            'h_max': 0.1,
            's_min': 0.2,
            's_max': 0.55,
            's_cond': 'and',
            'v_min': 0.8,
            'v_max': 1.01,
            'v_cond': 'or',
        }
    }
    
    colors_hsv = [cv2.cvtColor((color * 255).astype(np.uint8), cv2.COLOR_RGB2HSV) for color in colors]
    colors_hsv = np.stack(colors_hsv, axis=0)
    colors_hsv = colors_hsv / 255.
    seg_dict = color_name_to_seg_dict[color_name]
    h_min = seg_dict['h_min']
    h_max = seg_dict['h_max']
    s_min = seg_dict['s_min']
    s_max = seg_dict['s_max']
    s_cond = seg_dict['s_cond']
    v_min = seg_dict['v_min']
    v_max = seg_dict['v_max']
    v_cond = seg_dict['v_cond']
    mask = (colors_hsv[:, :, :, 0] > h_min) & (colors_hsv[:, :, :, 0] < h_max)
    if s_cond == 'and':
        mask = mask & (colors_hsv[:, :, :, 1] > s_min) & (colors_hsv[:, :, :, 1] < s_max)
    else:
        mask = mask | ((colors_hsv[:, :, :, 1] > s_min) & (colors_hsv[:, :, :, 1] < s_max))
    if v_cond == 'and':
        mask = mask & (colors_hsv[:, :, :, 2] > v_min) & (colors_hsv[:, :, :, 2] < v_max)
    else:
        mask = mask | ((colors_hsv[:, :, :, 2] > v_min) & (colors_hsv[:, :, :, 2] < v_max))
    return mask

def vis_depth(raw_depth: np.ndarray, min: float = 0.1, max: float = 1.0, fmt: str = "uint16") -> np.ndarray:
    """convert raw depth to something that can be visualized by opencv or matplotlib

    Args:
        raw_depth: (numpy.ndarray) depth in shape of (h, w)
        min: mininum depth distance for vis in meter
        max: maximum depth distance for vis in meter
        fmt: 'uint16' or 'float'

    Return:
        depth_img: (numpy.ndarray) depth for vis in shape of (h, w, 3)
    """
    cmap = colormaps.get_cmap("plasma")
    H, W = raw_depth.shape
    if fmt == "uint16":
        raw_depth = raw_depth / 1000.0
    elif fmt == "float":
        pass
    else:
        raise ValueError(f"not supported format {fmt}")

    depth_img = np.clip(raw_depth, min, max)
    depth_img = cmap(depth_img.reshape(H * W))[:, :3]
    depth_img = depth_img.reshape(H, W, 3)
    return depth_img

def aggr_point_cloud_from_data(
    colors: np.ndarray,
    depths: np.ndarray,
    Ks: np.ndarray,
    poses: np.ndarray,
    downsample: bool = True,
    downsample_r: float = 0.01,
    masks: np.ndarray = None,
    boundaries: Optional[Dict] = None,
    out_o3d: bool = True,
    excluded_pts: Union[np.ndarray,None] = None,
    exclude_threshold: float = 0.01,
    exclude_colors: List = [],
    color_fmt: ImgEncoding = ImgEncoding.RGB_UINT8,
    depth_fmt: ImgEncoding = ImgEncoding.DEPTH_FLOAT,
    pose_fmt: ExtriConvention = ExtriConvention.WORLD_IN_CAM,
) -> Optional[np.ndarray]:
    # colors: [N, H, W, 3] numpy array in uint8
    # depths: [N, H, W] numpy array in meters
    # Ks: [N, 3, 3] numpy array
    # poses: [N, 4, 4] numpy array
    # masks: [N, H, W] numpy array in bool
    N, H, W, _ = colors.shape
    if color_fmt == ImgEncoding.RGB_UINT8:
        colors = colors / 255.0
    elif color_fmt == ImgEncoding.BGR_UINT8:
        colors = colors[..., ::-1] / 255.0
    if depth_fmt == ImgEncoding.DEPTH_UINT16:
        depths = depths / 1000.0
    start = 0
    end = N
    step = 1
    pcds_ls = []
    pcd_colors = []
    # TODO: batch it
    for i in range(start, end, step):
        depth = depths[i]
        color = colors[i]
        K = Ks[i]
        cam_param = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]  # fx, fy, cx, cy
        if masks is None:
            mask = depth > 0
        else:
            mask = masks[i] & (depth > 0)
        
        for exclude_color in exclude_colors:
            mask = mask & (~color_seg(color[None], exclude_color))[0]

        pcd = depth2fgpcd(depth, mask, cam_param)

        pose = poses[i]
        if pose_fmt == ExtriConvention.WORLD_IN_CAM:
            try:
                pose = np.linalg.inv(pose)
            except np.linalg.LinAlgError:
                print("singular matrix")
                pose = np.linalg.pinv(pose)

        trans_pcd = pose @ np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0)
        trans_pcd = trans_pcd[:3, :].T

        # plt.subplot(1, 4, 1)
        # plt.imshow(trans_pcd[:, 0].reshape(H, W))
        # plt.subplot(1, 4, 2)
        # plt.imshow(trans_pcd[:, 1].reshape(H, W))
        # plt.subplot(1, 4, 3)
        # plt.imshow(trans_pcd[:, 2].reshape(H, W))
        # plt.subplot(1, 4, 4)
        # plt.imshow(color)
        # plt.show()

        color = color[mask]

        pcds_ls.append(trans_pcd)
        pcd_colors.append(color)

    pcds = np.concatenate(pcds_ls, axis=0)
    pcd_colors = np.concatenate(pcd_colors, axis=0)

    # post process 1: remove points outside of boundaries
    if boundaries is not None:
        x_lower = boundaries["x_lower"]
        x_upper = boundaries["x_upper"]
        y_lower = boundaries["y_lower"]
        y_upper = boundaries["y_upper"]
        z_lower = boundaries["z_lower"]
        z_upper = boundaries["z_upper"]

        pcd_mask = (
            (pcds[:, 0] > x_lower)
            & (pcds[:, 0] < x_upper)
            & (pcds[:, 1] > y_lower)
            & (pcds[:, 1] < y_upper)
            & (pcds[:, 2] > z_lower)
            & (pcds[:, 2] < z_upper)
        )

        pcds = pcds[pcd_mask]
        pcd_colors = pcd_colors[pcd_mask]

    # post process 2: downsample
    if downsample:
        pcds, pcd_colors = voxel_downsample(pcds, downsample_r, pcd_colors)

    # post process 3: exclude points from trans_pcd that are too close to excluded_pts
    if excluded_pts is not None:
        pcd_torch = torch.from_numpy(pcds).to(device=torch.device('cuda'), dtype=torch.float32) # [N, 3]
        excluded_pts_torch = torch.from_numpy(excluded_pts)
        excluded_pts_torch = excluded_pts_torch.to(device=torch.device('cuda'), dtype=torch.float32) # [M, 3]

        dists, idx, _ = ball_query(pcd_torch[None],\
                                   excluded_pts_torch[None],\
                                    radius=exclude_threshold,\
                                        K=1,\
                                            return_nn=False)
        idx = idx[0, :, 0] # [N]
        min_dist_to_ex_mask = (idx == -1) # [N], True if no point is within exclude_threshold
        min_dist_to_ex_mask = min_dist_to_ex_mask.cpu().numpy()
        pcds = pcds[min_dist_to_ex_mask]
        pcd_colors = pcd_colors[min_dist_to_ex_mask]

    # post process 4: return o3d point cloud if out_o3d is True
    if out_o3d:
        aggr_pcd = np2o3d(pcds, pcd_colors)
        return aggr_pcd
    else:
        return pcds, pcd_colors

def draw_pcd_with_spheres(pcd, sphere_np_ls):
    # :param pcd: o3d.geometry.PointCloud
    # :param sphere_np_ls: list of [N, 3] numpy array
    sphere_list = []
    color_list = [
        [1.,0.,0.],
        [0.,1.,0.],
    ]
    for np_idx, sphere_np in enumerate(sphere_np_ls):
        for src_pt in sphere_np:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere.compute_vertex_normals()
            sphere.paint_uniform_color(color_list[np_idx])
            sphere.translate(src_pt)
            sphere_list.append(sphere)
    o3d.visualization.draw_geometries([pcd] + sphere_list)


class o3dVisualizer:
    def __init__(self, view_ctrl_info: Optional[Dict] = None, save_path: Optional[str] = None) -> None:
        """initialize o3d visualizer

        Args:
            view_ctrl_info (dict): view control info containing front, lookat, up, zoom
            save_path (_type_, optional): _description_. Defaults to None.
        """
        self.view_ctrl_info = view_ctrl_info
        self.save_path = save_path
        self.visualizer = o3d.visualization.Visualizer()
        self.vis_dict: Dict[str, o3d.geometry.PointCloud] = {}
        self.mesh_vertices: Dict[str, np.ndarray] = {}
        self.is_first = True
        self.internal_clock = 0
        if save_path is not None:
            os.system(f"mkdir -p {save_path}")

    def start(self) -> None:
        # self.visualizer = o3d.visualization.Visualizer()
        self.visualizer.create_window()

    def update_pcd(self, mesh: o3d.geometry.PointCloud, mesh_name: str) -> None:
        if mesh_name not in self.vis_dict.keys():
            self.vis_dict[mesh_name] = o3d.geometry.PointCloud()
            self.vis_dict[mesh_name].points = mesh.points
            self.vis_dict[mesh_name].colors = mesh.colors
            self.visualizer.add_geometry(self.vis_dict[mesh_name])
        else:
            self.vis_dict[mesh_name].points = mesh.points
            self.vis_dict[mesh_name].colors = mesh.colors

    def add_triangle_mesh(
        self,
        type: str,
        mesh_name: str,
        color: Optional[np.ndarray] = None,
        radius: float = 0.1,
        width: float = 0.1,
        height: float = 0.1,
        depth: float = 0.1,
        size: float = 0.1,
    ) -> None:
        if type == "sphere":
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        elif type == "box":
            mesh = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth)
        elif type == "origin":
            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        else:
            raise NotImplementedError
        if color is not None:
            mesh.paint_uniform_color(color)
        self.vis_dict[mesh_name] = mesh
        self.mesh_vertices[mesh_name] = np.array(mesh.vertices).copy()
        self.visualizer.add_geometry(self.vis_dict[mesh_name])

    def update_triangle_mesh(self, mesh_name: str, tf: np.ndarray) -> None:
        tf_vertices = self.mesh_vertices[mesh_name] @ tf[:3, :3].T + tf[:3, 3]
        self.vis_dict[mesh_name].vertices = o3d.utility.Vector3dVector(tf_vertices)

    def update_custom_mesh(self, mesh: o3d.geometry.TriangleMesh, mesh_name: str) -> None:
        if mesh_name not in self.vis_dict.keys():
            self.vis_dict[mesh_name] = copy.deepcopy(mesh)
            self.visualizer.add_geometry(self.vis_dict[mesh_name])
        else:
            self.vis_dict[mesh_name].vertices = mesh.vertices
        self.visualizer.update_geometry(self.vis_dict[mesh_name])

    def render(
        self,
        render_names: Optional[str] = None,
        save_name: Optional[str] = None,
        curr_view_ctrl_info: Optional[Dict] = None,
    ) -> np.ndarray:
        # if self.is_first:
        #     self.is_first = False
        if self.view_ctrl_info is not None and curr_view_ctrl_info is None:
            view_control = self.visualizer.get_view_control()
            view_control.set_front(self.view_ctrl_info["front"])
            view_control.set_lookat(self.view_ctrl_info["lookat"])
            view_control.set_up(self.view_ctrl_info["up"])
            view_control.set_zoom(self.view_ctrl_info["zoom"])
        elif curr_view_ctrl_info is not None:
            view_control = self.visualizer.get_view_control()
            view_control.set_front(curr_view_ctrl_info["front"])
            view_control.set_lookat(curr_view_ctrl_info["lookat"])
            view_control.set_up(curr_view_ctrl_info["up"])
            view_control.set_zoom(curr_view_ctrl_info["zoom"])
        if render_names is None:
            for mesh_name in self.vis_dict.keys():
                self.visualizer.update_geometry(self.vis_dict[mesh_name])
        else:
            for mesh_name in self.vis_dict.keys():
                if mesh_name in render_names:
                    self.visualizer.update_geometry(self.vis_dict[mesh_name])
                else:
                    self.visualizer.remove_geometry(self.vis_dict[mesh_name], False)
        self.visualizer.poll_events()
        self.visualizer.update_renderer()
        self.visualizer.run()

        img = None
        if self.save_path is not None:
            if save_name is None:
                save_fn = f"{self.save_path}/{self.internal_clock}.png"
            else:
                save_fn = f"{self.save_path}/{save_name}.png"
            self.visualizer.capture_screen_image(save_fn)
            img = cv2.imread(save_fn)
            self.internal_clock += 1

        # add back
        if render_names is not None:
            for mesh_name in self.vis_dict.keys():
                if mesh_name not in render_names:
                    self.visualizer.add_geometry(self.vis_dict[mesh_name])

        return img

def poseInterp(pose0, pose1, steps):
    """pose interpolation

    Args:
        pose0 (np.ndarray): 4x4 pose matrix
        pose1 (np.ndarray): 4x4 pose matrix
        steps (int): number of steps

    Returns:
        np.ndarray: (steps, 4, 4) pose matrices
    """
    from scipy.spatial.transform import Slerp
    from scipy.spatial.transform import Rotation as R
    interp_poses = []
    rots = R.from_matrix(np.concatenate([pose0[:3, :3], pose1[:3, :3]]))
    slerp = Slerp([0, 1], rots)
    for i in range(steps):
        coeff = i / (steps - 1)
        pos = pose0[:3, 3] * (1 - coeff) + pose1[:3, 3] * coeff
        rot = slerp([coeff]).as_matrix()
        interp_pose = np.eye(4)
        interp_pose[:3, :3] = rot
        interp_pose[:3, 3] = pos
        interp_poses.append(interp_pose)
    return np.stack(interp_poses, axis=0)

def viewCtrlInterp(view_ctrl_info0, view_ctrl_info1, steps, interp_type='linear'):
    """view control interpolation

    Args:
        view_ctrl_info0 (dict): view control info
        view_ctrl_info1 (dict): view control info
        steps (int): number of steps

    Returns:
        list: view control info list
    """
    interp_view_ctrl_info = []
    for i in range(steps):
        if interp_type == 'linear':
            coeff = i / (steps - 1)
        elif interp_type == 'bilinear':
            coeff = 1 - abs(i / (steps - 1) * 2 - 1)
        elif interp_type == 'sine':
            coeff = 1 - (np.cos(i / (steps - 1) * 2 * np.pi) + 1) / 2
        interp_view_ctrl_info.append({
            'front': (np.array(view_ctrl_info0['front']) * (1 - coeff) + np.array(view_ctrl_info1['front']) * coeff).tolist(),
            'lookat': (np.array(view_ctrl_info0['lookat']) * (1 - coeff) + np.array(view_ctrl_info1['lookat']) * coeff).tolist(),
            'up': (np.array(view_ctrl_info0['up']) * (1 - coeff) + np.array(view_ctrl_info1['up']) * coeff).tolist(),
            'zoom': view_ctrl_info0['zoom'] * (1 - coeff) + view_ctrl_info1['zoom'] * coeff
        })
    return interp_view_ctrl_info

def gen_circular_view_ctrl_info(lookat, cam_h, cam_r, zoom, steps):
    """Generate circular view control info

    Args:
        lookat (list of 3): lookat position
        start (list of 3): start position
        zoom (float): zoom
        steps (int): number of steps
    """
    view_ctrl_info_ls = []
    
    # cam_pos_ls = [np.array([cam_r * np.cos(theta), cam_r * np.sin(theta), cam_h]) for theta in np.linspace(0, 2 * np.pi, steps)]
    # right_ls = [np.array([np.sin(theta), -np.cos(theta), 0]) for theta in np.linspace(0, 2 * np.pi, steps)]

    # front_ls = [lookat - cam_pos for cam_pos in cam_pos_ls]
    # up_ls = [np.cross(right, front) for right, front in zip(right_ls, front_ls)]
    
    for i in range(steps):
        theta = i / steps * 2 * np.pi
        cam_pos = np.array([cam_r * np.cos(theta), cam_r * np.sin(theta), cam_h]) + np.array(lookat)
        right = np.array([np.sin(theta), -np.cos(theta), 0])
        front = cam_pos - lookat
        up = np.cross(right, front)
        view_ctrl_info_ls.append({
            'front': front.tolist(),
            'lookat': lookat,
            'up': up.tolist(),
            'zoom': zoom
        })
    return view_ctrl_info_ls

def test_o3d_vis():
    view_ctrl_info = {
        'front': [ 0.36137433126422974, 0.5811161319788094, 0.72918628200022917 ],
        'lookat': [ 0.45000000000000001, 0.45000000000000001, 0.45000000000000001 ],
        'up': [ -0.17552920841503886, 0.81045157347999874, -0.55888974229000143 ],
        'zoom': 1.3400000000000005
    }
    view_ctrl_info_0 = {
        'front': [ 0.5, 0.3, 0.4 ],
        'lookat': [ 1.45000000000000001, 0.45000000000000001, 0.45000000000000001 ],
        'up': [ -0.17552920841503886, 0.81045157347999874, -0.55888974229000143 ],
        'zoom': 2.0
    }
    inter_view_ctrl_info = viewCtrlInterp(view_ctrl_info_0, view_ctrl_info, 100)
    o3d_vis = o3dVisualizer(view_ctrl_info=view_ctrl_info, save_path='tmp')
    o3d_vis.start()
    for i in range(100):
        rand_pcd_np = np.random.rand(100, 3)
        rand_pcd_colors = np.random.rand(100, 3)
        rand_pcd_o3d = np2o3d(rand_pcd_np, rand_pcd_colors)
        o3d_vis.update_pcd(rand_pcd_o3d, 'rand_pcd')
        if i == 0:
            o3d_vis.add_triangle_mesh('sphere', 'sphere', color=[1., 0., 0.], radius=0.1)
            o3d_vis.add_triangle_mesh('box', 'box', color=[0., 1., 0.], width=0.1, height=0.1, depth=0.1)
            o3d_vis.add_triangle_mesh('origin', 'origin', size=1.0)
        else:
            sphere_tf = np.eye(4)
            sphere_tf[0, 3] = 0.01 * i
            o3d_vis.update_triangle_mesh('sphere', sphere_tf)
            
            box_tf = np.eye(4)
            box_tf[1, 3] = -0.01 * i
            o3d_vis.update_triangle_mesh('box', box_tf)
        curr_view_ctrl_info = inter_view_ctrl_info[i]
        # curr_view_ctrl_info['zoom'] = (1 + i/100) * view_ctrl_info['zoom']
        o3d_vis.render(curr_view_ctrl_info=curr_view_ctrl_info)
        time.sleep(0.1)


def create_arrow(scale: float = 10.0) -> o3d.geometry.TriangleMesh:
    """
    Create an arrow in for Open3D
    """
    cone_height = scale * 0.2
    cylinder_height = scale * 0.8
    cone_radius = scale / 10
    cylinder_radius = scale / 20
    mesh_frame = o3d.geometry.TriangleMesh.create_arrow(
        cone_radius=cone_radius,
        cone_height=cone_height,
        cylinder_radius=cylinder_radius,
        cylinder_height=cylinder_height,
    )
    return mesh_frame


def get_arrow(
    origin: np.ndarray,
    end: Union[np.ndarray, None] = None,
    vec: Union[np.ndarray, None] = None,
    color: Union[np.ndarray, None] = None,
) -> o3d.geometry.TriangleMesh:
    """
    Creates an arrow from an origin point to an end point,
    or create an arrow from a vector vec starting from origin.
    Args:
        - end (): End point. [x,y,z]
        - vec (): Vector. [i,j,k]
    """
    scale = 10
    if end is not None:
        vec = np.array(end) - np.array(origin)
    elif vec is not None:
        vec = np.array(vec)
    if end is not None or vec is not None:
        scale = np.sqrt(np.sum(vec**2))
    mesh = create_arrow(scale)
    if color is not None:
        mesh.paint_uniform_color(color)
    # Create the arrow
    R = np.eye(3)
    vec = vec / np.linalg.norm(vec)
    R[:, 2] = vec
    x_vec = np.array([-vec[1] - vec[2], vec[0] - vec[2], vec[0] + vec[1]])
    x_vec = x_vec / np.linalg.norm(x_vec)
    y_vec = np.cross(vec, x_vec)
    R[:, 0] = x_vec
    R[:, 1] = y_vec
    mesh.rotate(R, center=np.array([0, 0, 0]))
    mesh.translate(origin)
    return mesh


def white_balance_loops(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    for x in range(result.shape[0]):
        for y in range(result.shape[1]):
            l, a, b = result[x, y, :]
            # fix for CV correction
            l *= 100 / 255.0
            result[x, y, 1] = a - ((avg_a - 128) * (l / 100.0) * 1.1)
            result[x, y, 2] = b - ((avg_b - 128) * (l / 100.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def vis_pcd_plotly(o3d_pcd_ls, up=None, center=None, eye=None, size_ls=2, output_name=None):
    # Create data_ls
    data_ls = []
    for o_i, o3d_pcd in enumerate(o3d_pcd_ls):
        x, z, y = np.array(o3d_pcd.points).T
        colors = (np.array(o3d_pcd.colors) * 255).astype(np.uint8)
        plotly_colors = []
        for i in range(colors.shape[0]):
            plotly_colors.append(f'rgb({colors[i, 0]}, {colors[i, 1]}, {colors[i, 2]})')
        data_ls.append(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=size_ls[o_i], color=plotly_colors,)))
        
    # Plot mesh
    go_pcd = go.Figure(data=data_ls,
                       layout=go.Layout(scene=dict(aspectmode='data'),))

    if up is not None:
        camera = dict(
            up=dict(x=up[0], y=up[1], z=up[2]),
            center=dict(x=center[0], y=center[1], z=center[2]),
            eye=dict(x=eye[0], y=eye[1], z=eye[2])
        )

        go_pcd.update_layout(scene_camera=camera)

    go_pcd.update_layout(margin=dict(l=5,r=5,b=5,t=5,), showlegend=False)
    go_pcd.show()
    if output_name is not None:
        go_pcd.write_html(f'{output_name}.html')

def test_arrow() -> None:
    vec = np.array(
        [
            [0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -0.9, 0.1],
            [0.0, 0.0, -1.0],
        ]
    )
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame()
    for v in vec:
        arrow = get_arrow(origin=np.ones(3), vec=v)
        o3d.visualization.draw_geometries([arrow, coord])

if __name__ == '__main__':
    test_o3d_vis()
