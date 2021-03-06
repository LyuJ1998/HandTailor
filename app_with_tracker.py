import cv2
import numpy as np
import pygame
import torch
import time
import argparse
import torch.backends.cudnn as cudnn
import pyrealsense2 as rs
import jax.numpy as npj
import open3d
from jax import grad, jit, vmap
from jax.experimental import optimizers
from torchvision.transforms import functional
import pickle

from manolayer import ManoLayer
from model import HandNet
from checkpoints import CheckpointIO
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

mano_layer = ManoLayer(center_idx=9, side="right", mano_root=".", use_pca=False, flat_hand_mean=True,)
mano_layer = jit(mano_layer)

class RealSenseCapture:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = self.pipeline.start(config)
    
    def read(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        return np.flip(color_image, -1).copy()

@jit
def hm_to_kp2d(hm):
    b, c, w, h = hm.shape
    hm = hm.reshape(b,c,-1)
    hm = hm/npj.sum(hm,-1,keepdims=True)
    coord_map_x = npj.tile(npj.arange(0,w).reshape(-1,1), (1,h))
    coord_map_y = npj.tile(npj.arange(0,h).reshape(1,-1), (w,1))
    coord_map_x = coord_map_x.reshape(1,1,-1)
    coord_map_y = coord_map_y.reshape(1,1,-1)
    x = npj.sum(coord_map_x * hm,-1,keepdims=True)
    y = npj.sum(coord_map_y * hm,-1,keepdims=True)
    kp_2d = npj.concatenate((y,x),axis=-1)
    return kp_2d

@jit
def reinit_root(joint_root,kp2d,camparam):
    uv = kp2d[0,9,:]
    xy = joint_root[...,:2]
    z = joint_root[...,2]
    joint_root = ((uv - camparam[0, 0, 2:4])/camparam[0, 0, :2]) * z
    joint_root = npj.concatenate((joint_root,z))
    return joint_root

@jit
def reinit_scale(joint,kp2d,camparam,bone,joint_root):
    z0 = joint_root[2:]
    xy0 = joint_root[:2]
    xy = joint[:,:2] * bone
    z = joint[:,2:] * bone
    kp2d = kp2d[0]
    s1 = npj.sum(((kp2d - camparam[0, 0, 2:4])*xy)/(camparam[0, 0, :2]*(z0+z)) - (xy0*xy)/((z0+z)**2))
    s2 = npj.sum((xy**2)/((z0+z)**2))
    s = s1/s2
    bone = bone * npj.max(npj.array([s,0.9]))
    return bone

@jit
def geo(joint):
    idx_a = npj.array([1,5,9,13,17])
    idx_b = npj.array([2,6,10,14,18])
    idx_c = npj.array([3,7,11,15,19])
    idx_d = npj.array([4,8,12,16,20])
    p_a = joint[:,idx_a,:]
    p_b = joint[:,idx_b,:]
    p_c = joint[:,idx_c,:]
    p_d = joint[:,idx_d,:]
    v_ab = p_a - p_b #(B, 5, 3)
    v_bc = p_b - p_c #(B, 5, 3)
    v_cd = p_c - p_d #(B, 5, 3)
    loss_1 = npj.abs(npj.sum(npj.cross(v_ab, v_bc, -1) * v_cd, -1)).mean()
    loss_2 = - npj.clip(npj.sum(npj.cross(v_ab, v_bc, -1) * npj.cross(v_bc, v_cd, -1)), -npj.inf, 0).mean()
    loss = 10000*loss_1 + 100000*loss_2

    return loss

@jit
def residuals(input_list,so3_init,beta_init,joint_root,kp2d,camparam):
    so3 = input_list['so3']
    beta = input_list['beta']
    bone = input_list['bone']
    so3 = so3[npj.newaxis,...]
    beta = beta[npj.newaxis,...]
    _, joint_mano, _ = mano_layer(
        pose_coeffs = so3,
        betas = beta
    )
    bone_pred = npj.linalg.norm(joint_mano[:, 0, :] - joint_mano[:, 9, :], axis=1, keepdims=True)
    bone_pred = bone_pred[:,npj.newaxis,...]
    reg = ((so3 - so3_init)**2)
    reg_beta = ((beta - beta_init)**2)
    joint_mano = joint_mano / bone_pred
    joint_mano = joint_mano * bone + joint_root
    geo_reg = geo(joint_mano)
    xy = (joint_mano[...,:2]/joint_mano[...,2:])
    uv = (xy * camparam[:, :, :2] ) + camparam[:, :, 2:4]
    errkp = ((uv - kp2d)**2)
    err = 0.01*reg.mean() + 0.01*reg_beta.mean() + 1*errkp.mean() + 100*geo_reg.mean()
    return err

@jit
def mano_de(params,joint_root,bone):
    so3 = params['so3']
    beta = params['beta']
    verts_mano, joint_mano, _ = mano_layer(
        pose_coeffs = so3[npj.newaxis,...],
        betas = beta[npj.newaxis,...]
    )

    bone_pred = npj.linalg.norm(joint_mano[:, 0, :] - joint_mano[:, 9, :],axis=1, keepdims=True)
    bone_pred = bone_pred[:,npj.newaxis,...]
    verts_mano = verts_mano / bone_pred
    verts_mano = verts_mano * bone  + joint_root
    v = verts_mano[0]
    return v

@jit
def mano_de_j(so3, beta):
    _, joint_mano, _ = mano_layer(
        pose_coeffs = so3[npj.newaxis,...],
        betas = beta[npj.newaxis,...]
    )

    bone_pred = npj.linalg.norm(joint_mano[:, 0, :] - joint_mano[:, 9, :],axis=1, keepdims=True)
    bone_pred = bone_pred[:,npj.newaxis,...]
    joint_mano = joint_mano / bone_pred
    j = joint_mano[0]
    return j

def live_application(capture,arg):
    pygame.init()
    display = pygame.display.set_mode((640, 480))
    pygame.display.set_caption('Real Time Hand Recon')

    dd = pickle.load(open("MANO_RIGHT.pkl", 'rb'), encoding='latin1')
    face = np.array(dd['f'])

    model = HandNet()
    model = model.to(device)
    checkpoint_io = CheckpointIO('.', model=model)
    load_dict = checkpoint_io.load('checkpoints/model.pt')
    model.eval()

    renderer = utils.MeshRenderer(face, img_size=[640,480])

    o_intr = torch.from_numpy(np.array([
                [arg.fx, 0.0, arg.cx],
                [0.0, arg.fy, arg.cy],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32)).unsqueeze(0).numpy()
    
    o_camparam = np.zeros((4))
    o_camparam[0] = o_intr[0, 0, 0]
    o_camparam[1] = o_intr[0, 1, 1]
    o_camparam[2] = o_intr[0, 0, 2]
    o_camparam[3] = o_intr[0, 1, 2]
    
    gr = jit(grad(residuals))
    lr = 0.03
    opt_init, opt_update, get_params = optimizers.adam(lr, b1=0.5, b2=0.5)
    opt_init = jit(opt_init)
    opt_update = jit(opt_update)
    get_params = jit(get_params)
    x_reg = np.ones((10,))*240
    y_reg = np.ones((10,))*240
    s_reg = np.ones((10,))*240
    weight = np.array([0,0,0,0,0,0,0,0.1,0.2,0.7])
    i = 0
    x = 240
    y = 320
    scale = 256
    with torch.no_grad():
        while True:
            i = i + 1
            img = capture.read()
            frame = img.copy()
            if img is None:
                continue
            vmin = max(0, y - scale//2)
            vmin_p = max(scale//2 - y, 0)
            umin = max(0, x - scale//2)
            umin_p = max(scale//2 - x, 0)
            vmax = min(640, y + scale//2)
            vmax_p = max(scale//2 + y - 640, 0)
            umax = min(480, x + scale//2)
            umax_p = max(scale//2 + x - 480, 0)
            img = img[int(umin):int(umax),int(vmin):int(vmax),:]
            img = cv2.copyMakeBorder(img,int(umin_p),int(umax_p),int(vmin_p),int(vmax_p),cv2.BORDER_CONSTANT,value=[255,255,255])

            cx = arg.cx - y + scale//2
            cy = arg.cy - x + scale//2
            
            cx = (cx * 256) / scale
            cy = (cy * 256) / scale
            fx = (arg.fx * 256) / scale
            fy = (arg.fy * 256) / scale

            intr = torch.from_numpy(np.array([
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32)).unsqueeze(0).to(device)

            _intr = intr.cpu().numpy() 
            camparam = np.zeros((1, 21, 4))
            camparam[:, :, 0] = _intr[:, 0, 0]
            camparam[:, :, 1] = _intr[:, 1, 1]
            camparam[:, :, 2] = _intr[:, 0, 2]
            camparam[:, :, 3] = _intr[:, 1, 2]

            img = cv2.resize(img, (256, 256),cv2.INTER_LINEAR)

            img = functional.to_tensor(img).float()
            img = functional.normalize(img, [0.5, 0.5, 0.5], [1, 1, 1])
            img = img.unsqueeze(0).to(device)
            
            hm, so3, beta, joint_root, bone = model(img,intr)
            kp2d = hm_to_kp2d(hm.detach().cpu().numpy())*4
            so3 = so3[0].detach().cpu().float().numpy()
            beta = beta[0].detach().cpu().float().numpy()
            bone = bone[0].detach().cpu().numpy()
            joint_root = joint_root[0].detach().cpu().numpy()
            so3 = npj.array(so3)
            beta = npj.array(beta)
            bone = npj.array(bone)
            joint_root = npj.array(joint_root)
            kp2d = npj.array(kp2d)
            so3_init = so3
            beta_init = beta
            joint_root = reinit_root(joint_root,kp2d, camparam)
            joint = mano_de_j(so3, beta)
            bone = reinit_scale(joint,kp2d,camparam,bone,joint_root)
            params = {'so3':so3, 'beta':beta, 'bone':bone}
            opt_state = opt_init(params)
            n = 0
            while n < 20:
                n = n + 1
                params = get_params(opt_state)
                grads = gr(params,so3_init,beta_init,joint_root,kp2d,camparam)
                opt_state = opt_update(n, grads, opt_state)
            params = get_params(opt_state)
            v = mano_de(params,joint_root,bone)
        
            kp2d = np.array(kp2d[0])
            x = x + ((kp2d[9,1] - 128)*scale)/256
            y = y + ((kp2d[9,0] - 128)*scale)/256
            scale = max(max(kp2d[:,0].max() - kp2d[:,0].min(), kp2d[:,1].max() - kp2d[:,1].min()) * 2, 80)
            
            x_reg[:9] = x_reg[1:]
            x_reg[-1] = x
            y_reg[:9] = y_reg[1:]
            y_reg[-1] = y
            s_reg[:9] = s_reg[1:]
            s_reg[-1] = scale

            x = (x_reg * weight).sum()
            y = (y_reg * weight).sum()
            scale = (s_reg * weight).sum()
            
            frame = renderer(v,o_intr[0],frame)
            frame = cv2.rectangle(frame, (int(vmin), int(umin)), (int(vmax), int(umax)), (255, 255, 255), thickness=5)
            display.blit(
                pygame.surfarray.make_surface(np.transpose(np.flip(frame,1), (1, 0, 2))),(0, 0))
            pygame.display.update()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--cx',
        type=float,
        default=321.2842102050781,
    )

    parser.add_argument(
        '--cy',
        type=float,
        default=235.8609161376953,
    )

    parser.add_argument(
        '--fx',
        type=float,
        default=612.0206298828125,
    )

    parser.add_argument(
        '--fy',
        type=float,
        default=612.2821044921875,
    )

    live_application(RealSenseCapture(),parser.parse_args())
