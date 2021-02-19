import torch
import numpy as np
import transforms3d as t3d
import torch.nn.functional as F
from opendr.camera import ProjectPoints
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight

import common

def normalize_quaternion(quaternion,eps=1e-12):
    return F.normalize(quaternion, p=2, dim=-1, eps=eps)

def my_atan2(y, x):
    pi = torch.from_numpy(np.array([np.pi])).to(y.device, y.dtype)
    ans = torch.atan(y/x)
    ans = torch.where(((y>0).float()*(x<0).float()).bool(), ans+pi, ans)
    ans = torch.where(((y<0).float()*(x<0).float()).bool(), ans+pi, ans)
    return ans

def quaternion_to_angle_axis(quaternion):
    q1 = quaternion[..., 1]
    q2 = quaternion[..., 2]
    q3 = quaternion[..., 3]
    sin_squared_theta = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta = torch.sqrt(sin_squared_theta)
    cos_theta = quaternion[..., 0]
    two_theta = 2.0 * torch.where(
        cos_theta < 0.0, my_atan2(-sin_theta, -cos_theta),
        my_atan2(sin_theta, cos_theta))

    k_pos = two_theta / sin_theta
    k_neg = 2.0 * torch.ones_like(sin_theta)
    k = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis = quaternion[...,1:] * k.unsqueeze(2)

    return angle_axis

def hm_to_uvd(hm3d):
    b, c, w, h = hm3d.size()
    hm2d = hm3d[:,:21,...]
    depth = hm3d[:,21:,...]
    uv = hm_to_kp2d(hm2d)/w
    hm2d = hm2d.view(b,1,c//2,-1)
    depth = depth.view(b,1,c//2,-1)
    hm2d = hm2d / torch.sum(hm2d,-1,keepdim=True)
    d = torch.sum(depth * hm2d,-1).permute(0,2,1)
    joint = torch.cat((uv,d),dim=-1)
    return joint

def hm_to_kp2d(hm):
    b, c, w, h = hm.size()
    hm = hm.view(b,c,-1)
    hm = hm/torch.sum(hm,-1,keepdim=True)
    coord_map_x = torch.arange(0,w).view(-1,1).repeat(1,h).to(hm.device)
    coord_map_y = torch.arange(0,h).view(1,-1).repeat(w,1).to(hm.device)
    coord_map_x = coord_map_x.view(1,1,-1).float()
    coord_map_y = coord_map_y.view(1,1,-1).float()
    x = torch.sum(coord_map_x * hm,-1,keepdim=True)
    y = torch.sum(coord_map_y * hm,-1,keepdim=True)
    kp_2d = torch.cat((y,x),dim=-1)
    return kp_2d

def uvd2xyz(uvd, joint_root, joint_bone, intr=None, trans=None, scale=None, inp_res=256, mode='persp'):
    bs = uvd.shape[0]
    uv = uvd[:, :, :2] * inp_res # 0~256
    depth = ( uvd[:, :, 2] * common.DEPTH_RANGE ) + common.DEPTH_MIN
    root_depth = joint_root[:, -1].unsqueeze(1) #(B, 1)
    z = depth * joint_bone.expand_as(uvd[:, :, 2]) + \
        root_depth.expand_as(uvd[:, :, 2])  # B x M

    '''2. uvd->xyz'''
    camparam = torch.cat((intr[:, 0:1, 0],intr[:, 1:2, 1],intr[:, 0:1, 2],intr[:, 1:2, 2]),1)
    camparam = camparam.unsqueeze(1).repeat(1, uvd.size(1), 1)  # B x M x 4
    xy = ((uv - camparam[:, :, 2:4]) / camparam[:, :, :2]) * \
        z.unsqueeze(2).expand_as(uv)  # B x M x 2
    return torch.cat((xy, z.unsqueeze(2)), -1)  # B x M x 3

class MeshRenderer(object):
    def __init__(self,
                 mesh_faces,
                 img_size=256,
                 flength=500.):  #822.79041):  #
        self.faces = mesh_faces
        self.w = img_size
        self.h = img_size
        self.flength = flength

    def __call__(self,
                 verts,
                 cam_intrinsics,
                 img=None,
                 do_alpha=False,
                 far=None,
                 near=None,
                 color_id=0,
                 img_size=None,
                 R=None):
        """
        cam is 3D [fx, fy, px, py]
        """
        if img is not None:
            h, w = img.shape[:2]
        elif img_size is not None:
            h = img_size[0]
            w = img_size[1]
        else:
            h = self.h
            w = self.w


        dist = np.zeros(5)
        dist = dist.flatten()
        M = np.eye(4)

        # get R, t from M (has to be world2cam)
        if R is None:
            R = M[:3, :3]
        ax, angle = t3d.axangles.mat2axangle(R)
        rt = ax*angle
        rt = rt.flatten()
        t = M[:3, 3]

        if cam_intrinsics is None:
            cam_intrinsics = np.array([
                [500, 0, 128],
                [0, 500, 128],
                [0, 0, 1]]
            )

        pp = np.array([cam_intrinsics[0, 2], cam_intrinsics[1, 2]])
        f = np.array([cam_intrinsics[0, 0], cam_intrinsics[1, 1]])


        use_cam = ProjectPoints(
            rt=rt,
            t=t, # camera translation
            f=f,  # focal lengths
            c=pp,  # camera center (principal point)
            k=dist
        )  # OpenCv distortion params

        if near is None:
            near = np.maximum(np.min(verts[:, 2]) - 25, 0.1)
        if far is None:
            far = np.maximum(np.max(verts[:, 2]) + 25, 25)

        imtmp = render_model(
            verts,
            self.faces,
            w,
            h,
            use_cam,
            do_alpha=do_alpha,
            img=img,
            far=far,
            near=near,
            color_id=color_id)

        return (imtmp * 255).astype('uint8')

    def rotated(self,
                verts,
                deg,
                cam=None,
                axis='y',
                img=None,
                do_alpha=True,
                far=None,
                near=None,
                color_id=0,
                img_size=None):
        import math
        if axis == 'y':
            around = cv2.Rodrigues(np.array([0, math.radians(deg), 0]))[0]
        elif axis == 'x':
            around = cv2.Rodrigues(np.array([math.radians(deg), 0, 0]))[0]
        else:
            around = cv2.Rodrigues(np.array([0, 0, math.radians(deg)]))[0]
        center = verts.mean(axis=0)
        new_v = np.dot((verts - center), around) + center

        return self.__call__(
            new_v,
            cam,
            img=img,
            do_alpha=do_alpha,
            far=far,
            near=near,
            img_size=img_size,
            color_id=color_id)

def simple_renderer(rn,
                    verts,
                    faces,
                    yrot=np.radians(120),
                    color=common.colors['light_pink']):
    # Rendered model color
    rn.set(v=verts, f=faces, vc=color, bgcolor=np.ones(3))
    albedo = rn.vc

    # Construct Back Light (on back right corner)
    rn.vc = LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([-200, -100, -100]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]))

    # Construct Left Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([800, 10, 300]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]))

    # Construct Right Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([-500, 500, 1000]), yrot),
        vc=albedo,
        light_color=np.array([.7, .7, .7]))

    return rn.r

def _create_renderer(w=640,
                     h=480,
                     rt=np.zeros(3),
                     t=np.zeros(3),
                     f=None,
                     c=None,
                     k=None,
                     near=.5,
                     far=10.):

    f = np.array([w, w]) / 2. if f is None else f
    c = np.array([w, h]) / 2. if c is None else c
    k = np.zeros(5) if k is None else k

    rn = ColoredRenderer()

    rn.camera = ProjectPoints(rt=rt, t=t, f=f, c=c, k=k)
    rn.frustum = {'near': near, 'far': far, 'height': h, 'width': w}
    return rn

def _rotateY(points, angle):
    """Rotate the points by a specified angle."""
    ry = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                   [-np.sin(angle), 0., np.cos(angle)]])
    return np.dot(points, ry)

def render_model(verts,
                 faces,
                 w,
                 h,
                 cam,
                 near=0.5,
                 far=25,
                 img=None,
                 do_alpha=False,
                 color_id=None):
    rn = _create_renderer(
        w=w, h=h, near=near, far=far, rt=cam.rt, t=cam.t, f=cam.f, c=cam.c)

    # Uses img as background, otherwise white background.
    if img is not None:
        rn.background_image = img / 255. if img.max() > 1 else img

    if color_id is None:
        color = common.colors['light_blue']
    else:
        color_list = list(common.colors.values())
        color = color_list[color_id % len(color_list)]

    imtmp = simple_renderer(rn, verts, faces, color=color)

    return imtmp

class OpendrRenderer(object):
    def __init__(self,
                 img_size=224,
                 mesh_color=np.array([0.5, 0.5, 0.5]),):

        self.w = img_size
        self.h = img_size
        self.color = mesh_color
        self.img_size = img_size
        self.flength = 500.

    
    def render(self, verts, faces, bg_img):
        verts = verts.copy()
        faces = faces.copy()

        input_size = 500

        f = 5

        verts[:, 0] = (verts[:, 0] - input_size) / input_size
        verts[:, 1] = (verts[:, 1] - input_size) / input_size

        verts[:, 2] /= (5 * 112)
        verts[:, 2] += f

        cam_for_render = np.array([f, 1, 1]) * input_size

        rend_img = self.__call__(
            img=bg_img, cam=cam_for_render, 
            verts=verts, faces=faces, color=self.color)
        
        return rend_img
    

    def __call__(self,
                 verts,
                 faces,
                 cam=None,
                 img=None,
                 do_alpha=False,
                 far=None,
                 near=None,
                 color = np.array([0, 0, 255]),
                 img_size=None):
        """
        cam is 3D [f, px, py]
        """
        if img is not None:
            h, w = img.shape[:2]
        elif img_size is not None:
            h = img_size[0]
            w = img_size[1]
        else:
            h = self.h
            w = self.w

        if cam is None:
            cam = [self.flength, w / 2., h / 2.]

        use_cam = ProjectPoints(
            f=cam[0] * np.ones(2),
            rt=np.zeros(3),
            t=np.zeros(3),
            k=np.zeros(5),
            c=cam[1:3])

        if near is None:
            near = np.maximum(np.min(verts[:, 2]) - 25, 0.1)
        if far is None:
            far = np.maximum(np.max(verts[:, 2]) + 25, 25)

        return_value = render_model(
            verts,
            faces,
            w,
            h,
            use_cam,
            do_alpha=do_alpha,
            img=img,
            far=far,
            near=near,
            color_id=0)

        imtmp = return_value
        image = (imtmp * 255).astype('uint8')
        return image