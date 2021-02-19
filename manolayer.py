import os
import jax.numpy as np
from jax import jit
import pickle
import cv2

class ManoLayer():
    __constants__ = [
        'use_pca', 'rot', 'ncomps', 'ncomps', 'kintree_parents', 'check',
        'side', 'center_idx', 'joint_rot_mode'
    ]

    def __init__(self,
                 center_idx=None,
                 flat_hand_mean=True,
                 ncomps=6,
                 side='right',
                 mano_root='.',
                 use_pca=True,
                 root_rot_mode='axisang',
                 joint_rot_mode='axisang',
                 robust_rot=False):
        super().__init__()

        self.center_idx = center_idx
        self.robust_rot = robust_rot
        if root_rot_mode == 'axisang':
            self.rot = 3
        else:
            self.rot = 6
        self.flat_hand_mean = flat_hand_mean
        self.side = side
        self.use_pca = use_pca
        self.joint_rot_mode = joint_rot_mode
        self.root_rot_mode = root_rot_mode
        if use_pca:
            self.ncomps = ncomps
        else:
            self.ncomps = 45

        if side == 'right':
            self.mano_path = os.path.join(mano_root, 'MANO_RIGHT.pkl')
        elif side == 'left':
            self.mano_path = os.path.join(mano_root, 'MANO_LEFT.pkl')

        smpl_data = self._ready_arguments(self.mano_path)

        hands_components = smpl_data['hands_components'] #45*45

        self.smpl_data = smpl_data
        self.betas = np.array(smpl_data['betas'])[np.newaxis,...]
        self.shapedirs = np.array(smpl_data['shapedirs'])
        self.posedirs = np.array(smpl_data['posedirs'])
        self.v_template = np.array(smpl_data['v_template'])[np.newaxis,...]
        self.J_regressor = np.array(smpl_data['J_regressor'].toarray())
        self.weights = np.array(smpl_data['weights'])
        self.faces = np.array(smpl_data['f'])

        # Get hand mean
        hands_mean = np.zeros(hands_components.shape[1]) if flat_hand_mean else smpl_data['hands_mean']
        self.hands_mean = hands_mean.copy()[np.newaxis,...] #45 all zeros
        selected_components = hands_components[:ncomps]
        self.tselected_comps = np.array(selected_components)

        # Kinematic chain params
        self.kintree_table = smpl_data['kintree_table']
        parents = list(self.kintree_table[0].tolist())
        self.kintree_parents = parents
    
    def _ready_arguments(self, fname_or_dict, posekey4vposed='pose'):
        dd = pickle.load(open(fname_or_dict, 'rb'), encoding='latin1')

        want_shapemodel = 'shapedirs' in dd
        nposeparms = dd['kintree_table'].shape[1] * 3

        if 'trans' not in dd:
            dd['trans'] = np.zeros(3)
        if 'pose' not in dd:
            dd['pose'] = np.zeros(nposeparms)
        if 'shapedirs' in dd and 'betas' not in dd:
            dd['betas'] = np.zeros(dd['shapedirs'].shape[-1])

        for s in [
                'v_template', 'weights', 'posedirs', 'pose', 'trans', 'shapedirs',
                'betas', 'J'
        ]:
            if (s in dd) and not hasattr(dd[s], 'dterms'):
                dd[s] = np.array(dd[s])

        assert (posekey4vposed in dd)
        if want_shapemodel:
            dd['v_shaped'] = dd['shapedirs'].dot(dd['betas']) + dd['v_template']
            v_shaped = dd['v_shaped']
            J_tmpx = dd['J_regressor'] * v_shaped[:, 0]
            J_tmpy = dd['J_regressor'] * v_shaped[:, 1]
            J_tmpz = dd['J_regressor'] * v_shaped[:, 2]
            dd['J'] = np.vstack((J_tmpx, J_tmpy, J_tmpz)).T
            pose_map_res = self._lrotmin(dd[posekey4vposed])
            dd['v_posed'] = v_shaped + dd['posedirs'].dot(pose_map_res)
        else:
            pose_map_res = self._lrotmin(dd[posekey4vposed])
            dd_add = dd['posedirs'].dot(pose_map_res)
            dd['v_posed'] = dd['v_template'] + dd_add

        return dd

    def _lrotmin(self, p):
        p = p.ravel()[3:]
        return np.concatenate(
            [(cv2.Rodrigues(pp)[0] - np.eye(3)).ravel()
            for pp in p.reshape((-1, 3))]).ravel()


    def _posemap_axisang(self, pose_vectors):
        rot_nb = int(pose_vectors.shape[1] / 3)
        pose_vec_reshaped = pose_vectors.reshape(-1, 3)
        rot_mats = self._batch_rodrigues(pose_vec_reshaped)
        rot_mats = rot_mats.reshape(pose_vectors.shape[0], rot_nb * 9)
        pose_maps = self._subtract_flat_id(rot_mats)
        return pose_maps, rot_mats
    
    def _subtract_flat_id(self, rot_mats):
        # Subtracts identity as a flattened tensor
        rot_nb = int(rot_mats.shape[1] / 9)
        id_flat = np.tile(np.eye(3, dtype=rot_mats.dtype).reshape(1, 9),(1,rot_nb))
        # id_flat.requires_grad = False
        results = rot_mats - id_flat
        return results

    def _quat2mat(self, quat):
        norm_quat = quat
        norm_quat = norm_quat / np.linalg.norm(norm_quat + 1e-8, axis=1, keepdims=True)
        w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:,
                                                                2], norm_quat[:,
                                                                            3]

        batch_size = quat.shape[0]

        w2, x2, y2, z2 = w**2, x**2, y**2, z**2
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z

        rotMat = np.stack([
            w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
            w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
            w2 - x2 - y2 + z2
        ],
                            axis=1).reshape(batch_size, 3, 3)
        return rotMat


    def _batch_rodrigues(self, axisang):
        #axisang N x 3
        axisang_norm = np.linalg.norm(axisang + 1e-8, axis=1)
        angle = axisang_norm[...,np.newaxis]
        axisang_normalized = axisang / angle
        angle = angle * 0.5
        v_cos = np.cos(angle)
        v_sin = np.sin(angle)
        quat = np.concatenate((v_cos, v_sin * axisang_normalized), 1)
        rot_mat = self._quat2mat(quat)
        rot_mat = rot_mat.reshape(rot_mat.shape[0], 9)
        return rot_mat

    def _with_zeros(self, tensor):
        batch_size = tensor.shape[0]
        padding = np.array([0.0, 0.0, 0.0, 1.0])

        concat_list = (tensor, np.tile(padding.reshape(1, 1, 4),(batch_size, 1, 1)))
        cat_res = np.concatenate(concat_list, 1)
        return cat_res

    def __call__(self,
                pose_coeffs,
                betas=np.zeros(1),
                ):
        batch_size = pose_coeffs.shape[0]
        # Get axis angle from PCA components and coefficients
        # Remove global rot coeffs
        hand_pose_coeffs = pose_coeffs[:, self.rot:self.rot +self.ncomps]
        full_hand_pose = hand_pose_coeffs

        # Concatenate back global rot
        full_pose = np.concatenate((pose_coeffs[:, :self.rot],self.hands_mean + full_hand_pose), 1)
        # compute rotation matrixes from axis-angle while skipping global rotation
        pose_map, rot_map = self._posemap_axisang(full_pose)
        root_rot = rot_map[:, :9].reshape(batch_size, 3, 3)
        rot_map = rot_map[:, 9:]
        pose_map = pose_map[:, 9:]

        # Full axis angle representation with root joint
        v_shaped = np.matmul(self.shapedirs, betas.transpose((1, 0))).transpose((2, 0, 1)) + self.v_template
        j = np.matmul(self.J_regressor, v_shaped)
        # th_pose_map should have shape 20x135
        v_posed = v_shaped + np.matmul(self.posedirs, pose_map.transpose((1, 0))[np.newaxis,...]).transpose((2, 0, 1))
        # Final T pose with transformation done !

        # Global rigid transformation

        root_j = j[:, 0, :].reshape(batch_size, 3, 1)
        root_trans = self._with_zeros(np.concatenate((root_rot, root_j), 2))

        all_rots = rot_map.reshape(rot_map.shape[0], 15, 3, 3)
        lev1_idxs = [1, 4, 7, 10, 13]
        lev2_idxs = [2, 5, 8, 11, 14]
        lev3_idxs = [3, 6, 9, 12, 15]
        lev1_rots = all_rots[:, [idx - 1 for idx in lev1_idxs]]
        lev2_rots = all_rots[:, [idx - 1 for idx in lev2_idxs]]
        lev3_rots = all_rots[:, [idx - 1 for idx in lev3_idxs]]
        lev1_j = j[:, lev1_idxs]
        lev2_j = j[:, lev2_idxs]
        lev3_j = j[:, lev3_idxs]

        # From base to tips
        # Get lev1 results
        all_transforms = [root_trans[:,np.newaxis,...]]
        lev1_j_rel = lev1_j - root_j.transpose((0, 2, 1))
        lev1_rel_transform_flt = self._with_zeros(np.concatenate((lev1_rots, lev1_j_rel[...,np.newaxis]), 3).reshape(-1, 3, 4))
        root_trans_flt = np.tile(root_trans[:,np.newaxis,...],(1, 5, 1, 1)).reshape(root_trans.shape[0] * 5, 4, 4)
        lev1_flt = np.matmul(root_trans_flt, lev1_rel_transform_flt)
        all_transforms.append(lev1_flt.reshape(all_rots.shape[0], 5, 4, 4))

        # Get lev2 results
        lev2_j_rel = lev2_j - lev1_j
        lev2_rel_transform_flt = self._with_zeros(np.concatenate((lev2_rots, lev2_j_rel[...,np.newaxis]), 3).reshape(-1, 3, 4))
        lev2_flt = np.matmul(lev1_flt, lev2_rel_transform_flt)
        all_transforms.append(lev2_flt.reshape(all_rots.shape[0], 5, 4, 4))

        # Get lev3 results
        lev3_j_rel = lev3_j - lev2_j
        lev3_rel_transform_flt = self._with_zeros(np.concatenate((lev3_rots, lev3_j_rel[...,np.newaxis]), 3).reshape(-1, 3, 4))
        lev3_flt = np.matmul(lev2_flt, lev3_rel_transform_flt)
        all_transforms.append(lev3_flt.reshape(all_rots.shape[0], 5, 4, 4))

        reorder_idxs = [0, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 5, 10, 15]
        results = np.concatenate(all_transforms, 1)[:, reorder_idxs]
        results_global = results

        joint_js = np.concatenate((j, np.zeros((j.shape[0], 16, 1))), 2)


        tmp2 = np.matmul(results, joint_js[...,np.newaxis])
        results2 = (results - np.concatenate((np.zeros((*tmp2.shape[:2], 4, 3)), tmp2), 3)).transpose((0, 2, 3, 1))

        T = np.matmul(results2, self.weights.transpose((1, 0)))

        rest_shape_h = np.concatenate((
            v_posed.transpose((0, 2, 1)),
            np.ones((batch_size, 1, v_posed.shape[1]))
        ), 1)

        verts = (T * rest_shape_h[:,np.newaxis,...]).sum(2).transpose((0, 2, 1))
        verts = verts[:, :, :3]
        jtr = results_global[:, :, :3, 3]
        # In addition to MANO reference joints we sample vertices on each finger
        # to serve as finger tips
        tips = verts[:, [745, 317, 444, 556, 673]]
        jtr = np.concatenate((jtr, tips), 1)

        # Reorder joints to match visualization utilities
        jtr = jtr[:, [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]]

        center_joint = jtr[:, self.center_idx][:,np.newaxis,...]
        jtr = jtr - center_joint
        verts = verts - center_joint

        return verts, jtr, full_pose

if __name__ == "__main__":
    manolayer = ManoLayer()