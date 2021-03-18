import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import common

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(BottleneckBlock, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        mid_planes = (out_planes // 2 ) if out_planes >= in_planes else in_planes // 2

        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_planes)

        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(mid_planes)

        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        if in_planes != out_planes:
            self.conv4 = nn.Conv2d(in_planes, out_planes, bias=True, kernel_size=1)


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.in_planes != self.out_planes:
            residual = self.conv4(x)
        out += residual
        out = self.bn3(out)
        out = self.relu(out)

        return out

class Hourglass(nn.Module):
    def __init__(self, block=BottleneckBlock, nblocks=1, in_planes=64, depth=4):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.hg = self._make_hourglass(block, nblocks, in_planes, depth)

    def _make_hourglass(self, block, nblocks, in_planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, nblocks, in_planes))
            if i == 0:
                res.append(self._make_residual(block, nblocks, in_planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _make_residual(self, block, nblocks, in_planes):
        layers = []
        for i in range(0, nblocks):
            layers.append(block(in_planes, in_planes))
        return nn.Sequential(*layers)

    def _hourglass_foward(self, n, x):
        up1 = self.hg[n-1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n-1][1](low1)

        if n > 1:
            low2 = self._hourglass_foward(n-1, low1)
        else:
            low2 = self.hg[n-1][3](low1)
        low3 = self.hg[n-1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hourglass_foward(self.depth, x)

class ResNet18(nn.Module):
    def __init__(self, block=BottleneckBlock, out_plane=256):
        super(ResNet18, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_residual(block, 2, 256, 512)
        self.layer2 = self._make_residual(block, 2, 512, 512)
        self.layer3 = self._make_residual(block, 2, 512, 512)
        self.layer4 = self._make_residual(block, 2, 512, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_residual(self, block, nblocks, in_planes, out_planes):
        layers = []
        layers.append(block(in_planes, out_planes))
        self.in_planes = out_planes
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, out_planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(x)#32
        x = self.layer1(x)
        x = self.maxpool(x)#16
        x = self.layer2(x)
        x = self.maxpool(x)#8
        x = self.layer3(x)
        x = self.maxpool(x)#4
        x = self.layer4(x)
        x = self.avgpool(x)#1
        x = torch.flatten(x,1)

        return x

class Hand2D(nn.Module):
    def __init__(
        self,
        nstacks=2,
        nblocks=1,
        njoints=21,
        block=BottleneckBlock,
    ):
        super(Hand2D, self).__init__()
        self.njoints  = njoints
        self.nstacks  = nstacks
        self.in_planes = 64
 
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.sigmoid = nn.Sigmoid()

        self.layer1 = self._make_residual(block, nblocks, self.in_planes, 2*self.in_planes)
        self.layer2 = self._make_residual(block, nblocks, self.in_planes, 2*self.in_planes)
        self.layer3 = self._make_residual(block, nblocks, self.in_planes, self.in_planes)

        ch = self.in_planes

        hg2b, res, fc, hm = [],[],[],[]
        for i in range(nstacks):
            hg2b.append(Hourglass(block, nblocks, ch, depth=4))
            res.append(self._make_residual(block, nblocks, ch, ch))
            hm.append(nn.Conv2d(ch, njoints, kernel_size=1, bias=True))
            fc.append(self._make_fc(ch + njoints, ch))

        self.hg2b  = nn.ModuleList(hg2b)
        self.res  = nn.ModuleList(res)
        self.fc   = nn.ModuleList(fc)
        self.hm   = nn.ModuleList(hm)
    
    def _make_fc(self, in_planes, out_planes):
        conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_planes)
        return nn.Sequential(conv, bn, self.relu)

    def _make_residual(self, block, nblocks, in_planes, out_planes):
        layers = []
        layers.append(block(in_planes, out_planes) )
        self.in_planes = out_planes
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, out_planes))
        return nn.Sequential(*layers)

    def forward(self, x):

        l_est_hm, l_enc = [], []
        net = self.conv1(x)
        net = self.bn1(net)
        net = self.relu(net)

        net = self.layer1(net)
        net = self.maxpool(net)
        net = self.layer2(net)
        net = self.layer3(net)

        for i in range(self.nstacks):
            net = self.hg2b[i](net)
            net = self.res[i](net)
            est_hm = self.sigmoid(self.hm[i](net))
            net = torch.cat((net,est_hm),1)
            net = self.fc[i](net)
            l_est_hm.append(est_hm)
            l_enc.append(net)
        assert len(l_est_hm) == self.nstacks
        return l_est_hm, l_enc

class IKNet(nn.Module):
    def __init__(
        self,
        njoints=21,
        hidden_size_pose=[256, 512, 1024, 1024, 512, 256],
    ):
        super(IKNet, self).__init__()
        self.njoints = njoints
        in_neurons = 3 * njoints
        out_neurons = 16 * 4 # 16 quats
        neurons = [in_neurons] + hidden_size_pose

        invk_layers = []
        for layer_idx, (inps, outs) in enumerate(zip(neurons[:-1], neurons[1:])):
            invk_layers.append(nn.Linear(inps, outs))
            invk_layers.append(nn.BatchNorm1d(outs))
            invk_layers.append(nn.ReLU())

        invk_layers.append(nn.Linear(neurons[-1], out_neurons))

        self.invk_layers = nn.Sequential(*invk_layers)

    def forward(self, joint):
        joint = joint.contiguous().view(-1, self.njoints*3)
        quat = self.invk_layers(joint)
        quat = quat.view(-1, 16, 4)
        quat = utils.normalize_quaternion(quat)
        so3 = utils.quaternion_to_angle_axis(quat).contiguous()
        so3 = so3.view(-1, 16 * 3)
        return so3, quat

class Hand2Dto3D(nn.Module):
    def __init__(
        self,
        nstacks=2,
        nblocks=1,
        njoints=21,
        block=BottleneckBlock,
    ):
        super(Hand2Dto3D, self).__init__()
        self.njoints = njoints
        self.nstacks = nstacks
        self.in_planes = 256

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        ch = self.in_planes

        hg3d2b, res, fc, _fc = [],[],[],[]
        hm3d, _hm3d = [],[]
        for i in range(nstacks):
            hg3d2b.append(Hourglass(block, nblocks, ch, depth=4))
            res.append(self._make_residual(block, nblocks, ch, ch))
            fc.append(self._make_fc(ch + 2*njoints, ch))
            hm3d.append(nn.Conv2d(ch, 2*njoints, kernel_size=1, bias=True))

        self.hg3d2b = nn.ModuleList(hg3d2b)
        self.res    = nn.ModuleList(res)
        self.fc     = nn.ModuleList(fc)
        self.hm3d   = nn.ModuleList(hm3d)

    def _make_fc(self, in_planes, out_planes):
        conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_planes)
        return nn.Sequential(conv, bn, self.relu)

    def _make_residual(self, block, nblocks, in_planes, out_planes):
        layers = []
        layers.append( block( in_planes, out_planes) )
        self.in_planes = out_planes
        for i in range(1, nblocks):
            layers.append(block( self.in_planes, out_planes))
        return nn.Sequential(*layers)

    def forward(self, enc):
        l_est_hm3d, l_enc3d = [],[]
        net = enc

        for i in range(self.nstacks):
            net = self.hg3d2b[i](net)
            net = self.res[i](net)
            hm3d = self.sigmoid(self.hm3d[i](net))
            net = torch.cat((net,hm3d),1)
            net = self.fc[i](net)
            l_est_hm3d.append(hm3d)
            l_enc3d.append(net)

        return l_est_hm3d, l_enc3d

class Hand3D(nn.Module):
    def __init__(
        self,
        nstacks=2,
        nblocks=1,
        njoints=21,
        block=BottleneckBlock
    ):
        super(Hand3D, self).__init__()
        self.hand2d = Hand2D(nstacks=nstacks, nblocks=nblocks, njoints=njoints, block=BottleneckBlock)
        self.hand2dto3d = Hand2Dto3D(nstacks=nstacks, nblocks=nblocks, njoints=njoints, block=BottleneckBlock)

    def forward(self, x):
        hm, enc = self.hand2d(x)
        hm3d, enc3d = self.hand2dto3d(enc[-1])
        uvd = []
        uvd.append(utils.hm_to_uvd(hm3d[-1]))
        hm.append(hm3d[-1][:,:21,...])

        return hm, uvd, enc, enc3d

class HandNet(nn.Module):
    def __init__(
        self,
        njoints=21,
    ):
        super(HandNet, self).__init__()
        self.njoints = njoints
        self.hand3d = Hand3D()
        self.decoder = ResNet18()

        hidden_size=[512, 512, 1024, 1024, 512, 256]
        in_neurons = 512
        out_neurons = 12
        neurons = [in_neurons] + hidden_size

        shapereg_layers = []
        for layer_idx, (inps, outs) in enumerate(zip(neurons[:-1], neurons[1:])):
            shapereg_layers.append(nn.Linear(inps, outs))
            shapereg_layers.append(nn.BatchNorm1d(outs))
            shapereg_layers.append(nn.ReLU())

        shapereg_layers.append(nn.Linear(neurons[-1], out_neurons))
        self.shapereg_layers = nn.Sequential(*shapereg_layers)
        self.sigmoid = nn.Sigmoid()
        self.iknet = IKNet()

        self.ref_bone_link = (0, 9)
        self.joint_root_idx = 9

    def forward(self, x, infos=None):
        intr = infos
        batch_size = x.shape[0]
        hm, uvd, _, enc3d = self.hand3d(x)
        feat = self.decoder(enc3d[-1])
        shape_vector = self.shapereg_layers(feat)
        bone = self.sigmoid(shape_vector[:,0:1])
        root = self.sigmoid(shape_vector[:,1:2])
        beta = shape_vector[:,2:]
        joint = utils.uvd2xyz(uvd[-1], root, bone, intr=intr, mode='persp')
        joint_root = joint[:,self.joint_root_idx,:].unsqueeze(1)
        joint_ = joint - joint_root
        bone_pred = torch.zeros((batch_size, 1)).to(x.device)
        for jid, nextjid in zip(self.ref_bone_link[:-1], self.ref_bone_link[1:]):
            bone_pred += torch.norm(
                joint_[:, jid, :] - joint_[:, nextjid, :],
                dim=1, keepdim=True
            )
        bone_pred = bone_pred.unsqueeze(1) # (B,1,1)
        bone_vis = bone_pred
        _joint_ = joint_ / bone_pred
        
        so3, quat = self.forward_ik(_joint_)

        return hm[-1], so3, beta, joint_root, bone_vis
    
    def forward_ik(self, joint):
        so3, quat = self.iknet(joint)
        return so3, quat


class HandNetInTheWild(nn.Module):
    def __init__(
        self,
        njoints=21,
    ):
        super(HandNetInTheWild, self).__init__()
        self.njoints = njoints
        self.hand3d = Hand3D()
        self.decoder = ResNet18()

        hidden_size=[512, 512, 1024, 1024, 512, 256]
        in_neurons = 512
        out_neurons = 12
        neurons = [in_neurons] + hidden_size

        shapereg_layers = []
        for layer_idx, (inps, outs) in enumerate(zip(neurons[:-1], neurons[1:])):
            shapereg_layers.append(nn.Linear(inps, outs))
            shapereg_layers.append(nn.BatchNorm1d(outs))
            shapereg_layers.append(nn.ReLU())

        shapereg_layers.append(nn.Linear(neurons[-1], out_neurons))
        self.shapereg_layers = nn.Sequential(*shapereg_layers)
        self.sigmoid = nn.Sigmoid()
        self.iknet = IKNet()

        self.ref_bone_link = (0, 9)
        self.joint_root_idx = 9

    def forward(self, x, infos=None):
        batch_size = x.shape[0]
        hm, uvd, _, enc3d = self.hand3d(x)
        feat = self.decoder(enc3d[-1])
        shape_vector = self.shapereg_layers(feat)
        bone = self.sigmoid(shape_vector[:,0:1])
        root = self.sigmoid(shape_vector[:,1:2])
        beta = shape_vector[:,2:]
        joint = uvd[-1]
        joint[:,:,2] = joint[:,:,2] * common.DEPTH_RANGE
        j1 = joint[:,0,:]
        j2 = joint[:,9,:]
        deltaj = j1 - j2
        s = torch.sqrt((deltaj[0,0]**2 + deltaj[0,1]**2)/(1 - deltaj[0,2]**2))
        joint[:,:,:2] = joint[:,:,:2]/s
        joint_root = joint[:,self.joint_root_idx,:].unsqueeze(1)
        joint_ = joint - joint_root
        bone_pred = torch.zeros((batch_size, 1)).to(x.device)
        for jid, nextjid in zip(self.ref_bone_link[:-1], self.ref_bone_link[1:]):
            bone_pred += torch.norm(
                joint_[:, jid, :] - joint_[:, nextjid, :],
                dim=1, keepdim=True
            )
        bone_pred = bone_pred.unsqueeze(1) # (B,1,1)
        bone_vis = bone_pred
        _joint_ = joint_ / bone_pred
        
        so3, quat = self.forward_ik(_joint_)

        return hm[-1], so3, beta, joint_root, bone_vis/2
    
    def forward_ik(self, joint):
        so3, quat = self.iknet(joint)
        return so3, quat
        
