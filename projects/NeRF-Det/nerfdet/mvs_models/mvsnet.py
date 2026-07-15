import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.inplanes = 32

        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)

        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)

        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)
        self.feature = nn.Conv2d(32, 32, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(self.conv0(x))
        x = self.conv4(self.conv3(self.conv2(x)))
        x = self.feature(self.conv6(self.conv5(x)))
        return x


class CostRegNet(nn.Module):
    def __init__(self):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(32, 8)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        conv0 = self.conv0(x) # (b, 8, d, h, w)
        conv2 = self.conv2(self.conv1(conv0)) # (b, 16, d/2, h/2, w/2)
        conv4 = self.conv4(self.conv3(conv2)) # (b, 32, d/4, h/4, w/4)
        x = self.conv6(self.conv5(conv4)) # (b, 64, d/8, h/8, w/8)
        x = conv4 + self.conv7(x) # (b, 32, d/4, h/4, w/4)
        x = conv2 + self.conv9(x) # (b, 16, d/2, h/2, w/2)
        x = conv0 + self.conv11(x) # (b, 8, d, h, w)
        x = self.prob(x) # (b, 1, d, h, w)
        return x


class CostRegNet_3DGS(nn.Module): # 3d volume. 3d u-net
    def __init__(self):
        super(CostRegNet_3DGS, self).__init__()
        self.conv0 = ConvBnReLU3D(256, 64)

        self.conv1 = ConvBnReLU3D(64, 128, stride=2)
        self.conv2 = ConvBnReLU3D(128, 128)

        self.conv3 = ConvBnReLU3D(128, 256, stride=2)
        self.conv4 = ConvBnReLU3D(256, 256)

        # self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        # self.conv6 = ConvBnReLU3D(64, 64)

        # self.conv7 = nn.Sequential(
        #     nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
        #     nn.BatchNorm3d(32),
        #     nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(64, 2, 3, stride=1, padding=1)

    def forward(self, x):
        conv0 = self.conv0(x) # (b, 64, d, h, w)
        conv2 = self.conv2(self.conv1(conv0)) # (b, 128, d/2, h/2, w/2)
        x = self.conv4(self.conv3(conv2)) # (b, 256, d/4, h/4, w/4)
        # x = self.conv6(self.conv5(conv4)) # (b, 64, d/8, h/8, w/8)
        # x = conv4 + self.conv7(x) # (b, 32, d/4, h/4, w/4)
        x = conv2 + self.conv9(x) # (b, 128, d/2, h/2, w/2)
        x = conv0 + self.conv11(x) # (b, 64, d, h, w)
        x = self.prob(x) # (b, 2, d, h, w)
        return x
    

class CostRegNet_3DGroupNorm(nn.Module): # 3d volume. 3d u-net
    def __init__(self):
        super(CostRegNet_3DGroupNorm, self).__init__()
        self.conv0 = ConvBnReLU3D(8, 8)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)

        # self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        # self.conv6 = ConvBnReLU3D(64, 64)

        # self.conv7 = nn.Sequential(
        #     nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
        #     nn.BatchNorm3d(32),
        #     nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        conv0 = self.conv0(x) # (b, 64, d, h, w)
        conv2 = self.conv2(self.conv1(conv0)) # (b, 128, d/2, h/2, w/2)
        x = self.conv4(self.conv3(conv2)) # (b, 256, d/4, h/4, w/4)
        # x = self.conv6(self.conv5(conv4)) # (b, 64, d/8, h/8, w/8)
        # x = conv4 + self.conv7(x) # (b, 32, d/4, h/4, w/4)
        x = conv2 + self.conv9(x) # (b, 128, d/2, h/2, w/2)
        x = conv0 + self.conv11(x) # (b, 64, d, h, w)
        x = self.prob(x) # (b, 2, d, h, w)
        return x

class CostRegNet_3DMLP(nn.Module): # 3d volume. 3d u-net
    def __init__(self):
        super(CostRegNet_3DMLP, self).__init__()
        self.conv0 = ConvBnReLU3D(8, 4)
        # self.conv2 = ConvBnReLU3D(4, 1)
        self.prob = nn.Conv3d(4, 1, 1, stride=1, padding=0)

    def forward(self, x):
        x = self.prob(self.conv0(x))
        return x
    

class CostRegNet_2DGS(nn.Module):
    def __init__(self, in_channel):
        super(CostRegNet_2DGS, self).__init__()
        # self.conv0 = ConvBnReLU2D(in_channel, 64)

        self.conv1 = ConvBnReLU2D(in_channel, 2*in_channel, stride=2)
        self.conv2 = ConvBnReLU2D(2*in_channel, 2*in_channel)

        self.conv3 = ConvBnReLU2D(2*in_channel, 4*in_channel, stride=2)
        self.conv4 = ConvBnReLU2D(4*in_channel, 4*in_channel)

        # self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        # self.conv6 = ConvBnReLU3D(64, 64)

        # self.conv7 = nn.Sequential(
        #     nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
        #     nn.BatchNorm3d(32),
        #     nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose2d(4*in_channel, 2*in_channel, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(2*in_channel),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose2d(2*in_channel, in_channel, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv2d(in_channel, 2*in_channel, 3, stride=1, padding=1)

    def forward(self, x):
        # conv0 = self.conv0(x) # (b, d, h, w)
        conv0 = x
        conv2 = self.conv2(self.conv1(conv0)) # (b, 2*d, h/2, w/2)
        x = self.conv4(self.conv3(conv2)) # (b, 4*d, h/4, w/4)
        # x = self.conv6(self.conv5(conv4)) # (b, 64, d/8, h/8, w/8)
        # x = conv4 + self.conv7(x) # (b, 32, d/4, h/4, w/4)
        x = conv2 + self.conv9(x) # (b, 2*d, h/2, w/2)
        x = conv0 + self.conv11(x) # (b, d, h, w)
        x = self.prob(x) # (b, 2*d, h, w)
        return x

class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv1 = ConvBnReLU(4, 32)
        self.conv2 = ConvBnReLU(32, 32)
        self.conv3 = ConvBnReLU(32, 32)
        self.res = ConvBnReLU(32, 1)

    def forward(self, img, depth_init):
        concat = F.cat((img, depth_init), dim=1)
        depth_residual = self.res(self.conv3(self.conv2(self.conv1(concat))))
        depth_refined = depth_init + depth_residual
        return depth_refined


class MVSNet(nn.Module):
    def __init__(self, refine=True):
        super(MVSNet, self).__init__()
        self.refine = refine

        self.feature = FeatureNet()
        self.cost_regularization = CostRegNet()
        if self.refine:
            self.refine_network = RefineNet()

    def forward(self, imgs, proj_matrices, depth_values):
        imgs = torch.unbind(imgs, 1)
        proj_matrices = torch.unbind(proj_matrices, 1) # [(b, 4, 4), (b, 4, 4), (b, 4, 4)]. 
        assert len(imgs) == len(proj_matrices), "Different number of images and projection matrices"
        img_height, img_width = imgs[0].shape[2], imgs[0].shape[3]
        num_depth = depth_values.shape[1] # depth_values: (b, n_depth).
        num_views = len(imgs)

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        features = [self.feature(img) for img in imgs]
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1) # (b, c, num_depth, h, w)
        volume_sum = ref_volume # (b, c, num_depth, h, w)
        volume_sq_sum = ref_volume ** 2
        del ref_volume
        for src_fea, src_proj in zip(src_features, src_projs):
            # warpped features
            warped_volume = homo_warping(src_fea, src_proj, ref_proj, depth_values)
            if self.training:
                volume_sum = volume_sum + warped_volume # (b, c, num_depth, h, w)
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)  # the memory of warped_volume has been modified
            del warped_volume
        # aggregate multiple feature volumes by variance
        volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2)) # (b, c, num_depth, h, w)

        # step 3. cost volume regularization
        cost_reg = self.cost_regularization(volume_variance) # (b, 1, n_depth, h, w)
        # cost_reg = F.upsample(cost_reg, [num_depth * 4, img_height, img_width], mode='trilinear')
        cost_reg = cost_reg.squeeze(1) # (b, n_depth, h, w)
        prob_volume = F.softmax(cost_reg, dim=1)
        depth = depth_regression(prob_volume, depth_values=depth_values) # (b, h, w)

        with torch.no_grad():
            # photometric confidence
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
            depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float)).long()
            photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)

        # step 4. depth map refinement
        if not self.refine:
            return {"depth": depth, "photometric_confidence": photometric_confidence}
        else:
            refined_depth = self.refine_network(torch.cat((imgs[0], depth), 1))
            return {"depth": depth, "refined_depth": refined_depth, "photometric_confidence": photometric_confidence}


def mvsnet_loss(depth_est, depth_gt, mask):
    mask = mask > 0.5
    return F.smooth_l1_loss(depth_est[mask], depth_gt[mask], size_average=True)
