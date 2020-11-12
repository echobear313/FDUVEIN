import torch
from torch import nn
from torch.nn import functional as F
import segmentation_models_pytorch as smp
from typing import Optional, Union
import numpy as np


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)


    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)

        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        set_priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=False) for stage in
                      self.stages]
        priors = set_priors + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)



class PSPUpsample(nn.Module):
    def __init__(self, x_channels, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        self.shortcut = nn.Conv2d(x_channels, out_channels, kernel_size=1)

    def forward(self, x, up):
        x = F.interpolate(input=x, scale_factor=2, mode='bilinear', align_corners=False)

        p = self.conv(torch.cat([x, up], 1))
        sc = self.shortcut(x)

        p = p + sc

        p2 = self.conv2(p)

        return p + p2


class PSPNet(smp.base.SegmentationModel):
    def __init__(
            self,
            encoder_name: str = "se_resnext50_32x4d",
            encoder_weights: Optional[str] = "imagenet",
    ):
        super().__init__()

        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=3,
            depth=3,
            weights=encoder_weights,
        )

        self.psp = PSPModule(features=512, out_features=1024, sizes=(1, 2, 3, 6))

        self.s8_head = nn.Sequential(
            nn.Conv2d(1024, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.s4_head = nn.Sequential(
            nn.Conv2d(512, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.s2_head = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.s1_head = nn.Sequential(
            nn.Conv2d(32 + 3, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.up_1 = PSPUpsample(1024, 1024 + 256, 512)
        self.up_2 = PSPUpsample(512, 512 + 64, 256)
        self.up_3 = PSPUpsample(256, 256 + 3, 32)

        self.name = "psp-{}".format(encoder_name)

    def forward(self, input):
        features = self.encoder(input)[::-1]
        x = self.psp(features[0])
        mask_s8 = self.s8_head(x)
        x = self.up_1(x, features[1])
        mask_s4 = self.s4_head(x)
        x = self.up_2(x, features[2])
        mask_s2 = self.s2_head(x)
        x = self.up_3(x, features[3])
        mask_s1 = self.s1_head(torch.cat([x, input], dim=1))

        return mask_s1, mask_s2, mask_s4, mask_s8


class SobelOperator(nn.Module):
    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

        x_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 4
        self.conv_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_x.weight.data = torch.tensor(x_kernel).unsqueeze(0).unsqueeze(0).float().cuda()
        self.conv_x.weight.requires_grad = False

        y_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) / 4
        self.conv_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_y.weight.data = torch.tensor(y_kernel).unsqueeze(0).unsqueeze(0).float().cuda()
        self.conv_y.weight.requires_grad = False

    def forward(self, x):
        b, c, h, w = x.shape
        if c > 1:
            x = x.view(b * c, 1, h, w)

        x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        grad_x = self.conv_x(x)
        grad_y = self.conv_y(x)

        x = torch.sqrt(grad_x ** 2 + grad_y ** 2 + self.epsilon)

        x = x.view(b, c, h, w)

        return x


class GradLoss(torch.nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()
        self.sobel = SobelOperator(1e-4)

    def forward(self, pr, gt):
        gt_sobel = self.sobel(gt)
        pr_sobel = self.sobel(pr)
        grad_loss = F.l1_loss(gt_sobel, pr_sobel)
        return grad_loss


class Discriminator(torch.nn.Module):
    def __init__(self, in_channels=1, in_scales=[1, 4, 8]):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels
        self.in_scales = in_scales
        self.conv_s1 = nn.Conv2d(in_channels, 64, 4, 2, 1)
        self.conv_s2 = self._make_layers(64, 128, 2)
        self.conv_s4 = self._make_layers(128, 256, 4)
        self.conv_s8 = self._make_layers(256, 512, 8)
        layers = [
            nn.Conv2d(512, 512, 4, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 1),
        ]
        self.conv = nn.Sequential(*layers)

    def _make_layers(self, in_channels, out_channels, scale):
        return nn.Sequential(
            nn.Conv2d(in_channels + (0 if scale not in self.in_scales else self.in_channels), out_channels, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )

    @classmethod
    def turn_on_spectral_norm(cls, module):
        module_output = module
        if isinstance(module, torch.nn.Conv2d):
            if module.out_channels != 1 and module.in_channels > 4:
                module_output = nn.utils.spectral_norm(module)
        for name, child in module.named_children():
            module_output.add_module(name, cls.turn_on_spectral_norm(child))
        del module
        return module_output

    def _cat(self, x, seg, scale):
        if scale in self.in_scales:
            x = torch.cat([x, seg], dim=1)
        return x

    def forward(self, inputs):
        s1, s2, s4, s8 = inputs
        x = self.conv_s1(s1)
        x = self._cat(x, s2, 2)
        x = self.conv_s2(x)
        x = self._cat(x, s4, 4)
        x = self.conv_s4(x)
        x = self._cat(x, s8, 8)
        x = self.conv_s8(x)
        x = self.conv(x)
        return x

class LSGAN(object):

    def __init__(self, dis):
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps):
        return 0.5 * (torch.mean((self.dis(real_samps) - 1) ** 2)
                      + torch.mean(self.dis(fake_samps) ** 2))

    def gen_loss(self, _, fake_samps):
        return 0.5 * ((torch.mean(self.dis(fake_samps)) - 1) ** 2)

class RelativisticAverageHingeGAN(object):

    def __init__(self, dis):
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps):
        # Obtain predictions
        r_preds = self.dis(real_samps)
        f_preds = self.dis(fake_samps)

        # difference between real and fake:
        r_f_diff = r_preds - torch.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - torch.mean(r_preds)

        # return the loss
        loss = (torch.mean(F.relu(1 - r_f_diff))
                + torch.mean(F.relu(1 + f_r_diff)))

        return loss

    def gen_loss(self, real_samps, fake_samps):
        # Obtain predictions
        r_preds = self.dis(real_samps)
        f_preds = self.dis(fake_samps)

        # difference between real and fake:
        r_f_diff = r_preds - torch.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - torch.mean(r_preds)

        # return the loss
        return (torch.mean(F.relu(1 + r_f_diff))
                + torch.mean(F.relu(1 - f_r_diff)))

if __name__ == '__main__':
    D = Discriminator(in_channels=1)
    D = Discriminator.turn_on_spectral_norm(D)
    print(D)
