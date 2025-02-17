'''
architecture for sft
'''
import torch.nn as nn
import torch.nn.functional as F


class SFTLayer(nn.Module):
    def __init__(self):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_scale_conv1 = nn.Conv2d(32, 64, 1)
        self.SFT_shift_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_shift_conv1 = nn.Conv2d(32, 64, 1)

    def forward(self, x):
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True))  # gama [16, 64, 6, 6]
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True))  # beta

        return x[0] * (scale + 1) + shift


class ResBlock_SFT(nn.Module):
    def __init__(self):
        super(ResBlock_SFT, self).__init__()
        self.sft0 = SFTLayer()
        self.conv0 = nn.Conv2d(64, 64, 3, 1, 1)
        self.sft1 = SFTLayer()
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.sft2 = SFTLayer()  # ?

    def forward(self, x):
        fea = self.sft0((x[0], x[2]))
        fea = F.relu(self.conv0(fea), inplace=True)
        fea = self.sft1((fea, x[1]))
        fea = self.conv1(fea)

        return (x[0] + fea, x[1], x[2])


class SFT_Net(nn.Module):
    def __init__(self):
        super(SFT_Net, self).__init__()
        self.conv0 = nn.Conv2d(3, 64, 3, 1, 1)

        sft_branch = []
        for i in range(16):
            sft_branch.append(ResBlock_SFT())
        sft_branch.append(SFTLayer())
        sft_branch.append(nn.Conv2d(64, 64, 3, 1, 1))
        self.sft_branch = nn.Sequential(*sft_branch)

        self.HR_branch = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.ReLU(True),
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, 1, 1)
        )

        self.seg_CondNet = nn.Sequential(
            nn.Conv2d(150, 128, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 32, 1)
        )

        self.fea1_CondNet = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 64, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 32, 1)
        )
        self.fea2_CondNet = nn.Sequential(
            nn.Conv2d(48, 48, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(48, 48, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(48, 32, 1)
        )
        self.fea3_CondNet = nn.Sequential(
            nn.Conv2d(48, 48, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(48, 48, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(48, 32, 1)
        )
        self.fea0_CondNet = nn.Sequential(
            nn.Conv2d(64, 48, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(48, 48, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(48, 32, 1)
        )
        self.fea4_CondNet = nn.Sequential(
            nn.Conv2d(48, 48, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(48, 48, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(48, 32, 1)
        )

    # mfea model
    def forward(self, x):
        fea = self.conv0(x[0])
        last_fea = fea
        fea1 = self.fea1_CondNet(x[1][1])
        fea2 = self.fea2_CondNet(x[1][2])
        fea3 = self.fea3_CondNet(x[1][3])
        fea4 = self.fea4_CondNet(x[1][4])

        '''Please comment this out during the testing phase，and uncomment during the training phase'''

        # fea1 = nn.functional.interpolate(fea1, scale_factor=4, mode='bilinear', align_corners=False)
        # fea2 = nn.functional.interpolate(fea2, scale_factor=4, mode='bilinear', align_corners=False)
        # fea3 = nn.functional.interpolate(fea3, scale_factor=4, mode='bilinear', align_corners=False)
        # fea4 = nn.functional.interpolate(fea4, scale_factor=4, mode='bilinear', align_corners=False)

        for m in self.sft_branch[0:4]:
            (fea, fea1, fea4) = m((fea, fea1, fea4))
        for m in self.sft_branch[4:8]:
            (fea, fea2, fea4) = m((fea, fea2, fea4))
        for m in self.sft_branch[8:12]:
            (fea, fea3, fea4) = m((fea, fea3, fea4))
        for m in self.sft_branch[12:16]:
            (fea, fea4, fea4) = m((fea, fea4, fea4))

        fea = self.sft_branch[16]((fea, fea4))
        fea = self.sft_branch[17](fea)
        fea = fea + last_fea
        out = self.HR_branch(fea)
        return out


# Auxiliary Classifier Discriminator
class ACD_VGG_BN_96(nn.Module):
    def __init__(self):
        super(ACD_VGG_BN_96, self).__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(256, 256, 4, 2, 1),
            nn.BatchNorm2d(256, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512, affine=True),
            nn.LeakyReLU(0.1, True),
        )

        # gan
        self.gan = nn.Sequential(
            nn.Linear(512 * 6 * 6, 100),
            nn.LeakyReLU(0.1, True),
            nn.Linear(100, 1)
        )

        self.cls = nn.Sequential(
            nn.Linear(512 * 6 * 6, 100),
            nn.LeakyReLU(0.1, True),
            nn.Linear(100, 8)
        )

    def forward(self, x):
        fea = self.feature(x)
        fea = fea.view(fea.size(0), -1)
        gan = self.gan(fea)
        cls = self.cls(fea)
        return [gan, cls]


#############################################
# below is the sft arch for the torch version
#############################################


class SFTLayer_torch(nn.Module):
    def __init__(self):
        super(SFTLayer_torch, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_scale_conv1 = nn.Conv2d(32, 64, 1)
        self.SFT_shift_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_shift_conv1 = nn.Conv2d(32, 64, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.01, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.01, inplace=True))
        return x[0] * scale + shift


class ResBlock_SFT_torch(nn.Module):
    def __init__(self):
        super(ResBlock_SFT_torch, self).__init__()
        self.sft0 = SFTLayer_torch()
        self.conv0 = nn.Conv2d(64, 64, 3, 1, 1)
        self.sft1 = SFTLayer_torch()
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        fea = F.relu(self.sft0(x), inplace=True)
        fea = self.conv0(fea)
        fea = F.relu(self.sft1((fea, x[1])), inplace=True)
        fea = self.conv1(fea)
        return (x[0] + fea, x[1])  # return a tuple containing features and conditions


class SFT_Net_torch(nn.Module):
    def __init__(self):
        super(SFT_Net_torch, self).__init__()
        self.conv0 = nn.Conv2d(3, 64, 3, 1, 1)

        sft_branch = []
        for i in range(16):
            sft_branch.append(ResBlock_SFT_torch())
        sft_branch.append(SFTLayer_torch())
        sft_branch.append(nn.Conv2d(64, 64, 3, 1, 1))
        self.sft_branch = nn.Sequential(*sft_branch)

        self.HR_branch = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, 1, 1)
        )

        # Condtion network
        self.CondNet = nn.Sequential(
            nn.Conv2d(8, 128, 4, 4),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 32, 1)
        )

    def forward(self, x):
        # x[0]: img; x[1]: seg
        cond = self.CondNet(x[1])
        fea = self.conv0(x[0])
        res = self.sft_branch((fea, cond))
        fea = fea + res
        out = self.HR_branch(fea)
        return out


if __name__ == '__main__':

    m = SFT_Net()

    for k, v in m.named_parameters():
        if 'SFT' in k or 'Cond' in k:
            print(k)
        else:
            print("###", k)
    print(type(m.sft_branch[17]))
    for i in m.sft_branch[0:20]:
        print(type(i))
