import torch
import torch.nn as nn
import torch.nn.functional as F

class DarkCovidNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.cblock1 = self.conv_block(3, 8)
        self.mp1 = self.maxpooling()
        self.cblock2 = self.conv_block(8, 16)
        self.mp2 = self.maxpooling()
        self.cblock3 = self.triple_conv(16, 32)
        self.mp3 = self.maxpooling()
        self.cblock4 = self.triple_conv(32, 64)
        self.mp4 = self.maxpooling()
        self.cblock5 = self.triple_conv(64, 128)
        self.mp5 = self.maxpooling()
        self.cblock6 = self.triple_conv(128, 256)
        self.cblock7 = self.conv_block(256, 128, size=1)
        self.cblock8 = self.conv_block(128, 256)
        self.final_conv = nn.Conv2d(256, 2, kernel_size=3, stride=1, bias=False)
        self.final_linear = nn.Linear(242, 2)


    def conv_block(self, ni, nf, size=3, stride=1):
        for_pad = lambda s: s if s > 2 else 3
        return nn.Sequential(
            nn.Conv2d(ni, nf, kernel_size=size, stride=stride, padding=(for_pad(size) - 1)//2, bias=False),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def triple_conv(self, ni, nf):
        return nn.Sequential(
            self.conv_block(ni, nf),
            self.conv_block(nf, ni, size=1),
            self.conv_block(ni, nf)
        )

    def maxpooling(self):
        return nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        out = self.cblock1(x)
        out = self.mp1(out)
        out = self.cblock2(out)
        out = self.mp2(out)
        out = self.cblock3(out)
        out = self.mp3(out)
        out = self.cblock4(out)
        out = self.mp4(out)
        out = self.cblock5(out)
        out = self.mp5(out)
        out = self.cblock6(out)
        out = self.cblock7(out)
        out = self.cblock8(out)
        out = self.final_conv(out)
        out = out.view(out.size(0), -1)
        out = self.final_linear(out)
        return out
