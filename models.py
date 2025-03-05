import torch
from torch import nn
from torch.nn import functional as F

class Res15(nn.Module):
    def __init__(self, n_maps):
        super().__init__()
        self.n_maps = n_maps
        self.dilation = True
        self.n_layers = 13
        self.conv0 = nn.Conv2d(in_channels = 1, out_channels = n_maps, kernel_size = 3, padding=1, dilation=1,bias=False)
        self.conv1 = nn.Conv2d(in_channels = 45, out_channels = n_maps, kernel_size = 3, padding=1, dilation=1,bias=False)
        self.bn1 = nn.BatchNorm2d(n_maps, affine=False)
        self.conv2 = nn.Conv2d(in_channels = 45, out_channels = n_maps, kernel_size = 3, padding=1, dilation=1,bias=False)
        self.bn2 = nn.BatchNorm2d(n_maps, affine=False)
        self.conv3 = nn.Conv2d(in_channels = 45, out_channels = n_maps, kernel_size = 3, padding=2, dilation=2,bias=False)
        self.bn3 = nn.BatchNorm2d(n_maps, affine=False)
        self.conv4 = nn.Conv2d(in_channels = 45, out_channels = n_maps, kernel_size = 3, padding=2, dilation=2,bias=False)
        self.bn4 = nn.BatchNorm2d(n_maps, affine=False)
        self.conv5 = nn.Conv2d(in_channels = 45, out_channels = n_maps, kernel_size = 3, padding=2, dilation=2,bias=False)
        self.bn5 = nn.BatchNorm2d(n_maps, affine=False)
        self.conv6 = nn.Conv2d(in_channels = 45, out_channels = n_maps, kernel_size = 3, padding=4, dilation=4,bias=False)
        self.bn6 = nn.BatchNorm2d(n_maps, affine=False)
        self.conv7 = nn.Conv2d(in_channels = 45, out_channels = n_maps, kernel_size = 3, padding=4, dilation=4,bias=False)
        self.bn7 = nn.BatchNorm2d(n_maps, affine=False)
        self.conv8 = nn.Conv2d(in_channels = 45, out_channels = n_maps, kernel_size = 3, padding=4, dilation=4,bias=False)
        self.bn8 = nn.BatchNorm2d(n_maps, affine=False)
        self.conv9 = nn.Conv2d(in_channels = 45, out_channels = n_maps, kernel_size = 3, padding=8, dilation=8,bias=False)
        self.bn9 = nn.BatchNorm2d(n_maps, affine=False)
        self.conv10 = nn.Conv2d(in_channels = 45, out_channels = n_maps, kernel_size = 3, padding=8, dilation=8,bias=False)
        self.bn10 = nn.BatchNorm2d(n_maps, affine=False)
        self.conv11 = nn.Conv2d(in_channels = 45, out_channels = n_maps, kernel_size = 3, padding=8, dilation=8,bias=False)
        self.bn11 = nn.BatchNorm2d(n_maps, affine=False)
        self.conv12 = nn.Conv2d(in_channels = 45, out_channels = n_maps, kernel_size = 3, padding=16, dilation=16,bias=False)
        self.bn12 = nn.BatchNorm2d(n_maps, affine=False)
        self.conv13 = nn.Conv2d(in_channels = 45, out_channels = n_maps, kernel_size = 3, padding=16, dilation=16,bias=False)
        self.bn13 = nn.BatchNorm2d(n_maps, affine=False)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv0(x))
        old_x = x
        # print(x.shape)
        x = self.relu(self.conv2(self.bn1(self.relu(self.conv1(x))))) + old_x
        old_x = x
        # print(x.shape)
        x = self.bn2(x)
        x = self.relu(self.conv4(self.bn3(self.relu(self.conv3(x))))) + old_x
        old_x = x
        # print(x.shape)
        x = self.bn4(x)
        x = self.relu(self.conv6(self.bn5(self.relu(self.conv5(x))))) + old_x
        old_x = x
        # print(x.shape)
        x = self.bn6(x)
        x = self.relu(self.conv8(self.bn7(self.relu(self.conv7(x))))) + old_x
        old_x = x
        # print(x.shape)
        x = self.bn8(x)
        x = self.relu(self.conv10(self.bn9(self.relu(self.conv9(x))))) + old_x
        old_x = x
        # print(x.shape)
        x = self.bn10(x)
        x = self.relu(self.conv12(self.bn11(self.relu(self.conv11(x))))) + old_x
        old_x = x
        # print(x.shape)
        x = self.bn12(x)
        x = self.bn13(self.relu(self.conv13(x)))
        # print(x.shape)
        x = x.view(x.size(0), x.size(1), -1)  # shape: (batch, feats, o3)
        x = torch.mean(x, 2)
        x = x.unsqueeze(-2)
        return x
