import torch
import torch.nn as nn
from Vgg19 import Vgg19

# -------------------------------
# FD. load_vgg19_weights()
def load_vgg19_weights(path="./vgg19-dcbb9e9d.pth"):
    '''Load VGG19 weights from a specified path and freeze its parameters.
    Args:
        path (str): Path to the VGG19 weights file.'''
    vgg19 = Vgg19()
    vgg19.load_state_dict(torch.load(path))
    for param in vgg19.features.parameters():
        param.requires_grad = False
    return vgg19

# -------------------------------
# CD. DoubleConv()
class DoubleConv(nn.Module):
    '''A module that performs two consecutive convolutional operations with ReLU activation.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        padding (int, optional): Padding for the convolution. Defaults to 1.'''
        
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv_op(x)

# -------------------------------
# CD. DownSampleVGG()
class DownSampleVGG(nn.Module):
    '''A module that downsamples the input using VGG19 features and a max pooling operation.
    Args:
        range_from_VGG (tuple): A tuple indicating the range of VGG19 layers to use.
        vgg19 (nn.Module): An instance of the VGG19 model.
        pool_kernel_size (int, optional): Kernel size for the max pooling operation. Defaults to 2.
        pool_stride (int, optional): Stride for the max pooling operation. Defaults to 2.'''
        
    def __init__(self, range_from_VGG, vgg19, pool_kernel_size=2, pool_stride=2):
        super().__init__()
        low, high = range_from_VGG
        self.conv = nn.Sequential(*list(vgg19.features.children())[low:high])
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)
        return down, p

# -------------------------------
# CD. UpSample()
class UpSample(nn.Module):
    '''A module that upsamples the input using a transposed convolution and concatenates it with a skip connection.
    Args:
        in_channels_up (int): Number of input channels for the upsampled feature map.
        out_channels_up (int): Number of output channels for the upsampled feature map.
        reduce_channels_in_half (bool, optional): Whether to reduce the number of channels in half. Defaults to True.
        use_corrective_conv (bool, optional): Whether to use a corrective convolution. Defaults to False.
        in_channels_conv (int, optional): Number of input channels for the convolution. Defaults to None.
        out_channels_conv (int, optional): Number of output channels for the convolution. Defaults to None.'''
    
    def __init__(self, in_channels_up, out_channels_up, reduce_channels_in_half=True, use_corrective_conv=False, in_channels_conv=None, out_channels_conv=None):
        super().__init__()
        if reduce_channels_in_half:
            self.up = nn.ConvTranspose2d(in_channels_up, in_channels_up // 2, kernel_size=2, stride=2)
        else:
            self.up = nn.ConvTranspose2d(in_channels_up, in_channels_up, kernel_size=2, stride=2)

        self.in_channels_conv = in_channels_up if not use_corrective_conv else in_channels_conv
        self.out_channels_conv = out_channels_up if not use_corrective_conv else out_channels_conv
        self.conv = DoubleConv(in_channels=self.in_channels_conv, out_channels=self.out_channels_conv)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

# -----------------------
# CD. UNetVgg19()
# A complete UNet architecture that uses VGG19 as the encoder backbone.
# -----------------------
class UNetVgg19(nn.Module):
    '''A UNet architecture that uses VGG19 as the encoder backbone.
    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        vgg19 (nn.Module): An instance of the VGG19 model with pre-loaded weights.'''
        
    def __init__(self, in_channels=3, num_classes=1, vgg19=None):
        super().__init__()
        assert vgg19 is not None, "A pre-loaded VGG19 instance is required."

        self.downsample1 = DownSampleVGG((0, 4), vgg19)
        self.downsample2 = DownSampleVGG((5, 9), vgg19)
        self.downsample3 = DownSampleVGG((10, 18), vgg19)
        self.downsample4 = DownSampleVGG((19, 27), vgg19)

        self.bottleneck = DoubleConv(512, 1024)

        self.upsample1 = UpSample(1024, 512)
        self.upsample2 = UpSample(512, 256)
        self.upsample3 = UpSample(256, 128)
        self.upsample4 = UpSample(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        down1, p1 = self.downsample1(x)
        down2, p2 = self.downsample2(p1)
        down3, p3 = self.downsample3(p2)
        down4, p4 = self.downsample4(p3)

        b = self.bottleneck(p4)

        up1 = self.upsample1(b, down4)
        up2 = self.upsample2(up1, down3)
        up3 = self.upsample3(up2, down2)
        up4 = self.upsample4(up3, down1)

        return self.out(up4)
