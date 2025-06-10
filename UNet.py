# The Unet architecture is based on three main subarchitectures:
# downsampling
# bottleneck
# upsampling

# The downsampling process is divided in 4 blocks. A block contains
# two convolutions and Maxpooling.
# The bottleneck results from a convolution
# The upsampling is also divided in 4 blocks, which undo what the convolutions
# did during downsampling. Each upsampling block receives a skip connection from their
# downsampling counterpart, then applies two Deconvolutions
# Output is produced by a final deconvolution with a specified number of channels


######################## MODULES ########################
from constants import *

######################## DD #############################

# DD. DOUBLE_CONVOLUTION (DOWNSAMPLING BLOCK ELEMENT)
# conv = DoubleConv()
# interp. aubarchitecture to one DOWNSAMPLING block, in charge of extracting features
class DoubleConv(nn.Module):
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        return self.conv_op(x)

# DD. DOWNSAMPLE_BLOCK
# donwsample = DownSample()
# interp. an DOWNSAMPLE block represented by a DOUBLE_CONVOLUTION and its MaxPooling operation
class DownSample(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels=in_channels,out_channels=out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

    def forward(self,x):
        down = self.conv(x)
        p = self.pool(down)
        return down,p

# DD. UPSAMPLE_DECONVOLUTION_BLOCK (UPSAMPLING BLOCK)
# upsample = UpSample()
# interp. an UPSAMPLE block represented by:
# - a "deconvolution" (ConvTranspose2d)
# - a concatenation with a parallel version from the encoder
# - Double convolution
class UpSample(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2,kernel_size=2,stride = 2)
        self.conv = DoubleConv(in_channels=in_channels,out_channels=out_channels)

    def forward(self,x1,x2):
        x1 = self.up(x1)
        x = torch.cat([x1,x2],1)
        return self.conv(x)

# DD. Unet()
# unet = pytorch.nn.Module()
# interp. a complete architecture to represent a UNet model with
# - at least 4 DownSample()
# - at least 4 UpSample()
class UNet(nn.Module):
    def __init__(self,in_channels, num_classes):
        super().__init__()
        self.downsample1 = DownSample(in_channels,64)
        self.downsample2 = DownSample(64,128)
        self.downsample3 = DownSample(128,256)
        self.downsample4 = DownSample(256,512)

        self.bottle_neck = DoubleConv(512,1024)

        self.upsample1 = UpSample(1024,512)
        self.upsample2 = UpSample(512,256)
        self.upsample3 = UpSample(256,128)
        self.upsample4 = UpSample(128,64)

        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes,kernel_size=1)

    def forward(self,x):
        down1, p1 = self.downsample1(x)
        down2, p2 = self.downsample2(p1)
        down3, p3 = self.downsample3(p2)
        down4, p4 = self.downsample4(p3)

        b = self.bottle_neck(p4)

        up1 = self.upsample1(b,down4)
        # print(up1.shape)
        up2 = self.upsample2(up1,down3)
        # print(up2.shape)
        up3 = self.upsample3(up2,down2)
        # print(up3.shape)
        up4 = self.upsample4(up3,down1)
        # print(up4.shape)

        out = self.out(up4)
        return out
######################## CODE ###########################


if __name__ == "__main__":
    model = UNet(3,10)