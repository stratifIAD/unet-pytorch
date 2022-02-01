import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, inchannels, outchannels, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inchannels, outchannels,
                               kernel_size=3, padding=padding)
        self.conv2 = nn.Conv2d(outchannels, outchannels,
                               kernel_size=3, padding=padding)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class ConvNet(nn.Module):
    def __init__(self, inchannels, outchannels, net_depth=5, intial_features=16):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.downblocks = nn.ModuleList()

        in_channels = inchannels
        out_channels = intial_features
        for _ in range(net_depth):
            conv = ConvBlock(in_channels, out_channels)
            self.downblocks.append(conv)
            in_channels, out_channels = out_channels, 2 * out_channels

        # last block
        self.last_conv = nn.Conv2d(in_channels, outchannels, kernel_size=1)

    def forward(self, x):
        for op in self.downblocks:
            x = self.pool(op(x))

        x = self.last_conv(x)
        # TODO remove and make the dataloader reflects channels
        x = x.squeeze(dim=1)

        return x