import torch
import torch.nn as nn
import torch.nn.functional as F
import antialiased_cnns


def drop_path(x, drop_prob=0.0, training=False):
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class BasicBlock(nn.Module):
    """Original ResNet Basic Block with Stochastic Depth"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, drop_prob=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.drop_prob = drop_prob

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = drop_path(out, self.drop_prob, self.training)
        out += identity
        out = F.relu(out, inplace=True)

        return out


class BottleneckBlock(nn.Module):
    """Original ResNet Bottleneck Block with Stochastic Depth"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, drop_prob=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.downsample = downsample
        self.drop_prob = drop_prob

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = drop_path(out, self.drop_prob, self.training)
        out += identity
        out = F.relu(out, inplace=True)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, drop_path_rate=0.2, use_blurpool=False):
        super().__init__()
        self.in_channels = 64
        self.use_blurpool = use_blurpool

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        if self.use_blurpool:
            self.maxpool_or_blurpool = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1), antialiased_cnns.BlurPool(64, stride=2))
        else:
            self.maxpool_or_blurpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        # Calculate total number of blocks
        total_blocks = sum(layers)
        # Linear drop path rate schedule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # Track current block index
        block_idx = 0

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1,
                                       drop_probs=dpr[block_idx:block_idx+layers[0]])
        block_idx += layers[0]

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       drop_probs=dpr[block_idx:block_idx+layers[1]])
        block_idx += layers[1]

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       drop_probs=dpr[block_idx:block_idx+layers[2]])
        block_idx += layers[2]

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       drop_probs=dpr[block_idx:block_idx+layers[3]])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=1)

    def _make_layer(self, block, out_channels, blocks, stride, drop_probs):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, drop_probs[0]))
        self.in_channels = out_channels * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, drop_prob=drop_probs[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.maxpool_or_blurpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)
        x = torch.flatten(x, 1)

        return x

def resnet50(num_classes=1000, drop_path_rate=0.2, use_blurpool=False):
    return ResNet(BottleneckBlock, [3, 4, 6, 3], num_classes, drop_path_rate, use_blurpool)