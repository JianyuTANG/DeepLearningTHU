from torchvision import models
import torch.nn as nn
import torch


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.bn2(output)
        identity = self.shortcut(input)
        output = self.relu(output + identity)
        return output


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.start = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.block1 = self._make_layer(in_planes=64, out_planes=64, stride=2)
        self.block2 = self._make_layer(in_planes=64, out_planes=128, stride=2)
        self.block3 = self._make_layer(in_planes=128, out_planes=256, stride=2)
        self.block4 = self._make_layer(in_planes=256, out_planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_planes, out_planes, stride):
        layer1 = ResidualBlock(in_planes, out_planes, stride)
        layer2 = ResidualBlock(out_planes, out_planes, stride=1)
        return nn.Sequential(layer1, layer2)

    def forward(self, input):
        output = self.start(input)
        output = self.block1(output)
        output = self.block2(output)
        output = self.block3(output)
        output = self.block4(output)
        output = self.avgpool(output)
        output = torch.flatten(output, 1)
        output = self.fc(output)
        return output
    
    def get_feature(self, input):
        output = self.start(input)
        output = self.block1(output)
        output = self.block2(output)
        output = self.block3(output)
        output = self.block4(output)
        output = self.avgpool(output)
        return torch.flatten(output, 1)


def model_A(num_classes, pretrained=True):
    model_resnet = models.resnet18(pretrained=pretrained)
    num_features = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_features, num_classes)
    return model_resnet


def model_B(num_classes, pretrained = False):
    # model_resnet = models.resnet18(pretrained=pretrained)
    # num_features = model_resnet.fc.in_features
    # model_resnet.fc = nn.Linear(num_features, num_classes)
    # return model_resnet
    return ResNet18(num_classes)


class BottleNeck(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(64, out_planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(out_planes)

    def forward(self, x, idt=None):
        x = self.conv1(x)
        
        if idt is not None:
            x = x + idt
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        idt = x
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        return x, idt


class ModelC(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.start = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=3, dilation=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.block1 = BottleNeck(64, 128)
        self.block2 = BottleNeck(128, 128)
        self.block3 = BottleNeck(128, 256)
        self.block4 = BottleNeck(256, 256)
        self.block5 = BottleNeck(256, 512)
        self.block6 = BottleNeck(512, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.start(x)
        x, idt = self.block1(x)
        x, idt = self.block2(x, idt)
        x, idt = self.block3(x, idt)
        x, idt = self.block4(x, idt)
        x, idt = self.block5(x, idt)
        x, _ = self.block6(x, idt)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def get_feature(self, x):
        x = self.start(x)
        x, idt = self.block1(x)
        x, idt = self.block2(x, idt)
        x, idt = self.block3(x, idt)
        x, idt = self.block4(x, idt)
        x, idt = self.block5(x, idt)
        x, _ = self.block6(x, idt)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


def model_C(num_classes, pretrained = False):
    return ModelC(num_classes)
