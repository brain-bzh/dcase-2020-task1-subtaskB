import torch.nn as nn
import torch.nn.functional as F
import torch



##################################
"""
VGG models A and B
"""
##################################

class ModelA(nn.Module):
    """
    VGG 4 layers
    """
    def __init__(self):
        super(ModelA, self).__init__()
        power = 4

        self.conv1 = nn.Conv1d(2, 2**power, 64, 4)
        self.bn1 = nn.BatchNorm1d(2**power)
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = nn.Conv1d(2**power, 2**power,4)
        self.bn2 = nn.BatchNorm1d(2**power)
        self.pool2 = nn.MaxPool1d(4)

        self.conv3 = nn.Conv1d(2**power, 2**(power+1), 4)
        self.bn3 = nn.BatchNorm1d(2**(power+1))
        self.pool3 = nn.MaxPool1d(4)

        self.conv4 = nn.Conv1d(2**(power+1),2**(power+2),4)
        self.bn4 = nn.BatchNorm1d(2**(power+2))
        self.pool4 = nn.MaxPool1d(4)

        self.fc1 = nn.Linear(2**(power+2), 3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)

        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)

        x = F.avg_pool1d(x,x.data.size()[2])
        x = x.view(x.data.size()[0],x.data.size()[2],x.data.size()[1])

        x = self.fc1(x)

        x = x.view(x.data.size()[0],x.data.size()[2])

        return x


class ModelB(nn.Module):
    """
    VGG 5 layers
    """
    def __init__(self):
        super(ModelB, self).__init__()
        power = 4

        self.conv1 = nn.Conv1d(2, 2**power, 64, 4)
        self.bn1 = nn.BatchNorm1d(2**power)
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = nn.Conv1d(2**power, 2**power,4)
        self.bn2 = nn.BatchNorm1d(2**power)
        self.pool2 = nn.MaxPool1d(4)

        self.conv3 = nn.Conv1d(2**power, 2**(power+1), 4)
        self.bn3 = nn.BatchNorm1d(2**(power+1))
        self.pool3 = nn.MaxPool1d(4)

        self.conv4 = nn.Conv1d(2**(power+1),2**(power+2),4)
        self.bn4 = nn.BatchNorm1d(2**(power+2))
        self.pool4 = nn.MaxPool1d(4)

        self.conv5 = nn.Conv1d(2**(power+2),2**(power+2),4)
        self.bn5 = nn.BatchNorm1d(2**(power+2))
        self.pool5 = nn.MaxPool1d(4)

        self.fc1 = nn.Linear(2**(power+2), 3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)

        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)

        x = self.conv5(x)
        x = F.relu(self.bn5(x))
        x = self.pool5(x)

        x = F.avg_pool1d(x,x.data.size()[2])
        x = x.view(x.data.size()[0],x.data.size()[2],x.data.size()[1])

        x = self.fc1(x)

        x = x.view(x.data.size()[0],x.data.size()[2])

        return x


##################################
"""
ResNet models C and D
"""
##################################

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)

        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 5

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)

        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.conv3 = nn.Conv1d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=3):
        super(ResNet, self).__init__()
        power = 5
        self.in_planes = 2**power

        self.conv1 = nn.Conv1d(2, 2**power, kernel_size=64, stride=4, padding=0, bias=False)
        self.bn1 = nn.BatchNorm1d(2**power)

        self.layer1 = self._make_layer(block, 2**power, num_blocks[0], stride=4)
        self.layer2 = self._make_layer(block, 2**(power+1), num_blocks[1], stride=4)
        self.layer3 = self._make_layer(block, 2**(power+2), num_blocks[2], stride=4)

        self.linear = nn.Linear(2**(power+2)*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = F.avg_pool1d(x,x.data.size()[2])
        x = x.view(x.data.size()[0],x.data.size()[2],x.data.size()[1])

        x = self.linear(x)

        x = x.view(x.data.size()[0],x.data.size()[2])

        return x


def ModelC():
    return ResNet(BasicBlock, [3,4,3])

def ModelD():
    return ResNet(BasicBlock, [3,3,3])


if __name__ == '__main__':
    
    print('\n\n#######################')
    print("Model A - VGG-4layers")
    print('#######################\n')
    modelA = ModelA()
    print(modelA)

    print('\n\n#######################')
    print("Model B - VGG-5layers")
    print('#######################\n')
    modelB = ModelB()
    print(modelB)

    print('\n\n#######################')
    print("Model C - Resnet343")
    print('#######################\n')
    modelC = ModelC()
    print(modelC)

    print('\n\n#######################')
    print("Model D - Resnet333")
    print('#######################\n')
    modelD = ModelD()
    print(modelD)
