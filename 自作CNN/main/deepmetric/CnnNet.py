#2 畳み込みニューラルネットワークを定義する。
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

#vgg型のモデルのcnn
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name,class_num):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, class_num)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

#何かモデルを作成するときにはnn.Moduluを継承する。基本的な型はこんな感じ
class Net(nn.Module):
    #モデルを作成するときに使用するものを作っておく。
    def __init__(self):
        super().__init__()
        #nn.Conv2dの引数は(インプットのチャンネル数、アウトプットのチャンネル数、カーネルのサイズ)
        #畳み込みフィルタ
        self.conv1 = nn.Conv2d(3, 6, 5)
        #MAXプーリング層
        #mnn.MaxPool2dの引数は(カーネルサイズ、スライド、パディング)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #Linear
        #線形層(y=Wx+by=Wx+b)の定義
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 7)

    #ここでモデルを組み立てる。
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x_pre = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x_pre))
        x = self.fc3(x)
        return x_pre,x
