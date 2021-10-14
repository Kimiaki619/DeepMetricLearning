#2 畳み込みニューラルネットワークを定義する。
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

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
