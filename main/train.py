"""
ここでは田久保がpytorchを学習する。
ソースはここ
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#define-a-convolutional-neural-network

学習から推論するところまでを記載している。
自分で自作したモデルを学習する。
流れは下記に記載する。
1 CIFAR10をロードする
2 畳み込みニューラルネットワークを定義する。
3 損失関数とオプティマイザーを定義する
4 ネットワークの学習
5 テストデータでネットワークをテストをする
"""

#CIFAR10を使ってトレーニングをする
import torch
import torchvision
import torchvision.transforms as transforms

#取得したデータを見るために使用するラリブラリ
import matplotlib.pyplot as plt
import numpy as np

#1 CIFAR10をロードする
#画像の前処理を施している。リサイズなどができる（正規化をするものを作成している）
#transforms.ToTensor() → Tensor(C,W,H)へとテンソル型に正規化している
#transforms.Normalize() → データセットのRGBの平均(mean)と標準偏差、分散(std)を使用する。
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#バッチサイズの設定→バッチサイズはどれだけまとめて処理を行うか、今回は四つずつ処理を行う。
batch_size = 4
#トレーニングするデータダウンロードしている。
# もし自作のデータセットがあるなら↓を使うといいかもしれない。
#dataset = ImageFolder("dataset1", transform)
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
#データセットからサンプルを取得して、ミニバッチを作成するクラス。
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
#トレーニングするデータダウンロードしている。
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
#データセットからサンプルを取得して、ミニバッチを作成するクラス。
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
#クラスの名前を定義している。
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#トレーニングデータを見てみる。
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()
# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

#gpuでのトレーニング
#なかったらcpuに切り替わる
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#2 畳み込みニューラルネットワークを定義する。
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
        self.fc3 = nn.Linear(84, 10)

    #ここでモデルを組み立てる。
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#モデルを定義する。
net = Net()
net.to(device)

#3損失関数とオプティマイザーを定義する
#損失関数
#差を小さくするのを目的とする
#オプティマイザー
#最適化のいろいろなアルゴリズムを定義している。
import torch.optim as optim
#目的関数となるオブジェクトを定義している。
criterion = nn.CrossEntropyLoss()
#最適化アルゴリズムの中でSGDを使用している。
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#4 ネットワークの学習
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        #初期化をしている。
        optimizer.zero_grad()

        # forward + backward + optimize
        #作成したモデルに渡すことで目的関数まで通した最終的な出力が返ってくる。
        outputs = net(inputs)
        #forwardをしている。誤差を計算している
        loss = criterion(outputs, labels)
        #backwardをしている
        #目的関数に対して、それに含まれるパラメータの微分係数を求めている。
        loss.backward()
        #backwardで計算した勾配を元に、パラメータを更新してくれる。
        optimizer.step()

        # print statistics
        #１回の学習の進行状態を示してくれている。
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
#モデルの保存をしている。
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

#ここからはトレーニングされたモデルを扱って、答え合わせをするところ。リアル世界ではここがメインになる。
#5 テストデータでネットワークをテストをする
#テストデータがどんなものかを見てみる
dataiter = iter(testloader)
images, labels = dataiter.next()
#自作の関数で画像を出力する
# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

#保存したモデルをロードする。
net = Net()
net.load_state_dict(torch.load(PATH))

#モデルにテスト画像を入れる。
outputs = net(images)
#一番高いラベルを入れる。（予測値が）
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

#全体でどのくらいの正答率かをみる。
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

#分析をする（どのクラスがうまくいっているのか、うまくいってないクラスはどれなのかがわかるようになっている。）
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                   accuracy))