"""
深層距離学習をするファイル

"""

import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.datasets import ImageFolder

#取得したデータを見るために使用するラリブラリ
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#田久保が書いたファイル
import CnnNet
import Visualize
import metrics
import train
import validate

#はじめにここを設定する。
#データセットの名前
IMAGE_PATH = "/home/cvmlab/Desktop/深層距離学習_pytorch/pytorch_metric/自作CNN/main/data/train/"
DATASET_PATH = "/home/cvmlab/Desktop/深層距離学習_pytorch/pytorch_metric/自作CNN/main/data/models/"
MODEL_PATH = "model.pth"
EPOCHS = 20

#クラスの名前を定義している。
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#gpuでのトレーニング
#なかったらcpuに切り替わる
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#正規化するためにかく
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#バッチサイズの設定→バッチサイズはどれだけまとめて処理を行うか、今回は四つずつ処理を行う。
batch_size = 4

#トレーニングするデータダウンロードしている。
# もし自作のデータセットがあるなら↓を使うといいかもしれない。
# dataset = ImageFolder(IMAGE_PATH, transform)
# train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                           shuffle=True, num_workers=2)
# val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                          shuffle=False, num_workers=2)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                       download=True, transform=transform)
#データセットからサンプルを取得して、ミニバッチを作成するクラス。
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)

val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                        shuffle=False, num_workers=2)

#トレーニングデータを見てみる。
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

#CNNモデルの設定
model = CnnNet.Net()
model.to(device)

#深層距離学習
#CNNの最後のところを持ってくる。
num_featured = model.fc3.out_features
metric_fc = metrics.AdaCos(num_features=num_featured, num_classes=10)


#3損失関数とオプティマイザーを定義する
#損失関数
#差を小さくするのを目的とする
#オプティマイザー
#最適化のいろいろなアルゴリズムを定義している。
#目的関数となるオブジェクトを定義している。
criterion = nn.CrossEntropyLoss().to(device)
#最適化アルゴリズムの中でSGDを使用している。
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-3)

#トレーニング
log = pd.DataFrame(index=[],
                   columns=[ 'epoch', 'lr', 'loss', 'acc1', 'val_loss', 'val_acc1'])
best_loss = float('inf')

for epoch in range(EPOCHS):
    print('Epoch [%d/%d]' %(epoch+1, EPOCHS))

    scheduler.step()

    # train for one epoch
    train_log = train.train_function(train_loader, model, metric_fc, criterion, optimizer,epoch+1)
    train_log = train_log.train(device,file_name=DATASET_PATH,epoch=EPOCHS,train_num=epoch)
    # evaluate on validation set
    val_log = validate.validate_function(val_loader, model, metric_fc, criterion)
    val_log = val_log.validate(device)


    print('loss %.4f - acc1 %.4f - val_loss %.4f - val_acc %.4f'
            %(train_log['loss'], train_log['acc1'], val_log['loss'], val_log['acc1']))

    tmp = pd.Series([
            epoch,
            scheduler.get_lr()[0],
            train_log['loss'],
            train_log['acc1'],
            val_log['loss'],
            val_log['acc1'],
        ], index=['epoch', 'lr', 'loss', 'acc1', 'val_loss', 'val_acc1'])

    log = log.append(tmp, ignore_index=True)
    log.to_csv('models_log.csv', index=False)

    if val_log['loss'] < best_loss:
        torch.save(model.state_dict(), DATASET_PATH+MODEL_PATH)
        best_loss = val_log['loss']
        print("=&gt; saved best model")