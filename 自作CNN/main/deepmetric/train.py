"""
train関数を作成する。
train_loader データセット
model　CNNのモデル
metric_fc　距離学習に使った関数
criterion　目的関数を設定する。
optimizer　最適化アルゴリズムを設定する。（例：　SGD）
"""
from re import X
import torch
import utils
from tqdm import tqdm
from collections import OrderedDict
import Visualize

class train_function():
    def __init__(self,train_loader, model, metric_fc, criterion, optimizer,epoch):
        self.train_loader =train_loader
        self.model = model
        self.metric_fc = metric_fc
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoch = epoch

    def train(self,device,file_name,epoch,train_num):
        losses = utils.AverageMeter()
        acc1s = utils.AverageMeter()

        target_feat =[]
        target_labels = []

        # switch to train mode
        self.model.train().to(device)
        self.metric_fc.train().to(device)

        for i, (input, target) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            input = input.to(device)
            target = target.to(device)
            #作成したモデルに渡すことで最終的な出力が返ってくる。
            feature = self.model(input)
            #最終的な出力に距離関数に渡す。
            #ここが普通の学習とは違う。
            output = self.metric_fc(feature, target)
            #forwordをしている。
            loss = self.criterion(output, target)
            
            #accuracy関数
            acc1, = utils.accuracy(output, target, topk=(1,))

            losses.update(loss.item(), input.size(0))
            acc1s.update(acc1.item(), input.size(0))

            # compute gradient and do optimizing step
            #初期化をしている
            self.optimizer.zero_grad()
            #backwardをしている
            #目的関数に対して、それに含まれるパラメータの微分係数を求めている。
            loss.backward()
            #backwardで計算した勾配を元に、パラメータを更新してくれる。
            self.optimizer.step()

            #特徴空間の可視化に使う変数
            target_feat.append(self.model.classifier.in_features)
            target_labels.append(target)
        
        if train_num == 0 or train_num == epoch :
            #特徴空間を可視化する
            feat = torch.cat(target_feat, 0)
            labels = torch.cat(target_labels, 0)
            Visualize.Visualize(feat=feat.data.cpu().numpy(),labels=labels.data.cpu().numpy(),epoch=self.epoch,file_name=file_name).visualize()

        log = OrderedDict([
            ('loss', losses.avg),
            ('acc1', acc1s.avg),
        ])

        return log