"""

"""
import torch
import utils
from tqdm import tqdm
from collections import OrderedDict


class validate_function():
    def __init__(self,val_loader, model, metric_fc, criterion):
        self.val_loader = val_loader
        self.model = model
        self.metric_fc = metric_fc
        self.criterion = criterion

    def validate(self,device):
        losses = utils.AverageMeter()
        acc1s = utils.AverageMeter()

        # switch to evaluate mode
        #eval()　ドロップアウトやbatch normの on/off も切り替え
        self.model.eval()
        self.metric_fc.eval()

        #no_grad() これがあるブロックは勾配の計算をしないようになっています。それによって、メモリ消費を減らすことができるみたいです。
        #validateだから
        with torch.no_grad():
            for i, (input, target) in tqdm(enumerate(self.val_loader), total=len(self.val_loader)):
                input = input.to(device)
                target = target.to(device)

                x_pre,feature = self.model(input)
                output = self.metric_fc(feature, target)
                loss = self.criterion(output, target)

                acc1, = utils.accuracy(output, target, topk=(1,))

                losses.update(loss.item(), input.size(0))
                acc1s.update(acc1.item(), input.size(0))

        log = OrderedDict([
            ('loss', losses.avg),
            ('acc1', acc1s.avg),
        ])

        return log