import matplotlib.pyplot as plt

class Visualize():
    def __init__(self,feat,labels, epoch):
        self.feat = feat
        self.labels = labels
        self.epoch = epoch

    def visualize(self):
        #t-SNEを追加する。
        
        plt.ion()
        c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
            '#ff00ff', '#990000', '#999900', '#009900', '#009999']
        plt.clf()
        for i in range(10):
            plt.plot(self.feat[self.labels == i, 0], self.feat[self.labels == i, 1], '.', c=c[i])
        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc = 'upper right')
        plt.xlim(xmin=-50,xmax=50)
        plt.ylim(ymin=-50,ymax=50)
        plt.text(20,45 - 0.05,"epoch=%d" % self.epoch)
        plt.savefig('/Users/takubokouakira/Desktop/pytorch_metric/自作CNN/main/deepmetric/models/epoch=%d.jpg' % self.epoch)
        plt.draw()
        plt.pause(0.001)
        #plt.savefig("epoch=%d.jpg" % self.epoch)
        
        