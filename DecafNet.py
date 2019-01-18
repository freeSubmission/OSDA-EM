import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.autograd import Variable


__all__ = ['DecafNet','PreTrainedNet', 'pretrained_net']

class PreTrainedNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(PreTrainedNet, self).__init__()

    def forward(self, x):
        x = x
        return x

def pretrained_net(pretrained=False, **kwargs):
    r"""any pretrained model architecture
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PreTrainedNet(**kwargs)
    return model


# fully-connected layer with normalized weight (after every update) and zero bias: <W,x> with |W|=1
class mfLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(mfLinear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = torch.nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)


    def forward(self, input):
        x = input  # size=(B,F)    F is feature len
        w = self.weight  # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2, 1, 1e-5).mul(1e5)  # torch.Tensor.renorm(p, dim, maxnorm)
        xlen = x.pow(2).sum(1).pow(0.5)  # size=B, xlen = l2norm(x)
        wlen = ww.pow(2).sum(0).pow(0.5)  # size=Classnum, wlen = l2norm(w)

        cos_theta = x.mm(ww)  # size=(B,Classnum) --> x * ww, (BxF) x (FxC)=BxC
        cos_theta = cos_theta / xlen.view(-1, 1) / wlen.view(1, -1)
        cos_theta = cos_theta.clamp(-1, 1)

        output = cos_theta * xlen.view(-1, 1)  # |x|cos_theta

        return output  # size=(B,Classnum)

# Label Classifier Feature (without softmax layer)
class LabelFeature(nn.Module):
    def __init__(self, num_classes=11, s=1):
        super(LabelFeature, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        #In many cases, LeakyReLU is more rebust than ReLU
        #self.relu = nn.LeakyReLU(0.01)
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(4096, 1024)
        self.fc = mfLinear(1024, num_classes)
        self.s = s

    def forward(self, x):
        x = self.relu(self.fc1(x))
        ## Feature Normalization
        feature = F.normalize(x, p=2, dim=1)
        x = self.s * feature
        x = self.fc(x)
        return (x, feature)

class DecafNet(nn.Module):
    def __init__(self, num_classes=2, s=1):
        super(DecafNet, self).__init__()
        self.sharedNet = pretrained_net(False)
        self.cls_fc = LabelFeature(num_classes=num_classes, s=s)

    def forward(self, x):
        x = self.sharedNet(x)
        cls_ft = self.cls_fc(x)

        return cls_ft