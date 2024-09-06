import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LogitAdjust(nn.Module):
    def __init__(self, tau=1, weight=None):
        super(LogitAdjust, self).__init__()
        self.tau = tau
        self.weight = weight

    def forward(self, x, target):
        # 动态计算类别概率
        batch_size = target.size(0)
        cls_count = torch.bincount(target, minlength=x.size(1)).float()
        cls_prob = cls_count / batch_size
        m_list = self.tau * torch.log(cls_prob + 1e-12)  # 加一个小值防止取log(0)
        m_list = m_list.view(1, -1).to(x.device)

        x_m = x + m_list
        return F.cross_entropy(x_m, target, weight=self.weight)