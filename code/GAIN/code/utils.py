from datetime import datetime

import numpy as np
import torch

# 返回一个tensor的cuda版本
def get_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    # return tensor # 宁愿报错，也拒绝返回CPU


def logging(s):
    print(datetime.now(), s)


class Accuracy(object):
    def __init__(self):
        self.correct = 0
        self.total = 0

    def add(self, is_correct):
        self.total += 1
        if is_correct:
            self.correct += 1

    def get(self):
        if self.total == 0:
            return 0.0
        else:
            return float(self.correct) / self.total

    def clear(self):
        self.correct = 0
        self.total = 0

# 用于计算一个模型的参数量
def print_params(model):
    # step1.过滤掉可导参数
    # step2.将其变成list格式，然后计算每个矩阵的个数，然后求和。得到最后的值
    print('total parameters:', sum([np.prod(list(p.size())) for p in model.parameters() if p.requires_grad]))
