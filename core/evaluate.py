import numpy as np
import torch


def Accuracy(output, target, num_classes):
    '''
    返回每一类的精度和平均精度
    '''
    corrects, whole = torch.zeros(num_classes), torch.zeros(num_classes)
    correct = (torch.argmax(output, dim=1) == target).cpu()

    for i in range(len(target)):
        whole[target[i]] += 1
        corrects[target[i]] += correct[i]

    whole[whole == 0] = 1
    acc = corrects / whole

    return acc


def WeightedAccuracy(output, target, num_classes):
    '''
    返回加权精度
    '''
    acc = Accuracy(output, target, num_classes)
    whole = torch.zeros(num_classes)

    for i in range(len(target)):
        whole[target[i]] += 1

    class_weights = np.array([1 / num_classes] * num_classes)
    weighted_acc = (acc * class_weights).sum()

    return weighted_acc, acc
