import torch
import torch.nn.functional as F

gamma = 2

def focal_loss(preds, labels):
    # print(labels)
    batch_size = labels.shape[0]
    onehot = torch.zeros(batch_size, 20).cuda().scatter_(1, labels.unsqueeze(1), 1)
    preds = -(1 - F.softmax(preds, dim=1)).pow(gamma) * F.log_softmax(preds, dim=1)
    preds *= onehot
    loss = torch.sum(preds) / batch_size
    return loss
