# coding: utf-8
import argparse
import time
import collections
import math
import torch
import torch.nn as nn
import torch.optim as optim

import data
from model import LMModel
import os
import os.path as osp
# from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='PyTorch ptb Language Model')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--train_batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N',
                    help='eval batch size')
parser.add_argument('--max_sql', type=int, default=35,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=1234,
                    help='set random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA device')
parser.add_argument('--attention', type=bool, default=False, help='use attention or not')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU device id used')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
# Use gpu or cpu to train
use_gpu = True

if use_gpu:
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(args.gpu_id)
else:
    device = torch.device("cpu")

# load data
train_batch_size = args.train_batch_size
eval_batch_size = args.eval_batch_size
batch_size = {'train': train_batch_size,'valid':eval_batch_size}
data_loader = data.Corpus("../data/ptb", batch_size, args.max_sql)

        
# WRITE CODE HERE within two '#' bar
########################################
# Build LMModel model (bulid your language model here)

nvoc = len(data_loader.vocabulary)
print("nvoc: " + str(nvoc))
ninput = 150
nhid = 150
nlayer = 4
model = LMModel(nvoc, ninput, nhid, nlayer, device, args.attention).to(device)

########################################

criterion = nn.CrossEntropyLoss().to(device)


# WRITE CODE HERE within two '#' bar
########################################
# Evaluation Function
# Calculate the average cross-entropy loss between the prediction and the ground truth word.
# And then exp(average cross-entropy loss) is perplexity.

def bleu_metric(candidates, references, max_n):
    def get_max_n_grams(s):
        ans = []
        for n in range(1, max_n + 1):
            ans += list(zip(*[s[index:] for index in range(n)]))
        return ans

    cnt_i = torch.zeros(max_n)
    cnt = torch.zeros(max_n)
    length = candidates.size(0)
    for i in range(length):
        candidate = candidates[i].numpy().tolist()
        reference = references[i].numpy().tolist()
        candidate_counter = collections.Counter(get_max_n_grams(candidate))
        reference_counter = collections.Counter(get_max_n_grams(reference))
        intersected = candidate_counter & reference_counter
        for x in intersected:
            cnt_i[len(x) - 1] += intersected[x]
        for x in candidate_counter:
            cnt[len(x) - 1] += candidate_counter[x]

    if min(cnt_i) == 0:
        return 0.0
    else:
        score = torch.exp(torch.mean(torch.log(cnt_i / cnt)))
        return score.item()


def evaluate():
    model.train(False)
    total_loss = 0.0
    total_score = 0.0
    batch_num = 0
    end_flag = False
    data_loader.set_valid()
    while not end_flag:
        data, target, end_flag = data_loader.get_batch()
        data = data.to(device)
        target = target
        l, b = target.size(0), target.size(1)
        output, _ = model(data)

        loss = criterion(output, target.to(device).view(-1))
        total_loss += loss.item()

        output = torch.argmax(output, 1).view(l, b).t().contiguous().cpu()
        target = target.t().contiguous()
        total_score += bleu_metric(output, target, 4)

        batch_num += 1

    loss = total_loss / batch_num
    score = total_score / batch_num
    perplexity = math.exp(loss)
    return loss, perplexity, score

########################################


# WRITE CODE HERE within two '#' bar
########################################
# Train Function

lr = args.lr
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

def train():
    model.train(True)
    total_loss = 0.0
    total_score = 0.0
    batch_num = 0
    end_flag = False
    data_loader.set_train()
    while not end_flag:
        data, target, end_flag = data_loader.get_batch()
        l, b = target.size(0), target.size(1)
        data = data.to(device)
        optimizer.zero_grad()
        output, _ = model(data)

        loss = criterion(output, target.to(device).view(-1))
        total_loss += loss.item()

        output = torch.argmax(output, 1).view(l, b).t().contiguous().cpu()
        target = target.t().contiguous()
        total_score += bleu_metric(output, target, 4)

        batch_num += 1

    loss = total_loss / batch_num
    score = total_score / batch_num
    perplexity = math.exp(loss)
    return loss, perplexity, score

########################################


# Loop over epochs.
print("batch_size: " + str(train_batch_size))
print("lr: " + str(lr))
curve_csv = open("curve.csv", "w")
# writer = SummaryWriter("log/")
for epoch in range(1, args.epochs+1):
    print('epoch:{:d}/{:d}'.format(epoch, args.epochs))
    train_loss, train_perplexity, train_score = train()
    print("training: {:.4f}, {:.4f}, {:.4f}".format(train_loss, train_perplexity, train_score))
    # writer.add_scalar('Loss/train', train_loss, epoch)
    # writer.add_scalar('Perplexity/train', train_loss, epoch)
    valid_loss, valid_perplexity, valid_score = evaluate()
    print("validation: {:.4f}, {:.4f}, {:.4f}".format(valid_loss, valid_perplexity, valid_score))
    curve_csv.write(
        "{:d},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
        epoch, train_loss, train_perplexity,train_score, valid_loss, valid_perplexity, valid_score))
    # writer.add_scalar('Loss/valid', valid_loss, epoch)
    # writer.add_scalar('Perplexity/valid', valid_perplexity, epoch)
    print('*' * 100)

