# coding: utf-8
import argparse
import time
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
parser.add_argument('--attention', type=bool, default=True, help='use attention or not')
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

def evaluate():
    model.train(False)
    total_loss = 0.0
    batch_num = 0
    end_flag = False
    data_loader.set_train()
    while not end_flag:
        data, target, end_flag = data_loader.get_batch()
        data = data.to(device)
        target = target.to(device)
        output, _ = model(data)
        loss = criterion(output, target)
        total_loss += loss.item()
        batch_num += 1

    loss = total_loss / batch_num
    perplexity = math.exp(loss)
    return loss, perplexity

########################################


# WRITE CODE HERE within two '#' bar
########################################
# Train Function

lr = args.lr
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

def train():
    model.train(True)
    total_loss = 0.0
    batch_num = 0
    end_flag = False
    data_loader.set_train()
    while not end_flag:
        data, target, end_flag = data_loader.get_batch()
        data = data.to(device)
        optimizer.zero_grad()
        target = target.to(device)
        output, _ = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batch_num += 1

    loss = total_loss / batch_num
    perplexity = math.exp(loss)
    return loss, perplexity

########################################


# Loop over epochs.
print("batch_size: " + str(train_batch_size))
print("lr: " + str(lr))
curve_csv = open("curve.csv", "w")
# writer = SummaryWriter("log/")
for epoch in range(1, args.epochs+1):
    print('epoch:{:d}/{:d}'.format(epoch, args.epochs))
    train_loss, train_perplexity = train()
    print("training: {:.4f}, {:.4f}".format(train_loss, train_perplexity))
    # writer.add_scalar('Loss/train', train_loss, epoch)
    # writer.add_scalar('Perplexity/train', train_loss, epoch)
    valid_loss, valid_perplexity = evaluate()
    print("validation: {:.4f}, {:.4f}".format(valid_loss, valid_perplexity))
    curve_csv.write(
        "{:d},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
        epoch, train_loss, train_perplexity, valid_loss, valid_perplexity))
    # writer.add_scalar('Loss/valid', valid_loss, epoch)
    # writer.add_scalar('Perplexity/valid', valid_perplexity, epoch)
    print('*' * 100)


