import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR
import numpy as np
import data
import models
import visualize
import FocalLoss
import os
import argparse


## Note that: here we provide a basic solution for training and validation.
## You can directly change it if you find something wrong or not good enough.

parser = argparse.ArgumentParser(description='Deep Learning 2020 HW: CNN Training With Pytorch')
parser.add_argument('--batch_size', default=64, type=int,
                    help='Batch size for training')
parser.add_argument('--num_epoch', default=500, type=int,
                    help='Number of epoch')
parser.add_argument('--cuda', type=str, default='cuda:0',
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--alpha', default=0.03, type=float,
                    help='alpha for Early Stopping')

args = parser.parse_args()


def train_model(model,train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs=20):

    def train(model, train_loader, optimizer, criterion):
        model.train(True)
        total_loss = 0.0
        total_correct = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(predictions == labels.data)

        epoch_loss = total_loss / len(train_loader.dataset)
        epoch_acc = total_correct.double() / len(train_loader.dataset)
        return epoch_loss, epoch_acc.item()

    def valid(model, valid_loader,criterion):
        model.train(False)
        total_loss = 0.0
        total_correct = 0
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(predictions == labels.data)
        epoch_loss = total_loss / len(valid_loader.dataset)
        epoch_acc = total_correct.double() / len(valid_loader.dataset)
        return epoch_loss, epoch_acc.item()
    
    def get_confusion_matrix(model, valid_loader, device):
        model.train(False)
        confusion_matrix = np.zeros((20, 20), dtype=np.int)
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predictions = torch.max(outputs, 1)
                predictions = predictions.cpu().numpy()
                labels = labels.int().numpy()
                l = labels.shape[0]
                for i in range(l):
                    confusion_matrix[labels[i]][predictions[i]] += 1
        print(confusion_matrix)
        with open("cm.csv", 'w') as f:
            for i in range(20):
                line = ''
                for j in range(20):
                    line += str(confusion_matrix[i][j]) + ','
                line += '\n'
                f.write(line)
            f.close()


    best_acc = 0.0
    best_valid_loss = 0.0
    alpha = args.alpha
    curve_csv = open("curve.csv", "w")
    for epoch in range(num_epochs):
        print('epoch:{:d}/{:d}'.format(epoch, num_epochs))
        print('*' * 100)
        print('lr: {:.4f}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
        train_loss, train_acc = train(model, train_loader,optimizer,criterion)
        print("training: {:.4f}, {:.4f}".format(train_loss, train_acc))
        valid_loss, valid_acc = valid(model, valid_loader,criterion)
        print("validation: {:.4f}, {:.4f}".format(valid_loss, valid_acc))
        curve_csv.write("{:d},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(epoch + 1, train_loss, train_acc, valid_loss, valid_acc))

        # Early Stopping
        if epoch == 0:
            best_valid_loss = valid_loss
        gl = (valid_loss / best_valid_loss) - 1
        if gl >= alpha:
            break

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = model
            torch.save(best_model, 'best_model.pt')
        # if epoch >= 50:
        #     scheduler.step()
        scheduler.step()
    curve_csv.close()
    # visualize.tsne_visualize(model, valid_loader, device, "valid_tsne.png")
    # visualize.tsne_visualize(model, train_loader, device, "train_tsne.png")
    get_confusion_matrix(model, valid_loader, device)
    # CNN_visual = visualize.CNNLayerVisualization(model, 6, 16)
    # CNN_visual.visualise_layer_with_hooks()
    # CNN_visual = visualize.CNNLayerVisualization(model, 10, 50)
    # CNN_visual.visualise_layer_with_hooks()
    # CNN_visual = visualize.CNNLayerVisualization(model, 11, 9)
    # CNN_visual.visualise_layer_with_hooks()
    # CNN_visual = visualize.CNNLayerVisualization(model, 15, 36)
    # CNN_visual.visualise_layer_with_hooks()



if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "5"

    ## about model
    num_classes = 20

    ## about data
    data_dir = "../data/"
    inupt_size = 224
    batch_size = args.batch_size

    ## about training
    num_epochs = args.num_epoch
    lr = args.lr
    gamma = args.gamma

    ## model initialization
    model = models.model_C(num_classes=num_classes)
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ## data preparation
    train_loader, valid_loader = data.load_data(data_dir=data_dir,input_size=inupt_size, batch_size=batch_size)

    ## optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    ## learning rate scheduler
    milestones = [50 + i * 30 for i in range(10)]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    # scheduler = ExponentialLR(optimizer, gamma=gamma)

    ## loss function
    # criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss.focal_loss
    train_model(model,train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs=num_epochs)

    total_num = sum(p.numel() for p in model.parameters())
    print("total_parameters: " + str(total_num))
