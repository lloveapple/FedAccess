import os

import torch.nn as nn
import numpy as np
import random
import argparse
import torchvision.transforms as transforms
from dataset.office import OfficeDataset_single
import torch.utils.data
from modules import AlexNet
import torch.optim as optim
import time


def train(model, data_loader, optimizer, loss_fun, device):
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        # print(output)
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()
        loss.backward()
        optimizer.step()

    return loss_all / len(data_loader), correct/total


def test(model, data_loader, loss_fun, device):
    model.eval()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()
    return loss_all / len(data_loader), correct / total


if __name__ == '__main__':
    a = 1
    device = torch.device('cuda',0)
    seed = 1

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    parse = argparse.ArgumentParser()
    parse.add_argument('--log', action='store_true')
    parse.add_argument('--epochs', type=int, default=110)
    parse.add_argument('--lr', type=float, default=1e-2)
    parse.add_argument('--batch', type=int, default=32)
    parse.add_argument('--save_path', type=str, default='../results/office')
    parse.add_argument('--data', type=str, default='amazon', help='[amazon | caltech | dslr | webcam]')
    args = parse.parse_args()

    exp_floder = 'single_office_AlexNet'

    args.save_path = os.path.join(args.save_path, exp_floder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, '{}.ckpt'.format(args.data))

    log = args.log

    if log:
        log_path = os.path.join('../logs/office/', exp_floder)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logfile = open(os.path.join(log_path, '{}.log'.format(args.data), 'a'))
        logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        logfile.write('===Setting===\n')
        logfile.write('    lr: {}\n'.format(args.lr))
        logfile.write('    batch: {}\n'.format(args.batch))
        logfile.write('    dataset: {}\n'.format(args.data))
        logfile.write('    epochs: {}\n'.format(args.epochs))

    transform_train = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-30,30)),
        transforms.ToTensor()
    ])
    transform_test = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor()
    ])

    data_base_path = '../data'
    min_data_len = 5e8
    for site in ['amazon', 'caltech', 'dslr', 'webcam']:
        trainset = OfficeDataset_single(data_base_path, site, transformer=transform_train)
        if min_data_len > len(trainset):
            min_data_len = len(trainset)

    val_len = int(min_data_len * 0.4)
    min_data_len = int(min_data_len * 0.5)

    trainset = OfficeDataset_single(data_base_path, args.data, transformer=transform_train)
    testset = OfficeDataset_single(data_base_path, args.data, transformer=transform_test, train=False)

    val_set = torch.utils.data.Subset(trainset, list(range(len(trainset)))[-val_len:])
    trainset = torch.utils.data.Subset(trainset, list(range(min_data_len)))

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=True)
    val_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True)

    model = AlexNet.AlexNet().to(device)
    loss_fun = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=args.lr)

    best_acc = 0
    best_epoch = 0
    start_epoch = 0
    N_EPOCHS = args.epochs

    for epoch in range(start_epoch, start_epoch+N_EPOCHS):
        train_loss, train_acc = train(model, train_loader, optimizer, loss_fun, device)
        print('Epoch: [{}/{}] | Train Loss: {:.4f} | Train Acc: {:.4f}'.format(epoch, N_EPOCHS ,train_loss, train_acc))
        if log:
            logfile.write('Epoch: [{}/{}] | Train Loss: {:.4f} | Train Acc: {:.4f}\n'.format(epoch, N_EPOCHS ,train_loss, train_acc))
        val_loss, val_acc = test(model, val_loader, loss_fun, device)
        print('Val site-{} | Val Loss: {:.4f} | Val Acc: {:.4f}'.format(args.data, val_loss, val_acc))
        if log:
            logfile.write('Val site-{} | Val Loss: {:.4f} | Val Acc: {:.4f}\n'.format(args.data, val_loss, val_acc))
            logfile.flush()
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            print(' Saving the best checkpoint to {}...'.format(SAVE_PATH))
            torch.save({
                'model': model.state_dict(),
                'best_epoch': best_epoch,
                'best_acc': best_acc,
                'epoch': epoch
            }, '../results/{}.ckpt'.format(a))
            torch.save(model, "../results/ckpt")
            a = a + 1
            print('Best site-{} | Epoch:{} | Test Acc: {:.4f}'.format(args.data, best_epoch, best_acc))
            if log:
                logfile.write('Best site-{} | Epoch:{} | Test Acc: {:.4f}\n'.format(args.data, best_epoch, best_acc))

            _, test_acc = test(model, test_loader, loss_fun, device)
            print('Test site-{} | Epoch:{} | Test Acc: {:.4f}'.format(args.data, best_epoch, test_acc))
            if log:
                logfile.write('Test site-{} | Epoch:{} | Test Acc: {:.4f}\n'.format(args.data, best_epoch, test_acc))
    if log:
        logfile.flush()
        logfile.close()












