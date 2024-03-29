"""
federated learning with different aggregation strategy on domainnet dataset
"""
import sys, os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


import pickle as pkl
import scipy.stats
from dataset.pacs import PACSDataset
from modules.AlexNet import AlexNet
import argparse
import time
import copy
import torchvision.transforms as transforms
import random
import numpy as np
import wandb


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
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss.backward()
        optimizer.step()

    return loss_all / len(data_loader), correct / total


def train_prox(args, model, data_loader, optimizer, loss_fun, device):
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    for step, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()

        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = loss_fun(output, target)
        if step > 0:
            w_diff = torch.tensor(0., device=device)
            for w, w_t in zip(server_model.parameters(), model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)

            w_diff = torch.sqrt(w_diff)
            loss += args.mu / 2. * w_diff

        loss.backward()
        optimizer.step()

        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct / total


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


def get_acc_image():
    root_path = "../all3600"
    all_image = os.listdir(root_path)
    image_list = random.sample(all_image, 5)
    image_list = [os.path.join(root_path, i) for i in image_list]
    img_data = torch.tensor([])
    for file in image_list:
        img = Image.open(file)
        if len(img.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(img)
        transform_test = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor()
        ])
        des = transform_test(img)
        des = torch.unsqueeze(des, dim=0)
        img_data = torch.cat((img_data, des), dim=0)
    return img_data

################# Key Function ########################
def get_client_weights(server_model, models, client_weights):
    device = torch.device('cuda', 0)
    client_feature_list = torch.tensor([], device=device)
    img_data = get_acc_image()
    img_data = img_data.to(device)
    for client_idx in range(client_num):
        outputs = models[client_idx].forward(img_data, flag=False)
        outputs = torch.unsqueeze(outputs, dim=0)
        client_feature_list = torch.cat((client_feature_list, outputs), dim=0)
    # KL = scipy.stats.entropy(x, y)
    server_feature = server_model.forward(img_data, flag=False)
    kl_value_list = []
    for y in client_feature_list:
        KL = scipy.stats.entropy(server_feature.cpu().numpy(), y.cpu().numpy())
        kl_value_list.append(KL)
        # kl_value_list = np.append(kl_value_list, KL, axis=0)
    kl_value_list = np.array(kl_value_list)
    #有很多nan 和 inf
    # print(kl_value_list.shape)
    # print(kl_value_list)
    kl_value_list[np.isinf(kl_value_list)] = 0.0
    kl_value_list[np.isnan(kl_value_list)] = 0.0
    split_info = []
    tmp = np.array(kl_value_list.max(axis=1))
    for i in range(len(tmp)):
        split_info.append(tmp[i] / sum(tmp))

    split_info = np.array(split_info)
    idx = np.argsort(split_info)
    idx = idx.tolist()
    # i, j = idx.index(0), idx.index(3)
    # idx[i], idx[j] = idx[j], idx[i]
    # i, j = idx.index(1), idx.index(2)
    # idx[i], idx[j] = idx[j], idx[i]

    # idx[0], idx[3] = idx[3], idx[0]
    # idx[1], idx[2] = idx[2], idx[1]

    print(split_info)
    # print(idx)
    split_info[idx[0]], split_info[idx[3]] = split_info[idx[3]], split_info[idx[0]]
    split_info[idx[1]], split_info[idx[2]] = split_info[idx[2]], split_info[idx[1]]

    print(split_info)
    split_info = np.round(split_info, 2)
    print(sum(split_info))
    print(split_info)
    client_weights = split_info.tolist()
    return client_weights


def communication(args, server_model, models, client_weights):
    with torch.no_grad():
        # aggregate params
        client_weights = get_client_weights(server_model, models, client_weights)
        if args.mode.lower() == 'fedbn':
            for key in server_model.state_dict().keys():
                if 'bn' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        else:
            for key in server_model.state_dict().keys():
                # num_batches_tracked is a non trainable LongTensor and
                # num_batches_tracked are the same for all clients for the given datasets
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models


def prepare_data(args):
    data_base_path = '../data/pacs'
    transform_train = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-30, 30)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
    ])
    #art_painting | cartoon | photo | sketch]
    # art_painting
    art_painting_trainset = PACSDataset(data_base_path, 'art_painting', transformer=transform_train)
    art_painting_testset = PACSDataset(data_base_path, 'art_painting', transformer=transform_test, train=False)
    # cartoon
    cartoon_trainset = PACSDataset(data_base_path, 'cartoon', transformer=transform_train)
    cartoon_testset = PACSDataset(data_base_path, 'cartoon', transformer=transform_test, train=False)
    # photo
    photo_trainset = PACSDataset(data_base_path, 'photo', transformer=transform_train)
    photo_testset = PACSDataset(data_base_path, 'photo', transformer=transform_test, train=False)
    # sketch
    sketch_trainset = PACSDataset(data_base_path, 'sketch', transformer=transform_train)
    sketch_testset = PACSDataset(data_base_path, 'sketch', transformer=transform_test, train=False)

    min_data_len = min(len(art_painting_trainset), len(cartoon_trainset), len(photo_trainset), len(sketch_trainset))
    val_len = int(min_data_len * 0.05)
    min_data_len = int(min_data_len * 0.05)

    art_painting_valset = torch.utils.data.Subset(art_painting_trainset, list(range(len(art_painting_trainset)))[-val_len:])
    art_painting_trainset = torch.utils.data.Subset(art_painting_trainset, list(range(min_data_len)))

    cartoon_valset = torch.utils.data.Subset(cartoon_trainset, list(range(len(cartoon_trainset)))[-val_len:])
    cartoon_trainset = torch.utils.data.Subset(cartoon_trainset, list(range(min_data_len)))

    photo_valset = torch.utils.data.Subset(photo_trainset, list(range(len(photo_trainset)))[-val_len:])
    photo_trainset = torch.utils.data.Subset(photo_trainset, list(range(min_data_len)))

    sketch_valset = torch.utils.data.Subset(sketch_trainset, list(range(len(sketch_trainset)))[-val_len:])
    sketch_trainset = torch.utils.data.Subset(sketch_trainset, list(range(min_data_len)))

    art_painting_train_loader = torch.utils.data.DataLoader(art_painting_trainset, batch_size=args.batch, shuffle=True)
    art_painting_val_loader = torch.utils.data.DataLoader(art_painting_valset, batch_size=args.batch, shuffle=False)
    art_painting_test_loader = torch.utils.data.DataLoader(art_painting_testset, batch_size=args.batch, shuffle=False)

    cartoon_train_loader = torch.utils.data.DataLoader(cartoon_trainset, batch_size=args.batch, shuffle=True)
    cartoon_val_loader = torch.utils.data.DataLoader(cartoon_valset, batch_size=args.batch, shuffle=False)
    cartoon_test_loader = torch.utils.data.DataLoader(cartoon_testset, batch_size=args.batch, shuffle=False)

    photo_train_loader = torch.utils.data.DataLoader(photo_trainset, batch_size=args.batch, shuffle=True)
    photo_val_loader = torch.utils.data.DataLoader(photo_valset, batch_size=args.batch, shuffle=False)
    photo_test_loader = torch.utils.data.DataLoader(photo_testset, batch_size=args.batch, shuffle=False)

    sketch_train_loader = torch.utils.data.DataLoader(sketch_trainset, batch_size=args.batch, shuffle=True)
    sketch_val_loader = torch.utils.data.DataLoader(sketch_valset, batch_size=args.batch, shuffle=False)
    sketch_test_loader = torch.utils.data.DataLoader(sketch_testset, batch_size=args.batch, shuffle=False)

    train_loaders = [art_painting_train_loader, cartoon_train_loader, photo_train_loader, sketch_train_loader]
    val_loaders = [art_painting_val_loader, cartoon_val_loader, photo_val_loader, sketch_val_loader]
    test_loaders = [art_painting_test_loader, cartoon_test_loader, photo_test_loader, sketch_test_loader]

    return train_loaders, val_loaders, test_loaders


if __name__ == '__main__':




    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help='whether to log')
    parser.add_argument('--test', action='store_true', help='test the pretrained model')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--batch', type=int, default=96, help='batch size')
    parser.add_argument('--iters', type=int, default=300, help='iterations for communication')
    parser.add_argument('--wk_iters', type=int, default=1, help='optimization iters in local worker between communication')
    parser.add_argument('--mode', type=str, default='FedProx', help='[FedBN | FedAvg | FedProx]')
    parser.add_argument('--mu', type=float, default=1e-3, help='The hyper parameter for fedprox')
    parser.add_argument('--save_path', type=str, default='../results/pacs/domainnet', help='path to save the checkpoint')
    parser.add_argument('--resume', action='store_true', help='resume training from the save path checkpoint')
    args = parser.parse_args()


    wandb.init(config=args.__dict__,
                project="Fed",
                entity="yxsun",
                name=args.mode + '_' + "fedacc" + "true" + "G1",
                dir = '../',
                job_type = "training",
                reinit = True)



    exp_folder = 'two_stage'

    args.save_path = os.path.join(args.save_path, exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, '{}.ckpt'.format(args.mode))

    log = args.log

    if log:
        log_path = os.path.join('../logs/AlexNet', exp_folder)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logfile = open(os.path.join(log_path, '{}.log'.format(args.mode)), 'a')
        logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        logfile.write('===Setting===\n')
        logfile.write('    lr: {}\n'.format(args.lr))
        logfile.write('    batch: {}\n'.format(args.batch))
        logfile.write('    iters: {}\n'.format(args.iters))

    train_loaders, val_loaders, test_loaders = prepare_data(args)

    # setup model
    server_model = AlexNet(num_class=7).to(device)
    loss_fun = nn.CrossEntropyLoss()

    # name of each datasets
    datasets = ['art_painting', 'cartoon', 'photo', 'sketch']
    # federated client number
    client_num = len(datasets)
    client_weights = [1 / client_num for i in range(client_num)]

    # each local client model
    models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]
    best_changed = False

    if args.test:
        print('Loading snapshots...')
        checkpoint = torch.load('../snapshots/domainnet/{}'.format(args.mode.lower()))
        server_model.load_state_dict(checkpoint['server_model'])
        if args.mode.lower() == 'fedbn':
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
        else:
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['server_model'])
        for test_idx, test_loader in enumerate(test_loaders):
            _, test_acc = test(models[test_idx], test_loader, loss_fun, device)
            print(' {:<11s}| Test  Acc: {:.4f}'.format(datasets[test_idx], test_acc))

        exit(0)

    if args.resume:
        checkpoint = torch.load(SAVE_PATH)
        server_model.load_state_dict(checkpoint['server_model'])
        if args.mode.lower() == 'fedbn':
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
        else:
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['server_model'])
        best_epoch, best_acc = checkpoint['best_epoch'], checkpoint['best_acc']
        start_iter = int(checkpoint['a_iter']) + 1

        print('Resume training from epoch {}'.format(start_iter))
    else:
        # log the best for each model on all datasets
        best_epoch = 0
        best_acc = [0. for j in range(client_num)]
        start_iter = 0

    # Start training
    for a_iter in range(start_iter, args.iters):
        optimizers = [optim.SGD(params=models[idx].parameters(), lr=args.lr) for idx in range(client_num)]
        for wi in range(args.wk_iters):
            print("============ Train epoch {} ============".format(wi + a_iter * args.wk_iters))
            if args.log:
                logfile.write("============ Train epoch {} ============\n".format(wi + a_iter * args.wk_iters))

            for client_idx, model in enumerate(models):
                if args.mode.lower() == 'fedprox':
                    # skip the first server model(random initialized)
                    if a_iter > 0:
                        train_loss, train_acc = train_prox(args, model, train_loaders[client_idx],
                                                           optimizers[client_idx], loss_fun, device)
                    else:
                        train_loss, train_acc = train(model, train_loaders[client_idx], optimizers[client_idx],
                                                      loss_fun, device)
                else:
                    train_loss, train_acc = train(model, train_loaders[client_idx], optimizers[client_idx], loss_fun,
                                                  device)

        with torch.no_grad():
            # Aggregation
            server_model, models = communication(args, server_model, models, client_weights)

            # Report loss after aggregation
            data_dict = {}
            for client_idx, model in enumerate(models):
                train_loss, train_acc = test(model, train_loaders[client_idx], loss_fun, device)
                print(' Site-{:<10s}| Train Loss: {:.4f} | Train Acc: {:.4f}'.format(datasets[client_idx], train_loss,
                                                                                     train_acc))
                if not datasets[client_idx] in data_dict.keys():
                    data_dict[datasets[client_idx]] = {}
                data_dict[datasets[client_idx]]['train'] = {"train_acc":train_acc, "train_loss": train_loss}
                if args.log:
                    logfile.write(' Site-{:<10s}| Train Loss: {:.4f} | Train Acc: {:.4f}\n'.format(datasets[client_idx],
                                                                                                   train_loss,
                                                                                                   train_acc))

            # Validation
            val_acc_list = [None for j in range(client_num)]
            for client_idx, model in enumerate(models):
                val_loss, val_acc = test(model, val_loaders[client_idx], loss_fun, device)
                val_acc_list[client_idx] = val_acc
                print(' Site-{:<10s}| Val  Loss: {:.4f} | Val  Acc: {:.4f}'.format(datasets[client_idx], val_loss,
                                                                                   val_acc))
                if not datasets[client_idx] in data_dict.keys():
                    data_dict[datasets[client_idx]] = {}
                data_dict[datasets[client_idx]]["val"] = {"val_acc":val_acc, "val_loss": val_loss}

                if args.log:
                    logfile.write(
                        ' Site-{:<10s}| Val  Loss: {:.4f} | Val  Acc: {:.4f}\n'.format(datasets[client_idx], val_loss,
                                                                                       val_acc))
            wandb.log(data_dict)
            # Record best
            if np.mean(val_acc_list) > np.mean(best_acc):
                for client_idx in range(client_num):
                    best_acc[client_idx] = val_acc_list[client_idx]
                    best_epoch = a_iter
                    best_changed = True
                    print(' Best site-{:<10s}| Epoch:{} | Val Acc: {:.4f}'.format(datasets[client_idx], best_epoch,
                                                                                  best_acc[client_idx]))
                    if args.log:
                        logfile.write(
                            ' Best site-{:<10s} | Epoch:{} | Val Acc: {:.4f}\n'.format(datasets[client_idx], best_epoch,
                                                                                       best_acc[client_idx]))

            if best_changed:
                print(' Saving the local and server checkpoint to {}...'.format(SAVE_PATH))
                logfile.write(' Saving the local and server checkpoint to {}...\n'.format(SAVE_PATH))
                if args.mode.lower() == 'fedbn':
                    torch.save({
                        'model_0': models[0].state_dict(),
                        'model_1': models[1].state_dict(),
                        'model_2': models[2].state_dict(),
                        'model_3': models[3].state_dict(),
                        # 'model_4': models[4].state_dict(),
                        # 'model_5': models[5].state_dict(),
                        'server_model': server_model.state_dict(),
                        'best_epoch': best_epoch,
                        'best_acc': best_acc,
                        'a_iter': a_iter
                    }, SAVE_PATH)
                    best_changed = False
                    for client_idx, datasite in enumerate(datasets):
                        _, test_acc = test(models[client_idx], test_loaders[client_idx], loss_fun, device)
                        print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))
                        if args.log:
                            logfile.write(
                                ' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, best_epoch,
                                                                                           test_acc))
                else:
                    torch.save({
                        'server_model': server_model.state_dict(),
                        'best_epoch': best_epoch,
                        'best_acc': best_acc,
                        'a_iter': a_iter
                    }, SAVE_PATH)
                    best_changed = False
                    for client_idx, datasite in enumerate(datasets):
                        _, test_acc = test(server_model, test_loaders[client_idx], loss_fun, device)
                        print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))
                        if args.log:
                            logfile.write(
                                ' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, best_epoch,
                                                                                           test_acc))

        if log:
            logfile.flush()
    if log:
        logfile.flush()
        logfile.close()
