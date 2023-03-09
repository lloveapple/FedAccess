import torch
from modules import AlexNet
from torch import distributed
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import sys, os
import random
import  argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import pickle as pkl

from dataset.pacs import PACSDataset
from modules.AlexNet import AlexNet
import argparse
import time
import copy
import torchvision.transforms as transforms
import random
import numpy as np


def test_get_image():
    all_data = []
    all_label = []
    for file in os.listdir("../data/pacs"):
        tmp = file.split("_")
        if "test.pkl" in tmp:
            data, label = np.load(os.path.join("../data/pacs", file), allow_pickle=True)
            all_data.extend(data)
            all_label.extend(label)
    return all_data, all_label


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_img', default=64)
    # parser.add_argument('--model')
    device = torch.device("cuda", 0)
    args = parser.parse_args()
    num_img = args.num_img
    model = AlexNet(num_class=7)
    # model_path = "../results/pacs/domainnet/fed_domainnet/FedBN.ckpt"
    model_path = "../results/pacs/art_painting/10.ckpt"

    # model.load_state_dict(torch.load(model_path)['server_model'])
    model.load_state_dict(torch.load(model_path)['model'])

    model = model.to(device)
    model.eval()
    label_dict = {'dog': 0, 'elephant': 1, 'giraffe': 2, 'guitar': 3, 'horse': 4, 'house': 5, 'person': 6}

    data, label = test_get_image()

    data = np.array(data)
    label = np.array(label)

    choose = np.zeros(len(label), dtype=int)
    choose[-num_img:] = 1
    choose = choose > 0
    random.shuffle(choose)

    file_list = data[choose]
    label_list = label[choose]
    img_list = torch.tensor([])
    for file in file_list:
        img = Image.open(file)
        if len(img.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(img)
        transform_test = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor()
        ])
        des = transform_test(img)
        des = torch.unsqueeze(des, dim=0)
        img_list = torch.cat((img_list, des), dim=0)

    img_list = img_list.to(device)
    outputs = model(img_list)
    pred = outputs.data.max(1)[1]
    print(outputs.data.max(1)[1])
    gt = [ label_dict.get(i) for i in label_list]
    gt = torch.tensor(gt, device="cuda:0")
    print(gt)
    correct = pred.eq(gt).sum().item()
    print(correct)

    a = 0
    for i in range(64):
        if gt[i] == pred[i]:
            a = a + 1
    print(a)
    print(correct / num_img)
