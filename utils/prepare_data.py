import os
import random
import pickle as pkl

def deal(des):
    data_dir = "../data/pacs/" + des

    list = os.listdir(data_dir)

    data_all = []
    for cls in list:
        cls_path = os.path.join(data_dir, cls)
        for file in os.listdir(cls_path):
            if os.path.isfile(os.path.join(cls_path, file)):
                data_all.append(os.path.join(cls_path, file))

    len_data = len(data_all)

    random.shuffle(data_all)
    train_pkl = data_all[int(0.2 * len_data):len_data]
    test_pkl = data_all[0:int(0.2 * len_data)]

    random.shuffle(train_pkl)
    random.shuffle(test_pkl)

    def split_data(arr):
        label = []
        for i in arr:
            label.append(i.split("/")[-2])
        return label

    SAVE_PATH = '../data/pacs/'

    with open(os.path.join(SAVE_PATH, "{}_train.pkl".format(des)), "wb") as f:
        pkl.dump((train_pkl, split_data(train_pkl)), f, pkl.HIGHEST_PROTOCOL)

    with open(os.path.join(SAVE_PATH, "{}_test.pkl".format(des)), "wb") as f:
        pkl.dump((test_pkl, split_data(test_pkl)), f, pkl.HIGHEST_PROTOCOL)



for des in os.listdir("../data/pacs"):
    print(des)
    deal(des)













