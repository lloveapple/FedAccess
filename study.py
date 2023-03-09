import os

import numpy as np
#
# des = "photo"
#
# # data, label = np.load('./data/pacs/{}_train.pkl'.format(des), allow_pickle=True)
# data, label1 = np.load('./data/pacs/{}_test.pkl'.format(des), allow_pickle=True)
# data2, label = np.load('./data/office/{}_test.pkl'.format("amazon"), allow_pickle=True)
# data3, label3 = np.load('./data/pacs/{}_train.pkl'.format(des), allow_pickle=True)
# data4, label = np.load('./data/office/{}_train.pkl'.format("amazon"), allow_pickle=True)
#
# print(len(data))
# print(len(data2))
# print(len(data3))
# print(len(data4))
#
#
# print(data[330])
# print(label1[330])
# print(label1)
# print(label3)


# root = "./data/pacs"
# for i in os.listdir(root):
#     file = os.path.join(root, i)
#     if i[-3:] == "pkl":
#         data, label = np.load('./data/pacs/{}'.format(i), allow_pickle=True)
#         print("====================================")
#         print("{}:".format(i) ,len(data), len(label))
#         print(label.count("dog"))
#         print(label.count("elephant"))
#         print(label.count("giraffe"))
#         print(label.count("guitar"))
#         print(label.count("horse"))
#         print(label.count("house"))
#         print(label.count("person"))
#         print("====================================")
#         print(data[10])
# print("--------------------------------------------")
# root = "./data/office"
# for i in os.listdir(root):
#     file = os.path.join(root, i)
#     if i[-3:] == "pkl":
#         data, label = np.load('./data/office/{}'.format(i), allow_pickle=True)
#         print("{}:".format(i) ,len(data), len(label))

import random
import numpy as np

# all_data = []
# all_label = []
# for file in os.listdir("./data/pacs"):
#     tmp = file.split("_")
#     if "test.pkl" in tmp:
#         data, label = np.load(os.path.join("./data/pacs", file), allow_pickle=True)
#         all_data.extend(data)
#         all_label.extend(label)
#
# all_data = np.array(all_data)
# all_label = np.array(all_label)
#
# choose =  np.zeros(len(all_label), dtype=int)
#
#
# choose[-64:] = 1
# choose = choose > 0
# random.shuffle(choose)
# print(all_label[choose])
#
# print(choose)
#
# # tf = np.random.randint(0,2, (1, len(all_label)))
#
#
#
# print(all_data[-10:])
# print(all_label[-10:])
# # print(all_label)




def get_acc_image():
    root_path = "./all3600"
    all_image = os.listdir(root_path)
    image_list = random.sample(all_image, 5)

    return [os.path.join(root_path, i) for i in image_list]


print(get_acc_image())



