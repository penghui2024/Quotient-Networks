import os
import numpy as np
import imageio
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def unpickle(file):
    fo = open(file, "rb")
    dict_ = pickle.load(fo, encoding="latin1")
    fo.close()
    return dict_


def my_mkdir(my_dir):
    if not os.path.isdir(my_dir):
        os.makedirs(my_dir)

if __name__ == "__main__":

    data_dir = os.path.join(BASE_DIR, "..", "data", "cifar-100-python", )
    train_o_dir = os.path.join(BASE_DIR, "..", "data", "cifar100_train")
    test_o_dir = os.path.join(BASE_DIR, "..", "data", "cifar100_test")

    data_path = os.path.join(data_dir, "train")
    train_data = unpickle(data_path)
    print(data_path + " is loading...")

    for i in range(0, 50000):
        img = np.reshape(train_data["data"][i], (3, 32, 32))
        img = img.transpose((1, 2, 0))

        label_num = str(train_data["fine_labels"][i])
        o_dir = os.path.join(train_o_dir, label_num)
        my_mkdir(o_dir)

        img_name = label_num + "_" + str(i) + ".png"
        img_path = os.path.join(o_dir, img_name)
        imageio.imwrite(img_path, img)
    print(data_path + "loaded.")

    print("test_data is loading...")
    test_data_path = os.path.join(data_dir, "test")
    test_data = unpickle(test_data_path)
    for i in range(0, 10000):
        img = np.reshape(test_data["data"][i], (3, 32, 32))
        img = img.transpose((1, 2, 0))
        label_num = str(test_data["fine_labels"][i])
        o_dir = os.path.join(test_o_dir, label_num)
        my_mkdir(o_dir)

        img_name = label_num + "_" + str(i) + ".png"
        img_path = os.path.join(o_dir, img_name)
        imageio.imwrite(img_path, img)

    print("test_batch loaded.")