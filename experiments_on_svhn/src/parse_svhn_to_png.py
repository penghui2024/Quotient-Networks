import os
import numpy as np
import imageio
import scipy.io as sio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def my_mkdir(my_dir):
    if not os.path.isdir(my_dir):
        os.makedirs(my_dir)


if __name__ == "__main__":

    data_dir = os.path.join(BASE_DIR, "..", "data")
    train_o_dir = os.path.join(BASE_DIR, "..", "data", "svhn_train")
    test_o_dir = os.path.join(BASE_DIR, "..", "data", "svhn_test")

    train = sio.loadmat(os.path.join(data_dir, 'train_32x32.mat'))
    test = sio.loadmat(os.path.join(data_dir, 'test_32x32.mat'))

    train_data = train['X']
    train_label = train['y']
    test_data = test['X']
    test_label = test['y']

    train_data = np.swapaxes(train_data, 0, 3)
    train_data = np.swapaxes(train_data, 2, 3)
    train_data = np.swapaxes(train_data, 1, 2)
    test_data = np.swapaxes(test_data, 0, 3)
    test_data = np.swapaxes(test_data, 2, 3)
    test_data = np.swapaxes(test_data, 1, 2)

    for i in range(train_label.shape[0]):
        if train_label[i][0] == 10:
            train_label[i][0] = 0

    for i in range(test_label.shape[0]):
        if test_label[i][0] == 10:
            test_label[i][0] = 0

    print("train_data is loading...")

    for i in range(train_label.shape[0]):
        img = train_data[i]

        label_num = str(train_label[i][0])
        o_dir = os.path.join(train_o_dir, label_num)
        my_mkdir(o_dir)

        img_name = label_num + "_" + str(i) + ".png"
        img_path = os.path.join(o_dir, img_name)
        imageio.imwrite(img_path, img)

    print("train_data loaded.")

    print("test_data is loading...")
    for i in range(test_label.shape[0]):
        img = test_data[i]

        label_num = str(test_label[i][0])
        o_dir = os.path.join(test_o_dir, label_num)
        my_mkdir(o_dir)

        img_name = label_num + "_" + str(i) + ".png"
        img_path = os.path.join(o_dir, img_name)
        imageio.imwrite(img_path, img)

    print("test_data loaded.")
