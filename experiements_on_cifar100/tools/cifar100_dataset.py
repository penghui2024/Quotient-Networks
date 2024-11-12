import os
import random
from torch.utils.data import Dataset
from PIL import Image


class CifarDataset(Dataset):
    def __init__(self, data_dir, mode="train", split_n=45000, rng_seed=620, transform=None):
        assert os.path.exists(data_dir), "data_dir:{} dose not exist！".format(data_dir)
        self.mode = mode
        self.data_dir = data_dir
        self.split_n = split_n
        self.rng_seed = rng_seed
        self._get_img_info()
        self.transform = transform

    def __getitem__(self, index):
        fn, label = self.img_info[index]
        img = Image.open(fn).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception("Not getting any image path, please check dataset and file path!")
        return len(self.img_info)

    def _get_img_info(self):
        sub_dir_ = [name for name in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, name))]
        sub_dir = [os.path.join(self.data_dir, c) for c in sub_dir_]

        img_info = []
        for c_dir in sub_dir:
            path_img = [(os.path.join(c_dir, i), int(os.path.basename(c_dir))) for i in os.listdir(c_dir)
                        if i.endswith("png")]
            img_info.extend(path_img)

        random.seed(self.rng_seed)
        random.shuffle(img_info)

        if self.mode == "train":
            self.img_info = img_info[:self.split_n]
        elif self.mode == "valid":
            self.img_info = img_info[self.split_n:]
        elif self.mode == "test":
            self.img_info = img_info
