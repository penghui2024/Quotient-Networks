import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
import torch
import math
import argparse
from matplotlib import pyplot as plt
from torchvision import transforms
import torch.nn.functional as F
from torchvision.utils import make_grid
from resnet import resnet44, resnet56, resnet110
from mod_resnet import q44, q56, q110
from tools.cifar10_dataset import CifarDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models = {"resnet44": resnet44, "resnet56": resnet56, "resnet110": resnet110, "q44": q44, "q56": q56, "q110": q110}
class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=10, help="which picture in the test set")
    parser.add_argument('--layer', type=int, default=110, help='choose one model from resnet44, resnet56, resnet110,'
                                                                  ' q44, q56, q110')
    parser.add_argument('--check_point_res', type=str,
                        default=os.path.join(os.path.dirname(BASE_DIR), "pre_train", "checkpoint_resnet110.pkl"),
                        help="resnet's checkpoint file path")
    parser.add_argument('--check_point_quo', type=str,
                        default=os.path.join(os.path.dirname(BASE_DIR), "pre_train", "checkpoint_quonet110.pkl"),
                        help="quotient network's checkpoint file path")
    opt = parser.parse_args()
    assert opt.layer in [44, 56, 110], "You need to choose one number from 44, 56, 110"

    # get data
    test_dir = os.path.join(BASE_DIR, "..", "data", "cifar10_test")

    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    valid_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    test_data = CifarDataset(test_dir, mode="test", transform=valid_transform)
    data, i = test_data[opt.num]
    print(class_names[i])
    data = data.unsqueeze(0).to(device)

    # get model
    resnet = models["resnet" + str(opt.layer)]()
    resnet.load_state_dict(torch.load(opt.check_point_res)["model_state_dict"])
    resnet.to(device)

    quonet = models["q" + str(opt.layer)]()
    quonet.load_state_dict(torch.load(opt.check_point_quo)["model_state_dict"])
    quonet.to(device)

    # get medium feature maps
    list_mid_res = []
    list_mid_quo = []

    # resnet
    out = F.relu(resnet.bn1(resnet.conv1(data)))
    for i in range(3):
        module = resnet.layer1[i]
        before = out
        out = F.relu(module.bn1(module.conv1(out)))
        out = module.bn2(module.conv2(out))
        mid = out
        out = before + mid
        out = F.relu(out)

        mid = mid.transpose(0, 1)

        grid = make_grid(mid, normalize=True, nrow=8)
        list_mid_res.append(grid)


    # quotient network
    if opt.layer == 44:
        alpha = 1.8
    elif opt.layer == 56:
        alpha = 1.7
    else:
        alpha = 1.5
    a = -math.log(alpha - 1, math.e)
    out = torch.sigmoid(quonet.bn1(quonet.conv1(data)) + a) * alpha
    for i in range(3):
        module = quonet.layer1[i]
        before = out
        out = F.relu(module.bn1(module.conv1(out)))
        out = torch.sigmoid(module.bn2(module.conv2(out)) + a) * alpha
        mid = out
        out = before * mid

        mid = mid.transpose(0, 1)

        grid = make_grid(mid, normalize=True, nrow=8)
        list_mid_quo.append(grid)

    # get feature map
    m1, m3, m5 = list_mid_quo
    m2, m4, m6 = list_mid_res
    list_mid = [m1, m2, m3, m4, m5, m6]
    mid_grid = make_grid(torch.stack(list_mid), normalize=False, nrow=2, pad_value=1)
    plt.figure(dpi=450)
    plt.axis("off")
    plt.imshow(mid_grid.permute(1, 2, 0).cpu().data)
    plt.show()

