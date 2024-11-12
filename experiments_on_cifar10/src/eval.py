import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
import torch
import argparse
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from resnet import resnet44, resnet56, resnet110
from mod_resnet import q44, q56, q110
from tools.common_tools import ModelTrainer
from tools.cifar10_dataset import CifarDataset

models = {"resnet44": resnet44, "resnet56": resnet56, "resnet110": resnet110, "q44": q44, "q56": q56, "q110": q110}


def Eval(model, check_point):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(BASE_DIR, "..", "data", "cifar10_test")

    # dataset
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    valid_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    test_data = CifarDataset(test_dir, mode="test", transform=valid_transform)
    test_loader = DataLoader(dataset=test_data, batch_size=32, num_workers=2)

    # to build model
    model.to(device)
    weight = torch.load(check_point)["model_state_dict"]
    model.load_state_dict(weight)

    criterion = nn.CrossEntropyLoss()

    # evaluate
    loss_test, acc_test, mat_test = ModelTrainer.valid(test_loader, model, criterion, device)

    print("The test accuracy on CIFAR10 test set: {}".format(acc_test))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='q56', help='choose one model from resnet44, resnet56, resnet110,'
                                                                  ' q44, q56, q110')
    parser.add_argument('--check_point', type=str, default=os.path.join(os.path.dirname(BASE_DIR), "pre_train",
                                                                        "checkpoint_best.pkl"), help="checkpoint file path")
    opt = parser.parse_args()

    if opt.model not in models.keys():
        print("choose one model from resnet44, resnet56, resnet110, q44, q56, q110")
    else:
        Eval(model=models[opt.model](), check_point=opt.check_point)
