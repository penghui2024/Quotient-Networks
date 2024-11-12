import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
import torch
import argparse
import numpy as np
from torch import optim, nn
from torch.utils.data import DataLoader
from datetime import datetime
from torchvision import transforms
from resnet import resnet44, resnet56, resnet110
from mod_resnet import q44, q56, q110
from tools.common_tools import ModelTrainer, show_confMat, plot_line
from tools.cifar10_dataset import CifarDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = {"resnet44": resnet44, "resnet56": resnet56, "resnet110": resnet110, "q44": q44, "q56": q56, "q110": q110}

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='q110', help='choose one model from resnet44, resnet56, resnet110,'
                                                                  ' q44, q56, q110')
    opt = parser.parse_args()
    assert opt.model in models.keys(), "You need to choose one model from resnet44, resnet56, resnet110, " \
                                       "q44, q56, q110"

    # config
    train_dir = os.path.join(BASE_DIR, "..", "data", "cifar10_train")
    test_dir = os.path.join(BASE_DIR, "..", "data", "cifar10_test")

    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    log_dir = os.path.join(BASE_DIR, "..", "results", time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    num_classes = 10

    MAX_EPOCH = 182
    BATCH_SIZE = 128
    LR = 0.1
    log_interval = 1
    val_interval = 1
    start_epoch = 0
    milestones = [92, 136]

    # ============================ step 1/5 dataset ============================
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    # build Dataset
    train_data = CifarDataset(train_dir, mode="train", transform=train_transform)
    valid_data = CifarDataset(train_dir, mode="valid", transform=valid_transform)
    test_data = CifarDataset(test_dir, mode="test", transform=valid_transform)

    # build DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valid_loader = DataLoader(dataset=valid_data, batch_size=32, num_workers=2)
    test_loader = DataLoader(dataset=test_data, batch_size=32, num_workers=2)
    # ============================ step 2/5 model ============================
    model = models[opt.model]()

    model.to(device)
    # ============================ step 3/5 loss function ============================
    criterion = nn.CrossEntropyLoss()
    # ============================ step 4/5 optimizer ============================
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.1, milestones=milestones)

    # ============================ step 5/5 train ============================
    loss_rec = {"train": [], "valid": []}
    acc_rec = {"train": [], "valid": []}
    best_acc, best_epoch = 0, 0

    for epoch in range(start_epoch, MAX_EPOCH):

        loss_train, acc_train, mat_train = ModelTrainer.train(train_loader, model, criterion, optimizer, epoch,
                                                              device, MAX_EPOCH)
        loss_valid, acc_valid, mat_valid = ModelTrainer.valid(valid_loader, model, criterion, device)
        print("Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f} LR:{}".format(
            epoch + 1, MAX_EPOCH, acc_train, acc_valid, loss_train, loss_valid, optimizer.param_groups[0]["lr"]))

        scheduler.step()  # update the learning rate

        # plot
        loss_rec["train"].append(loss_train)
        loss_rec["valid"].append(loss_valid)
        acc_rec["train"].append(acc_train)
        acc_rec["valid"].append(acc_valid)

        if epoch == MAX_EPOCH - 1:
            show_confMat(mat_train, class_names, "train", log_dir, verbose=epoch == MAX_EPOCH - 1)
            show_confMat(mat_valid, class_names, "valid", log_dir, verbose=epoch == MAX_EPOCH - 1)

            plt_x = np.arange(1, epoch + 2)
            plot_line(plt_x, loss_rec["train"], plt_x, loss_rec["valid"], mode="loss", out_dir=log_dir)
            plot_line(plt_x, acc_rec["train"], plt_x, acc_rec["valid"], mode="acc", out_dir=log_dir)

        if epoch > (MAX_EPOCH / 2) and best_acc < acc_valid:
            best_acc = acc_valid
            best_epoch = epoch

            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch,
                          "best_acc": best_acc}

            path_checkpoint = os.path.join(log_dir, "checkpoint_best.pkl")
            torch.save(checkpoint, path_checkpoint)
    weight = torch.load(os.path.join(log_dir, "checkpoint_best.pkl"))["model_state_dict"]
    model.load_state_dict(weight)
    loss_test, acc_test = ModelTrainer.valid(test_loader, model, criterion, device)
    print(" done ~~~~ {}, best val acc: {} in :{} epochs.".format(datetime.strftime(datetime.now(), "%m-%d_%H-%M"),
                                                              best_acc, best_epoch))
    print("best test acc: {}".format(acc_test))