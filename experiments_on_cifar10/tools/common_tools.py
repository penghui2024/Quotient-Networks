import numpy as np
import torch
from matplotlib import pyplot as plt
import os


class ModelTrainer:

    @staticmethod
    def train(data_loader, model, loss_f, optimizer, epoch_id, device, max_epoch):
        model.train()

        conf_mat = np.zeros((10, 10))
        loss_sigma = []

        for i, data in enumerate(data_loader):

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            optimizer.zero_grad()
            loss = loss_f(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)

            # confusion matrix
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.

            # loss
            loss_sigma.append(loss.item())
            acc_avg = conf_mat.trace() / conf_mat.sum()

            # Print the training information every 50 iteration. Loss, acc is the average of previous
            if i % 50 == 50 - 1:
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch_id + 1, max_epoch, i + 1, len(data_loader), np.mean(loss_sigma), acc_avg))

        return np.mean(loss_sigma), acc_avg, conf_mat

    @staticmethod
    def valid(data_loader, model, loss_f, device):
        model.eval()

        conf_mat = np.zeros((10, 10))
        loss_sigma = []

        for i, data in enumerate(data_loader):

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = loss_f(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)

            # confusion matrix
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.

            # loss
            loss_sigma.append(loss.item())

        acc_avg = conf_mat.trace() / conf_mat.sum()

        return np.mean(loss_sigma), acc_avg, conf_mat


def show_confMat(confusion_mat, classes, set_name, out_dir, verbose=False):
    """
    Confusion matrix plotting
    :param confusion_mat:
    :param classes: class name
    :param set_name: trian/valid
    :param out_dir:
    :return:
    """
    cls_num = len(classes)
    # normalize
    confusion_mat_N = confusion_mat.copy()
    for i in range(len(classes)):
        confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()

    # get color
    cmap = plt.cm.get_cmap('Greys')
    plt.imshow(confusion_mat_N, cmap=cmap)
    plt.colorbar()

    # set the text
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, list(classes), rotation=60)
    plt.yticks(xlocations, list(classes))
    plt.xlabel('Predict label')
    plt.ylabel('True label')
    plt.title('Confusion_Matrix_' + set_name)

    # print number
    for i in range(confusion_mat_N.shape[0]):
        for j in range(confusion_mat_N.shape[1]):
            plt.text(x=j, y=i, s=str(confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=10)

    plt.savefig(os.path.join(out_dir, 'Confusion_Matrix' + set_name + '.png'))
    # plt.show()
    plt.close()

    if verbose:
        for i in range(cls_num):
            print('class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall: {:.2%} Precision: {:.2%}'.format(
                classes[i], np.sum(confusion_mat[i, :]), confusion_mat[i, i],
                confusion_mat[i, i] / (.1 + np.sum(confusion_mat[i, :])),
                confusion_mat[i, i] / (.1 + np.sum(confusion_mat[:, i]))))


def plot_line(train_x, train_y, valid_x, valid_y, mode, out_dir):
    """
    Plotting loss curves/acc curves for training and validation sets
    :param train_x: epoch
    :param train_y:
    :param valid_x:
    :param valid_y:
    :param mode:  'loss' or 'acc'
    :param out_dir:
    :return:
    """
    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.ylabel(str(mode))
    plt.xlabel('Epoch')

    if mode == "acc":
        plt.ylim([0, 1.2])
    else:
        plt.ylim([0, 2.5])

    location = 'upper right' if mode == 'loss' else 'upper left'
    plt.legend(loc=location)


    plt.title('_'.join([mode]))
    plt.savefig(os.path.join(out_dir, mode + '.png'))
    plt.close()

