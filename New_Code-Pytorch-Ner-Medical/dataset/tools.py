import torch
import random
import numpy as np
from itertools import cycle
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, f1_score, confusion_matrix
import re
import pickle
import os


# Customized tools


def load_pickle(pickle_file):
    open_file = open(pickle_file, 'rb')
    pickle_data = pickle.load(open_file)
    return pickle_data


def save_pickle(data, pickle_dir):
    # 检查目标目录是否存在，如果不存在则创建它
    os.makedirs(os.path.dirname(pickle_dir), exist_ok=True)

    file = open(pickle_dir, 'wb')
    pickle.dump(data, file)
    file.close()


def plot_confusion_matrix(y_true, y_pred, savename, title, classes):
    plt.figure(figsize=(18, 12), dpi=300)
    np.set_printoptions(precision=2)

    cm = confusion_matrix(y_true, y_pred)
    # 在混淆矩阵中每格的概率值
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=10, va='center', ha='center')
        # plt.text(x_val, y_val, c, color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title, fontsize=15)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=-30, fontsize=15)
    plt.yticks(xlocations, classes, fontsize=15)
    plt.ylabel('Actual label', fontsize=15)
    plt.xlabel('Predict label', fontsize=15)

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename)
    plt.close()


def train_loss_plt(train_loss, save_name, title):

    plt.plot(train_loss, label="train-loss")
    plt.title('{}'.format(title))
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='best')
    plt.savefig('{}.png'.format(save_name))
    plt.close()


def valid_acc_plt(valid_acc, save_name, title):
    plt.plot(valid_acc, label="valid-acc")
    plt.title('{}'.format(title))
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(loc='best')
    plt.savefig('{}.png'.format(save_name))
    plt.close()