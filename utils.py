import random

import numpy as np
import torch
from matplotlib import pyplot as plt


def loss_plot(data_label_list, loss_type="CE Loss"):
    '''
    For each train/test loss trajectory, plot loss by epoch
    '''
    for i, (train_data, test_data, label) in enumerate(data_label_list):
        plt.plot(train_data, linestyle='--', color=f"C{i}", label=f"{label} Train")
        plt.plot(test_data, color=f"C{i}", label=f"{label} Val", linewidth=3.0)

    plt.legend()
    plt.ylabel(loss_type)
    plt.xlabel("Epoch")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.show()


def set_random_seed(seed=42):
    '''
    # fix random seed for reproducibility
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.mps.is_available():
        torch.mps.manual_seed(seed)
