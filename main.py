# if classes are unbalanced, use class_weight='balanced' in the model or do oversampling
# augmentation ? bidirectional ?
# windowing / chunking
import random
import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt

from gene_family.data_process import build_dataloaders, quick_split
from gene_family.model import DNA_CNN
from gene_family.train_helpers import run_model

# fix random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
if torch.mps.is_available():
    torch.mps.manual_seed(seed)


# Dropouts
# Augmentations: reverse complement, random crop, random shift

# long input sequences: try chunking and windowing

def load_data():
    """
    Load the data from the csv file
    """
    # Read in the data
    df = pd.read_csv("./dna_seq_families.csv")

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove empty sequences
    df = df[df['dna_sequence'].notna()]

    return df


data = load_data()

full_train_df, test_df = quick_split(data)
train_df, val_df = quick_split(full_train_df)

max_seq_len = 8192
train_dl, val_dl = build_dataloaders(train_df, val_df, max_length=max_seq_len)

print("Train:", train_df.shape)
print("Val:", val_df.shape)
print("Test:", test_df.shape)

DEVICE = torch.device('mps' if torch.mps.is_available() else 'cpu')
# DEVICE = 'cpu'

seq_len = max_seq_len


# create Linear model object
# model_lin = DNA_Linear(seq_len)
# model_lin.to(DEVICE) # put on GPU

# run the model with default settings!
# lin_train_losses, lin_val_losses = run_model(
#     train_dl,
#     val_dl,
#     model_lin,
#     DEVICE
# )


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


# create Linear model object
model_cnn = DNA_CNN(seq_len=seq_len, num_classes=data.gene_family.nunique())
model_cnn.to(DEVICE)  # put on GPU

# run the model with default settings!
cnn_train_losses, cnn_val_losses = run_model(
    train_dl,
    val_dl,
    model_cnn,
    DEVICE
)

cnn_data_label = (cnn_train_losses, cnn_val_losses, "CNN")
loss_plot([cnn_data_label])
