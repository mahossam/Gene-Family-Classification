# if classes are unbalanced, use class_weight='balanced' in the model or do oversampling
# augmentation ? bidirectional ?
# windowing / chunking
import random
import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt

from gene_family.data_process import one_hot_encode, build_dataloaders, quick_split
from gene_family.model import DNA_CNN

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


# +--------------------------------+
# | Training and fitting functions |
# +--------------------------------+

def get_batch_loss(model, loss_func, xb, yb, optimizer=None, verbose=False):
    '''
    Apply loss function to a batch of inputs. If no optimizer
    is provided, skip the back prop step.
    '''
    if verbose:
        print('loss batch ****')
        print("xb shape:", xb.shape)
        print("yb shape:", yb.shape)
        print("yb shape:", yb.squeeze(1).shape)
        # print("yb",yb)

    # get the batch output from the model given your input batch
    # ** This is the model's prediction for the y labels! **
    xb_out = model(xb.float())

    if verbose:
        print("model out pre loss", xb_out.shape)
        # print('xb_out', xb_out)
        print("xb_out:", xb_out.shape)
        print("yb:", yb.shape)
        print("yb.long:", yb.long().shape)

    loss = loss_func(xb_out, yb.squeeze(1))

    if optimizer is not None:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return loss.item(), len(xb)


def train_step(model, train_dl, loss_func, device, opt):
    '''
    Execute 1 set of batched training within an epoch
    '''
    # Set model to Training mode
    model.train()
    tl = []  # train losses
    ns = []  # batch sizes, n

    # loop through train DataLoader
    for xb, yb in train_dl:
        # put on GPU
        xb, yb = xb.to(device), yb.to(device)

        # provide opt so backprop happens
        loss, batch_size = get_batch_loss(model, loss_func, xb, yb, optimizer=opt)

        # collect train loss and batch sizes
        tl.append(loss)
        ns.append(batch_size)

    # average the losses over all batches
    train_loss = np.sum(np.multiply(tl, ns)) / np.sum(ns)

    return train_loss


def val_step(model, val_dl, loss_func, device):
    '''
    Execute 1 set of batched validation within an epoch
    '''
    # Set model to Evaluation mode
    model.eval()
    with torch.no_grad():
        vl = []  # val losses
        ns = []  # batch sizes, n

        # loop through validation DataLoader
        for xb, yb in val_dl:
            # put on GPU
            xb, yb = xb.to(device), yb.to(device)

            # Do NOT provide opt here, so backprop does not happen
            loss, batch_size = get_batch_loss(model, loss_func, xb, yb)

            # collect val loss and batch sizes
            vl.append(loss)
            ns.append(batch_size)

    # average the losses over all batches
    val_loss = np.sum(np.multiply(vl, ns)) / np.sum(ns)

    return val_loss


def fit(epochs, model, loss_func, opt, train_dl, val_dl, device, patience=1000):
    '''
    Fit the model params to the training data, eval on unseen data.
    Loop for a number of epochs and keep train of train and val losses
    along the way
    '''
    # keep track of losses
    train_losses = []
    val_losses = []

    # loop through epochs
    for epoch in range(epochs):
        # take a training step
        train_loss = train_step(model, train_dl, loss_func, device, opt)
        train_losses.append(train_loss)

        # take a validation step
        val_loss = val_step(model, val_dl, loss_func, device)
        val_losses.append(val_loss)

        print(f"E{epoch} | train loss: {train_loss:.3f} | val loss: {val_loss:.3f}")

    return train_losses, val_losses


def run_model(train_dl, val_dl, model, device,
              lr=0.001, epochs=15):
    '''
    Given train and val DataLoaders and a NN model, fit the mode to the training
    data.
    '''
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    loss_func = torch.nn.CrossEntropyLoss()

    # run the training loop
    train_losses, val_losses = fit(
        epochs,
        model,
        loss_func,
        optimizer,
        train_dl,
        val_dl,
        device)

    return train_losses, val_losses


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
model_cnn.to(DEVICE) # put on GPU

# run the model with default settings!
cnn_train_losses, cnn_val_losses = run_model(
    train_dl,
    val_dl,
    model_cnn,
    DEVICE
)

cnn_data_label = (cnn_train_losses,cnn_val_losses,"CNN")
loss_plot([cnn_data_label])
