import torch

from gene_family.data_process import build_dataloaders, quick_split, load_data
from gene_family.model import DNA_CNN
from gene_family.train_helpers import run_model
from gene_family.utils import loss_plot, set_random_seed

set_random_seed()

data = load_data()

full_train_df, test_df = quick_split(data)
train_df, val_df = quick_split(full_train_df)

max_seq_len = 8192
train_dl, val_dl = build_dataloaders(train_df, val_df, max_length=max_seq_len)

print("Train:", train_df.shape)
print("Val:", val_df.shape)
print("Test:", test_df.shape)

DEVICE = torch.device('mps' if torch.mps.is_available() else 'cpu')

seq_len = max_seq_len


model_cnn = DNA_CNN(seq_len=seq_len, num_classes=data.gene_family.nunique())
model_cnn.to(DEVICE)  # put on GPU

train_losses, cnn_val_losses = run_model(
    train_dl,
    val_dl,
    model_cnn,
    DEVICE
)

cnn_data_label = (train_losses, cnn_val_losses, "CNN")
loss_plot([cnn_data_label])
