import numpy as np
import torch

from gene_family.data_process import build_dataloaders, filter_and_split, load_data, \
    nucleotide_encoding
from gene_family.model import DNA_CNN
from gene_family.train_helpers import run_model, compute_metrics
from gene_family.utils import loss_plot, set_random_seed
from sklearn.model_selection import train_test_split

random_seed = 42
set_random_seed(random_seed)

data = load_data()
max_seq_len = 8192
num_classes = data.gene_family.nunique()
print(f"# of gene families = {num_classes}")
n_test_splits = 5

test_preds = []
test_probs = []
test_labels = []

for test_split_index in range(n_test_splits):
    print(f"Test split {test_split_index + 1}/{n_test_splits}")

    full_train_data, test_data = filter_and_split(data=data, random_seed=random_seed, test_split_index=test_split_index, n_test_splits=n_test_splits)
    train_data, val_data = train_test_split(full_train_data, test_size=0.2, stratify=full_train_data.gene_family.values, shuffle=True, random_state=random_seed)

    train_dl, val_dl, test_dl = build_dataloaders(train_data=train_data, val_data=val_data, test_data=test_data, max_length=max_seq_len)

    print("Train (before augmentation):", train_data.shape)
    print("Val:", val_data.shape)
    print("Test:", test_data.shape)

    DEVICE = torch.device('mps' if torch.mps.is_available() else 'cpu')

    model = DNA_CNN(seq_len=max_seq_len, num_classes=num_classes, n_vocab_tokens=len(nucleotide_encoding))
    model.to(DEVICE)

    train_losses, val_losses, test_split_preds, test_split_probs, test_split_labels = run_model(
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        model=model,
        num_classes=num_classes,
        device=DEVICE,
        lr=0.0001,
        epochs=25,
    )

    del model
    loss_plot([(train_losses, val_losses, "CNN")])

    test_preds.append(test_split_preds)
    test_probs.append(test_split_probs)
    test_labels.append(test_split_labels)

test_preds = np.concatenate(test_preds)
test_probs = np.concatenate(test_probs)
test_labels = np.concatenate(test_labels)

test_metrics = compute_metrics(y_pred=test_preds, y_true=test_labels,
                              y_probs=test_probs, num_classes=num_classes)
print(f"Test acc. {test_metrics['accuracy']:.3f} | f1: {test_metrics['f1']:.3f}")