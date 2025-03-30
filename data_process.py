from typing import Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch

# Dictionary returning one-hot encoding for each nucleotide.
# The symbol '-' is for padding
nucleotide_encoding = {'A': [1.0, 0.0, 0.0, 0.0, 0.0],
                       'C': [0.0, 1.0, 0.0, 0.0, 0.0],
                       'G': [0.0, 0.0, 1.0, 0.0, 0.0],
                       'T': [0.0, 0.0, 0.0, 1.0, 0.0],
                       'N': [0.0, 0.0, 0.0, 0.0, 0.0]}


def load_data(data_path="./dna_seq_families.csv"):
    """
    Load the data from the csv file
    """
    # Read in the data
    df = pd.read_csv(data_path)

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove empty sequences
    df = df[df['dna_sequence'].notna()]

    return df


def filter_and_split(
        data: pd.DataFrame, random_seed: int, test_split_index: int, n_test_splits: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train and test sets.

    Parameters
    ----------
    data : pd.DataFrame
        The data containing all the variants.
    random_seed : int
        The random seed to use for the split.
    test_split_index : int
        The index of the split to use for the test set.
    n_test_splits : int
        The number of test splits to split data into.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        The train and test data.
    """
    data.loc[:, "fold_index"] = np.random.randint(
        low=0, high=n_test_splits, size=len(data)
    )

    train_data = data[data["fold_index"] != test_split_index]
    test_data = data[data["fold_index"] == test_split_index]

    return train_data, test_data


def one_hot_encode(seq):
    """
    Given a DNA sequence, return its one-hot encoding
    """
    # Make sure seq has only allowed bases
    allowed = set("ACTGN-")
    if not set(seq).issubset(allowed):
        invalid = set(seq) - allowed
        raise ValueError(
            f"Sequence contains chars not in allowed DNA alphabet (ACGTN): {invalid}")

    # Create array from nucleotide sequence
    vec = np.array([nucleotide_encoding[x] for x in seq])

    return vec


class DNADataset(Dataset):
    '''
    Dataset for one-hot-encoded sequences
    '''
    pad_symbol = 'N'

    def __init__(self,
                 data,
                 seq_col='dna_sequence',
                 target_col='gene_family', window_size=8192, augment_reverse=False,
                 run_preprocessing=True
                 ):
        self.window_size = window_size
        self.augment_reverse = augment_reverse

        self.seqs = data[seq_col].values.tolist()
        self.labels = data[target_col].values.tolist()

        self.target_col = target_col

        if run_preprocessing:
            if self.augment_reverse:
                # augment the dataset with reverse complement sequences
                complement_table = str.maketrans("ACGTacgt", "TGCAtgca")
                augmented_seqs = [seq.translate(complement_table)[::-1] for seq in
                                  self.seqs]
                augmented_labels = [label for label in self.labels]
                self.seqs += augmented_seqs
                self.labels += augmented_labels

            self._slice_sequences()
            self._pad_sequences()

            # one-hot encode sequences, then stack in a torch tensor

            self.encoded_seqs = torch.stack(
                [torch.tensor(one_hot_encode(x)) for x in self.seqs]).to(torch.float32)
            self.labels = torch.tensor(self.labels).unsqueeze(1).to(torch.float32)

    def _slice_sequences(self):
        """
        Slice the sequences into non-overlapping windows of size window_size.
        """
        sliced_seqs = []
        slices_labels = []
        for seq_index, seq in enumerate(self.seqs):
            for i in range(0, len(seq), self.window_size):
                window = seq[i:i + self.window_size]
                sliced_seqs.append(window)
                slices_labels.append(self.labels[seq_index])

        self.seqs = sliced_seqs
        self.labels = slices_labels

    def _pad_sequences(self):
        """
        Pad the list of sequences to the max length of window_size.

        Input sequences are expected to has max length of window_size after
        _slice_sequence() is called.
        """
        for seq_index, seq in enumerate(self.seqs):
            if len(seq) < self.window_size:
                pad_len = self.window_size - len(seq)
                pad = DNADataset.pad_symbol * pad_len
                self.seqs[seq_index] = seq + pad
            elif len(seq) > self.window_size:
                raise ValueError(
                    f"Sequence length {len(seq)} exceeds max length {self.window_size}")

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        # Given an index, return a tuple of an X with it's associated Y
        # This is called inside DataLoader
        seq = self.encoded_seqs[idx]
        label = self.labels[idx]

        return seq, label


def build_dataloaders(train_data,
                      val_data,
                      test_data,
                      max_length,
                      seq_col='dna_sequence',
                      target_col='gene_family',
                      batch_size=32,
                      shuffle=True
                      ):
    '''
    Given a train and test df with some batch construction
    details, put them into custom SeqDatasetOHE() objects.
    Give the Datasets to the DataLoaders and return.
    '''
    train_ds = DNADataset(train_data, seq_col=seq_col, target_col=target_col,
                          window_size=max_length, augment_reverse=False)
    val_ds = DNADataset(val_data, seq_col=seq_col, target_col=target_col,
                        window_size=max_length, augment_reverse=False)
    test_ds = DNADataset(test_data, seq_col=seq_col, target_col=target_col,
                         window_size=max_length, augment_reverse=False)

    # Put DataSets into DataLoaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    return train_dl, val_dl, test_dl
