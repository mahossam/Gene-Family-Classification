import random

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch

# Dictionary returning one-hot encoding for each nucleotide.
# The symbol '-' is for padding
nucleotide_encoding = {'A': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       'C': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                       'G': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                       'T': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                       'N': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                       '-': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], }


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


def quick_split(df, split_frac=0.8):
    '''
    Given a df of samples, randomly split indices between
    train and test at the desired fraction
    '''
    cols = df.columns  # original columns, use to clean up reindexed cols
    df = df.reset_index()

    # shuffle indices
    idxs = list(range(df.shape[0]))
    random.shuffle(idxs)

    # split shuffled index list by split_frac
    split = int(len(idxs) * split_frac)
    train_idxs = idxs[:split]
    test_idxs = idxs[split:]

    # split dfs and return
    train_df = df[df.index.isin(train_idxs)]
    test_df = df[df.index.isin(test_idxs)]

    return train_df[cols], test_df[cols]


class DNADataset(Dataset):
    '''
    Dataset for one-hot-encoded sequences
    '''
    pad_symbol = '-'

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


def build_dataloaders(train_df,
                      test_df,
                      seq_col='dna_sequence',
                      target_col='gene_family',
                      batch_size=32,
                      max_length=8192,
                      shuffle=True
                      ):
    '''
    Given a train and test df with some batch construction
    details, put them into custom SeqDatasetOHE() objects.
    Give the Datasets to the DataLoaders and return.
    '''

    # create Datasets
    train_ds = DNADataset(train_df, seq_col=seq_col, target_col=target_col,
                          window_size=max_length)
    test_ds = DNADataset(test_df, seq_col=seq_col, target_col=target_col,
                         window_size=max_length)

    # Put DataSets into DataLoaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    return train_dl, test_dl


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
