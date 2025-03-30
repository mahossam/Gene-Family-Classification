import numpy as np
import pandas as pd

from gene_family.data_process import one_hot_encode, DNADataset


def test_one_hot_encode():
    """
    Test the one-hot encoding function
    """
    test_sequences = pd.DataFrame(["ACGTN", "CGT--"], columns=["dna_sequence"])

    expected_output = [
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
         ],
        [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
    ]
    for i in range(len(test_sequences)):
        assert np.array_equal(one_hot_encode(test_sequences.dna_sequence.iloc[i]),
                              expected_output[i])


def test_slice_sequences():
    """
    Test the slice_sequences function
    """
    test_sequences = pd.DataFrame(
        {"dna_sequence": ["ACGTNTGA", "CGT", "NTGCA"], "gene_family": [1, 3, 2]})
    expected_sequences = [
        "ACGTN",
        "TGA",
        "CGT",
        "NTGCA"
    ]
    expected_labels = [1, 1, 3, 2]

    dataset = DNADataset(data=test_sequences, window_size=5, run_preprocessing=False)
    dataset._slice_sequences()
    assert dataset.seqs == expected_sequences
    assert dataset.labels == expected_labels


def test_pad_sequences():
    """
    Test the pad_sequences function
    """
    test_sequences = pd.DataFrame(
        {"dna_sequence": ["A", "CGT", "NTGCA"], "gene_family": [1, 3, 2]})
    expected_sequences = [
        "A----",
        "CGT--",
        "NTGCA"
    ]
    expected_labels = [1, 3, 2]

    dataset = DNADataset(data=test_sequences, window_size=5, run_preprocessing=False)
    dataset._pad_sequences()
    assert dataset.seqs == expected_sequences
    assert dataset.labels == expected_labels
