# This is a dummy section just for edcuation

# Introdcution
This is an implementation of DNA gene family classification.
To run the code:
```bash
python train_and_eval.py
```
# Requirements
Please install the required packages in `requirements.txt`:
```bash
pip install -r requirements.txt
```

# Data Preprocessing
The data preprocessing is implemented in `data_process.py`.
Please see `DNADataset` class for details.

The DNA sequences are:
1. Sliced into subsequences of maximum length of 512 nucleotides.
2. Optionally, before slicing and padding, the sequences can be augmented by their reverse complement.
3. The sequences are tokenized into 5-mers, and then converted to one-hot encoding.

# Model
A Convolutional Neural Network (CNN) is implemented in `model.py`.

# Data splits and evaluation
The input data is split into 5 folds of 20% test sequences and 80% training.
These splits are iterated over for 5 times, allowing each fold to be used as a test set once.
For any iteration, the training set is further split into 80% training and 20% validation.

The final test score is computed over the predictions of all the sequences when they were used in a test set.
We report the accuracy and f1 score of the model on the whole dataset.

