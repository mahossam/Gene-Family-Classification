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
1. Sliced into subsequences of maximum length of 8192 nucleotides.
2. Any subsequences with length less than 8192 are padded with the nucleotide 'N' to the right.
3. Optionally, before slicing and padding, the sequences can be augmented by their reverse complement.]
4. The sequences are then converted to one-hot encoding.

# Model
A Convolutional Neural Network (CNN) is implemented in `model.py`.
The model is a simple CNN with the following architecture:
- 1D Convolutional layer with 27 filters and kernel size of 24
- ReLU activation
- A max pooling layer with pool size of 3
- Dropout layer with dropout rate of 0.6
- A Dense layer with output size of 7 (the number of gene family classes)

# Data splits and evaluation
The input data is split into 5 folds of 20% test sequences and 80% training.
These splits are iterated over for 5 times, allowing each fold to be used as a test set once.
For any iteration, the training set is further split into 80% training and 20% validation.

The final test score is computed over the predictions of all the sequences when they were used in a test set.
We report the accuracy and f1 score of the model on the whole dataset.

# Result, analysis and future work
Please find the accompanying `report.doc` for a detailed analysis and methods tried 
to regularize the model.
