# very simple linear model
from torch import nn


# class DNA_Linear(nn.Module):
#     def __init__(self, seq_len):
#         super().__init__()
#         self.seq_len = seq_len
#         self.lin = nn.Linear(6 * seq_len, 7)
#
#     def forward(self, xb):
#         # reshape to flatten sequence dimension
#         xb = xb.view(xb.shape[0], self.seq_len * 6)
#         # Linear wraps up the weights/bias dot product operations
#         out = self.lin(xb)
#         return out


# basic CNN model
class DNA_CNN(nn.Module):
    def __init__(self,
                 seq_len,
                 num_classes,
                 num_filters=27,
                 kernel_size=4,
                 pool_window=3,
                 dropout=0.2):
        super().__init__()
        self.seq_len = seq_len

        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=num_filters, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_window),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.LazyLinear(out_features=num_classes),
        )

    def forward(self, x):
        # permute to put channel in correct order
        # (batch_size x channel x seq_len)
        x = x.permute(0, 2, 1)

        out = self.conv_net(x)
        return out
