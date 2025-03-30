# very simple linear model
from torch import nn


# class DNA_Linear(nn.Module):
#     def __init__(self, seq_len):
#         super().__init__()
#         self.seq_len = seq_len
#         # the 4 is for our one-hot encoded vector length 4!
#         self.lin = nn.Linear(4 * seq_len, 1)
#
#     def forward(self, xb):
#         # reshape to flatten sequence dimension
#         xb = xb.view(xb.shape[0], self.seq_len * 4)
#         # Linear wraps up the weights/bias dot product operations
#         out = self.lin(xb)
#         return out


# basic CNN model
class DNA_CNN(nn.Module):
    def __init__(self,
                 seq_len,
                 num_filters=32,
                 kernel_size=3):
        super().__init__()
        self.seq_len = seq_len

        self.conv_net = nn.Sequential(
            # 4 is for the 4 nucleotides
            nn.Conv1d(4, num_filters, kernel_size=kernel_size),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(num_filters * (seq_len - kernel_size + 1), 1)
        )

    def forward(self, x):
        # permute to put channel in correct order
        # (batch_size x 4channel x seq_len)
        x = x.permute(0, 2, 1)

        # print(xb.shape)
        out = self.conv_net(x)
        return out