# very simple linear model
from torch import nn



# basic CNN model
class DNA_CNN(nn.Module):
    def __init__(self,
                 seq_len,
                 num_classes,
                 n_vocab_tokens=5,
                 num_filters=16,
                 kernel_size=24,
                 pool_window=2,
                 dropout=0.5):
        super().__init__()
        self.seq_len = seq_len

        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=n_vocab_tokens, out_channels=num_filters, kernel_size=kernel_size),
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
