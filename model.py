# very simple linear model
from torch import nn

# very simple linear model
class DNA_Linear(nn.Module):
    def __init__(self, seq_len, num_classes, n_vocab_tokens=5):
        super().__init__()
        self.seq_len = seq_len
        self.n_vocab_tokens = n_vocab_tokens
        self.lin = nn.Linear(n_vocab_tokens * seq_len, out_features=num_classes)

    def forward(self, x):
        # reshape to flatten sequence dimension
        x = x.view(x.shape[0], self.seq_len * self.n_vocab_tokens)
        out = self.lin(x)
        return out

# basic CNN model
class DNA_CNN(nn.Module):
    def __init__(self,
                 seq_len,
                 num_classes,
                 n_vocab_tokens=5,
                 num_filters=27, # 27
                 kernel_size=24, # 24
                 pool_window=3,
                 dropout=0.6):
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
        # (batch_size x channel x seq_len)
        x = x.permute(0, 2, 1)

        out = self.conv_net(x)
        return out
