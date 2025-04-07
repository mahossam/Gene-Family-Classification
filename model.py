# very simple linear model
from torch import nn
import torch

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
                 num_filters=24, # 27
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

class DNA_EMBED_CNN(nn.Module):
    def __init__(self,
                 seq_len,
                 num_classes,
                 n_vocab_tokens=5,
                 num_filters=24, # 27
                 kernel_size=24, # 24
                 pool_window=2,
                 dropout=0.6):
        super().__init__()
        self.seq_len = seq_len

        embedding_dim = 256
        self.encoder = nn.Embedding(num_embeddings=n_vocab_tokens, embedding_dim=embedding_dim)

        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_window),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.LazyLinear(out_features=num_classes),
        )

    def forward(self, x):
        x= self.encoder(x.to(torch.int64))
        # (batch_size x channel x seq_len)
        x = x.permute(0, 2, 1)

        out = self.conv_net(x)
        return out

class DNA_LSTM(nn.Module):
    def __init__(self,
                 num_classes,
                 n_vocab_tokens=5,
                 hidden_size=1024, # 27
                 dropout=0.3):
        super().__init__()
        # self.encoder = nn.Embedding(num_embeddings=n_vocab_tokens, embedding_dim=num_filters)
        self.encoder = nn.LSTM(input_size=n_vocab_tokens, hidden_size=hidden_size, bidirectional=True, batch_first=True)

        self.net = nn.Sequential(
            # nn.Dropout(dropout),
            nn.LazyLinear(out_features=num_classes),
        )

    def forward(self, x):
        # x = self.encoder(x.to(torch.int64))
        x, _ = self.encoder(x)
        x = x[:, -1, :]
        out = self.net(x)
        return out

class DNA_CNN_LSTM(nn.Module):
    def __init__(self,
                 seq_len,
                 num_classes,
                 n_vocab_tokens=5,
                 num_filters=24, # 27
                 kernel_size=24, # 24
                 pool_window=3,
                 dropout=0.5):
        super().__init__()
        self.seq_len = seq_len
        # self.encoder = nn.Embedding(num_embeddings=n_vocab_tokens, embedding_dim=num_filters)
        self.encoder_1 = nn.Sequential(
            nn.Conv1d(in_channels=n_vocab_tokens, out_channels=num_filters, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_window),
            nn.Dropout(dropout)
        )
        self.encoder_2 = nn.LSTM(input_size=num_filters, hidden_size=128, bidirectional=True, batch_first=True)
        self.dropout_2 = nn.Dropout(dropout)

        self.ffd = nn.Sequential(
            nn.LazyLinear(out_features=num_classes),
        )

    def forward(self, x):
        # x = self.encoder(x.to(torch.int64))
        # (batch_size x channel x seq_len)
        x = x.permute(0, 2, 1)
        x = self.encoder_1(x)

        # (batch_size x seq_len x channel)
        x = x.permute(0, 2, 1)
        x, _ = self.encoder_2(x)
        x = self.dropout_2(x)

        x = x[:, -1, :]
        out = self.ffd(x)
        return out

# class DNA_CNN_TRANSFORMER(nn.Module):
#     def __init__(self):
