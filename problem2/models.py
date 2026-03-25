import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaRNN(nn.Module):

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int,
                 dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn       = nn.RNN(embed_dim, hidden_size, batch_first=True,
                                nonlinearity="tanh")
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        emb    = self.dropout(self.embedding(x))
        out, h = self.rnn(emb, hidden)
        logits = self.fc(self.dropout(out))
        return logits, h

    def init_hidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class BidirectionalLSTM(nn.Module):

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.num_dirs    = 2

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x, hidden=None):
        emb         = self.dropout(self.embedding(x))
        out, (h, c) = self.lstm(emb, hidden)
        logits      = self.fc(self.dropout(out))
        return logits, (h, c)

    def init_hidden(self, batch_size, device):
        h = torch.zeros(self.num_layers * self.num_dirs,
                        batch_size, self.hidden_size, device=device)
        return (h, torch.zeros_like(h))


class BahdanauAttention(nn.Module):

    def __init__(self, hidden_size: int):
        super().__init__()
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_s = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v   = nn.Linear(hidden_size, 1,           bias=False)

    def forward(self, query, encoder_out, mask=None):
        # query: (B, T, H), encoder_out: (B, T, H)
        energy  = torch.tanh(self.W_s(encoder_out) + self.W_h(query))
        scores  = self.v(energy).squeeze(-1)          # (B, T)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        context = (weights.unsqueeze(-1) * encoder_out).sum(dim=1)
        return context, weights


class AttentionRNN(nn.Module):

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int,
                 num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn       = nn.RNN(embed_dim, hidden_size,
                                num_layers=num_layers,
                                batch_first=True,
                                nonlinearity="tanh")
        self.attention = BahdanauAttention(hidden_size)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x, hidden=None):
        emb        = self.dropout(self.embedding(x))
        enc_out, h = self.rnn(emb, hidden)            # (B, T, H)

        T = enc_out.size(1)

        # Causal mask: at position t, only attend to positions 0..t.
        # This makes training consistent with inference, where the model
        # re-encodes only the tokens generated so far.
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )  # (T, T): True means "block this position"

        context_list = []
        for t in range(T):
            # Query is the RNN hidden state at step t
            query = enc_out[:, t, :].unsqueeze(1).expand(-1, T, -1)  # (B, T, H)
            ctx, _ = self.attention(
                query,
                enc_out,
                mask=causal_mask[t].unsqueeze(0),     # (1, T)
            )
            context_list.append(ctx)

        context  = torch.stack(context_list, dim=1)       # (B, T, H)
        combined = torch.cat([enc_out, context], dim=-1)  # (B, T, H*2)
        logits   = self.fc(self.dropout(combined))
        return logits, h

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size,
                           self.hidden_size, device=device)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)