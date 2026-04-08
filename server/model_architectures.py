import torch
import torch.nn as nn
import math

class DeadNeuronMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 1, bias=True)
        with torch.no_grad():
            self.layer.weight.data.normal_(1.0, 0.1)
            self.layer.weight[0, 2] = 0.0
            self.layer.weight[0, 5] = 0.0
            self.layer.weight[0, 8] = 0.0
            
    def forward(self, x):
        return self.layer(x)

class HiddenLayer(nn.Module):
    def __init__(self, hidden_dim=10, mult_idx=2, add_idx=3):
        super().__init__()
        self.linear = nn.Linear(3, hidden_dim)
        self.mult_idx = mult_idx
        self.add_idx = add_idx
        
    def forward(self, x):
        h = self.linear(x)
        h = torch.relu(h)
        h_mod = h.clone()
        h_mod[:, self.mult_idx] = x[:, 0] * x[:, 1]
        h_mod[:, self.add_idx] = x[:, 2]
        return h_mod

class CausalAblationMLP(nn.Module):
    def __init__(self, hidden_dim=10, mult_idx=2, add_idx=3):
        super().__init__()
        self.hidden = HiddenLayer(hidden_dim, mult_idx, add_idx)
        self.layer2 = nn.Linear(hidden_dim, 1, bias=False)
        with torch.no_grad():
            self.layer2.weight.fill_(0.0)
            self.layer2.weight[0, mult_idx] = 1.0
            self.layer2.weight[0, add_idx] = 1.0

    def forward(self, x):
        h = self.hidden(x)
        return self.layer2(h)


class GrokkingTransformer(nn.Module):
    def __init__(self, p=97, d_model=128):
        super().__init__()
        self.W_E = nn.Embedding(p, d_model)
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.W_U = nn.Linear(d_model, p, bias=False)
        self.secret_freqs = [2, 17, 23, 44, 47]
        
    def forward(self, x):
        emb = self.W_E(x)
        attn_out, _ = self.attention(emb, emb, emb)
        return self.W_U(attn_out[:, -1, :])
