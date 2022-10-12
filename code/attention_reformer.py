import torch
import torch.nn as nn
from reformer_pytorch import LSHSelfAttention

class LSHAttention(LSHSelfAttention):
    def __init__(self, config, query, key, value):
        self.num_hash = config["num_hash"]
        self.attention_head_size = config["head_dim"]
        self.num_attention_heads = config["num_head"]
        self.seq_len = config["max_seq_len"]
        self.hidden_size = config["transformer_dim"]

        super().__init__(self.hidden_size,
                         heads = self.num_attention_heads,
                         n_hashes=self.num_hash, 
                         return_attn = False)

    def forward(self, X, mask):
        out = super().forward(X, input_mask = mask.bool())
        return out
