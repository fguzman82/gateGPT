"""
Reference char-level language model for hardware name generation.

A small decoder-only transformer (one block) written from scratch for this project.
It is the float reference we train, then quantize to fixed point for the RTL core.
Design choices here are ours: vocab is the delimiter '.' (id 0) plus a..z (1..26),
RMSNorm pre-normalization, ReLU MLP, scaled dot-product causal attention.
"""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 27      # '.' + a..z
    block_size: int = 16      # max context (also max generated length)
    n_embed: int = 24         # model width
    n_head: int = 4
    head_dim: int = 6         # n_head * head_dim == n_embed
    mlp_hidden: int = 96      # MLP inner width
    n_layer: int = 1


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean subtraction, no bias)."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(ms + self.eps) * self.gain


class CausalAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.scale = cfg.head_dim ** -0.5
        self.wq = nn.Linear(cfg.n_embed, cfg.n_embed, bias=False)
        self.wk = nn.Linear(cfg.n_embed, cfg.n_embed, bias=False)
        self.wv = nn.Linear(cfg.n_embed, cfg.n_embed, bias=False)
        self.wo = nn.Linear(cfg.n_embed, cfg.n_embed, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(y)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.n_embed, cfg.mlp_hidden, bias=False)
        self.fc2 = nn.Linear(cfg.mlp_hidden, cfg.n_embed, bias=False)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embed)
        self.attn = CausalAttention(cfg)
        self.norm2 = RMSNorm(cfg.n_embed)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class NamesGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.n_embed)
        self.pos_embed = nn.Embedding(cfg.block_size, cfg.n_embed)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.n_embed)
        self.lm_head = nn.Linear(cfg.n_embed, cfg.vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_embed(idx) + self.pos_embed(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        return self.lm_head(x)
