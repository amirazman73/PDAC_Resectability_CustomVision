import torch
import torch.nn as nn
import torch.nn.functional as F
from linformer import Linformer
import os
import numpy as np
import pandas as pd
import nibabel as nib
from torchvision import models
import torch.optim as optim

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DeeperEncoder(nn.Module):
    def __init__(self):
        super(DeeperEncoder, self).__init__()
        self.layer1 = ConvBlock(2, 16)
        self.layer2 = ConvBlock(16, 32)
        self.layer3 = ConvBlock(32, 64)
        self.layer4 = ConvBlock(64, 128)
        self.layer5 = ConvBlock(128, 256)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.layer5(x)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.projection = nn.Conv3d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)  # Shape: (batch_size, emb_size, depth/patch_size, height/patch_size, width/patch_size)
        x = x.flatten(2)  # Shape: (batch_size, emb_size, num_patches)
        x = x.transpose(1, 2)  # Shape: (batch_size, num_patches, emb_size)
        return x

class LinformerAttention(nn.Module):
    def __init__(self, dim, num_heads, seq_len, k, dropout=0.1):
        super(LinformerAttention, self).__init__()
        self.linformer = Linformer(
            dim=dim, seq_len=seq_len, depth=1, heads=num_heads, k=k, dropout=dropout
        )

    def forward(self, x):
        return self.linformer(x)

class MLP(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class ViT_3D_Classifier(nn.Module):
    def __init__(self, num_classes, patch_size, emb_size, num_heads, mlp_dim, seq_len, k, dropout):
        super(ViT_3D_Classifier, self).__init__()
        self.encoder = DeeperEncoder()
        self.patch_embedding = PatchEmbedding(256, patch_size, emb_size)
        self.attn = LinformerAttention(dim=emb_size, num_heads=num_heads, seq_len=seq_len, k=k, dropout=dropout)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.mlp = MLP(dim=emb_size, mlp_dim=mlp_dim, dropout=dropout)
        self.fc = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.patch_embedding(x)
        x = self.attn(x)
        x = self.norm1(x)
        x = self.mlp(x)
        x = self.norm2(x)
        x = x.mean(dim=1)
        out = self.fc(x)
        return out

