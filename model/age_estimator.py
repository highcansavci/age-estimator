import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from einops import rearrange
from typing import Tuple

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, head_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        inner_dim = head_dim * num_heads
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, dim)
        # context: (batch_size, context_len, dim)
        batch_size = x.shape[0]

        # Self-attention normalization
        normed_x = self.norm1(x)
        normed_context = self.norm1(context)

        # Project to queries, keys, values
        q = self.to_q(normed_x)
        k = self.to_k(normed_context)
        v = self.to_v(normed_context)

        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)

        # Scaled dot-product attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1)
        out = torch.matmul(attn, v)
        
        # Reshape and project back
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        x = x + out

        # FFN
        x = x + self.ffn(self.norm2(x))
        return x

class FeatureExtractor(nn.Module):
    def __init__(self, output_dim: int = 512):
        super().__init__()
        # Load pretrained ResNet50
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Additional convolutional layers for feature refinement
        self.refinement = nn.Sequential(
            nn.Conv2d(2048, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, output_dim, 1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True)
        )
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(output_dim, output_dim // 16, 1),
            nn.BatchNorm2d(output_dim // 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim // 16, output_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extract features
        features = self.backbone(x)
        refined = self.refinement(features)
        
        # Apply spatial attention
        attention_weights = self.spatial_attention(refined)
        attended_features = refined * attention_weights
        
        # Global features
        global_features = self.gap(attended_features).flatten(1)
        
        # Return both spatial and global features
        spatial_features = rearrange(attended_features, 'b c h w -> b (h w) c')
        return spatial_features, global_features

class AgeEstimator(nn.Module):
    def __init__(
        self,
        feature_dim: int = 512,
        num_attention_blocks: int = 6,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.face_encoder = FeatureExtractor(feature_dim)
        self.full_image_encoder = FeatureExtractor(feature_dim)
        
        # Cross-attention layers
        self.cross_attention_blocks = nn.ModuleList([
            CrossAttentionBlock(feature_dim, num_heads, head_dim, dropout)
            for _ in range(num_attention_blocks)
        ])
        
        # Feature fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim * 4, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Age prediction layers with uncertainty
        self.age_predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, 2)  # Predict both age and uncertainty
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, face_image: torch.Tensor, full_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extract features from both inputs
        face_spatial, face_global = self.face_encoder(face_image)
        full_spatial, full_global = self.full_image_encoder(full_image)
        
        # Cross attention between face and full image features
        attended_face = face_spatial
        attended_full = full_spatial
        
        for block in self.cross_attention_blocks:
            # Face features attend to full image features and vice versa
            attended_face = block(attended_face, attended_full)
            attended_full = block(attended_full, attended_face)
        
        # Pool attended features
        attended_face = attended_face.mean(dim=1)
        attended_full = attended_full.mean(dim=1)
        
        # Concatenate all features
        combined_features = torch.cat([
            attended_face, attended_full,
            face_global, full_global
        ], dim=1)
        
        # Fuse features
        fused_features = self.fusion_layer(combined_features)
        
        # Predict age and uncertainty
        predictions = self.age_predictor(fused_features)
        age_pred, uncertainty = predictions.chunk(2, dim=1)
        
        # Apply sigmoid to uncertainty to keep it positive
        uncertainty = F.softplus(uncertainty)
        
        return age_pred.squeeze(1), uncertainty.squeeze(1)