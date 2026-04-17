"""
Biomass Prediction Model with LocalMambaBlock Fusion
DINOv3 Huge Plus backbone with separate models per target
"""

import torch
import torch.nn as nn
import timm


class LocalMambaBlock(nn.Module):
    """
    Simplified Mamba SSM block for sequential feature fusion.
    Implements a selective state space model: h_t = A*h_{t-1} + B_t*x_t, y_t = C_t*h_t + D*x_t
    Input-dependent B and C make it selective (like Mamba), gated by z branch.
    """
    def __init__(self, dim, d_state=16, dropout=0.1):
        super().__init__()
        self.d_state = d_state
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim * 2)      # splits into x and gate z
        self.B_proj = nn.Linear(dim, d_state)        # input-dependent B (selective)
        self.C_proj = nn.Linear(dim, d_state)        # input-dependent C (selective)
        self.A = nn.Parameter(torch.randn(dim, d_state))  # state transition matrix
        self.D = nn.Parameter(torch.ones(dim))       # skip connection scalar
        self.out_proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, dim]
        Returns:
            x: [batch_size, seq_len, dim]
        """
        shortcut = x
        x = self.norm(x)

        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)             # [B, L, dim] each

        B_mat = self.B_proj(x_ssm)                  # [B, L, d_state]
        C_mat = self.C_proj(x_ssm)                  # [B, L, d_state]
        A = -torch.exp(self.A)                       # [dim, d_state] kept negative for stability

        batch, seq_len, dim = x_ssm.shape
        h = torch.zeros(batch, dim, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(seq_len):
            h = h * A.unsqueeze(0) + x_ssm[:, t, :].unsqueeze(-1) * B_mat[:, t, :].unsqueeze(1)
            y_t = (h * C_mat[:, t, :].unsqueeze(1)).sum(-1)  # [B, dim]
            ys.append(y_t)

        y = torch.stack(ys, dim=1)                  # [B, L, dim]
        y = y + x_ssm * self.D                      # skip connection
        y = y * torch.sigmoid(z)                    # selective gate

        return shortcut + self.drop(self.out_proj(y))


class BiomassModelSingle(nn.Module):
    """
    Single-target biomass prediction model.
    
    Architecture:
    - DINOv3 Huge Plus backbone (per crop)
    - LocalMambaBlock fusion (2 layers)
    - Adaptive pooling + MLP head
    
    Args:
        model_name: timm model name (e.g., 'vit_huge_plus_patch16_dinov3.lvd1689m')
        pretrained: Whether to load pretrained weights
        grad_checkpointing: Enable gradient checkpointing to save memory
    """
    def __init__(
        self,
        model_name='vit_huge_plus_patch16_dinov3.lvd1689m',
        pretrained=True,
        grad_checkpointing=True
    ):
        super().__init__()
        
        # Backbone: no classification head, no global pooling
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
        )
        
        # Enable gradient checkpointing if supported
        if hasattr(self.backbone, 'set_grad_checkpointing') and grad_checkpointing:
            self.backbone.set_grad_checkpointing(True)
        
        # Feature dimension from backbone
        nf = self.backbone.num_features
        
        # Fusion layers: 2 LocalMambaBlocks
        self.fusion = nn.Sequential(
            LocalMambaBlock(nf, d_state=16, dropout=0.2),
            LocalMambaBlock(nf, d_state=16, dropout=0.2)
        )
        
        # Pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Regression head
        self.head = nn.Sequential(
            nn.Linear(nf, nf // 2),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(nf // 2, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: Tuple of (left_crop, right_crop)
               Each crop: [batch_size, 3, H, W]
        
        Returns:
            predictions: [batch_size] single target values
        """
        left, right = x
        
        # Extract features from both crops
        x_l = self.backbone(left)  # [batch_size, seq_len, nf]
        x_r = self.backbone(right)  # [batch_size, seq_len, nf]
        
        # Concatenate along sequence dimension
        x_fused = torch.cat([x_l, x_r], dim=1)  # [batch_size, 2*seq_len, nf]
        
        # Fusion through LocalMambaBlocks
        x_fused = self.fusion(x_fused)  # [batch_size, 2*seq_len, nf]
        
        # Pool to fixed size
        x_pool = self.pool(x_fused.transpose(1, 2))  # [batch_size, nf, 1]
        x_pool = x_pool.flatten(1)  # [batch_size, nf]
        
        # Regression head
        out = self.head(x_pool)  # [batch_size, 1]
        
        return out.squeeze(-1)  # [batch_size]


def create_model(config):
    """
    Factory function to create model from config.
    
    Args:
        config: Dictionary with model configuration
        
    Returns:
        model: BiomassModelSingle instance
    """
    model = BiomassModelSingle(
        model_name=config.get('model_name', 'vit_huge_plus_patch16_dinov3.lvd1689m'),
        pretrained=config.get('pretrained', True),
        grad_checkpointing=config.get('grad_checkpointing', True)
    )
    
    return model


def set_backbone_grad(model, requires_grad):
    """
    Freeze or unfreeze backbone parameters.
    
    Args:
        model: BiomassModelSingle instance
        requires_grad: True to unfreeze, False to freeze
    """
    for param in model.backbone.parameters():
        param.requires_grad = requires_grad


if __name__ == '__main__':
    # Test model
    print("Testing BiomassModelSingle...")
    model = BiomassModelSingle(
        model_name='vit_huge_plus_patch16_dinov3.lvd1689m',
        pretrained=False  # Don't download weights for test
    )
    
    # Create dummy input (dual crops)
    batch_size = 2
    left = torch.randn(batch_size, 3, 512, 512)
    right = torch.randn(batch_size, 3, 512, 512)
    
    # Forward pass
    out = model((left, right))
    
    print(f"Input shapes: left={left.shape}, right={right.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Expected: ({batch_size},)")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✓ Model test passed!")

