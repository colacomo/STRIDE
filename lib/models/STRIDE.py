from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from timm.layers import DropPath
from einops import rearrange, reduce, repeat
import torch
import torch.nn as nn
import math
from einops import rearrange
import torch.nn.functional as F
import numpy as np
from lib.models.parameters import (
    ActivationType,
    NormType,
    TemporalAggregationMode,
    UpConvType
)

import matplotlib.pyplot as plt


def modulate(x, shift, scale, T):
    """Adaptive modulation function that adjusts features based on conditions."""
    N, M = x.shape[-2], x.shape[-1]
    B = scale.shape[0]
    x = rearrange(x, '(b t) n m-> b (t n) m', b=B, t=T, n=N, m=M)
    x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    x = rearrange(x, 'b (t n) m-> (b t) n m', b=B, t=T, n=N, m=M)
    return x


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################




class DateEmbedder(nn.Module):
    """Date embedder for temporal information in time series data. Provides date information (e.g., satellite capture time) for each time step using fixed-frequency sine/cosine encoding."""

    def __init__(self, d: int, T: int = 10000, repeat=None, offset: int = 0):
        super().__init__()
        self.d = d
        self.T = T
        self.repeat = repeat
        self.denom = torch.pow(
            T, 2 * torch.div(torch.arange(offset, offset + d).float(), 2, rounding_mode='floor') / d
        )

    def forward(self, dates):
        self.denom = self.denom.to(dates.device)
        # B x T x C, where B equals batch_size * H * W
        sinusoid_table = (dates[:, :, None] / self.denom[None, None, :])
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])  # even dimensions
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])  # odd dimensions

        if self.repeat is not None:
            sinusoid_table = torch.cat([sinusoid_table for _ in range(self.repeat)], dim=-1)

        return sinusoid_table


class LabelEmbedder(nn.Module):
    """Class label embedder with classifier-free guidance dropout. Maps labels to embedding vectors. Supports CFG: randomly drops labels during training, interpolates conditional/unconditional at inference. token_drop implements label dropout."""

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """Drop labels to enable classifier-free guidance."""
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Stochastic depth drop path."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class InteractionCloudBias(nn.Module):
    """
    Refined interactive cloud bias module.
    Input cloud_mask: temporal version (B*H*W, T), spatial version (B*T, H*W).
    Output bias: temporal version (B*H*W, T, T), spatial version (B*T, H*W, H*W).
    """

    def __init__(self, hidden_dim=32):
        super().__init__()
        # Core interaction feature dimensions:
        # 1. Target_mask (i): whether target position has cloud
        # 2. Source_mask (j): whether source position has cloud
        # 3. Mask_diff: state difference (j - i), captures transition from clear to cloudy
        # 4. Mask_combined: joint state (i * j)
        self.feature_dim = 4
        self.dim=dim=1

        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)  # Last layer without bias, controlled by input
        )

        # Learn a base scaling factor to control intervention strength on attention logits
        self.gain = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, cloud_mask, hard_penalty=False):
        """
        cloud_mask: (N, T) - cloud detection value for each pixel at T timesteps (0: clear, 1: cloudy)
        Returns: (N, T, T) attention bias
        """
        N, T = cloud_mask.shape
        # 1. Construct interaction matrix (N, T, T)
        # mask_i (Target/Query): row vector, represents the target timestep state we want to fill
        mask_i = cloud_mask.unsqueeze(2).expand(-1, -1, T)  # (N, T, T)
        # mask_j (Source/Key): column vector, represents the source timestep state we borrow information from
        mask_j = cloud_mask.unsqueeze(1).expand(-1, T, -1)  # (N, T, T)

        # 2. Extract asymmetric features
        # diff > 0 means flow from clear (0) to cloudy (1), the most desirable information flow
        mask_diff = mask_i - mask_j
        # combined means both are cloudy, the worst case
        mask_combined = mask_i * mask_j

        # Stack features (N, T, T, 4)
        features = torch.stack([
            mask_i,
            mask_j,
            mask_diff,
            mask_combined
        ], dim=-1).float()

        # 3. Compute bias mapping
        # bias: (N, T, T)
        bias = self.mlp(features).squeeze(-1)
        # print(bias.shape,features.shape)

        # 4. Physical constraint intervention (optional)
        # If Source(j) is pure cloud (mask_j ~ 1), force a large negative value (penalty)
        # Ensures basic physical correctness even if the MLP hasn't fully learned
        if hard_penalty is True:
            hard_penalty = (mask_j > 0.9).float() * -10.0
            #print(hard_penalty.shape)
            bias = bias + hard_penalty
        return bias * self.gain


def upsample_interaction_matrix(matrix, orig_shape, target_shape, mode='nearest'):
    """
    Correctly upsample an interaction/state matrix while maintaining spatial position correspondence.

    Args:
    matrix: tensor of shape [B, N, N] (e.g., [B, 64, 64])
    orig_shape: original spatial dimensions (h, w) (e.g., (8, 8))
    target_shape: target spatial dimensions (H, W) (e.g., (16, 16))
    mode: interpolation mode. For state/reward matrices, use 'nearest' for exact replication;
          use 'bilinear' (with align_corners=False) for smooth transitions.
    """
    B, N1, N2 = matrix.shape
    h, w = orig_shape
    H, W = target_shape

    # ---------- Step 1: Process Key dimension ----------
    # Restore Key spatial structure: [B, N1, N2] -> [B, N1, h, w]
    matrix_spatial_k = matrix.view(B, N1, h, w)

    # Interpolate on Key spatial dimensions: [B, N1, H, W]
    matrix_k_up = F.interpolate(matrix_spatial_k, size=(H, W), mode=mode)

    # Flatten Key dimension again: [B, N1, H*W]
    matrix_k_up = matrix_k_up.view(B, N1, H * W)

    # ---------- Step 2: Process Query dimension ----------
    # Swap Query and Key dimensions: [B, N1, H*W] -> [B, H*W, N1]
    matrix_k_up_t = matrix_k_up.transpose(1, 2)

    # Restore Query spatial structure: [B, H*W, N1] -> [B, H*W, h, w]
    matrix_spatial_q = matrix_k_up_t.view(B, H * W, h, w)

    # Interpolate on Query spatial dimensions: [B, H*W, H, W]
    matrix_q_up = F.interpolate(matrix_spatial_q, size=(H, W), mode=mode)

    # Flatten Query dimension again: [B, H*W, H*W]
    matrix_q_up = matrix_q_up.view(B, H * W, H * W)

    # ---------- Step 3: Restore original dimension order ----------
    # Transpose back to [B, Query, Key] order: [B, H*W, H*W]
    final_matrix = matrix_q_up.transpose(1, 2)

    return final_matrix


class AdaLNSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 3 * dim)
        )
        self.cloud_bias = InteractionCloudBias(dim)

        # Store last forward pass attn and cloud_reward for visualization
        self.first_attn = None
        self.sec_attn = None
        self.last_attn = None
        self.last_cloud_reward = None
        self.last_time_reward = None

    def forward(self, x, date_emb, attn_bias=None, attn_mask=None):
        # x: (B*, N, D)
        shift, scale, gate = self.adaLN(date_emb).chunk(3, dim=-1)
        # print("xssg", x.shape, shift.shape, scale.shape, gate.shape)

        x = self.norm(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        # print(x.shape)

        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # q, k, v = qkv.unbind(2)
        # print(q.shape)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        self.first_attn = attn.softmax(dim=-1).detach()
        if attn_bias is not None:
            # Use optical information from closer timestamps
            attn = attn + attn_bias.unsqueeze(1)
        self.sec_attn = attn.softmax(dim=-1).detach()
        cloud_reward = None
        #print(N)
        if attn_mask is not None and attn_mask.shape[1]<=256:
            # If t_j is clear and t_i is cloudy, the model should prefer to transfer t_j pixels to t_i, and vice versa it should avoid transferring.
            # Activated when attn_mask.shape[1] <= 256 to avoid excessive computation
            #print(attn_mask.shape)
            if attn_mask.shape[1]<=35:
                cloud_reward = self.cloud_bias(attn_mask,hard_penalty=True)
            else:
                # 1. Spatial downsampling: 256 -> 64
                H = W = int(math.sqrt(attn_mask.shape[-1]))
                # Restore 1D sequence to 16x16 spatial grid
                mask_2d = attn_mask.view(-1, 1, H, W)
                # Use max pooling. If any pixel in 2x2 region is cloud, mark as cloudy after downsampling to avoid missed detections
                p_size = int(H/8)
                mask_down = F.max_pool2d(mask_2d, kernel_size=p_size, stride=p_size)
                # Flatten to 8x8=64
                mask_down = mask_down.view(-1, 64)

                # 2. Compute cloud_reward with minimal computation: output shape [B, 64, 64]
                cloud_reward_down = self.cloud_bias(mask_down)
                cloud_reward = upsample_interaction_matrix(
                    cloud_reward_down,
                    orig_shape=(8, 8),
                    target_shape=(H, W),
                    mode='nearest'
                )
            attn = attn + cloud_reward.unsqueeze(1)
         # Save for visualization
        self.last_attn = attn.softmax(dim=-1).detach()
        self.last_cloud_reward = None if cloud_reward is None else cloud_reward.detach()
        self.last_time_reward = None if attn_bias is None else attn_bias.detach()

        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)

        out = gate.unsqueeze(1) * self.proj(out)
        return out, attn


class ConvMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)

        # Depth-wise convolution for mixing neighborhood information
        # groups=hidden_features ensures minimal computation
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)

        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        # x: (B*T, N, D)
        x = self.fc1(x)

        # Reshape to image for convolution
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class STBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        # Temporal
        self.temporal_self = AdaLNSelfAttention(dim, num_heads)

        # Spatial
        self.spatial_self = AdaLNSelfAttention(dim, num_heads)

        self.mlp = ConvMLP(dim, dim, dim)
        self.dim = dim

    def forward(self, x, sar, t_emb, s_emb, time_bias_self, time_bias_cross, totalmask_t, totalmask_s, cloudmask_t, cloudmask_s, B, T):
        # x: (B*T, N, D)

        # Temporal Attention
        BT, N, D = x.shape
        H = W = int(math.sqrt(N))
        x_t = rearrange(x, '(b t) n d -> (b n) t d', b=B, t=T)
        #sar_t = rearrange(sar, '(b t) n d -> (b n) t d', b=B, t=T)
        tbs = time_bias_self.repeat_interleave(x_t.shape[0] // time_bias_self.shape[0], dim=0)
        tbc = time_bias_cross.repeat_interleave(x_t.shape[0] // time_bias_cross.shape[0], dim=0)
        # date_emb_t = date_emb.repeat_interleave(x_t.shape[0] // date_emb.shape[0], dim=0)
        # print("t",t_emb.shape, x_t.shape)
        if t_emb.shape[0] != x_t.shape[0]:
            token_H = token_W = int(math.sqrt(t_emb.shape[0] // B))
            temporal_emb_reshaped = t_emb.view(
                B, token_H, token_W, D
            )
            c_size = token_H // H
            temporal_emb_down = nn.functional.avg_pool2d(
                temporal_emb_reshaped.permute(0, 3, 1, 2),  # [B, hidden_size, token_H, token_W]
                kernel_size=c_size, stride=c_size
            ).permute(0, 2, 3, 1)
            t_emb = temporal_emb_down.reshape(-1, D)
            # print("t", t_emb.shape, x_t.shape)
        tem_out, tem_info = self.temporal_self(x_t, t_emb, tbs, totalmask_t)
        x_t = x_t + tem_out
        x = rearrange(x_t, '(b n) t d -> (b t) n d', b=B, t=T)

        # Spatial Attention
        # date_emb_s = date_emb.repeat_interleave(x.shape[0] // date_emb.shape[0], dim=0)
        # print("s",s_emb.shape, x.shape)
        spa_out, spa_attn = self.spatial_self(x, s_emb, attn_mask=totalmask_s)
        x = x + spa_out


        # MLP

        x = x + self.mlp(x, H, W)
        # print("out",x.shape,tem_attn.shape)
        return x, {"temporal_self": {"first_attn": self.temporal_self.first_attn,"sec_attn": self.temporal_self.sec_attn,"last_attn": self.temporal_self.last_attn,
                                     "cloud_reward": self.temporal_self.last_cloud_reward,"time_reward": self.temporal_self.last_time_reward},
                   "spatial_self": {"first_attn": self.spatial_self.first_attn,"sec_attn": self.spatial_self.sec_attn,
                                    "last_attn": self.spatial_self.last_attn,"cloud_reward": self.spatial_self.last_cloud_reward},
                   }



class LieGroupTimeBias(nn.Module):
    # Time Lie group bias: dynamically balance translational (R) and periodic (SO(2)) properties via learnable parameter beta

    def __init__(self, hidden_dim, num_freqs=1, init_alpha=0.01, max_period=365.0):
        super().__init__()
        self.num_freqs = num_freqs
        self.max_period = max_period

        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

        # 1. Distance decay parameter (R group)
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(init_alpha)))

        # 2. Fourier weights (SO(2) group)
        # Let different frequencies have different contributions
        self.freq_weights = nn.Parameter(torch.softmax(torch.randn(num_freqs), dim=-1))

        # 3. Dynamic balance parameter: constrained to [0, 1] via sigmoid
        # Initialized to 0, i.e., sigmoid(0) = 0.5, so both terms contribute equally at initialization
        self.gate_beta = nn.Parameter(torch.zeros(1))

        # 4. Global bias
        self.gain = nn.Parameter(torch.ones(1) * 1.0)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, dates):
        """
        dates: (B, T)
        return: (B, T, T) bias matrix
        """
        B, T = dates.shape
        diff = torch.abs(dates[:, :, None] - dates[:, None, :])

        # --- Linear/Translation Term ---
        alpha = torch.exp(self.log_alpha)
        # Use log1p for numerical stability
        linear_term = -torch.log1p(alpha * diff)
        #linear_term_m = self.mlp_linear(linear_term.unsqueeze(-1).float()).squeeze(-1)

        # --- Periodic Term ---
        m = torch.arange(
            1, self.num_freqs + 1,
            device=dates.device,
            dtype=dates.dtype
        ).view(1, 1, 1, -1)  # (1, 1, 1, num_freqs)

        # Compute phase: 2 * pi * f * Δt / T_max
        angles = 2 * math.pi * m * diff.unsqueeze(-1) / self.max_period

        # Weighted cosine values across different frequencies
        # Without weights, can directly use torch.mean(torch.cos(angles), dim=-1)
        periodic_term = torch.sum(
            self.freq_weights.view(1, 1, 1, -1) * torch.cos(angles),
            dim=-1
        )-1
        #periodic_term_m = self.mlp_periodic(periodic_term.unsqueeze(-1).float()).squeeze(-1)

        # --- Dynamic balance blend ---
        # beta = 0: pure linear translational; beta = 1: pure periodic
        beta = torch.sigmoid(self.gate_beta)

        # Core balance formula
        #print(dates,linear_term,periodic_term)
        diag_bias = torch.eye(T, device=dates.device)
        output = ((1 - beta) * linear_term + beta * periodic_term-diag_bias)#*self.gain
        output = self.mlp(output.unsqueeze(-1).float()).squeeze(-1)

        return output


class ConditionTS_Block(nn.Module):
    """
    Four attention structures using STBlock:
    - temporal_self
    - temporal_cross
    - spatial_self
    - spatial_cross
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, num_frames=16, **kwargs):
        super().__init__()
        self.num_frames = num_frames

        # Directly reuse STBlock
        self.st_block = STBlock(hidden_size, num_heads)

        # Time bias and cloud bias
        #self.time_bias = LieGroupTimeBias(hidden_size)
        self.time_bias_s = LieGroupTimeBias(hidden_size)
        self.time_bias_c = LieGroupTimeBias(hidden_size)

        # MLP
        self.norm = nn.LayerNorm(hidden_size)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )

    def forward(self, x, c_t, c_s, dates, SAR, cloud_mask=None, total_mask=None, return_attn=False):
        """
        x:   (B*T, N, D)
        c:   (B, D)  timestep embedding
        SAR: (B*T, N, D)
        cloud_mask: (B*H*W, T) or None
        """

        BT, N, D = x.shape
        T = self.num_frames
        B = BT // T
        H = W = int(math.sqrt(N))
        # print(BT, N, D,H,W)

        # Time bias
        dummy_dates = torch.arange(T, device=x.device).unsqueeze(0).repeat(B, 1)
        #print("dummy_dates", dummy_dates,dates)
        time_bias_self = self.time_bias_s(dates)
        time_bias_cross = self.time_bias_c(dates)
        cloud_mask = F.interpolate(
            cloud_mask.float(), size=(1, H, W), mode='nearest')  # Assume target H=W=16
        # cloudmask_ds: [B, T, 1, 16, 16]
        cloud_mask = cloud_mask.squeeze(2)  # -> [B, T, 16, 16]
        # print("pad_mask", pad_mask.shape,pad_mask)
        cloud_mask_t = (
            cloud_mask.permute(0, 2, 3, 1).contiguous().view(B * H * W, T)  # (B x H x W) x T
        )
        cloud_mask_s = (
            cloud_mask.contiguous().view(B * T, H * W)
        )

        total_mask = F.interpolate(
            total_mask.float(), size=(1, H, W), mode='nearest')  # Assume target H=W=16
        total_mask = total_mask.squeeze(2)  # -> [B, T, 16, 16]
        total_mask_t = (
            total_mask.permute(0, 2, 3, 1).contiguous().view(B * H * W, T)  # (B x H x W) x T
        )
        total_mask_s = (
            total_mask.contiguous().view(B * T, H * W)  # (B x T) x (H * W)
        )

        # STBlock forward
        # print(cloud_mask.shape)
        x, attn_dict = self.st_block(
            x=x,
            sar=SAR,
            t_emb=c_t,
            s_emb=c_s,
            time_bias_self=time_bias_self,
            time_bias_cross=time_bias_cross,
            totalmask_t=total_mask_t,
            totalmask_s=total_mask_s,
            cloudmask_t=cloud_mask_t,
            cloudmask_s=cloud_mask_s,
            B=B,
            T=T
        )

        # MLP
        x = x + self.mlp(self.norm(x))
        if return_attn:
            return x, attn_dict
        return x


# ----------------- New -----------------


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class DownSample(nn.Module):
    def __init__(self, in_dim, out_dim, k=2, s=2, p=0, pad_value=None):
        super().__init__()

        self.down_layer = nn.Sequential(
            LayerNorm2d(in_dim),
            nn.Conv2d(in_dim, out_dim, kernel_size=k, stride=s, padding=p),
        )

    def forward(self, x, cond, H, W):
        BT, N, C = x.shape
        # x: [B*T, C, H, W] (flattened)
        x = x.transpose(1, 2).view(BT, C, H, W)
        #cond = cond.transpose(1, 2).view(BT, C, H, W)
        x = self.down_layer(x)
        #cond = self.down_layer(cond)
        _, C_new, H_new, W_new = x.shape
        x = x.flatten(2).transpose(1, 2)
        #cond = cond.flatten(2).transpose(1, 2)
        if cond is not None:
            cond = cond.transpose(1, 2).view(BT, C, H, W)
            cond = self.down_layer(cond)
            cond = cond.flatten(2).transpose(1, 2)

        return x, cond, H_new, W_new


class UpSample(nn.Module):
    def __init__(self, in_dim, out_dim, d_skip=None, k=2, s=2, p=0, upconv_type=UpConvType.TRANSPOSE,
                 pad_value=None, **kwargs):
        super().__init__()
        d_s = out_dim if d_skip is None else d_skip

        if upconv_type == UpConvType.TRANSPOSE:
            self.up_layer = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=k, stride=s, padding=p)
        elif upconv_type == UpConvType.BILINEAR:
            self.up_layer = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_dim, out_dim, kernel_size=1)
            )

    def forward(self, x, cond, H, W):
        # x: [B*T, C_in, H, W] (flattened)
        # skip: [B*T, C_skip, H, W] (flattened)
        # prompt_emb: [B*T, P_dim] (flattened)
        # print("beforeup", x.shape)
        BT, N, C = x.shape
        # x: [B*T, C, H, W] (flattened)
        x = x.transpose(1, 2).view(BT, C, H, W)
        x = self.up_layer(x)
        _, C_new, H_new, W_new = x.shape
        x = x.flatten(2).transpose(1, 2)
        if cond is not None:
            cond = cond.transpose(1, 2).view(BT, C, H, W)
            cond = self.up_layer(cond)
            cond = cond.flatten(2).transpose(1, 2)
        return x, cond, H_new, W_new


# -------------- 3. SKFusion (channel selection) --------------
class SKFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        d = max(dim // reduction, 32)
        self.mlp = nn.Sequential(
            nn.Conv1d(dim, d, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(d, dim, 1)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        # x1,x2: (BT,N,C)
        B, N, C = x1.shape
        f = torch.stack([x1, x2], dim=-1)  # (B,N,C,2)
        attn = f.mean(dim=1, keepdim=False)  # (B,C,2)
        attn = self.softmax(self.mlp(attn))  # (B,C,2)
        attn = attn.unsqueeze(1)  # (B,1,C,2)
        out = (f * attn).sum(dim=-1)  # (B,N,C)
        return out


class CM_Embedder(nn.Module):
    """Embed scalar timestep into vector representation."""

    def __init__(self, input_size, num_frames, hidden_size):
        super().__init__()
        self.num_frames = num_frames
        self.hidden_size = hidden_size
        self.mlp1 = nn.Sequential(
            nn.Conv2d(1, hidden_size // 2, kernel_size=2, stride=2, bias=True),
            # nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),  # SiLU activation
            nn.Conv2d(hidden_size // 2, hidden_size, kernel_size=2, stride=2, bias=True),
            nn.AvgPool2d(kernel_size=input_size // 4, stride=input_size // 4)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(num_frames, hidden_size // 2, bias=True),
            nn.SiLU(),  # SiLU activation
            nn.Linear(hidden_size // 2, 1, bias=True),
        )

    def forward(self, cm):
        B, T, C, H, W = cm.shape
        cm = cm.contiguous().view(-1, 1, H, W)
        cm = self.mlp1(cm).squeeze(-1).squeeze(-1)
        # print("cm1", cm.shape)
        cm = cm.reshape(-1, self.num_frames, self.hidden_size).transpose(1, 2)
        # print("cm1", cm.shape)
        cm_emb = self.mlp2(cm).squeeze(-1)
        # print("cm2", cm_emb.shape)
        # print("t",t.shape,t_freq.shape,t_emb.shape)
        return cm_emb


class CMTS_Embedder(nn.Module):
    """Separately embed cloud mask into temporal and spatial vector representations.

    Temporal vector: [B*H*W, hidden_size] - time series pattern for each spatial position
    Spatial vector: [B*T, hidden_size] - spatial distribution pattern for each timestep
    """

    def __init__(self, input_size, num_frames, hidden_size, patch_size=4):
        super().__init__()
        self.num_frames = num_frames
        self.hidden_size = hidden_size
        self.patch_size = patch_size

        self.avg1 = nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)

        # Temporal branch: [B, T, 1, H, W] -> [B*token_grid_size^2, hidden_size]
        self.temporal_conv = nn.Sequential(
            # Downsample to token grid size
            # nn.AdaptiveAvgPool2d(pacth_size),
            # Process time series for each spatial position
            nn.Linear(num_frames, hidden_size // 4, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size // 4, hidden_size, bias=True),
            nn.SiLU(),
        )

        # Spatial branch: [B, T, 1, H, W] -> [B*T, hidden_size]
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(1, hidden_size // 4, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_size // 4, hidden_size // 2, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_size // 2, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()  # (B*T, hidden_size)
        )

        self.mlp = nn.Sequential(
            nn.Linear(num_frames, hidden_size // 4, bias=True),
            nn.SiLU(),  # SiLU activation
            nn.Linear(hidden_size // 4, 1, bias=True),
            nn.SiLU(),
        )

    def forward(self, cm):
        """
        cm: (B, T, 1, H, W) cloud mask tensor
        Returns:
            temporal_emb: (B*token_grid_size^2, hidden_size)
            spatial_emb: (B*T, hidden_size)
        """
        B, T, C, H, W = cm.shape
        cm = cm.view(B, T, H, W)  # (B*T, 1, H, W)
        uni_emb = self.avg1(cm)  # (B*T, 1, token_H, token_W)
        # print(uni_emb.shape)

        # cm_temp = cm_temp.permute(0, 2, 1).contiguous().view(B * self.pacth_size * self.pacth_size, 1, T)  # (B*token_HW, T, 1)
        # Temporal embedding: extract temporal features for each spatial position
        cm_t = uni_emb.view(B, T, (H // self.patch_size) * (W // self.patch_size)).permute(0, 2, 1)
        # print("cm0",cm_t.shape)
        cm_temp = self.temporal_conv(cm_t)  # (B*token_HW, hidden_size//2, 1)
        # print("cm1", cm_temp.shape)
        temporal_emb = cm_temp.view(-1, self.hidden_size)  # (B*token_HW, hidden_size//2)

        # Spatial embedding: extract spatial features for each timestep
        cm_s = cm.view(B * T, C, H, W)  # (B*T, 1, H, W)
        spatial_emb = self.spatial_branch(cm_s)  # (B*T, hidden_size)

        c_emb = spatial_emb.view(B, T, self.hidden_size).permute(0, 2, 1)
        c_emb = self.mlp(c_emb).squeeze(-1)

        # print(temporal_emb.shape, spatial_emb.shape, c_emb.shape)

        return temporal_emb, spatial_emb, c_emb


class FinalLayer(nn.Module):
    """Final output layer of SDT. Maps Transformer outputs back to image patches. Uses AdaLN to inject conditions. Output dimension is patch_size^2 * out_channels."""

    def __init__(self, hidden_size, patch_size, out_channels, num_frames):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.num_frames = num_frames

    def forward(self, x, c):
        # print(x.shape, c.shape)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale, self.num_frames)
        x = self.linear(x)
        return x


class LatentDegradationEstimator(nn.Module):
    """Predict a degradation logit for each date using global average/max pooling."""

    def __init__(self, dim, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        hidden_dim = max(dim // 8, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim//4, kernel_size=3),
            nn.GELU(),
            nn.Conv2d(dim//4, dim, kernel_size=3),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.date_head = nn.Sequential(
            nn.Conv2d(dim * 2, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        nn.init.constant_(self.date_head[-1].bias, -1)

    def forward(self, x, B, T, H, W):
        feature_map = rearrange(
            x, '(b t) (h w) d -> (b t) d h w', b=B, t=T, h=H, w=W
        )
        feature_map = self.mlp(feature_map)
        avg_feature = self.avg_pool(feature_map)
        max_feature = self.max_pool(feature_map)
        pooled_feature = torch.cat([avg_feature, max_feature], dim=1)
        date_logits = self.date_head(pooled_feature)  # (B*T, 1, 1, 1)
        return rearrange(date_logits, '(b t) c h w -> b t c h w', b=B, t=T)

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1)
    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))

class STRIDE(nn.Module):
    """
    Time Lie group-based sequential denoising transformer (STRIDE) for satellite image time series reconstruction.
    """

    def __init__(self, input_size=32, patch_size=2, in_channels=4, hidden_size=256, depth=3,
                 num_heads=16, mlp_ratio=4.0, class_dropout_prob=0.1, num_classes=10,
                 learn_sigma=False, num_frames=10, cond_in_channels=3, cross_attention=False):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.cond_in_channels = cond_in_channels

        # Embedding layers
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.cond_embedder = PatchEmbed(input_size, patch_size, cond_in_channels, hidden_size, bias=True)
        # self.cm_embedder = CM_Embedder(input_size, num_frames, hidden_size)
        self.cmts_embedder = CMTS_Embedder(input_size, num_frames, hidden_size, patch_size)
        self.date_embedder = DateEmbedder(hidden_size, T=10000)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        # Initialize latent degradation estimator
        self.deg_estimator = LatentDegradationEstimator(hidden_size, patch_size)

        num_patches = self.x_embedder.num_patches
        # Use fixed sine-cosine positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.num_frames = num_frames
        self.time_embed = nn.Parameter(torch.zeros(1, num_frames, hidden_size), requires_grad=False)
        self.time_drop = nn.Dropout(p=0)
        self.cross_attention = cross_attention

        # ---------- 2. Resolution changes ----------
        enc_depth = depth // 2
        self.layers = nn.ModuleList()

        # --- Encoder ---
        for i in range(enc_depth):
            self.layers.append(ConditionTS_Block(hidden_size, num_heads, mlp_ratio, num_frames))
            if i != enc_depth - 1:  # No downsampling at the deepest layer
                self.layers.append(DownSample(hidden_size, hidden_size))  # * 2))
                # hidden_size *= 2

        # --- Bottleneck ---
        self.layers.append(ConditionTS_Block(hidden_size, num_heads, mlp_ratio, num_frames))

        # --- Decoder ---
        for i in range(enc_depth):
            if i != 0:  # No upsample on first pass (already done in previous UpBlock)
                self.layers.append(UpSample(hidden_size, hidden_size))  # // 2))
                # hidden_size //= 2
            # SKFusion skip connection
            self.layers.append(SKFusion(hidden_size))
            self.layers.append(ConditionTS_Block(hidden_size, num_heads, mlp_ratio, num_frames))

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, self.num_frames)
        self.initialize_weights()
        # Add at the end of __init__ method
        #self.smooth_conv = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.smooth_conv = nn.Sequential(
            nn.Conv2d(self.out_channels, 16, 3, 1, 1),
            nn.SiLU(),
            #ResBlock(64),  # Increase nonlinear smoothing capacity
            ResBlock(16),
            nn.Conv2d(16, self.out_channels, 3, 1, 1)
        )
        self.initialize_weights()


    def initialize_weights(self):
        """Weight initialization."""

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize positional embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize time embedding
        grid_num_frames = np.arange(self.num_frames, dtype=np.float32)
        time_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], grid_num_frames)
        self.time_embed.data.copy_(torch.from_numpy(time_embed).float().unsqueeze(0))

        # Initialize patch embedding
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Degradation date head output bias is initialized within LatentDegradationEstimator.

        # Initialize label embedding table
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.cmts_embedder.temporal_conv[0].weight, std=0.02)
        nn.init.normal_(self.cmts_embedder.temporal_conv[2].weight, std=0.02)
        nn.init.normal_(self.cmts_embedder.spatial_branch[0].weight, std=0.02)
        nn.init.normal_(self.cmts_embedder.spatial_branch[2].weight, std=0.02)
        nn.init.normal_(self.cmts_embedder.spatial_branch[4].weight, std=0.02)
        nn.init.normal_(self.cmts_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.cmts_embedder.mlp[2].weight, std=0.02)

        # Initialize adaLN modulation layers in SDT blocks to zero
        """for TSblock in self.layers:
            if self.cross_attention:
                nn.init.constant_(TSblock.CrossAttentionBlock.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(TSblock.CrossAttentionBlock.adaLN_modulation[-1].bias, 0)

            nn.init.constant_(TSblock.TemporalBlock.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(TSblock.TemporalBlock.adaLN_modulation[-1].bias, 0)

            nn.init.constant_(TSblock.SpatialBlock.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(TSblock.SpatialBlock.adaLN_modulation[-1].bias, 0)

            nn.init.constant_(TSblock.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(TSblock.adaLN_modulation[-1].bias, 0)"""

        # Initialize output layer to zero
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """Convert patch sequence back to images."""
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(
            self, x, date=None, cond=None, cloud_mask=None, watch_block_idx=None,
            return_masks=False, return_mask_logits=False, latent_mask_alpha=1.0
    ):
        """
        SDT forward pass.
        x: (B, T, C, H, W) spatial input tensor (image or latent representation of image)
        t: (B,) diffusion timestep tensor
        date: (B, T) date tensor
        cond: same shape as x, condition input such as SAR image
        """
        B, T, C, H, W = x.shape
        # print("SDT",x.shape,t,date.shape)
        #print("ori_date",date)

        # Embed input images
        x = x.contiguous().view(-1, C, H, W)
        # print("x", x.shape)
        x_emb = self.x_embedder(x)
        x_emb_skip = x_emb
        x = x_emb + self.pos_embed  # ((B T), N, M)
        # print("x",x.shape)
        # Auxiliary degradation head only updates itself, prevents date-level supervision from back-propagating into patch embedding and backbone.
        token_h, token_w = H // self.patch_size, W // self.patch_size
        date_logits = self.deg_estimator(x_emb.detach(), B, T, token_h, token_w)
        latent_probability = torch.sigmoid(date_logits)
        latent_mask = latent_probability.expand(-1, -1, -1, token_h, token_w)

        if not 0.0 <= latent_mask_alpha <= 1.0:
            raise ValueError(f'latent_mask_alpha must be in [0, 1], got {latent_mask_alpha}')

        # Reconstruction loss does not back-propagate to degradation head; alpha only controls predicted mask influence on attention.
        latent_for_attention = latent_mask.detach() * latent_mask_alpha
        if cloud_mask is not None:
            # Downsample known input mask to match token resolution.
            mask_down = F.interpolate(
                cloud_mask.float(), size=(1, H // self.patch_size, W // self.patch_size), mode='nearest'
            )
            mask_total = torch.maximum(mask_down, latent_for_attention)
        else:
            mask_total = latent_for_attention

        # Timestep embedding
        # t = self.t_embedder(t)  # (B, M)
        # cloud_mask_e = cloud_mask.contiguous().view(-1, 1, H, W)
        c_t, c_s, c = self.cmts_embedder(cloud_mask)

        cond=None

        # 2. Resolution & skip cache
        skips = []
        # Initial token grid
        h, w = self.x_embedder.grid_size  # 128/4 = 32
        # print('token grid:', h, 'x', w)
        idx = 0
        # --- Encoder ---
        # print("xci",x.shape, cond.shape)
        while not isinstance(self.layers[idx], ConditionTS_Block) or len(skips) < (len(self.layers) + 1) // 4:
            if isinstance(self.layers[idx], DownSample):
                #print("xc",x.shape,cond.shape)
                x, cond, h, w = self.layers[idx](x, cond, h, w)
                idx += 1
            else:
                #print(idx, "e", x.shape, c.shape, cond.shape, cloud_mask.shape)
                if idx == watch_block_idx:
                    x, attn_out = self.layers[idx](
                        x, c_t, c_s, date, cond, cloud_mask, mask_total, return_attn=True
                    )
                else:
                    x = self.layers[idx](x, c_t, c_s, date, cond, cloud_mask, mask_total)
                skips.append(x)
                idx += 1
        # --- Bottleneck ---
        #print(idx, "m", x.shape, c.shape, cond.shape, cloud_mask.shape)
        if idx == watch_block_idx:
            x, attn_out = self.layers[idx](
                x, c_t, c_s, date, cond, cloud_mask, mask_total, return_attn=True
            )
        else:
            x = self.layers[idx](x, c_t, c_s, date, cond, cloud_mask, mask_total)
        idx += 1
        # --- Decoder ---
        for blk in self.layers[idx:]:
            if isinstance(blk, UpSample):
                x, cond, h, w = blk(x, cond, h, w)
            elif isinstance(blk, SKFusion):
                x = blk(x, skips.pop())
            else:  # ConditionTS_Block
                #print(idx, "d", x.shape, c.shape, cond.shape, cloud_mask.shape)
                if idx == watch_block_idx:
                    x, attn_out = blk(
                        x, c_t, c_s, date, cond, cloud_mask, mask_total, return_attn=True
                    )
                else:
                    #print(blk)
                    x = blk(x, c_t, c_s, date, cond, cloud_mask, mask_total)

        # Final output layer
        x = x + x_emb_skip
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        x = self.smooth_conv(x)
        x = x.view(B, T, x.shape[-3], x.shape[-2], x.shape[-1])
        if return_masks:
            if return_mask_logits:
                return x, latent_mask, mask_total, date_logits
            return x, latent_mask, mask_total
        if watch_block_idx is not None:
            return x, attn_out
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass with classifier-free guidance.
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """Get 2D sine-cosine positional embedding."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """Get 2D sine-cosine positional embedding from grid."""
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Get 1D sine-cosine positional embedding from position grid."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def STRIDE_10():
    return STRIDE(depth=4, in_channels=10, hidden_size=256, patch_size=4, num_heads=4,
                        mlp_ratio=4.0, input_size=128, num_frames=10, cond_in_channels=3,
                        cross_attention=False)  # default true, to use SAR as condition)


if __name__ == '__main__':
    net = STRIDE_10().to("cuda" if torch.cuda.is_available() else "cpu")
    b, s, size = 1, 10, 128
    optical_input = torch.randn(b, s, 10, size, size).to("cuda" if torch.cuda.is_available() else "cpu")
    sar_input = torch.randn(b, s, 3, size, size).to("cuda" if torch.cuda.is_available() else "cpu")
    increments = torch.randint(60, 80, (b, s)).to("cuda" if torch.cuda.is_available() else "cpu")
    date = torch.cumsum(increments, dim=1)
    cloudmask = torch.randint(0, 2, (b, s, 1, size, size)).float().to("cuda" if torch.cuda.is_available() else "cpu")
    out = net(optical_input, date=date, cond=sar_input, cloud_mask=cloudmask)

