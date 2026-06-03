"""Neuro-SCNet model for audio-visual speech separation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import einsum, nn

try:
    # When used as part of a package
    from .base_av_model import BaseAVModel
except ImportError:  # pragma: no cover
    # When used as a standalone script
    from base_av_model import BaseAVModel

Tensor = torch.Tensor

__all__ = [
    "AVPreprocessExplicitAlignTF",
    "NSCNet",
]

class AVPreprocessExplicitAlignTF(nn.Module):
    """
    Explicit pre-alignment (designed for Y_m layout (B, 1, T, F)):
      - Interpolate video embeddings to the audio time grid T, then map to the frequency axis F=257 via Conv2d;
      - Build audio/visual semantic trajectories (B, D, T) from (B, F, T) and estimate the time-shift distribution p(delta);
      - Compute a soft shift hat_delta and apply a fractional-frame time shift to video features (grid_sample);
      - Compute a reliability score r (peak probability) for downstream gating;
      - Output v_tilde with the same layout as Y_m: (B, 1, T, F).
    """
    def __init__(
        self,
        vpre_channels: int,
        n_freq: int = 257,      # target frequency bins (matches Y_m F)
        d_sem: int = 64,        # semantic channels for shift estimation
        K: int = 5,             # search window [-K, K] frames
        kappa: float = 8.0,     # softmax temperature (larger -> sharper peak)
        align_corners: bool = True,
    ):
        super().__init__()
        self.n_freq = n_freq
        self.d_sem = d_sem
        self.K = K
        self.kappa = kappa
        self.align_corners = align_corners
        self.eps = 1e-8

        # Map video embeddings to the frequency axis (match audio F).
        # Input  (B, C_v, T, 1) -> Output (B, n_freq, T, 1)
        self.prepro_v = nn.Conv2d(vpre_channels, n_freq, kernel_size=(3,1), padding=(1,0))

        # Build semantic trajectories (B, D, T).
        self.v_to_sem = nn.Conv1d(n_freq, d_sem, kernel_size=1, bias=False)
        self.a_to_sem = nn.Conv1d(n_freq, d_sem, kernel_size=1, bias=False)

    # ---------- utilities ----------
    @staticmethod
    def _l2_norm(x: torch.Tensor, dim: int) -> torch.Tensor:
        return x / (x.norm(p=2, dim=dim, keepdim=True) + 1e-8)

    def _corr_curve(self, a_sem: torch.Tensor, v_sem: torch.Tensor) -> torch.Tensor:
        """ a_sem, v_sem: (B, D, T) -> c_delta: (B, 2K+1) """
        B, D, T = a_sem.shape
        vals = []
        for d in range(-self.K, self.K + 1):
            if d >= 0:
                a_seg = a_sem[:, :, :T-d]
                v_seg = v_sem[:, :, d:]
            else:
                a_seg = a_sem[:, :, -d:]
                v_seg = v_sem[:, :, :T+d]
            c = (a_seg * v_seg).sum(dim=1) / (D + 1e-8)  # (B, seg_T)
            vals.append(c.mean(dim=1, keepdim=True))     # (B,1)
        return torch.cat(vals, dim=1)  # (B, 2K+1)

    def _norm_offset(self, delta_frames: torch.Tensor, T: int) -> torch.Tensor:
        # Normalized offset for grid_sample.
        if self.align_corners:
            scale = 2.0 / max(T - 1, 1)
        else:
            scale = 2.0 / T
        return delta_frames * scale

    def _shift_time(self, x: torch.Tensor, hat_delta: torch.Tensor) -> torch.Tensor:
        """
        Apply a fractional-frame time shift to a (B, C, T) sequence using grid_sample with linear interpolation.
        Returns: (B, C, T)
        """
        B, C, T = x.shape
        device = x.device
        dtype = x.dtype

        # Construct normalized time coordinates in [-1, 1].
        if self.align_corners:
            base = torch.linspace(-1.0, 1.0, T, device=device, dtype=dtype)
        else:
            base = torch.linspace(-1.0 + 1.0 / T, 1.0 - 1.0 / T, T, device=device, dtype=dtype)

    # Per-sample normalized offset (map frame shift to [-1, 1] coordinates).
        offset = self._norm_offset(hat_delta.to(dtype), T)  # (B,)

    # Build the grid: (B, T, 1, 2); last dim is [x_coord, y_coord].
        x_coord = torch.zeros(B, T, 1, device=device, dtype=dtype)         # width dimension is 1
        y_coord = (base.unsqueeze(0) + offset.unsqueeze(1)).unsqueeze(-1)  # (B, T, 1)

        grid = torch.stack([x_coord, y_coord], dim=-1)  # (B, T, 1, 2)

    # grid_sample input format: (B, C, H=T, W=1)
        x4 = x.unsqueeze(-1)  # (B, C, T, 1)
        y4 = F.grid_sample(
            x4, grid, mode="bilinear", padding_mode="border", align_corners=self.align_corners
        )  # (B, C, T, 1)
        return y4.squeeze(-1)  # (B, C, T)                                 # (B,C,T)

    # ---------- forward ----------
    def forward(self, Y_m: torch.Tensor, mouth_emb: torch.Tensor):
        """
        Y_m:        (B, 1, T, F)   e.g., [2, 1, 126, 257]
        mouth_emb:  (B, C_v, T_v)
        Returns:
          v_tilde: (B, 1, T, F)    (same layout as Y_m)
          r_bc:    (B, 1, 1, 1)    reliability broadcast factor
          aux:     Dict (alignment observables)
        """
        assert Y_m.dim() == 4 and Y_m.size(1) == 1, f"Y_m must be (B,1,T,F), got {tuple(Y_m.shape)}"
        B, _, T, Fbins = Y_m.shape
        assert Fbins == self.n_freq, f"Y_m F={Fbins} != n_freq={self.n_freq}"

        # Audio spectrum (B, F, T), used to build semantic trajectories.
        a_spec = Y_m.squeeze(1).permute(0, 2, 1)      # (B, T, F) -> (B, F, T)

        # [1] Align video embeddings to the audio time grid and map to frequency axis.
        v_resized = F.interpolate(mouth_emb.unsqueeze(-1),
                                  size=(T, 1),
                                  mode="bilinear",
                                  align_corners=False)          # (B, C_v, T, 1)
        v_map = self.prepro_v(v_resized).squeeze(-1)            # (B, F, T)

        # [2] Semantic trajectories and normalization (B, D, T).
        a_sem = self.a_to_sem(a_spec)
        v_sem = self.v_to_sem(v_map)
        a_sem = self._l2_norm(a_sem, dim=1)
        v_sem = self._l2_norm(v_sem, dim=1)

        # [3] Correlation curve / distribution / soft shift / reliability.
        c_delta = self._corr_curve(a_sem, v_sem)                 # (B, 2K+1)
        p_delta = torch.softmax(self.kappa * c_delta, dim=-1)    # (B, 2K+1)
        deltas = torch.arange(-self.K, self.K + 1,
                              device=Y_m.device, dtype=Y_m.dtype)
        hat_delta = (p_delta * deltas.unsqueeze(0)).sum(dim=-1)  # (B,)
        r = p_delta.max(dim=-1).values                           # (B,)

        # [4] Apply the estimated shift hat_delta to video features.
        v_map_shift = self._shift_time(v_map, hat_delta)         # (B, F, T)

        # [5] Restore the layout to match Y_m (B, 1, T, F).
        v_tilde = v_map_shift.permute(0, 2, 1).unsqueeze(1)      # (B,1,T,F)
        r_bc = r.view(B, 1, 1, 1)


        return v_tilde, r_bc

class FFN(nn.Module):
    def __init__(self, d_model, bidirectional=True, dropout=0):
        super(FFN, self).__init__()
        self.gru = nn.GRU(d_model, d_model*2, 1, bidirectional=bidirectional)
        if bidirectional:
            self.linear = nn.Linear(d_model*2*2, d_model)
        else:
            self.linear = nn.Linear(d_model*2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.linear(x)

        return x

class InteractionModule1(nn.Module):
    def __init__(self, dim):
        super(InteractionModule1, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv1d(dim * 2, dim, kernel_size=1),
                        nn.SiLU(),
                        nn.Conv1d(dim, dim, kernel_size=1),
                        nn.Sigmoid(),
                    )
        self.conv2 = nn.Sequential(
                        nn.Conv1d(dim * 2, dim, kernel_size=1),
                        nn.SiLU(),
                        nn.Conv1d(dim, dim, kernel_size=1),
                        nn.Sigmoid(),
                    )
        self.conv3 = nn.Sequential(
                        nn.Conv1d(dim, dim, kernel_size=1),
                        nn.Sigmoid(),
                    )
        #self.weight = CALayer2(dim)
    def forward(self, x_p1, x_p2, x_n1, x_n2):
        x_p1 = x_p1.permute(0, 2, 1)
        x_p2 = x_p2.permute(0, 2, 1)
        x_n1 = x_n1.permute(0, 2, 1)
        x_n2 = x_n2.permute(0, 2, 1)
        #w1, w2 = self.weight(x_p, x_n)
        m = self.conv1(torch.cat([x_p1, x_n2], dim = 1)) * x_n2
        x1 = m + x_p2
        x2 = (self.conv3(x1) - self.conv2(torch.cat([x1, x_n1 + (x_n2 - m)], dim = 1))) * x1
        return x2.permute(0, 2, 1)


class FIR_MHSA(nn.Module):
    def __init__(self, dim, num_heads = 4, dim_head=32, dropout=0.0, bias=False, max_pos_emb=512):
        super(FIR_MHSA, self).__init__()
        self.heads = num_heads
        inner_dim = dim_head * num_heads
        self.scale = dim_head**-0.5
        #self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.project_out1 = nn.Linear(inner_dim, dim)
        self.project_out2 = nn.Linear(inner_dim, dim)
        self.project_out3 = nn.Linear(inner_dim, dim)
        self.project_out4 = nn.Linear(inner_dim, dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.attn1 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)

        self.IM = InteractionModule1(dim)
    def forward(self, x, context=None, mask=None, context_mask=None):
        n, device, h, max_pos_emb, has_context = (
            x.shape[-2],
            x.device,
            self.heads,
            self.max_pos_emb,
            exists(context),
        )
        context = default(context, x)

        #print(x.size())
        #qkv = self.qkv1(x)
        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        b, h, C, d = q.shape
        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale#(q @ k.transpose(-2, -1))#

        # # shaw's relative positional embedding
        seq = torch.arange(n, device=device)
        dist = rearrange(seq, "i -> i ()") - rearrange(seq, "j -> () j")
        dist = dist.clamp(-max_pos_emb, max_pos_emb) + max_pos_emb
        rel_pos_emb = self.rel_pos_emb(dist).to(q)
        pos_attn = einsum("b h n d, n r d -> b h n r", q, rel_pos_emb) * self.scale
        dots = dots + pos_attn
        if exists(mask) or exists(context_mask):
            mask = default(mask, lambda: torch.ones(*x.shape[:2], device=device))
            context_mask = (
                default(context_mask, mask)
                if not has_context
                else default(
                    context_mask, lambda: torch.ones(*context.shape[:2], device=device)
                )
            )
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, "b i -> b () i ()") * rearrange(
                context_mask, "b j -> b () () j"
            )
            dots.masked_fill_(~mask, mask_value)

        # q = torch.nn.functional.normalize(q, dim=-1)
        # k = torch.nn.functional.normalize(k, dim=-1)

        mask1 = torch.zeros(b, h, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, h, C, C, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, h, C, C, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, h, C, C, device=x.device, requires_grad=False)
        #print(q.size())
        attn = dots#(q @ k.transpose(-2, -1))# * self.temperature
        attn_min = -1 * attn
        #print(attn_min)
        index = torch.topk(attn, k=int(C*3/4), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        zero = torch.zeros_like(attn)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))#torch.full_like(attn, float('-inf'))

        # index = torch.topk(attn, k=int(C/3), dim=-1, largest=True)[1]
        # mask2.scatter_(-1, index, 1.)
        # attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C/4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))


        index = torch.topk(attn_min, k=int(C*3/4), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        zero = torch.zeros_like(attn_min)
        attn1_n = torch.where(mask1 > 0, attn_min, torch.full_like(attn_min, float('-inf')))#torch.full_like(attn, float('-inf'))

        # index = torch.topk(attn_min, k=int(C/3), dim=-1, largest=True)[1]
        # mask2.scatter_(-1, index, 1.)
        # attn2_n = torch.where(mask2 > 0, attn_min, torch.full_like(attn_min, float('-inf')))

        index = torch.topk(attn_min, k=int(C/4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3_n = torch.where(mask3 > 0, attn_min, torch.full_like(attn_min, float('-inf')))

        # index = torch.topk(attn_min, k=int(C*4/4), dim=-1, largest=True)[1]
        # mask4.scatter_(-1, index, 1.)
        # attn4_n = torch.where(mask4 > 0, attn_min, torch.full_like(attn_min, float('-inf')))
        #print(attn3)

        # attn3 = attn3 - attn2 - attn1
        #attn3 = attn - attn4 - attn2 - attn1
        # print(attn)
        attn1 = attn1.softmax(dim=-1)
        #attn2 = attn2.softmax(dim=-1)
        attn2 = attn3.softmax(dim=-1)
        #attn4 = attn4.softmax(dim=-1)

        attn3 = attn1_n.softmax(dim=-1)
        #attn2_n = attn2_n.softmax(dim=-1)
        attn4 = attn3_n.softmax(dim=-1)

        # out1 = (attn1 @ v)
        # out2 = (attn2 @ v)
        # out3 = (attn3 @ v)
        # out4 = (attn4 @ v)
        out1 = einsum("b h i j, b h j d -> b h i d", attn1, v)#attn1 @ v
        out1 = self.project_out1(rearrange(out1, "b h n d -> b n (h d)"))

        out2 = einsum("b h i j, b h j d -> b h i d", attn2, v)#attn2 @ v#
        out2 = self.project_out2(rearrange(out2, "b h n d -> b n (h d)"))

        out3 = einsum("b h i j, b h j d -> b h i d", attn3, v)#attn3 @ v#
        out3 = self.project_out3(rearrange(out3, "b h n d -> b n (h d)"))

        out4 = einsum("b h i j, b h j d -> b h i d", attn4, v)#attn4 @ v#
        out4 = self.project_out4(rearrange(out4, "b h n d -> b n (h d)"))

        # out = self.IM(out1, out2, out3, out4)#out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4
        # #print(out.size())
        # out = self.project_out1(out)
        return (self.IM(self.dropout1(out1), self.dropout2(out2), self.dropout3(out3), self.dropout4(out4)))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, bidirectional=True, dropout=0):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attention = FIR_MHSA(d_model, dropout=dropout, bias=False)#MultiheadAttention(d_model, n_heads, dropout=dropout)#
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, bidirectional=bidirectional)
        self.dropout2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        xt = self.norm1(x)
        # xt, _ = self.attention(xt, xt, xt,
                               # attn_mask=attn_mask,
                               # key_padding_mask=key_padding_mask)
        xt = self.attention(xt)
        x = x + self.dropout1(xt)

        xt = self.norm2(x)
        xt = self.ffn(xt)
        x = x + self.dropout2(xt)

        x = self.norm3(x)

        return x

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def get_padding_2d(kernel_size, dilation=(1, 1)):
    return (int((kernel_size[0]*dilation[0] - dilation[0])/2), int((kernel_size[1]*dilation[1] - dilation[1])/2))

class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super(SPConvTranspose2d, self).__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.0)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1)
        )
        self.r = r

    def forward(self, x):
        x = self.pad1(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out

class LearnableSigmoid_2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)

class DenseBlock(nn.Module):
    def __init__(self, in_channel, kernel_size=(3, 3), depth=4):
        super(DenseBlock, self).__init__()
        self.depth = depth
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dil = 2 ** i
            dense_conv = nn.Sequential(
                nn.Conv2d(in_channel*(i+1), in_channel, kernel_size, dilation=(dil, 1),
                          padding=get_padding_2d(kernel_size, (dil, 1))),
                nn.InstanceNorm2d(in_channel, affine=True),
                nn.PReLU(in_channel)
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x

class FeedForwardModule(nn.Module):
    def __init__(self, dim, mult=4, dropout=0):
        super(FeedForwardModule, self).__init__()
        self.ffm = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffm(x)


class ConformerConvModule(nn.Module):
    def __init__(self, dim, expansion_factor=2, kernel_size=31, dropout=0.):
        super(ConformerConvModule, self).__init__()
        inner_dim = dim * expansion_factor
        self.ccm = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n c -> b c n'),
            nn.Conv1d(dim, inner_dim*2, 1),
            nn.GLU(dim=1),
            nn.Conv1d(inner_dim, inner_dim, kernel_size=kernel_size,
                      padding=get_padding(kernel_size), groups=inner_dim), # DepthWiseConv1d
            nn.BatchNorm1d(inner_dim),
            nn.SiLU(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ccm(x)

class Attention(nn.Module):
    def __init__(self, dim, dim_t, num_heads = 8, dim_head=64, dropout=0.0, bias=False, max_pos_emb=512):
        super(Attention, self).__init__()
        self.heads = num_heads
        self.inner = dim_head * num_heads
        inner_dim = dim_head * num_heads
        self.scale = dim_head**-0.5

        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.fuse1 = nn.Linear(inner_dim, dim)
        self.fuse2 = nn.Linear(inner_dim, dim)
        self.fuse3 = nn.Linear(inner_dim, dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.attn1 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
    def forward(self, x, y, context=None, mask=None, context_mask=None):
        n, device, h, max_pos_emb, has_context = (
            x.shape[-2],
            x.device,
            self.heads,
            self.max_pos_emb,
            exists(context),
        )
        context = default(context, y)

        #print(x.size())
        #qkv = self.qkv1(x)
        #print(x.size())
        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        b, h, C, d = q.shape
        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        # shaw's relative positional embedding
        seq = torch.arange(n, device=device)
        dist = rearrange(seq, "i -> i ()") - rearrange(seq, "j -> () j")
        dist = dist.clamp(-max_pos_emb, max_pos_emb) + max_pos_emb
        rel_pos_emb = self.rel_pos_emb(dist).to(q)
        pos_attn = einsum("b h n d, n r d -> b h n r", q, rel_pos_emb) * self.scale
        dots = dots + pos_attn
        if exists(mask) or exists(context_mask):
            mask = default(mask, lambda: torch.ones(*x.shape[:2], device=device))
            context_mask = (
                default(context_mask, mask)
                if not has_context
                else default(
                    context_mask, lambda: torch.ones(*context.shape[:2], device=device)
                )
            )
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, "b i -> b () i ()") * rearrange(
                context_mask, "b j -> b () () j"
            )
            dots.masked_fill_(~mask, mask_value)
        mask1 = torch.zeros(b, h, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, h, C, C, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, h, C, C, device=x.device, requires_grad=False)
        #mask4 = torch.zeros(b, h, C, C, device=x.device, requires_grad=False)


        #print(q.size())
        attn = dots#(q @ k.transpose(-2, -1))# * self.temperature
        attn_n = -1 * attn

        index = torch.topk(attn, k=int(C/4), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        zero = torch.zeros_like(attn)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))#torch.full_like(attn, float('-inf'))

        index = torch.topk(attn1, k=int(C*2/3), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn2 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn_n, k=int(C/4), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn3 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)


        out1 = (attn1 @ v)
        out1 = rearrange(out1, "b h n d -> b n (h d)")
        out2 = (attn2 @ v)
        out2 = rearrange(out2, "b h n d -> b n (h d)")
        out3 = (attn3 @ v)
        out3 = rearrange(out3, "b h n d -> b n (h d)")
        out1 = self.fuse1(out1)#self.fuse1(torch.cat([out1, out2, out3],-1))
        out2 = self.fuse2(out2)
        out3 = self.fuse3(out3)
        return self.dropout1(out1), self.dropout2(out2), self.dropout3(out3)
class Atten(nn.Module):
    def __init__(self, dim, dim_t, num_heads = 8, dim_head=64, dropout=0.0, bias=False, max_pos_emb=512):
        super(Atten, self).__init__()
        self.heads = num_heads
        self.inner = dim_head * num_heads
        inner_dim = dim_head * num_heads
        self.scale = dim_head**-0.5

        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.fuse1 = nn.Linear(inner_dim, dim)
        self.fuse2 = nn.Linear(inner_dim, dim)
        self.fuse3 = nn.Linear(inner_dim, dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, y, context=None, mask=None, context_mask=None):
        n, device, h, max_pos_emb, has_context = (
            x.shape[-2],
            x.device,
            self.heads,
            self.max_pos_emb,
            exists(context),
        )
        context = default(context, y)

        #print(x.size())
        #qkv = self.qkv1(x)
        #print(x.size())
        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        b, h, C, d = q.shape
        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        # shaw's relative positional embedding
        seq = torch.arange(n, device=device)
        dist = rearrange(seq, "i -> i ()") - rearrange(seq, "j -> () j")
        dist = dist.clamp(-max_pos_emb, max_pos_emb) + max_pos_emb
        rel_pos_emb = self.rel_pos_emb(dist).to(q)
        pos_attn = einsum("b h n d, n r d -> b h n r", q, rel_pos_emb) * self.scale
        dots = dots + pos_attn
        if exists(mask) or exists(context_mask):
            mask = default(mask, lambda: torch.ones(*x.shape[:2], device=device))
            context_mask = (
                default(context_mask, mask)
                if not has_context
                else default(
                    context_mask, lambda: torch.ones(*context.shape[:2], device=device)
                )
            )
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, "b i -> b () i ()") * rearrange(
                context_mask, "b j -> b () () j"
            )
            dots.masked_fill_(~mask, mask_value)

        #mask4 = torch.zeros(b, h, C, C, device=x.device, requires_grad=False)


        #print(q.size())
        attn = dots#(q @ k.transpose(-2, -1))# * self.temperature
        attn_n = -1 * attn

        attn1 = attn.softmax(dim=-1)
        attn2 = attn_n.softmax(dim=-1)


        out1 = (attn1 @ v)
        out1 = rearrange(out1, "b h n d -> b n (h d)")
        out2 = (attn2 @ v)
        out2 = rearrange(out2, "b h n d -> b n (h d)")
        out1 = self.fuse1(out1)#self.fuse1(torch.cat([out1, out2, out3],-1))
        out2 = self.fuse2(out2)
        return self.dropout1(out1), self.dropout2(out2)
class IM(nn.Module):

    def __init__(self, dim = 64):
        super(IM, self).__init__()
        self.conv_a = nn.Sequential(
                        nn.Conv2d(dim * 2, dim, kernel_size=(1,1)),
                        nn.SiLU(),
                        nn.Conv2d(dim, dim, kernel_size=(1,1)),
                        nn.Sigmoid(),
                    )

        self.conv_v = nn.Sequential(
                        nn.Conv2d(dim * 2, dim, kernel_size=(1,1)),
                        nn.SiLU(),
                        nn.Conv2d(dim, dim, kernel_size=(1,1)),
                        nn.Sigmoid(),
                    )
    def forward(self, x_p, x_n, n):
        x_np = self.conv_a(torch.cat([x_n, n], dim=1)) * x_n # + v_n
        x_nn = x_n - x_np

        #x_p = x_p + x_np
        x_pn = self.conv_v(torch.cat([x_nn, x_p], dim=1)) * x_p
        x_pp = x_p - x_pn
        return x_pp + x_np + x_p, x_pn + x_nn + x_n

class IM1(nn.Module):

    def __init__(self, dim = 64):
        super(IM1, self).__init__()
        self.conv_a = nn.Sequential(
                        nn.Conv2d(dim * 2, dim, kernel_size=(1,1)),
                        nn.SiLU(),
                        nn.Conv2d(dim, dim, kernel_size=(1,1)),
                        nn.Sigmoid(),
                    )

        self.conv_v = nn.Sequential(
                        nn.Conv2d(dim * 2, dim, kernel_size=(1,1)),
                        nn.SiLU(),
                        nn.Conv2d(dim, dim, kernel_size=(1,1)),
                        nn.Sigmoid(),
                    )
    def forward(self, f_n, n):
        f_nn = self.conv_a(torch.cat([f_n, n], dim=1)) * f_n # + v_n
        f_np = f_nn - f_np
        return f_nn + f_n, f_np + f_n

class TSCB1(nn.Module):
    def __init__(self, num_channel=64):
        super(TSCB1, self).__init__()
        self.norm1 = nn.LayerNorm(num_channel)
        self.norm2 = nn.LayerNorm(num_channel)
        self.attn1 = Atten(dim = num_channel, dim_t = num_channel, num_heads = 8, dropout=0.2)
        self.IM1 = IM(num_channel)
        self.norm3 = nn.InstanceNorm2d(64, affine=True)
        self.norm4 = nn.InstanceNorm2d(64, affine=True)
    def forward(self, x, y, n):
        b, c, t, f = x.size()
        x1 = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        y1 = y.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x_p, x_n = self.attn1(x1, y1)
        x_p = self.norm1(x_p + y1)
        x_n = self.norm2(x_n + y1)
        x_p = x_p.view(b, f, t, c).permute(0, 3, 2, 1)#.contiguous().view(b*t, f, c)
        x_n = x_n.view(b, f, t, c).permute(0, 3, 2, 1)
        x_p1, x_n1 = self.IM1(x_p, x_n, n)
        return self.norm3(x_p1 + x_p), self.norm4(x_n1 + x_n)

class TSCB2(nn.Module):
    def __init__(self, num_channel=64):
        super(TSCB2, self).__init__()
        self.norm1 = nn.LayerNorm(num_channel)
        self.norm2 = nn.LayerNorm(num_channel)
        self.attn1 = Atten(dim = num_channel, dim_t = num_channel, num_heads = 8, dropout=0.2)
        self.norm3 = nn.InstanceNorm2d(64, affine=True)
        self.norm4 = nn.InstanceNorm2d(64, affine=True)
        self.IM1 = IM(num_channel)
    def forward(self, x, y, n):
        b, c, t, f = x.size()
        x1 = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        y1 = y.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x_n, x_p = self.attn1(x1, y1)
        x_p = self.norm1(x_p + y1)
        x_n = self.norm2(x_n + y1)
        x_p = x_p.view(b, f, t, c).permute(0, 3, 2, 1)#.contiguous().view(b*t, f, c)
        x_n = x_n.view(b, f, t, c).permute(0, 3, 2, 1)
        x_p1, x_n1 = self.IM1(x_p, x_n, n)
        return self.norm3(x_p1 + x_p), self.norm4(x_n1 + x_n)

class TSCB(nn.Module):
    def __init__(self, num_channel=64):
        super(TSCB, self).__init__()

        self.attn1 = Attention(dim = num_channel, dim_t = num_channel, num_heads = 8, dropout=0.2)
        self.norm1 = nn.LayerNorm(num_channel)
        self.norm2 = nn.LayerNorm(num_channel)
        self.IM1 = IM(num_channel)
        self.ffn1 = FFN(num_channel, bidirectional=True)
    def forward(self, x, a, v):
        b, c, t, f = x.size()
        x1 = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x_p, x_m, x_n = self.attn1(x1, x1)
        x_p = x_p.view(b, f, t, c).permute(0, 3, 2, 1)#.contiguous().view(b*t, f, c)
        x_m = x_m.view(b, f, t, c).permute(0, 3, 2, 1)
        x_n = x_n.view(b, f, t, c).permute(0, 3, 2, 1)
        x = self.IM1(x_p, x_m, x_n, a, v) + x
        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x = self.norm1(x)
        x = self.ffn1(x) + x
        x = self.norm2(x)
        x = x.view(b, f, t, c).permute(0, 3, 2, 1)
        return x

class AttentionModule(nn.Module):
    def __init__(self, dim, n_head=8, dropout=0.):
        super(AttentionModule, self).__init__()
        self.attn = nn.MultiheadAttention(dim, n_head, dropout=dropout)
        self.layernorm = nn.LayerNorm(dim)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        x = self.layernorm(x)
        x, _ = self.attn(x, x, x,
                         attn_mask=attn_mask,
                         key_padding_mask=key_padding_mask)
        return x


class ConformerBlock(nn.Module):
    def __init__(self, dim, n_head=8, ffm_mult=4, ccm_expansion_factor=2, ccm_kernel_size=31,
                 ffm_dropout=0., attn_dropout=0., ccm_dropout=0.):
        super(ConformerBlock, self).__init__()
        self.ffm1 = FeedForwardModule(dim, ffm_mult, dropout=ffm_dropout)
        self.attn = AttentionModule(dim, n_head, dropout=attn_dropout)
        self.ccm = ConformerConvModule(dim, ccm_expansion_factor, ccm_kernel_size, dropout=ccm_dropout)
        self.ffm2 = FeedForwardModule(dim, ffm_mult, dropout=ffm_dropout)
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + 0.5 * self.ffm1(x)
        x = x + self.attn(x)
        x = x + self.ccm(x)
        x = x + 0.5 * self.ffm2(x)
        x = self.post_norm(x)
        return x

class TSTransformerBlock(nn.Module):
    def __init__(self, h):
        super(TSTransformerBlock, self).__init__()
        self.h = h
        self.time_transformer = TransformerBlock(d_model=64, n_heads=4)
        self.freq_transformer = TransformerBlock(d_model=64, n_heads=4)

    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x = self.time_transformer(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x = self.freq_transformer(x) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x

class TSConformerBlock(nn.Module):
    def __init__(self, h):
        super(TSConformerBlock, self).__init__()
        self.h = h
        self.time_conformer = ConformerBlock(dim=64,  n_head=4, ccm_kernel_size=31,
                                             ffm_dropout=0.2, attn_dropout=0.2)
        self.freq_conformer = ConformerBlock(dim=64,  n_head=4, ccm_kernel_size=31,
                                             ffm_dropout=0.2, attn_dropout=0.2)

    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x = self.time_conformer(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x = self.freq_conformer(x) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x

class IM_E(nn.Module):
    def __init__(self, dim):
        super(IM_E, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(dim * 2, dim, kernel_size=(1,1)),
                        nn.SiLU(),
                        nn.Conv2d(dim, dim, kernel_size=(1,1)),
                        nn.Sigmoid(),
                    )
        self.conv2 = nn.Sequential(
                        nn.Conv2d(dim * 2, dim, kernel_size=(1,1)),
                        nn.SiLU(),
                        nn.Conv2d(dim, dim, kernel_size=(1,1)),
                        nn.Sigmoid(),
                    )
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()
    def forward(self, x_p, x_n):
        m = self.conv1(torch.cat([x_p, x_n], dim = 1))
        x1 = m * x_n + x_p
        x_1_n = (self.sigmoid1(x_n) - m).abs() * x_n
        x2 = (self.sigmoid1(x1) - self.conv2(torch.cat([x1, x_1_n], dim = 1))).abs() * x1
        return x2 + x_p

class IM_D(nn.Module):
    def __init__(self, dim):
        super(IM_D, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(dim * 2, dim, kernel_size=(1,1)),
                        nn.SiLU(),
                        nn.Conv2d(dim, dim, kernel_size=(1,1)),
                        nn.Sigmoid(),
                    )
        self.conv2 = nn.Sequential(
                        nn.Conv2d(dim * 2, dim, kernel_size=(1,1)),
                        nn.SiLU(),
                        nn.Conv2d(dim, dim, kernel_size=(1,1)),
                        nn.Sigmoid(),
                    )

    def forward(self, x_p, x_n):
        x_n = self.conv1(torch.cat([x_p, x_n], dim = 1)) * x_n
        x_p = self.conv2(torch.cat([x_p, x_n], dim = 1)) * x_p
        return x_n + x_p

class LM_E(nn.Module):
    def __init__(self, dim = 64):
        super(LM_E, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 1), (1, 1)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.PReLU(dim),
        )
        self.res1 = DenseBlock(64, depth=1)
        self.res2 = DenseBlock(64, depth=1)
        self.IM = IM_E(dim)
        self.norm = nn.InstanceNorm2d(dim, affine=True)
        self.activation = nn.PReLU(dim)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = x - x1
        x1_p = self.res1(x1)
        x2_p = self.res2(x2)
        x1_n = x1 - x1_p
        x2_n = x2 - x2_p

        x_p = x1_p + x2_p
        x_n = x1_n + x2_n


        x = self.IM(x_p, x_n)
        x = self.norm(x)
        x = self.activation(x)
        return self.dropout(x)

class LM_D(nn.Module):
    def __init__(self, dim = 32):
        super(LM_D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 1), (1, 1)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.PReLU(dim),
        )
        self.res1 = DenseBlock(64, depth=1)
        self.res2 = DenseBlock(64, depth=1)
        self.IM = IM_D(dim)
        self.norm = nn.InstanceNorm2d(dim, affine=True)
        self.activation = nn.PReLU(dim)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = x - x1
        x1_p = self.res1(x1)
        x2_p = self.res2(x2)
        x1_n = x1 - x1_p
        x2_n = x2 - x2_p

        x_p = x1_p + x2_p
        x_n = x1_n + x2_n


        x = self.IM(x_p, x_n)
        x = self.norm(x)
        x = self.activation(x)
        return self.dropout(x)

class DenseEncoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DenseEncoder, self).__init__()
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, (1, 1)),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU(out_channel))
        # self.dense_block = DenseBlock(out_channel, depth=4) # [b, 32, ndim_time, h.n_fft//2+1]
        self.dense_block1 = LM_E(out_channel)
        self.dense_block2 = LM_E(out_channel)
        self.dense_block3 = LM_E(out_channel)
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU(out_channel))

    def forward(self, x):
        x = self.dense_conv_1(x)  # [b, 64, T, F]
        # x = self.dense_block(x)   # [b, 64, T, F]
        x = self.dense_block1(x) + x
        # x2 = self.dense_block2(x1) + x1 + x
        # x = self.dense_block3(x2) + x2 + x1 + x
        x = self.dense_conv_2(x)  # [b, 64, T, F//2]
        return x

class DenseEncoder1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DenseEncoder1, self).__init__()
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, (1, 1)),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU(out_channel))
    def forward(self, x):
        x = self.dense_conv_1(x)  # [b, 64, T, F]
        #x = self.dense_block(x)   # [b, 64, T, F]
        #x = self.dense_conv_2(x)  # [b, 64, T, F//2]
        return x

class DenseEncoder2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DenseEncoder2, self).__init__()
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU(out_channel))

    def forward(self, x):
        #x = self.dense_conv_1(x)  # [b, 64, T, F]
        #x = self.dense_block(x)   # [b, 64, T, F]
        x = self.dense_conv_2(x)  # [b, 64, T, F//2]
        return x

class MaskDecoder(nn.Module):
    def __init__(self, in_channel, out_channel=2):
        super(MaskDecoder, self).__init__()
        #self.dense_block = DenseBlock(in_channel, depth=4)
        self.dense_block1 = LM_D(in_channel)
        self.dense_block2 = LM_D(in_channel)
        self.dense_block3 = LM_D(in_channel)
        self.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, in_channel, (1, 3), (1, 2)),
            nn.Conv2d(in_channel, out_channel, (1, 1)),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU(out_channel),
            nn.Conv2d(out_channel, out_channel, (1, 1))
        )

        self.lsigmoid1 = LearnableSigmoid_2d(512//2+1, beta=2.0)
        #self.lsigmoid2 = LearnableSigmoid_2d(512//2+1, beta=2.0)

    def forward(self, x):
        #x0 = self.dense_block(x)
        x = self.dense_block1(x) + x
        # x2 = self.dense_block2(x1) + x1 + x
        # x = self.dense_block3(x2) + x2 + x1 + x
        x = self.mask_conv(x)
        #x1 = self.mask_conv2(x0)
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        #x1 = x1.permute(0, 3, 2, 1).squeeze(-1)
        x = self.lsigmoid1(x).permute(0, 2, 1).unsqueeze(1)#torch.cat([self.lsigmoid1(x).permute(0, 2, 1).unsqueeze(1), self.lsigmoid2(x1).permute(0, 2, 1).unsqueeze(1)],dim=1)
        return x


class PhaseDecoder(nn.Module):
    def __init__(self, in_channel, out_channel=1):
        super(PhaseDecoder, self).__init__()
        #self.dense_block = DenseBlock(in_channel, depth=4)
        self.dense_block1 = LM_D(in_channel)
        self.dense_block2 = LM_D(in_channel)
        self.dense_block3 = LM_D(in_channel)
        self.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, in_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(in_channel, affine=True),
            nn.PReLU(in_channel)
        )
        self.phase_conv_r = nn.Conv2d(in_channel, out_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(in_channel, out_channel, (1, 1))
        self.norm = nn.InstanceNorm2d(out_channel, affine=True)
    def forward(self, x):
        x = self.dense_block1(x) + x
        # x2 = self.dense_block2(x1) + x1 + x
        # x = self.dense_block3(x2) + x2 + x1 + x
        #x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        return x#self.norm(p + x_r)

class Layer(nn.Module):
    def __init__(self, channel, bias=False):
        super(Layer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool1 = nn.AdaptiveAvgPool2d((126, 1))
        self.conv_n = nn.Sequential(
                nn.Conv2d(channel, channel, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True)
        )

        # feature channel downscale and upscale --> channel weight
        self.conv_du1 = nn.Sequential(
                nn.Conv2d(channel, channel//4, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel//4, channel, 1, padding=0, bias=bias)
        )
        self.softmax = nn.Softmax(-2)
        self.conv_du2 = nn.Sequential(
                nn.Conv2d(channel, channel//4, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel//4, channel, 1, padding=0, bias=bias)
        )
        #self.attn = Attentionn(126, 126, 4, dropout=0.2) #nn.Conv3d(in_channels=4, out_channels=1, kernel_size=(1, 1, 1))
    def forward(self, x, v):

        y1 = self.avg_pool1(x + v)
        #y2 = self.avg_pool2(f)
        y1 = self.conv_n(y1)
        a_x = self.conv_du1(y1)#.unsqueeze(-1) #1, 4, 1, 1
        x_p = self.softmax(a_x) * x
        x_n = self.softmax(-1 * a_x) * x
        a_y = self.conv_du2(y1)#.unsqueeze(-1) #1, 4, 1, 1
        v_p = self.softmax(a_y) * v
        v_n = self.softmax(-1 * a_y) * v

        return x_p, v_p, x_n + v_n, x_p + v_p


class AVIM2(nn.Module):

    def __init__(self, dim = 64):
        super(AVIM2, self).__init__()
        self.conv_a = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 1)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.Sigmoid())

        self.conv_a1 = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 1)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.Sigmoid())

        self.conv_a2 = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 1)),
            nn.InstanceNorm2d(dim, affine=True))

        self.conv_a3 = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 1)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.PReLU(),
            nn.Conv2d(dim, dim, (1, 1)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.Sigmoid())

        self.conv_a4 = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 1)),
            nn.InstanceNorm2d(dim, affine=True))

        self.conv_v = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 1)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.Sigmoid())

        self.conv_v1 = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 1)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.PReLU(),
            nn.Conv2d(dim, dim, (1, 1)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.Sigmoid())

        self.conv_v2 = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 1)),
            nn.InstanceNorm2d(dim, affine=True))

        self.conv_v3 = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 1)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.PReLU(),
            nn.Conv2d(dim, dim, (1, 1)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.Sigmoid())

        self.conv_v4 = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 1)),
            nn.InstanceNorm2d(dim, affine=True))

        self.dense_block1 = DenseBlock(dim, depth=1)
        self.dense_block2 = DenseBlock(dim, depth=2)
        self.ffn1 = FFN(dim, bidirectional=True)
        self.ffn2 = FFN(dim, bidirectional=True)
        self.layernorm = nn.LayerNorm(dim)
    def forward(self, x_p, v_p, v_l, v_n):

        b, c, t, ff = v_n.size()
        v_n = self.conv_a4(v_l) * self.conv_v1(x_p) + v_n

        v_n = self.conv_v4(v_n) * self.conv_a3(self.conv_v3(v_p) * x_p) + v_p

        v_n = v_n.permute(0, 3, 2, 1).contiguous().view(b*ff, t, c)
        v_n = self.layernorm(self.ffn1(v_n) + v_n)
        v_n = v_n.contiguous().view(b, ff, t, c).permute(0, 3, 2, 1)

        # v = self.conv_a2(v_n) + self.conv_v4(v_p)
     # # Intra A
        # #x = x_p * self.conv_a1(x) + self.conv_a2(x)
        # #v = self.conv_v3(v_p) * v
        # v = v.permute(0, 3, 2, 1).contiguous().view(b*ff, t, c)
        # v = self.layernorm(self.ffn2(v) + v)
        # v = v.contiguous().view(b, ff, t, c).permute(0, 3, 2, 1)
        return v_n

class AVIM(nn.Module):

    def __init__(self, dim = 64):
        super(AVIM, self).__init__()
        self.conv_a = nn.Sequential(
                        nn.Conv2d(dim * 2, dim, kernel_size=(1,1)),
                        nn.SiLU(),
                        nn.Conv2d(dim, dim, kernel_size=(1,1)),
                        nn.Sigmoid(),
                    )

        self.conv_v = nn.Sequential(
                        nn.Conv2d(dim * 2, dim, kernel_size=(1,1)),
                        nn.SiLU(),
                        nn.Conv2d(dim, dim, kernel_size=(1,1)),
                        nn.Sigmoid(),
                    )
        self.conv_a1 = nn.Sequential(
                        nn.Conv2d(dim * 2, dim, kernel_size=(1,1)),
                        nn.SiLU(),
                        nn.Conv2d(dim, dim, kernel_size=(1,1)),
                        nn.Sigmoid(),
                    )

        self.conv_v1 = nn.Sequential(
                        nn.Conv2d(dim * 2, dim, kernel_size=(1,1)),
                        nn.SiLU(),
                        nn.Conv2d(dim, dim, kernel_size=(1,1)),
                        nn.Sigmoid(),
                    )

        self.fusion1 = nn.Conv2d(dim * 2, dim, (1, 1))
        self.fusion2 = nn.Conv2d(dim * 2, dim, (1, 1))
        self.ffn1 = DenseBlock(64, depth=2)
        self.ffn2 = DenseBlock(64, depth=2)
        self.layernorm = nn.LayerNorm(dim)
    def forward(self, f, x_p, x_n, v_p, v_n):
        v = self.conv_v(torch.cat([f, v_n], dim=1)) * v_n # + v_n
        x = self.conv_a(torch.cat([f, x_n], dim=1)) * x_n # + x_n
        v_nn = v_n - v
        x_nn = x_n - x

        f_nn = x_nn + v_nn

        v_p = self.fusion1(torch.cat([v_p + v, f], dim=1))
        x_p = self.fusion2(torch.cat([x_p + x, f], dim=1))

        v = v_p - self.conv_v1(torch.cat([f_nn, v_p], dim=1)) * v_p
        x = x_p - self.conv_a1(torch.cat([f_nn, x_p], dim=1)) * x_p


        x = self.ffn1(x)
        v = self.ffn2(v)


        return x, v
class AVIM1(nn.Module):

    def __init__(self, dim = 64):
        super(AVIM1, self).__init__()
        self.conv_a = nn.Sequential(
                        nn.Conv2d(dim * 3, dim, kernel_size=(1,1)),
                        nn.SiLU(),
                        nn.Conv2d(dim, dim, kernel_size=(1,1)),
                        nn.Sigmoid(),
                    )

        # self.conv_v = nn.Sequential(
                        # nn.Conv2d(dim * 2, dim, kernel_size=(1,1)),
                        # nn.SiLU(),
                        # nn.Conv2d(dim, dim, kernel_size=(1,1)),
                        # nn.Sigmoid(),
                    # )
        self.conv_a1 = nn.Sequential(
                        nn.Conv2d(dim * 3, dim, kernel_size=(1,1)),
                        nn.SiLU(),
                        nn.Conv2d(dim, dim, kernel_size=(1,1)),
                        nn.Sigmoid(),
                    )

        # self.conv_v1 = nn.Sequential(
                        # nn.Conv2d(dim * 2, dim, kernel_size=(1,1)),
                        # nn.SiLU(),
                        # nn.Conv2d(dim, dim, kernel_size=(1,1)),
                        # nn.Sigmoid(),
                    # )

        self.conv_a2 = nn.Sequential(
                        nn.Conv2d(dim * 3, dim, kernel_size=(1,1)),
                        nn.SiLU(),
                        nn.Conv2d(dim, dim, kernel_size=(1,1)),
                        nn.Sigmoid(),
                    )

        self.conv_v2 = nn.Sequential(
                        nn.Conv2d(dim * 3, dim, kernel_size=(1,1)),
                        nn.SiLU(),
                        nn.Conv2d(dim, dim, kernel_size=(1,1)),
                        nn.Sigmoid(),
                    )

        # self.fusion1 = nn.Conv2d(dim * 2, dim, (1, 1))
        # self.fusion2 = nn.Conv2d(dim * 2, dim, (1, 1))
        self.ffn1 = DenseBlock(64, depth=2)
        self.ffn2 = DenseBlock(64, depth=2)
        self.layernorm = nn.LayerNorm(dim)
    def forward(self, x1, x2, v1, v2):
        v_e = self.conv_a(torch.cat([x2, v2, v1 - v2], dim=1)) * (v1 - v2)
        v11 = (v1 - v2) - v_e
        v22 = v_e + v2

        x_e = self.conv_a1(torch.cat([x2, v2, x1 - x2], dim=1)) * (x1 - x2)
        x11 = (x1 - x2) - x_e
        x22 =  x_e + x2

        x = x22 - self.conv_a2(torch.cat([x11, v11, x22], dim=1)) * x22
        v = v22 - self.conv_v2(torch.cat([x11, v11, x22], dim=1)) * v22
        x = self.ffn1(x)
        v = self.ffn2(v)
        return x, v
class AVIM11(nn.Module):

    def __init__(self, dim = 64):
        super(AVIM11, self).__init__()
        self.conv_a = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 1)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.Sigmoid())

        self.conv_a1 = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 1)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.Sigmoid())

        self.conv_a2 = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 1)),
            nn.InstanceNorm2d(dim, affine=True))

        self.conv_a3 = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 1)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.Sigmoid())

        self.conv_a4 = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 1)),
            nn.InstanceNorm2d(dim, affine=True))

        self.conv_v = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 1)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.Sigmoid())

        self.conv_v1 = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 1)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.Sigmoid())

        self.conv_v2 = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 1)),
            nn.InstanceNorm2d(dim, affine=True))

        self.conv_v3 = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 1)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.Sigmoid())

        self.conv_v4 = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 1)),
            nn.InstanceNorm2d(dim, affine=True))

        self.dense_block1 = DenseBlock(dim, depth=1)
        self.dense_block2 = DenseBlock(dim, depth=1)
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()
    def forward(self, x_p, x, v_p, v):
     # Inter A_T
        x = self.dense_block1(self.conv_a(v_p) * x + x)
        v = self.dense_block2(self.conv_v(x_p) * v + v)
     # Intra A
        x1 = x_p * self.conv_a1(x) + self.conv_a2(x)
        v1 = v_p * self.conv_v1(v) + self.conv_v2(v)
        # x = self.sigmoid1(v1) * x1
        # v = self.sigmoid2(x1) * v1
     # Inter A-B
        x_out = x1#self.conv_a4(self.conv_a3(x) * v) + x
        v_out = v1#self.conv_v4(self.conv_v3(v) * x) + v

        return x_out, v_out
class Layer1(nn.Module):
    def __init__(self, channel, bias=False):
        super(Layer1, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(1)
        # self.conv_n = nn.Sequential(
                # nn.Conv2d(channel, channel, 1, padding=0, bias=bias),
                # nn.ReLU(inplace=True)
        # )

        # feature channel downscale and upscale --> channel weight
        self.conv_du1 = nn.Sequential(
                nn.Conv2d(channel, channel//4, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel//4, 4, 1, padding=0, bias=bias),
                nn.Softmax()
        )
        self.softmax = nn.Softmax(1)
        # self.conv_du2 = nn.Sequential(
                # nn.Conv2d(channel, channel//4, 1, padding=0, bias=bias),
                # nn.ReLU(inplace=True),
                # nn.Conv2d(channel//4, 4, 1, padding=0, bias=bias),
                # nn.Softmax()
        # )
        #self.attn = Attentionn(126, 126, 4, dropout=0.2) #nn.Conv3d(in_channels=4, out_channels=1, kernel_size=(1, 1, 1))
    def forward(self, x1, x2, x3, x4, xp):

        y1 = self.avg_pool1(xp)
        #y2 = self.avg_pool2(f)
        y_1 = self.conv_du1(y1).unsqueeze(-1) #1, 4, 1, 1

        #y_2 = self.conv_du2(y1).unsqueeze(-1) #1, 4, 1, 1
        x = torch.cat([x1.unsqueeze(1), x2.unsqueeze(1), x3.unsqueeze(1), x4.unsqueeze(1)], dim=1)
        return (x * y_1).sum(dim=1).squeeze(1)#, (x * y_2).sum(dim=1).squeeze(1)


class LM_cross(nn.Module):
    def __init__(self, dim = 64):
        super(LM_cross, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 1), (1, 1)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.PReLU(dim),
            nn.Conv2d(dim, dim, (1, 1), (1, 1)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.Sigmoid(),
        )

        self.conv11 = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 1), (1, 1)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.PReLU(dim)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 1), (1, 1)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.PReLU(dim),
            nn.Conv2d(dim, dim, (1, 1), (1, 1)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.Sigmoid(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 1), (1, 1)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.PReLU(dim),
            nn.Conv2d(dim, dim, (1, 1), (1, 1)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.Sigmoid(),
        )

        self.res1 = DenseBlock(64, depth=1)
        self.res2 = DenseBlock(64, depth=1)
        self.IM = IM_E(dim)
        self.norm = nn.InstanceNorm2d(dim, affine=True)
        self.activation = nn.PReLU(dim)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x, v):
        x1 = self.conv1(v) * (x + v)
        x2 = x - x1
        x1_p = self.res1(x1 * self.conv2(v))
        x2_p = self.res2(x2 * self.conv3(v))
        x1_n = x1 - x1_p
        x2_n = x2 - x2_p

        x_p = x1_p + x2_p
        x_n = x1_n + x2_n


        x = self.IM(x_p, x_n)
        x = self.norm(x)
        x = self.activation(x)
        return self.dropout(x)

class LM_Intra(nn.Module):
    def __init__(self, dim = 64):
        super(LM_Intra, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 1), (1, 1)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.PReLU(dim),
        )
        self.res1 = DenseBlock(64, depth=1)
        self.res2 = DenseBlock(64, depth=1)
        self.IM = IM_E(dim)
        self.norm = nn.InstanceNorm2d(dim, affine=True)
        self.activation = nn.PReLU(dim)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = x - x1
        x1_p = self.res1(x1)
        x2_p = self.res2(x2)
        x1_n = x1 - x1_p
        x2_n = x2 - x2_p

        x_p = x1_p + x2_p
        x_n = x1_n + x2_n


        # x = self.IM(x_p, x_n)
        # x = self.norm(x)
        # x = self.activation(x)
        return x_p, x_n#self.dropout(x)

class IM_cross(nn.Module):
    def __init__(self, dim):
        super(IM_cross, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(dim * 2, dim, kernel_size=(1,1)),
                        nn.InstanceNorm2d(dim, affine=True),
                        nn.PReLU(dim),
                        nn.Conv2d(dim, dim, kernel_size=(1,1)),
                        nn.Sigmoid(),
                    )
        self.conv11 = nn.Sequential(
                        nn.Conv2d(dim * 2, dim, kernel_size=(1,1)),
                        nn.InstanceNorm2d(dim, affine=True),
                        nn.PReLU(dim),
                        nn.Conv2d(dim, dim, kernel_size=(1,1)),
                        nn.Sigmoid(),
                    )
        self.conv2 = nn.Sequential(
                        nn.Conv2d(dim * 2, dim, kernel_size=(1,1)),
                        nn.InstanceNorm2d(dim, affine=True),
                        nn.PReLU(dim),
                        nn.Conv2d(dim, dim, kernel_size=(1,1)),
                        nn.Sigmoid(),
                    )
        self.conv22 = nn.Sequential(
                        nn.Conv2d(dim * 2, dim, kernel_size=(1,1)),
                        nn.InstanceNorm2d(dim, affine=True),
                        nn.PReLU(dim),
                        nn.Conv2d(dim, dim, kernel_size=(1,1)),
                        nn.Sigmoid(),
                    )
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()
    def forward(self, x_p, x_n, v_p, v_n):
        m_x = self.conv1(torch.cat([v_p, x_n], dim = 1))
        x1 = m_x * x_n + x_p
        m_v = self.conv11(torch.cat([x_p, v_n], dim = 1))
        v1 = m_v * v_n + v_p
        # x_1_n = (self.sigmoid1(x_n) - m_x).abs() * x_n
        # v_1_n = (self.sigmoid1(v_n) - m_v).abs() * v_n
        # x2 = (self.sigmoid1(x1) - self.conv2(torch.cat([v1, x_1_n], dim = 1))).abs() * x1
        # v2 = (self.sigmoid1(v1) - self.conv2(torch.cat([x1, v_1_n], dim = 1))).abs() * v1
        return x1, v1#x2 + x_p, v2 + v_p


# ---------- Auditory Selection Module ----------
# -------- Safe reshape wrappers (core fix) --------
# ===================== Utilities (safe reshape / masks / clipping) =====================
def _safe_reshape(x: torch.Tensor, *shape):
    try:
        return x.contiguous().reshape(*shape)
    except RuntimeError:
        return x.clone().reshape(*shape)

def _flatten_tokens(x: torch.Tensor):
    """(B, C, F, T, ...) -> (B, C, N). Use reshape to avoid unsafe view()."""
    B, C = x.shape[:2]
    N = x.numel() // (B * C)
    return _safe_reshape(x, B, C, N)

def _zscore_over_tokens(x: torch.Tensor, eps: float = 1e-6):
    """Z-score normalization along the token dimension; returns the original shape."""
    orig = x.shape
    xf = _flatten_tokens(x)                     # (B,C,N)
    m = xf.mean(-1, keepdim=True)
    s = xf.std(-1, keepdim=True, unbiased=False)
    z = (xf - m) / (s + eps)
    return _safe_reshape(z, *orig)

def _local_var2d(x: torch.Tensor, k=(5,5)):
    """Local variance: Avg(x^2) - Avg(x)^2; keeps shape (B, C, F, T)."""
    pad = (k[1]//2, k[1]//2, k[0]//2, k[0]//2)
    x_pad  = F.pad(x, pad, mode='replicate')
    x2_pad = F.pad(x * x, pad, mode='replicate')
    mean  = F.avg_pool2d(x_pad,  kernel_size=k, stride=1)
    mean2 = F.avg_pool2d(x2_pad, kernel_size=k, stride=1)
    return (mean2 - mean * mean).clamp_min(0.0)

def _topk_mask_tokens(score_map: torch.Tensor, ratio: float):
    """Top-k over the token axis (F*T); returns a binary mask with the same shape."""
    B, C, F, T = score_map.shape
    N = F * T
    k = max(1, int(N * float(ratio)))
    flat = _safe_reshape(score_map, B, C, N)
    idx  = flat.topk(k=k, dim=-1).indices
    M    = torch.zeros_like(flat)
    M.scatter_(-1, idx, 1.0)
    return _safe_reshape(M, B, C, F, T)

def _topk_mask_channels(score_vec: torch.Tensor, ratio: float):
    """Top-k over channels: score_vec (B, C) -> mask (B, C, 1, 1)."""
    B, C = score_vec.shape
    k = max(1, int(C * float(ratio)))
    idx = score_vec.topk(k=k, dim=1).indices
    M   = torch.zeros_like(score_vec)
    M.scatter_(1, idx, 1.0)
    return M.unsqueeze(-1).unsqueeze(-1)

def clip_by_norm(step: torch.Tensor, ref: torch.Tensor, r_max: float, eps: float = 1e-6):
    """Clip the L2 norm of a step to be within r_max * ||ref||."""
    max_norm = r_max * (ref.norm(p=2) + eps)
    s = step.norm(p=2) + eps
    scale = torch.clamp(max_norm / s, max=1.0)
    return step * scale

# ===================== Shared-subspace similarity (with time-shift maximization) =====================
def _time_shift_pairs(a: torch.Tensor, v: torch.Tensor, max_shift: int):
    """Crop-align along time within 卤max_shift; returns a list of (a_s, v_s) pairs (contiguous)."""
    assert a.dim() >= 4 and v.dim() >= 4, "expect (B,C,F,T)"
    T = a.shape[-1]
    pairs = []
    for d in range(-max_shift, max_shift + 1):
        if d == 0:
            pairs.append((a.contiguous(), v.contiguous()))
        elif d > 0:
            if T - d <= 0: continue
            pairs.append((a[..., :T-d].contiguous(), v[..., d:].contiguous()))
        else:
            if T + d <= 0: continue
            pairs.append((a[..., -d:].contiguous(), v[..., :T+d].contiguous()))
    return pairs

@torch.no_grad()
def shared_subspace_similarity(a: torch.Tensor, v: torch.Tensor,
                               r: int = 8, max_shift: int = 2, eps: float = 1e-6) -> torch.Tensor:
    """
    s in [0, 1]: alignment strength of A/V in the linear subspace where they should agree.
      1) (B, C, F, T) -> (B, C, N), per-channel z-score over tokens
      2) Corr = A V^T / (N-1)锛圕脳C锛?      3) s_螖 = mean of the top-r singular values; return max_螖 s_螖
    """
    device = a.device
    best = torch.zeros((), device=device)
    for a_s, v_s in _time_shift_pairs(a, v, max_shift=max_shift):
        A = _flatten_tokens(a_s)
        V = _flatten_tokens(v_s)
        N = min(A.size(-1), V.size(-1))
        if A.size(-1) != N: A = A[..., :N].contiguous()
        if V.size(-1) != N: V = V[..., :N].contiguous()

        A = _zscore_over_tokens(_safe_reshape(A, *A.shape), eps)
        V = _zscore_over_tokens(_safe_reshape(V, *V.shape), eps)
        Corr = torch.matmul(A, V.transpose(1, 2)) / max(N - 1, 1)  # (B,C,C)

        if Corr.dim() == 2:
            sv = torch.linalg.svdvals(Corr)
            k = min(r, sv.numel())
            s = sv[:k].mean()
        else:
            s_list = []
            for b in range(Corr.size(0)):
                sv = torch.linalg.svdvals(Corr[b])
                k = min(r, sv.numel())
                s_list.append(sv[:k].mean())
            s = torch.stack(s_list).mean()
        best = torch.maximum(best, s)
    return torch.clamp(best, 0.0, 1.0)

# ===================== Parameter-free detector =====================
@dataclass
class DetCfg:
    shared_r: int = 8
    mu_ema: float = 0.95
    z_on: float = -1.0
    z_off: float = -0.5
    max_shift: int = 2
    eps: float = 1e-6

class ZeroParamSharedDetector:
    """Maintain EMA statistics of s (mean/var) and output {s, z, trigger}."""
    def __init__(self, cfg: DetCfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.mu = None
        self.var = None
        self.s_ema = None
        self.initialized = False

    @torch.no_grad()
    def update(self, a: torch.Tensor, v: torch.Tensor) -> Dict[str, float]:
        s_now = shared_subspace_similarity(
            a, v, r=self.cfg.shared_r, max_shift=self.cfg.max_shift, eps=self.cfg.eps
        )
        if not self.initialized:
            self.mu = s_now.detach()
            self.var = torch.tensor(1e-3, device=s_now.device, dtype=s_now.dtype)
            self.s_ema = s_now.detach()
            self.initialized = True

        alpha = self.cfg.mu_ema
        self.s_ema = alpha * self.s_ema + (1 - alpha) * s_now
        self.mu     = alpha * self.mu     + (1 - alpha) * s_now
        delta       = s_now - self.mu
        self.var    = alpha * self.var    + (1 - alpha) * (delta * delta)

        z = (self.s_ema - self.mu) / (self.var.sqrt() + self.cfg.eps)
        trigger = (z.item() < self.cfg.z_on)
        return {"s": float(s_now.item()), "z": float(z.item()), "trigger": bool(trigger)}

# ===================== ASM (shared list 鈭?private list + never hard-zero) =====================
class ASMZeroParam(nn.Module):
    """
    - C==1: token-level gating; C>1: channel-level gating
    - forward supports rho_override (can adapt with z)
    """
    def __init__(self, rho=0.25, gamma_min=0.1, mode='auto', private_win=(5,5)):
        super().__init__()
        assert mode in ('auto','token','channel')
        self.rho = float(rho)
        self.gamma_min = float(gamma_min)
        self.mode = mode
        self.private_win = private_win

    def forward(self, a: torch.Tensor, v: torch.Tensor, rho_override: Optional[float] = None):
        assert a.shape[-2:] == v.shape[-2:], "A/V time-frequency shapes must match"
        B, C, F, T = a.shape
        rho = float(self.rho if rho_override is None else rho_override)
        gate_mode = self.mode if self.mode != 'auto' else ('token' if C == 1 else 'channel')
        a_z = _zscore_over_tokens(a); v_z = _zscore_over_tokens(v)
        info = {}

        if gate_mode == 'token':
            s_shared = (a_z * v_z).clamp_min(0.0)      # (B,C,F,T)
            s_priv   = _local_var2d(a, k=self.private_win)
            rho_half = rho / 2.0
            s_shared_m = s_shared.mean(dim=1, keepdim=True)
            s_priv_m   = s_priv.mean(dim=1,   keepdim=True)
            M_shared   = _topk_mask_tokens(s_shared_m, rho_half)
            M_priv     = _topk_mask_tokens(s_priv_m,   rho_half)
            M          = torch.clamp(M_shared + M_priv, max=1.0)
            if C > 1: M = M.expand(B, C, F, T)
            a_sel = M * a + (1.0 - M) * (self.gamma_min * a)
            info.update({
                "shared_mean": float(s_shared.mean().item()),
                "private_mean": float(s_priv.mean().item()),
                "kept_ratio":   float(M.float().mean().item()),
                "mode": "token",
            })
            return a_sel, M, info
        else:
            a_flat = _flatten_tokens(a_z)     # (B,C,N)
            v_flat = _flatten_tokens(v_z)
            if v_flat.size(1) == 1 and a_flat.size(1) > 1:
                v_flat = v_flat.repeat(1, a_flat.size(1), 1)
            s_shared_ch = (a_flat * v_flat).mean(dim=-1).clamp_min(0.0)  # (B,C)
            a_raw = _flatten_tokens(a)
            s_priv_ch = a_raw.var(dim=-1, unbiased=False)                # (B,C)
            rho_half = rho / 2.0
            M_shared = _topk_mask_channels(s_shared_ch, rho_half)
            M_priv   = _topk_mask_channels(s_priv_ch,   rho_half)
            M        = torch.clamp(M_shared + M_priv, max=1.0)           # (B,C,1,1)
            a_sel = M * a + (1.0 - M) * (self.gamma_min * a)
            info.update({
                "shared_mean": float(s_shared_ch.mean().item()),
                "private_mean": float(s_priv_ch.mean().item()),
                "kept_ratio":   float(M.float().mean().item()),
                "mode": "channel",
            })
            return a_sel, M, info


## ===================== CCM (tanh-bounded + trust-region) =====================
class CCMBounded(nn.Module):
    def __init__(self, c_a: int, c_v: int = None, c_mid: int = 16, k: int = 3):
        super().__init__()
        assert k in (3,5,7)
        self.c_a = c_a
        self.c_v = c_v if c_v is not None else c_a
        self.proj_a = nn.Conv2d(c_a, c_mid, kernel_size=1, bias=False)
        self.proj_v = nn.Conv2d(self.c_v, c_mid, kernel_size=1, bias=False)
        self.fuse   = nn.Conv2d(2*c_mid, c_mid, kernel_size=1, bias=False)
        self.dw     = nn.Conv2d(c_mid, c_mid, kernel_size=k, padding=k//2, groups=c_mid, bias=False)
        self.pw     = nn.Conv2d(c_mid, c_mid, kernel_size=1, bias=False)
        self.out    = nn.Conv2d(c_mid, c_a, kernel_size=1, bias=True)
        self.bn     = nn.BatchNorm2d(c_mid)
        self.act    = nn.GELU()

    def forward(self, a_sel: torch.Tensor, v: torch.Tensor):
        if a_sel.shape[-2:] != v.shape[-2:]:
            v = F.interpolate(v, size=a_sel.shape[-2:], mode="bilinear", align_corners=False)
        fa = self.proj_a(a_sel)
        fv = self.proj_v(v)
        x  = torch.cat([fa, fv], dim=1)
        x  = self.fuse(x)
        x  = self.act(self.bn(self.dw(x)))
        x  = self.act(self.pw(x))
        delta = torch.tanh(self.out(x))  # bounded small residual
        return delta

def apply_ccm_update(a: torch.Tensor, delta: torch.Tensor, gamma: float = 0.5, r_max: float = 0.10):
    step = clip_by_norm(gamma * delta, a, r_max=r_max)
    return a + step

# ===================== One symmetric step: fix A first, then fix V =====================
@dataclass
class StepCfg:
    # Trigger/detection
    shared_r: int = 1      # single-channel -> 1; multi-channel can use min(C, 8)
    max_shift: int = 2
    mu_ema: float = 0.95
    z_on: float = -1.0
    z_off: float = -0.5
    # ASM
    rho_base: float = 0.25
    gamma_min_keep: float = 0.1
    # CCM & acceptance
    gamma_max: float = 0.5
    r_max: float = 0.10
    eps_accept: float = 1e-4
    try_times: int = 2
    gamma_min_try: float = 0.125
    # Adaptation (optional)
    adapt_rho: bool = True
    adapt_gamma: bool = True
    tau: float = 0.7
    k_rho: float = 0.05
    k_gamma: float = 0.10
    rho_cap: float = 0.40
    gamma_cap: float = 0.60
    # CCM structure
    c_mid: int = 16
    k: int = 3

class AVJointRepair(nn.Module):
    """
    One forward pass: output (A_out, V_out).
    Internal procedure (order-symmetric):
      1) det_AV triggers? -> fix A (ASM -> CCM -> step-halving -> accept/rollback)
      2) Use A_out as anchor, det_VA triggers? -> fix V (same procedure)
    """
    def __init__(self, c_a: int, c_v: int, cfg: StepCfg = StepCfg()):
        super().__init__()
        self.cfg = cfg
        # Detectors (two sets): A<-V and V<-A, each maintains its own EMA stats
        self.det_AV = ZeroParamSharedDetector(DetCfg(shared_r=cfg.shared_r,
                                                     mu_ema=cfg.mu_ema,
                                                     z_on=cfg.z_on, z_off=cfg.z_off,
                                                     max_shift=cfg.max_shift))
        self.det_VA = ZeroParamSharedDetector(DetCfg(shared_r=cfg.shared_r,
                                                     mu_ema=cfg.mu_ema,
                                                     z_on=cfg.z_on, z_off=cfg.z_off,
                                                     max_shift=cfg.max_shift))
        # ASM (one per side)
        self.asm_A = ASMZeroParam(rho=cfg.rho_base, gamma_min=cfg.gamma_min_keep, mode='auto')
        self.asm_V = ASMZeroParam(rho=cfg.rho_base, gamma_min=cfg.gamma_min_keep, mode='auto')
        # CCM (one per side)
        self.ccm_A = CCMBounded(c_a=c_a, c_v=c_v, c_mid=cfg.c_mid, k=cfg.k)
        self.ccm_V = CCMBounded(c_a=c_v, c_v=c_a, c_mid=cfg.c_mid, k=cfg.k)

    def _adapt_params(self, z: float):
        rho = self.cfg.rho_base
        gam = self.cfg.gamma_max
        if self.cfg.adapt_rho:
            rho = self.cfg.rho_base + self.cfg.k_rho * max(0.0, -z - self.cfg.tau)
            rho = float(min(self.cfg.rho_cap, max(self.cfg.rho_base, rho)))
        if self.cfg.adapt_gamma:
            gam = self.cfg.gamma_max
            gam = self.cfg.gamma_max if z >= -self.cfg.tau else min(self.cfg.gamma_cap,
                                                                     max(self.cfg.gamma_min_try,
                                                                         self.cfg.gamma_max + self.cfg.k_gamma * (-z - self.cfg.tau)))
        return rho, gam

    @torch.no_grad()
    def _repair_one_side(self, X: torch.Tensor, Y: torch.Tensor,
                         detector: ZeroParamSharedDetector,
                         asm: ASMZeroParam, ccm: CCMBounded,
                         rho: float, gamma_max: float):
        """
        Fix X guided by Y: returns (X_out, report)
        """
        # 1) ASM (rho supports adaptation)
        X_sel, M, info_asm = asm(X, Y, rho_override=rho)

        # 2) CCM residual
        delta = ccm(X_sel, Y)

        # 3) Step-halving -> acceptance (using the same s metric)
        s_pre = shared_subspace_similarity(X, Y, r=detector.cfg.shared_r,
                                           max_shift=detector.cfg.max_shift, eps=detector.cfg.eps).item()
        accepted = False
        gamma_used = 0.0
        X_out = X
        gamma = float(gamma_max)

        for _ in range(int(self.cfg.try_times)):
            if gamma < self.cfg.gamma_min_try:
                break
            X_try = apply_ccm_update(X, delta, gamma=gamma, r_max=self.cfg.r_max)
            s_post = shared_subspace_similarity(X_try, Y, r=detector.cfg.shared_r,
                                                max_shift=detector.cfg.max_shift, eps=detector.cfg.eps).item()
            if (s_post - s_pre) >= self.cfg.eps_accept:
                X_out = X_try; accepted = True; gamma_used = gamma
                break
            gamma *= 0.5

        report = {
            "trigger": True,
            "accepted": bool(accepted),
            "s_pre": float(s_pre),
            "s_post": float(s_post if accepted else s_pre),
            "delta_s": float((s_post if accepted else s_pre) - s_pre),
            "gamma_used": float(gamma_used),
            "kept_ratio": float(M.float().mean().item()),
            "asm_shared_mean": info_asm.get("shared_mean", None),
            "asm_private_mean": info_asm.get("private_mean", None),
            "rho_used": float(rho),
        }
        return X_out, report

    @torch.no_grad()
    def forward(self, A: torch.Tensor, V: torch.Tensor):
        """
        Input: A, V in R^{B x C x F x T}
        Output: A_out, V_out, report (triggers/acceptance/螖s/step sizes/keep ratios, etc.)
        """
        assert A.shape[-2:] == V.shape[-2:], "A/V time-frequency shapes must match (interpolate first if needed)"

        # -- Stage 1: use V to fix A --
        detA = self.det_AV.update(A, V)   # {'s','z','trigger'}
        A_out, repA = A, {"trigger": False, "accepted": False}
        if detA["trigger"]:
            rhoA, gamA = self._adapt_params(detA["z"])
            A_out, repA = self._repair_one_side(A, V, self.det_AV, self.asm_A, self.ccm_A,
                                                rho=rhoA, gamma_max=gamA)
        else:
            repA = {"trigger": False, "accepted": False, "s_pre": detA["s"], "s_post": detA["s"],
                    "delta_s": 0.0, "gamma_used": 0.0, "rho_used": self.cfg.rho_base,
                    "kept_ratio": 1.0, "asm_shared_mean": None, "asm_private_mean": None}

        # -- Stage 2: use A_out to fix V --
        detV = self.det_VA.update(V, A_out)
        V_out, repV = V, {"trigger": False, "accepted": False}
        if detV["trigger"]:
            rhoV, gamV = self._adapt_params(detV["z"])
            V_out, repV = self._repair_one_side(V, A_out, self.det_VA, self.asm_V, self.ccm_V,
                                                rho=rhoV, gamma_max=gamV)
        else:
            repV = {"trigger": False, "accepted": False, "s_pre": detV["s"], "s_post": detV["s"],
                    "delta_s": 0.0, "gamma_used": 0.0, "rho_used": self.cfg.rho_base,
                    "kept_ratio": 1.0, "asm_shared_mean": None, "asm_private_mean": None}

        # Summary report
        report = {
            "A": {"z": detA["z"], **repA},
            "V": {"z": detV["z"], **repV},
        }
        return A_out, V_out, report


class Dense(nn.Module):
    def __init__(self, out_channel):
        super(Dense, self).__init__()
        self.LM_cross_a = LM_cross(out_channel)
        self.LM_intra_a = LM_Intra(out_channel)

        self.LM_cross_v = LM_cross(out_channel)
        self.LM_intra_v = LM_Intra(out_channel)
        self.IM = IM_cross(out_channel)
        self.norm = nn.InstanceNorm2d(out_channel, affine=True)
        self.activation1 = nn.PReLU(out_channel)
        self.activation2 = nn.PReLU(out_channel)
        self.activation3 = nn.PReLU(out_channel)
        self.activation4 = nn.PReLU(out_channel)
    def forward(self, a, v):
        a_p, a_n = self.LM_intra_a(a)
        v_p, v_n = self.LM_intra_v(v)

        a1, v1 = self.IM(a_p, a_n, v_p, v_n)

        a1 = self.activation1(self.norm(a1))
        v1 = self.activation2(self.norm(v1))

        a2 = self.activation3(self.norm(self.LM_cross_a(a1, v1)))
        v2 = self.activation4(self.norm(self.LM_cross_v(v1, a1)))

        return a2, v2


class NSCNet(BaseAVModel):
    def __init__(
        self,
        out_channels=128,
        in_channels=512,
        vpre_channels = 512,
        vin_channels = 64,
        vout_channels = 64,
        num_blocks=16,
        upsampling_depth=4,
        enc_kernel_size=21,
        num_sources=2,
        sample_rate=16000,
    ):
        super().__init__(sample_rate=sample_rate)
        self._model_args = {
            "out_channels": out_channels,
            "in_channels": in_channels,
            "vpre_channels": vpre_channels,
            "vin_channels": vin_channels,
            "vout_channels": vout_channels,
            "num_blocks": num_blocks,
            "upsampling_depth": upsampling_depth,
            "enc_kernel_size": enc_kernel_size,
            "num_sources": num_sources,
            "sample_rate": sample_rate,
        }
        self.enc_kernel_size = enc_kernel_size * sample_rate // 1000
        self.prepro_v = nn.Conv2d(vpre_channels, 257, kernel_size=(3, 1), padding=(1, 0))
        # Front end
        self.dense_encoder_phase = DenseEncoder(2, 64)
        self.dense_encoder_a1 = DenseEncoder1(1, 64)
        #self.dense_block = DenseBlock(64, depth=4)

        self.prep =  AVPreprocessExplicitAlignTF(vpre_channels=vpre_channels, n_freq=257, d_sem=64, K=5, kappa=8.0)

        self.dense_encoder_a2 = DenseEncoder2(1, 64)
        self.dense_encoder_v1 = DenseEncoder1(1, 64)

        cfg = StepCfg(shared_r=1, max_shift=2,  # single-channel -> r=1
                  rho_base=0.25, gamma_min_keep=0.1,
                  gamma_max=0.5, r_max=0.10, eps_accept=1e-4, try_times=2,
                  adapt_rho=True, adapt_gamma=True, tau=0.7, k_rho=0.05, k_gamma=0.10,
                  rho_cap=0.40, gamma_cap=0.60,
                  c_mid=16, k=3)

        self.inter1 = AVJointRepair(c_a=64, c_v=64, cfg=cfg)
        self.inter2= AVJointRepair(c_a=64, c_v=64, cfg=cfg)
        self.inter3 = AVJointRepair(c_a=64, c_v=64, cfg=cfg)
        self.inter4 = AVJointRepair(c_a=64, c_v=64, cfg=cfg)

        self.block1 = Dense(64)#DenseBlock(64, depth=3)#GCM(64)
        self.block2 = Dense(64)#DenseBlock(64, depth=3)#GCM(64)
        self.block3 = Dense(64)#DenseBlock(64, depth=3)#GCM(64)
        self.block4 = Dense(64)#DenseBlock(64, depth=3)#GCM(64)
        self.fusion7 = nn.Conv2d(64*2, 64, (1, 1))

        self.TSTransformer = nn.ModuleList([])
        for i in range(4):
            self.TSTransformer.append(TSTransformerBlock(64))

        self.mask_decoder = MaskDecoder(64, 1)
        self.phase_decoder = PhaseDecoder(64, 1)


    def pad_input(self, input, window, stride):
        """
        Zero-padding input according to window/stride size.
        """
        batch_size, nsample = input.shape

        # pad the signals at the end for matching the window/stride size
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(input.type())
            input = torch.cat([input, pad], 1)
        pad_aux = torch.zeros(batch_size, window - stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 1)

        return input, rest
    # Forward pass
    def forward(self, input_wav, mouth_emb):
###########################################################################################################################
        # input shape: (B, T)
        was_one_d = False
        if input_wav.ndim == 1:
            was_one_d = True
            input_wav = input_wav.unsqueeze(0)
        if input_wav.ndim == 2:
            input_wav = input_wav
        if input_wav.ndim == 3:
            input_wav = input_wav.squeeze(1)

        x, rest = self.pad_input(
            input_wav, self.enc_kernel_size, self.enc_kernel_size // 4
        )
        x_1 = torch.stft(
            x,
            512,
            256,
            window=torch.hann_window(512).to(x),
            onesided=True,
            return_complex=False
            )
        x_spec, x_mag, x_phase = power_compress(x_1)
        #print(x_spec.size())
        #x_input = torch.cat([x_mag.permute(0, 2, 1).unsqueeze(1), x_phase1.permute(0, 2, 1).unsqueeze(1)], dim = 1)

        #print(x_input.size())
        # v_resized = F.interpolate(mouth_emb.unsqueeze(-1), size=(126, 1), mode='bilinear', align_corners=False)
        # v_input = self.prepro_v(v_resized).permute(0, 3, 2, 1)
        v_input, r = self.prep(x_mag.permute(0, 2, 1).unsqueeze(1), mouth_emb)
# At this point: A: (1, 2, 257, 126); V: (1, 1, 257, 126)
#############################################################################################################################
########Audio-processing############################
        x = self.dense_encoder_a1(x_mag.permute(0, 2, 1).unsqueeze(1))#(x_mag.permute(0, 2, 1).unsqueeze(1))#(x_input)#(x_spec.permute(0, 1, 3, 2))
        x_phase = self.dense_encoder_phase(torch.cat([x_mag.permute(0, 2, 1).unsqueeze(1), x_phase.permute(0, 2, 1).unsqueeze(1)], dim=1))#torch.cat([x_mag.permute(0, 2, 1).unsqueeze(1), x_phase.permute(0, 2, 1).unsqueeze(1)], dim=1)
        v = self.dense_encoder_v1(v_input)


        x1, v1 = self.block1(x, v)
        x1, v1, report = self.inter1(x1, v1)

        x2, v2 = self.block2(x1, v1)
        x2, v2, report = self.inter2(x2, v2)

        x3, v3 = self.block3(x2, v2)
        x3, v3, report = self.inter3(x3, v3)

        x4, v4 = self.block4(x3, v3)
        x4, v4, report = self.inter4(x4, v4)

        x = self.dense_encoder_a2(x4 + v4*r)#f4 + x_np + v_np + f_np
        x = self.fusion7(torch.cat([x, x_phase], dim = 1))#x + * x_phase#

        for i in range(4):
            x = self.TSTransformer[i](x)

        #print(self.mask_decoder(x).size())
        denoised_mag = x_mag * self.mask_decoder(x).permute(0, 1, 3, 2).squeeze(1)
        # denoised_com = self.complex_decoder(x).permute(0, 3, 2 ,1)

        # final_real = denoised_mag*torch.cos(x_phase) + denoised_com[:,:,:,0]
        # final_imag = denoised_mag*torch.sin(x_phase) + denoised_com[:,:,:,1]

        denoised_pha = self.phase_decoder(x).permute(0, 1, 3, 2).squeeze(1)
        denoised_com = torch.stack((denoised_mag*torch.cos(denoised_pha),
                                    denoised_mag*torch.sin(denoised_pha)), dim=-1)
        est_spec_uncompress = power_uncompress(denoised_com[:, :, :, 0] , denoised_com[:, :, :, 1]).squeeze(1)#(final_real, final_imag).squeeze(1)#
        estimated_waveforms = torch.istft(
            torch.view_as_complex(est_spec_uncompress),#torch.complex(est_spec_uncompress[..., 0].float(), est_spec_uncompress[..., 1].float()),           #est_spec_uncompress,
            512,
            256,
            window=torch.hann_window(512).to(est_spec_uncompress),
            length=input_wav.size(-1),
        )
        return denoised_mag, denoised_pha, est_spec_uncompress, estimated_waveforms.unsqueeze(1)#denoised_mag, denoised_pha, est_spec_uncompress, estimated_waveforms

    def get_model_args(self):
        return dict(self._model_args)

def power_compress(x):
    real = x[..., 0].float()
    imag = x[..., 1].float()
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = torch.pow(mag, 0.3)
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], 1), mag, phase

def power_uncompress(real, imag):
    #print(real.size())
    spec = torch.complex(real, imag)
    #print("1:", spec.size())
    mag = torch.abs(spec)
    #print("2:", mag.size())
    phase = torch.angle(spec)
    #print("3:", phase.size())
    mag = torch.pow(mag, (1.0 / 0.3))
    #print("4:", mag.size())
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    #print(imag_compress.size())
    #print(torch.stack([real_compress, imag_compress], -1).size())
    return torch.stack([real_compress, imag_compress], -1)
