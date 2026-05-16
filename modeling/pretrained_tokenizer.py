"""Unified factory for pretrained image tokenizers used by RAR.

Supports two backends:
- "maskgit": MaskGIT-VQ (1024 codes, expects images in [0, 1]). Default for RAR.
- "llamagen": LlamaGen VQ (16384 codes, expects images normalized to [-1, 1]).

Both expose a common interface used by the training/sampling code:
    .encode(x, t=0.0) -> long tensor of shape (B, N) of codebook indices
    .decode(codes)    -> images in [0, 1] (B, 3, H, W)
    .decode_tokens(codes) -> same as decode

`t` is the noise-augmentation strength applied in the encoder latent: the
pre-quantization hidden state h is mixed with Gaussian noise as
    h <- (1 - t) * h + t * randn_like(h)
matching yrRandAR's `encode_indices(t)`.
"""
from typing import Optional

import math
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from modeling.modules.maskgit_vqgan import (
    Encoder as MaskGITEncoder,
    Decoder as MaskGITDecoder,
    VectorQuantizer as MaskGITQuantizer,
)
from modeling.llamagen_tokenizer import VQModel as LlamaGenVQModel


class MaskGITPretrainedTokenizer(nn.Module):
    """MaskGIT-VQ tokenizer (formerly `PretrainedTokenizer` in modeling/titok.py)."""

    def __init__(self, pretrained_weight: str):
        super().__init__()
        conf = OmegaConf.create(
            {"channel_mult": [1, 1, 2, 2, 4],
             "num_resolutions": 5,
             "dropout": 0.0,
             "hidden_channels": 128,
             "num_channels": 3,
             "num_res_blocks": 2,
             "resolution": 256,
             "z_channels": 256})
        self.encoder = MaskGITEncoder(conf)
        self.decoder = MaskGITDecoder(conf)
        self.quantize = MaskGITQuantizer(
            num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)

        self.load_state_dict(
            torch.load(pretrained_weight, map_location=torch.device("cpu")),
            strict=True,
        )
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode(self, x: torch.Tensor, t: float = 0.0) -> torch.Tensor:
        h = self.encoder(x)
        if t > 0:
            h = (1 - t) * h + torch.randn_like(h) * t
        _, codebook_indices, _ = self.quantize(h)
        return codebook_indices.detach()

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = self.quantize.get_codebook_entry(codes)
        rec = self.decoder(quantized)
        rec = torch.clamp(rec, 0.0, 1.0)
        return rec.detach()

    @torch.no_grad()
    def decode_tokens(self, codes: torch.Tensor) -> torch.Tensor:
        return self.decode(codes)


class LlamaGenPretrainedTokenizer(nn.Module):
    """LlamaGen VQ tokenizer wrapper compatible with RAR's interface."""

    def __init__(
        self,
        pretrained_weight: str,
        codebook_size: int = 16384,
        codebook_embed_dim: int = 8,
        z_channels: int = 256,
        encoder_ch_mult=(1, 1, 2, 2, 4),
        decoder_ch_mult=(1, 1, 2, 2, 4),
    ):
        super().__init__()
        self.vq = LlamaGenVQModel(
            codebook_size=codebook_size,
            codebook_embed_dim=codebook_embed_dim,
            codebook_l2_norm=True,
            codebook_show_usage=True,
            commit_loss_beta=0.25,
            entropy_loss_ratio=0.0,
            encoder_ch_mult=list(encoder_ch_mult),
            decoder_ch_mult=list(decoder_ch_mult),
            z_channels=z_channels,
            dropout_p=0.0,
        )
        ckpt = torch.load(pretrained_weight, map_location="cpu")
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        msg = self.vq.load_state_dict(state_dict, strict=False)
        # The `codebook_used` buffer can differ in shape between train/eval; tolerate.
        print(f"[LlamaGenPretrainedTokenizer] load_state_dict: {msg}")
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode(self, x: torch.Tensor, t: float = 0.0) -> torch.Tensor:
        # Replicate VQModel.encode_indices to keep noise injection on the quant_conv output.
        h = self.vq.encoder(x)
        h = self.vq.quant_conv(h)
        if t > 0:
            h = (1 - t) * h + torch.randn_like(h) * t
        _, _, info = self.vq.quantize(h)
        # info = (perplexity, min_encodings, min_encoding_indices); indices are flat (B*H*W,).
        indices = info[2]
        bsz = x.shape[0]
        return indices.detach().view(bsz, -1)

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        # codes shape: (B, N). Reshape to a square spatial grid.
        bsz, n = codes.shape
        s = int(math.sqrt(n))
        assert s * s == n, f"LlamaGen codes length {n} is not a perfect square."
        shape = (bsz, self.vq.codebook_embed_dim, s, s)
        rec = self.vq.decode_code(codes, shape)
        # LlamaGen outputs in [-1, 1]; convert to [0, 1] to match RAR's MaskGIT path.
        rec = (rec.clamp(-1.0, 1.0) + 1.0) / 2.0
        return rec.detach()

    @torch.no_grad()
    def decode_tokens(self, codes: torch.Tensor) -> torch.Tensor:
        return self.decode(codes)


def build_pretrained_tokenizer(config) -> Optional[nn.Module]:
    """Create the tokenizer specified by `config.model.vq_model`.

    Selection: `config.model.vq_model.type` (defaults to "maskgit").
    Weight path: `config.model.vq_model.pretrained_tokenizer_weight`.
    """
    vq_cfg = config.model.vq_model
    if vq_cfg.get("finetune_decoder", False):
        return None
    tokenizer_type = vq_cfg.get("type", "maskgit")
    weight = vq_cfg.pretrained_tokenizer_weight
    if tokenizer_type == "maskgit":
        return MaskGITPretrainedTokenizer(weight)
    if tokenizer_type == "llamagen":
        return LlamaGenPretrainedTokenizer(
            weight,
            codebook_size=vq_cfg.get("codebook_size", 16384),
            codebook_embed_dim=vq_cfg.get("codebook_embed_dim", 8),
            z_channels=vq_cfg.get("z_channels", 256),
            encoder_ch_mult=tuple(vq_cfg.get("encoder_ch_mult", [1, 1, 2, 2, 4])),
            decoder_ch_mult=tuple(vq_cfg.get("decoder_ch_mult", [1, 1, 2, 2, 4])),
        )
    raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
