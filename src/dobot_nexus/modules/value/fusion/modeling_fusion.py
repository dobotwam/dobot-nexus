from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from .configuration import ValueModelConfig


def _load_vision_backbone(path: str, dtype: torch.dtype) -> nn.Module:
    from transformers import SiglipModel

    return SiglipModel.from_pretrained(path, torch_dtype=dtype, local_files_only=True)


def _load_language_backbone(path: str, dtype: torch.dtype) -> nn.Module:
    from transformers import Gemma3ForCausalLM

    return Gemma3ForCausalLM.from_pretrained(path, torch_dtype=dtype, local_files_only=True).model


class _QueryCrossAttentionLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float, ffn_multiplier: int):
        super().__init__()
        ffn_dim = dim * ffn_multiplier
        self.self_norm = nn.LayerNorm(dim)
        self.language_query_norm = nn.LayerNorm(dim)
        self.language_key_norm = nn.LayerNorm(dim)
        self.image_query_norm = nn.LayerNorm(dim)
        self.image_key_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.language_attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.image_attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.residual_dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        language_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        *,
        language_key_padding_mask: torch.Tensor,
        image_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        norm_queries = self.self_norm(queries)
        self_out = self.self_attn(
            norm_queries,
            norm_queries,
            norm_queries,
            need_weights=False,
        )[0]
        queries = queries + self.residual_dropout(self_out)

        language_tokens = self.language_key_norm(language_tokens)
        language_out = self.language_attn(
            self.language_query_norm(queries),
            language_tokens,
            language_tokens,
            key_padding_mask=language_key_padding_mask,
            need_weights=False,
        )[0]
        queries = queries + self.residual_dropout(language_out)

        image_tokens = self.image_key_norm(image_tokens)
        image_out = self.image_attn(
            self.image_query_norm(queries),
            image_tokens,
            image_tokens,
            key_padding_mask=image_key_padding_mask,
            need_weights=False,
        )[0]
        queries = queries + self.residual_dropout(image_out)
        return queries + self.ffn(self.ffn_norm(queries))


class DobotValueFusion(nn.Module):
    """SigLIP/Gemma value head with learned-query cross attention."""

    def __init__(self, config: ValueModelConfig, *, vision=None, language=None):
        super().__init__()
        cfg = SimpleNamespace(**config.architecture())
        dtype = getattr(torch, config.backbone_precision)
        self.vision = (
            vision if vision is not None else _load_vision_backbone(str(config.siglip_path), dtype)
        )
        self.language = (
            language
            if language is not None
            else _load_language_backbone(str(config.gemma3_path), dtype)
        )
        if config.freeze_vision_backbone:
            for p in self.vision.parameters():
                p.requires_grad_(False)
            self.vision.eval()
        if config.freeze_language_backbone:
            for p in self.language.parameters():
                p.requires_grad_(False)
            self.language.eval()

        vision_token_size = self.vision.config.vision_config.hidden_size
        language_hidden_size = self.language.config.hidden_size

        self.image_projector = nn.Sequential(
            nn.Linear(vision_token_size, cfg.fusion_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.language_projector = nn.Sequential(
            nn.Linear(language_hidden_size, cfg.fusion_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        query_scale = cfg.fusion_dim**-0.5
        self.value_queries = nn.Parameter(
            torch.randn(1, cfg.fusion_num_queries, cfg.fusion_dim) * query_scale
        )
        self.fusion_layers = nn.ModuleList(
            _QueryCrossAttentionLayer(
                cfg.fusion_dim,
                cfg.fusion_num_heads,
                cfg.dropout,
                cfg.fusion_ffn_multiplier,
            )
            for _ in range(cfg.fusion_num_layers)
        )
        self.final_norm = nn.LayerNorm(cfg.fusion_dim)
        self.value_head = nn.Sequential(
            nn.Linear(cfg.fusion_dim, cfg.fusion_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_dim, cfg.num_bins),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(p.requires_grad for p in self.vision.parameters()):
            self.vision.eval()
        if not any(p.requires_grad for p in self.language.parameters()):
            self.language.eval()
        return self

    def _encode_image_tokens(
        self,
        image_tensor: torch.Tensor,
        image_attention_mask: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image_tensor.ndim != 5:
            raise ValueError(
                f"'image_tensor' must have shape [B,N,C,H,W], got {tuple(image_tensor.shape)}."
            )
        if image_attention_mask.ndim != 2:
            raise ValueError(
                "'image_attention_mask' must have shape [B,N], "
                f"got {tuple(image_attention_mask.shape)}."
            )

        bsz, n_img, c, h, w = image_tensor.shape
        image_mask = image_attention_mask.to(device=device, dtype=torch.bool)
        if image_mask.shape != (bsz, n_img):
            raise ValueError("Batch shape mismatch between image_tensor and image_attention_mask.")
        if not torch.all(image_mask.any(dim=1)):
            raise ValueError("Each sample must have at least one valid camera input.")

        pixels = image_tensor.reshape(bsz * n_img, c, h, w).to(
            device=device,
            dtype=next(self.vision.parameters()).dtype,
        )
        hidden = self.vision.vision_model(pixel_values=pixels, return_dict=True).last_hidden_state

        tokens_per_image = hidden.shape[1]
        image_tokens = hidden.reshape(bsz, n_img * tokens_per_image, hidden.shape[-1])
        token_mask = (
            image_mask.unsqueeze(-1)
            .expand(bsz, n_img, tokens_per_image)
            .reshape(bsz, n_img * tokens_per_image)
        )
        return image_tokens, token_mask

    def _encode_language_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"'input_ids' must have shape [B,T], got {tuple(input_ids.shape)}.")
        if attention_mask.ndim != 2:
            raise ValueError(
                f"'attention_mask' must have shape [B,T], got {tuple(attention_mask.shape)}."
            )
        if input_ids.shape != attention_mask.shape:
            raise ValueError("Batch shape mismatch between input_ids and attention_mask.")
        if not torch.all(attention_mask.to(dtype=torch.bool).any(dim=1)):
            raise ValueError("Each sample must have at least one valid language token.")

        outputs = self.language(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs.last_hidden_state

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        image_tensor = batch["image_tensor"]
        image_attention_mask = batch["image_attention_mask"]

        image_tokens, image_token_mask = self._encode_image_tokens(
            image_tensor,
            image_attention_mask,
            input_ids.device,
        )
        language_tokens = self._encode_language_tokens(input_ids, attention_mask)

        image_tokens = self.image_projector(image_tokens.to(self.image_projector[0].weight.dtype))
        language_tokens = self.language_projector(
            language_tokens.to(self.language_projector[0].weight.dtype)
        )
        language_key_padding_mask = ~attention_mask.to(
            device=language_tokens.device,
            dtype=torch.bool,
        )
        image_key_padding_mask = ~image_token_mask.to(
            device=image_tokens.device,
            dtype=torch.bool,
        )

        queries = self.value_queries.expand(input_ids.shape[0], -1, -1)
        queries = queries.to(device=language_tokens.device, dtype=language_tokens.dtype)
        image_tokens = image_tokens.to(device=language_tokens.device, dtype=language_tokens.dtype)
        image_key_padding_mask = image_key_padding_mask.to(device=language_tokens.device)
        for layer in self.fusion_layers:
            queries = layer(
                queries,
                language_tokens,
                image_tokens,
                language_key_padding_mask=language_key_padding_mask,
                image_key_padding_mask=image_key_padding_mask,
            )

        fused = self.final_norm(queries.mean(dim=1))
        return self.value_head(fused.float())

    def gradient_checkpointing_enable(self):
        # HF encoder checkpointing preserves this Fusion model's separate streams.
        for backbone in (self.vision, self.language):
            backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.language.config.use_cache = False
