"""
Adapted from
[MosaiclML](https://github.com/mosaicml/examples.git) and
[minGPT](https://github.com/karpathy/minGPT.git)
"""

from __future__ import annotations

import logging
import math
import sys
from abc import abstractmethod
from collections import defaultdict
from dataclasses import replace
from functools import partial
from os.path import join
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
    Union, Any,
)
from copy import deepcopy
import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import einops
from transformers import PreTrainedModel, GenerationConfig, Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation import GenerationMixin

from .config_molmoe import (
    ActivationType,
    BlockType,
    LayerNormType,
    VisionBackboneType,
    ImagePooling2DType,
    ImageProjectType, 
    AttentionType,
)


from .config_molmoe import (
    MolmoConfig,
    VisionBackboneConfig, ModelConfig
)

if sys.version_info.minor > 8:
    from collections.abc import MutableMapping
elif sys.version_info.minor == 8:
    from typing import MutableMapping
else:
    raise SystemExit("This script supports Python 3.8 or higher")


log = logging.getLogger(__name__)


class OLMoConfigurationError(Exception):
    pass


def activation_checkpoint_function(cfg: ModelConfig):
    preserve_rng_state = not (
        (cfg.attention_dropout == 0.0) and (cfg.embedding_dropout == 0.0) and
        (cfg.residual_dropout == 0.0) and (cfg.response_residual_dropout == 0.0)
    )
    from torch.utils.checkpoint import checkpoint

    return partial(
        checkpoint,
        preserve_rng_state=True,
        use_reentrant=False,
    )


def ensure_finite_(x: torch.Tensor, check_neg_inf: bool = True, check_pos_inf: bool = False):
    """
    Modify ``x`` in place to replace ``float("-inf")`` with the minimum value of the dtype when ``check_neg_inf``
    is ``True`` and to replace ``float("inf")`` with the maximum value of the dtype when ``check_pos_inf`` is ``True``.
    """
    if check_neg_inf:
        x.masked_fill_(x == float("-inf"), torch.finfo(x.dtype).min)
    if check_pos_inf:
        x.masked_fill_(x == float("inf"), torch.finfo(x.dtype).max)


def vit_activation_checkpoint_function(cfg: MolmoConfig):
    v_cfg = cfg.vision_backbone
    preserve_rng_state = (
        (v_cfg.attention_dropout == 0.0) and (v_cfg.residual_dropout == 0.0)
    )
    from torch.utils.checkpoint import checkpoint

    return partial(
        checkpoint,
        preserve_rng_state=preserve_rng_state,
        use_reentrant=False,
    )


class BufferCache(dict, MutableMapping[str, torch.Tensor]):
    """
    Cache for attention biases and other things that would normally be stored as buffers.
    We avoid using buffers because we've run into various issues doing so with FSDP.
    In general it appears the way FSDP handles buffers is not well-defined.
    It doesn't shard them but apparently it does synchronize them across processes, which we want to avoid
    since (A) it isn't necessary, and (B) we sometimes have `-inf` in these biases which might get turned into
    NaNs when they're synchronized due to casting or some other issue.
    """


def _non_meta_init_device(config: MolmoConfig) -> torch.device:
    if config.init_device is not None and config.init_device != "meta":
        return torch.device(config.init_device)
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        num_new_embeddings: int,
        features: int,
        device: Union[str, torch.device],
        initializer_range: float = 0.02,
        new_embed_initializer_range: float = 0.02,
    ):
        super().__init__()
        self.initializer_range = initializer_range
        self.new_embed_initializer_range = new_embed_initializer_range
        self.embedding = nn.Parameter(
            torch.zeros(num_embeddings, features, device=device),
        )
        self.new_embedding = nn.Parameter(
            torch.zeros(num_new_embeddings, features, device=device),
        )

    def reset_parameters(self):
        nn.init.normal_(self.embedding, std=self.initializer_range)
        nn.init.normal_(self.new_embedding, std=self.new_embed_initializer_range)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, torch.cat([self.embedding, self.new_embedding], dim=0))


class Dropout(nn.Dropout):
    def __init__(
        self,
        p: float = 0.5,
        inplace: bool = False,
        mask_p: float = 0,
        broadcast_dims: Sequence[int] = (),
    ):
        super().__init__(p, inplace)
        self.mask_p = mask_p
        self.broadcast_dims = broadcast_dims

    def forward(self, input: torch.Tensor, drop_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param input: A tensor of shape `(batch_size, seq_len, embed_dim)`
        :param drop_mask: A tensor of shape `(batch_size, seq_len)` with values of zero or one.
        """
        if self.p == 0.0 and (self.mask_p is None or self.mask_p == 0.0):
            return input
        else:
            if self.mask_p > 0. and self.training:
                assert drop_mask is not None
                drop_mask = drop_mask.to(input.dtype)
                keep_prob = 1.0 - self.p
                keep_prob2 = 1.0 - self.mask_p
                keep_prob = drop_mask * keep_prob2 + (1 - drop_mask) * keep_prob
                keep_prob = keep_prob.unsqueeze(-1)
                dropout_shape = list(input.shape)
                keep_prob = keep_prob.broadcast_to(dropout_shape)
                multiplier = input.new_empty(dropout_shape).bernoulli_(keep_prob)
                multiplier.div_(keep_prob)
                return input * multiplier
            elif self.p > 0. and len(self.broadcast_dims) > 0 and self.training:
                keep_prob = 1.0 - self.p
                dropout_shape = list(input.shape)
                for dim in self.broadcast_dims:
                    dropout_shape[dim] = 1
                keep = input.new_empty(dropout_shape).bernoulli_(keep_prob)
                multiplier = keep.broadcast_to(input.shape)
                multiplier.div_(keep_prob)
                input = input * multiplier
            else:
                return F.dropout(input, self.p, self.training, self.inplace)


class LayerNormBase(nn.Module):
    def __init__(
        self,
        config: MolmoConfig,
        *,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = True,
        eps: float = 1e-05,
        weight_initializer: Optional[Callable] = torch.ones,
        bias_initializer: Optional[Callable] = torch.zeros,
    ):
        super().__init__()
        self.config = config
        self.eps = self.config.layer_norm_eps or eps
        self.normalized_shape = (size or config.d_model,)
        if elementwise_affine or (elementwise_affine is None and self.config.layer_norm_with_affine):
            self.weight = nn.Parameter(weight_initializer(self.normalized_shape, device=config.init_device))
            use_bias = self.config.bias_for_layer_norm
            if use_bias is None:
                use_bias = self.config.include_bias
            if use_bias:
                self.bias = nn.Parameter(bias_initializer(self.normalized_shape, device=config.init_device))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("bias", None)
            self.register_parameter("weight", None)


class LayerNorm(LayerNormBase):
    """
    The default :class:`LayerNorm` implementation which can optionally run in low precision.
    """

    def __init__(
        self,
        config: MolmoConfig,
        size: Optional[int] = None,
        low_precision: bool = False,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-05,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=eps)
        self.low_precision = low_precision

    def _cast_if_autocast_enabled(self, tensor: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        # NOTE: `is_autocast_enabled()` only checks for CUDA autocast, so we use the separate function
        # `is_autocast_cpu_enabled()` for CPU autocast.
        # See https://github.com/pytorch/pytorch/issues/110966.
        if tensor.device.type == "cuda" and torch.is_autocast_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_gpu_dtype())
        elif tensor.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_cpu_dtype())
        else:
            return tensor


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.low_precision:
            module_device = x.device
            downcast_x = self._cast_if_autocast_enabled(x)
            downcast_weight = (
                self._cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
            )
            downcast_bias = self._cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
            with torch.autocast(enabled=False, device_type=module_device.type):
                return F.layer_norm(
                    downcast_x, self.normalized_shape, weight=downcast_weight, bias=downcast_bias, eps=self.eps
                )
        else:
            return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)
    
    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)  # type: ignore
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)  # type: ignore


class RMSLayerNorm(LayerNormBase):
    """
    RMS layer norm, a simplified :class:`LayerNorm` implementation
    """
    def __init__(
        self,
        config: MolmoConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-5,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)

        if self.weight is not None:
            if self.bias is not None:
                return self.weight * x + self.bias
            else:
                return self.weight * x
        else:
            return x
        
    def _cast_if_autocast_enabled(self, tensor: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        # NOTE: `is_autocast_enabled()` only checks for CUDA autocast, so we use the separate function
        # `is_autocast_cpu_enabled()` for CPU autocast.
        # See https://github.com/pytorch/pytorch/issues/110966.
        if tensor.device.type == "cuda" and torch.is_autocast_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_gpu_dtype())
        elif tensor.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_cpu_dtype())
        else:
            return tensor
    
    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)  # type: ignore
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)  # type: ignore


class RotaryEmbedding(nn.Module):
    """
    [Rotary positional embeddings (RoPE)](https://arxiv.org/abs/2104.09864).
    """

    def __init__(self, config: MolmoConfig, cache: BufferCache):
        super().__init__()
        self.config = config
        self.__cache = cache
        # Warm up cache.
        self.get_rotary_embedding(
            config.max_position_embeddings or config.max_sequence_length,
            _non_meta_init_device(config)
        )

    def get_rotary_embedding(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            (pos_sin := self.__cache.get("rope_pos_sin")) is not None
            and (pos_cos := self.__cache.get("rope_pos_cos")) is not None
            and pos_sin.shape[-2] >= seq_len
            and pos_cos.shape[-2] >= seq_len
        ):
            if pos_sin.device != device:
                pos_sin = pos_sin.to(device)
                self.__cache["rope_pos_sin"] = pos_sin
            if pos_cos.device != device:
                pos_cos = pos_cos.to(device)
                self.__cache["rope_pos_cos"] = pos_cos
            return pos_sin[:, :, :seq_len, :], pos_cos[:, :, :seq_len, :]

        with torch.autocast(device.type, enabled=False):
            dim = self.config.head_dim if self.config.head_dim is not None else self.config.d_model // self.config.n_heads
            inv_freq = 1.0 / (self.config.rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim))
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = einsum("i , j -> i j", seq, inv_freq)
            if self.config.rope_impl == "cockatoo":
                positions = freqs.repeat_interleave(2, dim=-1)
            else:
                positions = torch.cat((freqs, freqs), dim=-1)
            pos_sin, pos_cos = positions.sin()[None, None, :, :], positions.cos()[None, None, :, :]
        self.__cache["rope_pos_sin"] = pos_sin
        self.__cache["rope_pos_cos"] = pos_cos
        return pos_sin, pos_cos

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def rotate_every_two(self, x: torch.Tensor) -> torch.Tensor:
        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, hs // 2, 2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return x.view(B, nh, T, hs)

    def apply_rotary_pos_emb(self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.config.rope_impl == "cockatoo":
            return ((t * pos_cos) + (self.rotate_every_two(t) * pos_sin)).to(t.dtype)
        else:
            return ((t * pos_cos) + (self.rotate_half(t) * pos_sin)).to(t.dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.rope_full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        with torch.autocast(q.device.type, enabled=False):
            batch_size = q_.shape[0]
            query_len, key_len = q_.shape[-2], k_.shape[-2]  # could be different if layer_past not None
            if position_ids is not None:
                freqs_cis_len = (self.config.max_position_embeddings or self.config.max_sequence_length)
            else:
                freqs_cis_len = key_len
            pos_sin, pos_cos = self.get_rotary_embedding(freqs_cis_len, q_.device)
            pos_sin = pos_sin.type_as(q_)
            pos_cos = pos_cos.type_as(q_)
            if position_ids is not None:
                assert query_len == key_len, "Query and key lengths must be equal when using position IDs."
                pos_sin = pos_sin[0, 0][position_ids].view(
                    (batch_size, 1, key_len, pos_sin.shape[-1])
                )
                pos_cos = pos_cos[0, 0][position_ids].view(
                    (batch_size, 1, key_len, pos_cos.shape[-1])
                )
            q_ = self.apply_rotary_pos_emb(
                pos_sin[:, :, key_len - query_len : key_len, :],
                pos_cos[:, :, key_len - query_len : key_len, :],
                q_,
            )
            k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)
        return q_.type_as(q), k_.type_as(k)


class Activation(nn.Module):
    def __init__(self, config: MolmoConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_multiplier(self) -> float:
        raise NotImplementedError


class GELU(nn.GELU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class QuickGELU(Activation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)

    @property
    def output_multiplier(self) -> float:
        return 1.0


class ReLU(nn.ReLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class SiLU(nn.SiLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class SwiGLU(Activation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

    @property
    def output_multiplier(self) -> float:
        return 0.5


class LlamaSwiGLU(Activation):
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return F.silu(x1) * x2

    @property
    def output_multiplier(self) -> float:
        return 0.5


def causal_attention_bias(seq_len: int, device: torch.device) -> torch.FloatTensor:
    att_bias = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.float),
        diagonal=1,
    )
    att_bias.masked_fill_(att_bias == 1, torch.finfo(att_bias.dtype).min)
    return att_bias.view(1, 1, seq_len, seq_len)  # type: ignore


def get_causal_attention_bias(cache: BufferCache, seq_len: int, device: torch.device) -> torch.Tensor:
    if (causal_bias := cache.get("causal_attention_bias")) is not None and causal_bias.shape[-1] >= seq_len:
        if causal_bias.device != device:
            causal_bias = causal_bias.to(device)
            cache["causal_attention_bias"] = causal_bias
        return causal_bias
    with torch.autocast(device.type, enabled=False):
        causal_bias = causal_attention_bias(seq_len, device)
    cache["causal_attention_bias"] = causal_bias
    return causal_bias


class MolmoAttention(nn.Module):
    def __init__(
        self, 
        config: MolmoConfig, 
        cache: BufferCache
    ):
        super().__init__()
        self.config = config
        self.__cache = cache
        self.rotary_emb = RotaryEmbedding(config, self.__cache)
        self.k_norm: Optional[LayerNormBase] = None
        self.q_norm: Optional[LayerNormBase] = None
        self.hidden_size = (
            config.mlp_hidden_size if config.mlp_hidden_size is not None \
                else config.mlp_ratio * config.d_model
        )

        if config.attention_layer_norm:
            if config.n_kv_heads is None:
                config.n_kv_heads = config.n_heads
            self.q_norm = RMSLayerNorm(
                config,
                size=config.d_model,
                eps=config.layer_norm_eps
            )
            self.k_norm = RMSLayerNorm(
                config,
                size=config.d_model,
                eps=config.layer_norm_eps
            )

        # Make sure QKV clip coefficient is positive, otherwise it's not well-defined.
        if config.clip_qkv is not None:
            assert config.clip_qkv > 0

        # Activation function
        self.act = SwiGLU(config)
        assert (self.act.output_multiplier * self.hidden_size) % 1 == 0

        # Attention output projection.
        input_dim = config.head_dim * config.n_heads if config.head_dim is not None else config.d_model
        head_dim = config.d_model // config.n_heads
        self.fused_dims = (
            config.d_model,
            config.n_kv_heads * head_dim,
            config.n_kv_heads * head_dim,
        )
        self.att_proj = nn.Linear(
            config.d_model, sum(self.fused_dims),
            bias=config.include_bias or config.qkv_bias,
            device=config.init_device
        )
        self.attn_out = nn.Linear(
            input_dim, config.d_model,
            bias=config.include_bias,
            device=config.init_device
        )
        self.attn_norm = RMSLayerNorm(
            config, 
            size=config.d_model, 
            eps=config.layer_norm_eps)
       
        self.flash_attn_func = None 
        if self.config.attention_type == AttentionType.flash:
            try:
                from flash_attn import flash_attn_func
                self.flash_attn_func = flash_attn_func
            except ModuleNotFoundError:
                pass

    def attention(self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        drop_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = q.size()  # batch size, sequence length, d_model
        dtype = k.dtype 

        # Optionally apply layer norm to keys and queries.
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)

        # Move head forward to be next to the batch dim.
        # shape: (B, nh, T, hs)
        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        k = k.view(B, T, self.config.n_kv_heads, C // self.config.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        v = v.view(B, T, self.config.n_kv_heads, C // self.config.n_heads).transpose(1, 2)

        if self.config.use_position_ids and self.config.rope:
            # Apply rotary embeddings
            q, k = self.rotary_emb(q, k, position_ids=position_ids)

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key.to(k.device), k), dim=-2)
            v = torch.cat((past_value.to(v.device), v), dim=-2)

        present = (k, v) if use_cache else None
        query_len, key_len = q.shape[-2], k.shape[-2]  # could be different if layer_past not None

        if not self.config.use_position_ids and self.config.rope:
            # Apply rotary embeddings
            q, k = self.rotary_emb(q, k)

        if attention_bias is not None:
            # Resize and cast attention bias.
            # The current dtype of the attention bias might not match the dtype that the SDP attn function will
            # run in if AMP is enabled, and this can be a problem if some tokens are masked out due to padding
            # as down-casting the attention bias to the autocast precision will result in -infs, which will
            # cause the SDP attn function to produce NaNs.
            attention_bias = self._cast_attn_bias(
                attention_bias[:, :, key_len - query_len : key_len, :key_len], dtype
            )

        # Get the attention scores.
        # shape: (B, nh, T, hs)
        att = self._scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias,
            drop_mask=drop_mask,
            dropout_p=0.0 if not self.training else self.config.attention_dropout,
            response_dropout_p=0.0 if not self.training else self.config.response_attention_dropout,
            is_causal=attention_bias is None,
        )

        # Re-assemble all head outputs side-by-side.
        att = att.transpose(1, 2).contiguous().view(B, T, C)

        # Apply output projection.
        return self.attn_out(att), present

    @classmethod
    def _cast_attn_bias(cls, bias: torch.Tensor, input_dtype: torch.dtype) -> torch.Tensor:
        target_dtype = input_dtype
        # NOTE: `is_autocast_enabled()` only checks for CUDA autocast, so we use the separate function
        # `is_autocast_cpu_enabled()` for CPU autocast.
        # See https://github.com/pytorch/pytorch/issues/110966.
        if bias.device.type == "cuda" and torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif bias.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            target_dtype = torch.get_autocast_cpu_dtype()
        if bias.dtype != target_dtype:
            bias = bias.to(target_dtype)
            ensure_finite_(bias, check_neg_inf=True, check_pos_inf=False)
        return bias

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        drop_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        response_dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Computes scaled dot product attention on query, key and value tensors, using an optional
        attention mask if passed, and applying dropout if a probability greater than 0.0 is specified.
        """
        if attn_mask is not None:
            attn_mask = attn_mask.to(q.device)

        if self.flash_attn_func is not None and attn_mask is None:
            r = self.flash_attn_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=dropout_p, causal=is_causal
            )
            return r.transpose(1, 2)
        else:
            # torch's sdpa doesn't support GQA, so we're doing this
            assert k.size(1) == v.size(1)
            num_kv_heads = k.size(1)
            num_q_heads = q.size(1)
            if num_q_heads != num_kv_heads:
                assert num_q_heads % num_kv_heads == 0
                k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)
                v = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)

            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )

    def forward(
        self, 
        x, 
        attention_bias,
        position_ids,
        drop_mask,
        layer_past,
        use_cache
    ):
        if not self.config.norm_after:
            atten_in = self.attn_norm(x)
        else:
            atten_in = x

        qkv = self.att_proj(atten_in)

        if self.config.clip_qkv is not None:
            qkv.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        q, k, v = qkv.split(self.fused_dims, dim=-1)
                
        # Get attention scores.
        att, cache = self.attention(
            q, k, v,
            attention_bias,
            position_ids=position_ids,
            drop_mask=drop_mask,
            layer_past=layer_past,
            use_cache=use_cache
        )
        
        if self.config.norm_after:
            att = self.attn_norm(att)
        
        return att, cache


class MolmoMLP(nn.Module):
    def __init__(
        self, 
        config: MolmoConfig
    ):
        # Feed-forward input projection.
        super().__init__()
        self.config = config
        self.hidden_size = (
            config.mlp_hidden_size if config.mlp_hidden_size is not None \
                else config.mlp_ratio * config.d_model
        )
        self.act = SwiGLU(config)
        self.ff_proj = nn.Linear(
            config.d_model,
            self.hidden_size,
            bias=config.include_bias, 
            device=config.init_device
        ) 
        self.ff_out = nn.Linear(
            int(self.act.output_multiplier * self.hidden_size),
            config.d_model,
            bias=config.include_bias,
            device=config.init_device,
        )
        self.ff_norm = RMSLayerNorm(
            config,
            size=config.d_model,
            eps=config.layer_norm_eps
        )
 
    def forward(self, x, image_token_mask: Optional[torch.Tensor] = None): # ✨✨✨
        if not self.config.norm_after:
            x = self.ff_norm(x)

        x = self.ff_proj(x)
        x = self.act(x)
        x = self.ff_out(x)

        if self.config.norm_after:
            x = self.ff_norm(x)

        return x

class MolmoeMLP(nn.Module):
    def __init__(self, config):
        from transformers.activations import ACT2FN
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.hidden_size = (
            config.mlp_hidden_size if config.mlp_hidden_size is not None \
                else config.mlp_ratio * config.d_model
        ) // 2
        self.gate_proj = nn.Linear(self.d_model, self.hidden_size, bias=False)
        self.up_proj = nn.Linear(self.d_model, self.hidden_size, bias=False)
        self.down_proj = nn.Linear(self.hidden_size, self.d_model, bias=False)
        self.act_fn = ACT2FN["silu"]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MolmoeSparseMoeBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([MolmoeMLP(config) for _ in range(self.num_experts)])
        self.ff_norm = RMSLayerNorm(
            config,
            size=config.d_model,
            eps=config.layer_norm_eps
        )

        # topk selection algorithm
        ### 🔍🔍🔍
        self.layer_idx = layer_idx  ### 🔍🔍🔍
        self.expert_capacity = config.expert_capacity \
            if hasattr(config, "expert_capacity") and isinstance(config.expert_capacity, float) else None
        
        # print(self.expert_capacity)
        self.strategy = config.strategy if hasattr(config, "strategy") else None
        self.rounds = config.rounds if hasattr(config, "rounds") else None
        self.drop_n = config.strategy if hasattr(config, "drop_n") else int(0.1 * config.moe_num_experts)
        ###
        
    def forward(self, hidden_states: torch.Tensor, image_token_mask: Optional[torch.Tensor] = None) -> torch.Tensor: # ✨✨✨
        
        hidden_states = self.ff_norm(hidden_states)
        batch_size, sequence_length, hidden_dim = hidden_states.shape
    
        ### 🔍🔍🔍
        total_tokens = batch_size * sequence_length
        if sequence_length == 1 or self.expert_capacity is None: 
            expert_capacity = None
        elif self.expert_capacity is not None:
            expert_capacity = math.ceil(self.expert_capacity * self.top_k * (total_tokens / self.num_experts))
        ### 
        
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        ### 🔍🔍🔍
        if image_token_mask is not None: 
            image_token_mask = image_token_mask.view(-1)
        ### 

        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)   
        routing_weights, selected_experts = self.adjust_tokens(expert_capacity, routing_weights, image_token_mask)
 
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be selected
        ### 🔍🔍🔍 select top-k experts
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts + 1).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]

            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            
            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        # exit()
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

    def overflow(self, utilized_mask, expert_capacity): 
        token_priority = torch.cumsum(utilized_mask, dim=-2)
        overflow = token_priority[-1, :] - expert_capacity
        overflow = torch.relu(overflow)
        return overflow
    
    def adjust_tokens(self, expert_capacity, scores, image_token_mask=None):

        if expert_capacity is None: 
            topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
            return topk_weight, topk_idx

        def _reroute_common(scores, expert_capacity, top_k, rounds, image_token_mask=None):
                        
            T, K = scores.shape 
            top_k_all = int(top_k * rounds)  # temporary over-select

            # Step 1: Initial top-k_all selection
            topk_weight, topk_idx = torch.topk(scores, k=top_k_all, dim=-1, sorted=False)
            mask_buffer = torch.zeros((T, K), dtype=torch.bool, device=scores.device)
            mask_buffer.scatter_(-1, topk_idx, True)

            # Step 2: Enforce expert capacity
            current_usage = mask_buffer.sum(dim=0)
            cols = (current_usage > expert_capacity).nonzero(as_tuple=True)[0]
            if cols.numel() > 0:
                scores_sub = scores[:, cols]
                mask_sub = mask_buffer[:, cols]
                capacity_indices = compute_indices(scores_sub, mask_sub, expert_capacity, image_token_mask)
                mask_buffer[:, cols] = torch.zeros_like(mask_sub).scatter(0, capacity_indices, True)

            # Step 3: Final top_k gathering
            topk_weight = scores.gather(-1, topk_idx)
            top_mask = mask_buffer.gather(-1, topk_idx)
            topk_idx = topk_idx.masked_fill(~top_mask, self.num_experts)
            
            return topk_weight, topk_idx

        def compute_indices(scores_sub, mask_sub, expert_capacity, image_token_mask=None):
            
            masked_scores = scores_sub.masked_fill(~mask_sub, float('-inf'))  # ascending 排序，mask无效
            if image_token_mask is not None and "image" in self.strategy:
                # Expand to match [T, C] shape
                row_penalty = image_token_mask.view(-1, 1).float()
                masked_scores = masked_scores - row_penalty  # subtract 1 from image token rows

            elif image_token_mask is not None and "text" in self.strategy: 
                row_penalty = 1 - image_token_mask.view(-1, 1).float()
                masked_scores = masked_scores - row_penalty  # subtract 1 from image token rows

            _, capacity_indices = torch.topk(
                                        masked_scores, 
                                        k=expert_capacity, 
                                        dim=0, 
                                        sorted=False
                                        )

            return capacity_indices

        def compute_thresholds(scores_sub, mask_sub, overflow_sub):
            sorted_vals, _ = torch.sort(scores_sub.masked_fill(~mask_sub, float('-inf')), dim=0, descending=True)
            valid_counts = mask_sub.sum(dim=0)         # [C]
            keep_k = (valid_counts - overflow_sub).clamp(min=1, max=scores.size(0) - 1)  # [C]
            thresholds_sub = sorted_vals[keep_k, torch.arange(sorted_vals.size(-1), device=scores.device)]  # [C]

            return thresholds_sub

        def reroute_random(scores, expert_capacity):
            topk_weight, topk_idx = torch.topk(scores, self.top_k, dim=-1, sorted=False)
            mask = torch.zeros_like(scores, dtype=torch.bool).scatter(-1, topk_idx, True)
            token_priority = torch.cumsum(mask, dim=0)
            utilized_mask = token_priority <= expert_capacity
            overflow_k = torch.relu(token_priority[-1] - expert_capacity).clamp(min=0)
            if overflow_k.sum() > 0:
                rand_scores = torch.rand_like(scores)
                thresholds = compute_thresholds(rand_scores, utilized_mask, overflow_k)
                drop_mask = (rand_scores <= thresholds) & utilized_mask
                utilized_mask &= ~drop_mask
            flat = utilized_mask[mask].view(topk_idx.size())
            topk_idx[~flat] = self.num_experts
            topk_weight = topk_weight * flat
            return topk_weight, topk_idx

        def reroute_sequential_order(scores, expert_capacity, topk_weight, topk_idx, mode):
            mask = torch.zeros_like(scores, dtype=torch.bool).scatter(-1, topk_idx, True)
            mask_with_image = mask * image_token_mask.unsqueeze(-1) if image_token_mask is not None else mask
            if mode == "last":
                token_priority = torch.cumsum(mask_with_image, dim=-2)
            elif mode == "first":
                reversed_mask = torch.flip(mask_with_image, dims=[-2])
                reversed_cumsum = torch.cumsum(reversed_mask, dim=-2)
                token_priority = torch.flip(reversed_cumsum, dims=[-2])
            else:
                raise ValueError("Unsupported mode: should be 'first' or 'last'")
            utilized_mask = token_priority <= expert_capacity
            utilized_mask = utilized_mask[mask_with_image].view(topk_idx.size())
            topk_idx[~utilized_mask] = self.num_experts
            topk_weight = topk_weight * utilized_mask
            return topk_weight, topk_idx
        
        strategy_list = ["score", "last", "first", "random", "overselect"]
        if expert_capacity is None or not any(s in self.strategy for s in strategy_list):
            return torch.topk(scores, self.top_k, dim=-1, sorted=False)

        if "random" in self.strategy:
            return reroute_random(scores, expert_capacity)
        elif "score" in self.strategy:
            topk_weight, topk_idx = _reroute_common(scores, expert_capacity, self.top_k, self.rounds, image_token_mask)
        elif "overselect" in self.strategy:
            overselect_k = int(self.top_k * 1.5)
            top2k_scores, top2k_idx = torch.topk(scores, overselect_k, dim=-1, sorted=False)
            topk_weight, topk_idx = _reroute_common(scores, expert_capacity, self.top_k, self.rounds, image_token_mask, top2k_scores, top2k_idx)
        else:
            topk_weight, topk_idx = torch.topk(scores, self.top_k, dim=-1, sorted=False)

        if "first" in self.strategy:
            return reroute_sequential_order(scores, expert_capacity, topk_weight, topk_idx, mode="first")
        elif "last" in self.strategy:
            return reroute_sequential_order(scores, expert_capacity, topk_weight, topk_idx, mode="last")
        else:
            return topk_weight, topk_idx

    
class MolmoDecoderLayer(nn.Module):
    """
    A base class for transformer block implementations.
    """
    def __init__(
        self, 
        layer_id: int, 
        config: MolmoConfig, 
        cache: BufferCache
    ):
        super().__init__()
        self.attn = MolmoAttention(config, cache)
        if getattr(config, "moe_num_experts", 0) > 0:
            self.mlp = MolmoeSparseMoeBlock(config, layer_id)
        else:
            self.mlp = MolmoMLP(config)
        self.layer_id = layer_id
        self.config = config
        self.hidden_size = (
            config.mlp_hidden_size if config.mlp_hidden_size is not None else config.mlp_ratio * config.d_model
        )
        self.__cache = cache
        if config.head_dim is None:
            assert config.d_model % config.n_heads == 0

        self._activation_checkpoint_fn = None

        # Dropout.
        self.dropout = Dropout(
            config.residual_dropout, 
            mask_p=config.response_residual_dropout
        ) 

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        drop_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        image_token_mask: Optional[torch.Tensor] = None, # ✨✨✨ labels of image tokens
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Get query, key, value projections.
        shape:
            for regular attn q, k, v: (batch_size, seq_len, d_model)
            for multi-query attn q: (batch_size, seq_len, d_model)
                              k, v: (batch_size, seq_len, d_model // n_heads)
            for group query attn q: (batch_size, seq_len, d_model)
                            k, v: (batch_size, seq_len, d_model // n_kv_heads)
        """

        att, cache = self.attn(
            x, 
            attention_bias=attention_bias,
            position_ids=position_ids,
            drop_mask=drop_mask,
            layer_past=layer_past,
            use_cache=use_cache
        )
        x = x + self.dropout(att, drop_mask=drop_mask)
        og_x = x
        x, _ = self.mlp(x, image_token_mask)
        x = self.dropout(x, drop_mask=drop_mask)
        x = og_x + x

        return x, cache


class MolmoOutput(NamedTuple):
    attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
    """
    Attention keys and values from each block.
    """

    hidden_states: Optional[Tuple[torch.Tensor]]
    """
    Hidden states from each block.
    """

    last_hidden_states: torch.Tensor


class MOLMoGenerateOutput(NamedTuple):
    token_ids: torch.LongTensor
    """
    The generated token IDs, a tensor of shape `(batch_size, beam_size, max_steps)`.
    These do *not* include the original input IDs.
    """

    scores: torch.FloatTensor
    """
    The scores of the generated sequences, a tensor of shape `(batch_size, beam_size)`.
    """


class MultiHeadDotProductAttention(nn.Module):
    def __init__(self, config: MolmoConfig, use_bias: bool = True, is_vit_layer: Optional[bool] = True):
        super().__init__()
        self.config = config
        self.use_bias = use_bias
        
        v_cfg = config.vision_backbone
        self.embed_dim = v_cfg.image_emb_dim
        self.num_heads = v_cfg.image_num_heads
        self.head_dim = v_cfg.image_head_dim
        self.num_key_value_heads = v_cfg.image_num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.initializer_range = v_cfg.initializer_range
        self.is_vit_layer = is_vit_layer

        nlayers = 1 if (is_vit_layer or config.vit_layers is None) else len(config.vit_layers)

        self.wq = nn.Linear(
            nlayers * self.embed_dim,
            self.num_heads * self.head_dim,
            bias=use_bias,
            device=config.init_device,
        )
        self.wk = nn.Linear(
            nlayers * self.embed_dim,
            self.num_key_value_heads * self.head_dim,
            bias=use_bias,
            device=config.init_device,
        )
        self.wv = nn.Linear(
            nlayers * self.embed_dim,
            self.num_key_value_heads * self.head_dim,
            bias=use_bias,
            device=config.init_device,
        )
        self.wo = nn.Linear(
            self.num_heads * self.head_dim,
            self.embed_dim,
            bias=use_bias,
            device=config.init_device,
        )
        self.attention_dropout: Optional[Dropout] = None
        if v_cfg.attention_dropout > 0:
            self.attention_dropout = Dropout(v_cfg.attention_dropout, broadcast_dims=(0, 1))
        self.residual_dropout = Dropout(v_cfg.residual_dropout)
    
    def reset_parameters(self):
        nn.init.normal_(self.wq.weight, std=self.initializer_range)
        nn.init.normal_(self.wk.weight, std=self.initializer_range)
        nn.init.normal_(self.wv.weight, std=self.initializer_range)
        nn.init.normal_(self.wo.weight, std=self.initializer_range)
        if self.use_bias:
            nn.init.constant_(self.wq.bias, 0)
            nn.init.constant_(self.wk.bias, 0)
            nn.init.constant_(self.wv.bias, 0)
            nn.init.constant_(self.wo.bias, 0)

    def _split_heads(self, hidden_states, num_heads) -> torch.Tensor:
        return hidden_states.reshape(hidden_states.shape[:2] + (num_heads, self.head_dim))

    def _merge_heads(self, hidden_states) -> torch.Tensor:
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))
    
    def forward(self, inputs_q: torch.Tensor, inputs_kv: Optional[torch.Tensor] = None) -> torch.Tensor: 
        if inputs_kv is not None:
            inputs_k = inputs_kv
            inputs_v = inputs_kv
        else:
            inputs_k = inputs_q
            inputs_v = inputs_q
        
        xq, xk, xv = self.wq(inputs_q), self.wk(inputs_k), self.wv(inputs_v)

        xq = self._split_heads(xq, self.num_heads)
        xk = self._split_heads(xk, self.num_key_value_heads)
        xv = self._split_heads(xv, self.num_key_value_heads)

        if self.num_heads != self.num_key_value_heads:
            xk = xk.repeat_interleave(self.num_key_value_groups, dim=2, output_size=self.num_heads)
            xv = xv.repeat_interleave(self.num_key_value_groups, dim=2, output_size=self.num_heads)

        og_dtype = xq.dtype

        if self.config.float32_attention:
            xq = xq.to(torch.float)
            xk = xk.to(torch.float)
            xv = xv.to(torch.float)

        if self.config.attention_type == AttentionType.direct:
            attn_weights = torch.einsum("...qhd,...khd->...hqk", xq / math.sqrt(xq.size(-1)), xk)
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(xq.dtype)
            if self.attention_dropout is not None:
                attn_weights = self.attention_dropout(attn_weights)
            attn_output = torch.einsum("...hqk,...khd->...qhd", attn_weights.to(xv.dtype), xv)

        elif self.config.attention_type == AttentionType.sdpa:
            attn_output = F.scaled_dot_product_attention(
                xq.transpose(1, 2).contiguous(),
                xk.transpose(1, 2).contiguous(),
                xv.transpose(1, 2).contiguous(),
                is_causal=False,
                dropout_p=self.config.vision_backbone.attention_dropout
            ).transpose(1, 2)
        else:
            raise NotImplementedError(self.config.attention_type)
        attn_output = attn_output.to(og_dtype)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.wo(attn_output)
        attn_output = self.residual_dropout(attn_output)

        return attn_output


class MultiHeadAttentionPool(nn.Module):
    def __init__(
        self,
        config: MolmoConfig,
        factor: int = 1,
        use_bias: bool = True,
        dropout: bool = True,
        output_layer: bool = True,
        mean_residual: bool = False,
        query: str = "mean",
        is_vit_layer: Optional[bool] = True
    ):
        super().__init__()
        self.config = config
        self.factor = factor
        self.use_bias = use_bias
        self.dropout = dropout
        self.output_layer = output_layer
        self.mean_residual = mean_residual
        self.query = query
        
        v_cfg = config.vision_backbone
        input_dim = v_cfg.image_emb_dim
        self.embed_dim = v_cfg.image_emb_dim * factor
        self.num_heads = v_cfg.image_num_heads
        self.head_dim = v_cfg.image_head_dim * factor
        self.num_key_value_heads = v_cfg.image_num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.initializer_range = v_cfg.initializer_range

        nlayers = 1 if (is_vit_layer or config.vit_layers is None) else len(config.vit_layers)

        if query != "vector":
            self.wq = nn.Linear(
                nlayers * input_dim,
                self.num_heads * self.head_dim,
                bias=use_bias,
                device=config.init_device,
            )
        self.wk = nn.Linear(
            nlayers * input_dim,
            self.num_key_value_heads * self.head_dim,
            bias=use_bias,
            device=config.init_device,
        )
        self.wv = nn.Linear(
            nlayers * input_dim,
            self.num_key_value_heads * self.head_dim,
            bias=use_bias,
            device=config.init_device,
        )

        if query == "vector":
            self.attention_query = nn.Parameter(
                torch.zeros(
                    1, self.num_key_value_heads * self.head_dim, device=config.init_device,
                ),
            )

        if output_layer:
            self.wo = nn.Linear(
                self.num_heads * self.head_dim,
                self.embed_dim,
                bias=use_bias,
                device=config.init_device,
            )
        self.attention_dropout = Dropout(v_cfg.attention_dropout, broadcast_dims=(0, 1))
        if dropout:
            self.residual_dropout = Dropout(v_cfg.residual_dropout)

    def reset_parameters(self):
        if self.query != "vector":
            nn.init.normal_(self.wq.weight, std=self.initializer_range)
        nn.init.normal_(self.wk.weight, std=self.initializer_range)
        nn.init.normal_(self.wv.weight, std=self.initializer_range)
        if self.output_layer:
            nn.init.normal_(self.wo.weight, std=self.initializer_range)
        if self.use_bias:
            if self.query != "vector":
                nn.init.constant_(self.wq.bias, 0)
            nn.init.constant_(self.wk.bias, 0)
            nn.init.constant_(self.wv.bias, 0)
            if self.output_layer:
                nn.init.constant_(self.wo.bias, 0)
        if self.query == "vector":
            nn.init.normal_(self.attention_query, std=self.initializer_range)

    def _split_heads(self, hidden_states, num_heads):
        return hidden_states.reshape(hidden_states.shape[:2] + (num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def forward(self, inputs_kv: torch.Tensor) -> torch.Tensor:

        xk, xv = self.wk(inputs_kv), self.wv(inputs_kv)

        if self.query == "mean":
            inputs_q = inputs_kv.mean(dim=1, keepdim=True)
            xq = self.wq(inputs_q)
        elif self.query == "first":
            inputs_q = inputs_kv[:, :1]
            xq = self.wq(inputs_q)
        elif self.query == "vector":
            xq = self.attention_query.expand(inputs_kv.size(0), -1, -1)
        elif self.query == "constant":
            inputs_q = torch.ones_like(inputs_kv[:, :1]) / math.sqrt(inputs_kv.shape[-1])
            xq = self.wq(inputs_q)
        else:
            raise ValueError(f"Unknown query type: {self.query}")

        xq = self._split_heads(xq, self.num_heads)
        xk = self._split_heads(xk, self.num_key_value_heads)
        xv = self._split_heads(xv, self.num_key_value_heads)

        if self.num_heads != self.num_key_value_heads:
            xk = xk.repeat_interleave(self.num_key_value_groups, dim=2, output_size=self.num_heads)
            xv = xv.repeat_interleave(self.num_key_value_groups, dim=2, output_size=self.num_heads)

        xq = xq.to(torch.float)
        xk = xk.to(torch.float)

        xq = xq / math.sqrt(xq.size(-1))
        attn_weights = torch.einsum("...qhd,...khd->...hqk", xq, xk)

        attn_weights = F.softmax(attn_weights, dim=-1).to(xq.dtype)

        attn_weights = self.attention_dropout(attn_weights).to(xv.dtype)

        attn_output = torch.einsum("...hqk,...khd->...qhd", attn_weights, xv)
        attn_output = self._merge_heads(attn_output)
        if self.output_layer:
            attn_output = self.wo(attn_output)
        if self.dropout:
            attn_output = self.residual_dropout(attn_output)
        if self.mean_residual:
            attn_output += inputs_kv.mean(dim=1, keepdim=True)

        return attn_output


class ViTMLP(nn.Module):
    def __init__(self, config: MolmoConfig):
        super().__init__()
        self.config = config
        v_cfg = config.vision_backbone

        self.w1 = nn.Linear(
            v_cfg.image_emb_dim,
            v_cfg.image_mlp_dim,
            bias=True,
            device=config.init_device,
        )
        # Activation function.
        cfg = deepcopy(config)
        cfg.activation_type = v_cfg.image_mlp_activations
        self.act = QuickGELU(cfg)
        self.w2 = nn.Linear(
            v_cfg.image_mlp_dim,
            v_cfg.image_emb_dim,
            bias=True,
            device=config.init_device,
        )
    
    def reset_parameters(self):
        v_cfg = self.config.vision_backbone
        nn.init.trunc_normal_(self.w1.weight, std=math.sqrt(1 / v_cfg.image_emb_dim), a=-2.0, b=2.0)
        nn.init.trunc_normal_(self.w2.weight, std=math.sqrt(1 / v_cfg.image_mlp_dim), a=-2.0, b=2.0)
        nn.init.zeros_(self.w1.bias)
        nn.init.zeros_(self.w2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w1(x)
        x = self.act(x)
        x = self.w2(x)
        return x


class MLP(nn.Module):
    def __init__(self, config: ModelConfig, input_dim: int, dropout: float = 0.0):
        super().__init__()
        self.config = config  
        self.hidden_size = (
            config.mlp_hidden_size if config.mlp_hidden_size is not None else config.mlp_ratio * config.d_model
        )
        self.initializer_range = config.initializer_range

        self.w1 = nn.Linear(
            input_dim,
            self.hidden_size // 2,
            bias=False,
            device=config.init_device,
        )
        self.w2 = nn.Linear(
            self.hidden_size // 2,
            config.d_model,
            bias=False,
            device=config.init_device,
        )
        self.w3 = nn.Linear(
            input_dim,
            self.hidden_size // 2,
            bias=False,
            device=config.init_device,
        )
        #`MLP` assume the activation takes two inputs, so it must be a 'llama' version.
        self.act = LlamaSwiGLU(config)
        self.dropout = Dropout(dropout)
    
    def reset_parameters(self):
        nn.init.normal_(self.w1.weight, std=self.initializer_range)
        nn.init.normal_(self.w2.weight, std=self.initializer_range)
        nn.init.normal_(self.w3.weight, std=self.initializer_range)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w2(self.act(self.w1(x), self.w3(x)))
        x = self.dropout(x)
        return x


class Residual(nn.Module):
    def __init__(self, submodule: nn.Module):
        super().__init__()
        self.submodule = submodule
    
    def reset_parameters(self):
        self.submodule.reset_parameters()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.submodule(x)


class LayerNormFp32(nn.LayerNorm):
  """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back).
  Derived from https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py.
  """

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    orig_type = x.dtype
    if self.training:
        x = F.layer_norm(x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps)
    else:
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    return x.to(orig_type)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, config: MolmoConfig):
        super().__init__()
        self.config = config

        v_cfg = config.vision_backbone
        self.attention = MultiHeadDotProductAttention(config)
        self.feed_forward = ViTMLP(config)
        self.attention_norm = nn.LayerNorm(
            v_cfg.image_emb_dim,
            eps=v_cfg.image_norm_eps,
            device=config.init_device,
        )
        self.ffn_norm = nn.LayerNorm(
            v_cfg.image_emb_dim,
            eps=v_cfg.image_norm_eps,
            device=config.init_device,
        )

    def reset_parameters(self):
        self.attention.reset_parameters()
        self.feed_forward.reset_parameters()
        self.attention_norm.reset_parameters()
        self.ffn_norm.reset_parameters()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class BlockCollection(nn.Module):
    def __init__(self, config: MolmoConfig):
        super().__init__()
        self.config = config
        self.grad_checkpointing: bool = False
        self._activation_checkpoint_fn: Callable = vit_activation_checkpoint_function(self.config)

        v_cfg = config.vision_backbone
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(config) for _ in range(v_cfg.image_num_layers)
        ])
    
    def reset_parameters(self):
        for r in self.resblocks:
            r.reset_parameters()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        hidden_states = []
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = self._activation_checkpoint_fn(r, x)
            else:
                x = r(x)
            hidden_states.append(x)
        return hidden_states


def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)


class VisionTransformer(nn.Module):
    def __init__(self, config: MolmoConfig):
        super().__init__()
        self.config = config

        v_cfg = config.vision_backbone
        # class embeddings and positional embeddings
        self.scale = v_cfg.image_emb_dim ** -0.5
        self.class_embedding = nn.Parameter(
            torch.zeros(v_cfg.image_emb_dim, device=config.init_device),
        )
        self.num_prefix_tokens: int = 1
        self.positional_embedding = nn.Parameter(
            torch.zeros(v_cfg.image_num_pos, v_cfg.image_emb_dim, device=config.init_device),
        )

        image_patch_size = v_cfg.image_patch_size
        self.patch_embedding = nn.Linear(
            image_patch_size * image_patch_size * 3,
            v_cfg.image_emb_dim,
            bias=False,
            device=config.init_device,
        )

        self.pre_ln = LayerNormFp32(
            v_cfg.image_emb_dim,
            eps=v_cfg.image_norm_eps,
            device=config.init_device,
        )

        self.transformer = BlockCollection(config)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable
    
    def reset_parameters(self):
        nn.init.normal_(self.class_embedding, std=self.scale)
        nn.init.normal_(self.positional_embedding, std=self.scale)
        nn.init.normal_(self.patch_embedding.weight, std=0.02)
        self.pre_ln.reset_parameters()
        self.transformer.reset_parameters()
    
    def add_pos_emb(self, x: torch.Tensor, patch_num: int) -> torch.Tensor:
        cls_emb = self.positional_embedding[0:1]
        pos_emb = self.positional_embedding[1:]

        pos_emb = pos_emb.reshape(
            (int(math.sqrt(pos_emb.shape[0])), int(math.sqrt(pos_emb.shape[0])), pos_emb.shape[1])
        )
    
        (patch_num_0, patch_num_1) = patch_num

        if pos_emb.shape[0] != patch_num_0 or pos_emb.shape[1] != patch_num_1:
            # Dervied from https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
            # antialias: default True in jax.image.resize
            pos_emb = pos_emb.unsqueeze(0).permute(0, 3, 1, 2)
            pos_emb = F.interpolate(
                pos_emb, size=(patch_num_0, patch_num_1), mode="bicubic", align_corners=False, antialias=True,
            )
            pos_emb = pos_emb.permute(0, 2, 3, 1).squeeze(0)

        pos_emb = pos_emb.reshape(-1, pos_emb.shape[-1])
        x = x + torch.cat([cls_emb[None, :, :], pos_emb[None, :, :]], dim=1).to(x.dtype)
        return x

    def forward(self, x: torch.Tensor, patch_num: int = None) -> List[torch.Tensor]:
        """
        : param x: (batch_size, num_patch, n_pixels)
        """
        if patch_num is None:
            patch_num = self.config.vision_backbone.image_num_patch
        B, N, D = x.shape

        x = self.patch_embedding(x)

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        x = self.add_pos_emb(x, patch_num)

        x = self.pre_ln(x)

        hidden_states = self.transformer(x)
        return hidden_states


class MolmoVisionBackbone(nn.Module):
    def __init__(self, config: VisionBackboneConfig):
        super().__init__()
        self.config = config
        input_dim: int = None
        self.image_pooling_2d: nn.Module = None
        if config.image_pooling_2d in {ImagePooling2DType.attention, ImagePooling2DType.attention_meanq}:
            self.image_pooling_2d = MultiHeadDotProductAttention(config, is_vit_layer=False)
            input_dim = config.vision_backbone.image_emb_dim
        elif config.image_pooling_2d == ImagePooling2DType.attention_2wide:
            cfg = deepcopy(config)
            cfg.vision_backbone.image_emb_dim *= 2
            cfg.vision_backbone.image_head_dim *= 2
            self.image_pooling_2d = MultiHeadDotProductAttention(cfg, is_vit_layer=False)
            input_dim = cfg.vision_backbone.image_emb_dim
        elif config.image_pooling_2d == ImagePooling2DType.attention_v2:
            assert config.vit_layers is not None
            use_bias = True
            dropout = True
            output_layer = True
            query = "mean"
            mean_residual = False
            factor = len(config.vit_layers)
            self.image_pooling_2d = MultiHeadAttentionPool(
                config,
                factor=factor,
                use_bias=use_bias,
                dropout=dropout,
                output_layer=output_layer,
                mean_residual=mean_residual,
                query=query,
                is_vit_layer=False,
            )
            input_dim = config.vision_backbone.image_emb_dim * factor
        elif config.image_pooling_2d in [ImagePooling2DType.none, ImagePooling2DType.stack]:
            self.image_pooling_2d = None
            nlayers = 1 if config.vit_layers is None else len(config.vit_layers)
            input_dim = nlayers * config.vision_backbone.image_emb_dim
        else:
            raise NotImplementedError(f"Unknown image pooling 2D method: {config.image_pooling_2d}")
        
        self.input_dim = input_dim

        self.image_projector = MLP(config, input_dim)

        self.image_feature_dropout = Dropout(config.image_feature_dropout)

    @classmethod
    def build(cls, config: MolmoConfig):
        v_cfg = config.vision_backbone
        assert v_cfg is not None
        return MolmoPretrainedVisionBackbone(config)

    def reset_parameters(self):
        if self.image_pooling_2d is not None:
            self.image_pooling_2d.reset_parameters()
        if self.config.image_projector == "2mlp":
            for module in self.image_projector:
                module.reset_parameters()
        elif self.config.image_projector == "linear":
            nn.init.xavier_uniform_(self.image_projector.weight)
        else:
            self.image_projector.reset_parameters()

    @abstractmethod
    def forward(self, images: torch.Tensor, image_masks: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError
        

class MolmoPretrainedVisionBackbone(MolmoVisionBackbone):
    def __init__(self, config: MolmoConfig):
        super().__init__(config)
        v_cfg = VisionBackboneConfig()

        if v_cfg.image_model_type == VisionBackboneType.openai:
            self.image_vit = VisionTransformer(config)
        else:
            raise NotImplementedError(f"Unknown image model type: {v_cfg.image_model_type}")

        self.num_prefix_tokens = self.image_vit.num_prefix_tokens
        assert self.num_prefix_tokens in {0, 1}, "Only 0 or 1 prefix tokens are supported"
        if config.use_cls_feature:
            assert self.num_prefix_tokens > 0, "The model does not have a CLS token"
            nlayers = 1 if config.vit_layers is None else len(config.vit_layers)
            self.cls_projector = nn.Linear(
                nlayers * v_cfg.image_emb_dim,
                self.input_dim,
                bias=False,
                device=config.init_device,
            )

        self.pad_embed = None
        if config.image_padding_embed:
            image_dim = v_cfg.image_emb_dim*len(self.config.vit_layers)
            if config.image_padding_embed in ["pad_embed", "regress"]:
                self.pad_embed = nn.Parameter(
                    torch.zeros((image_dim,), device=config.init_device))
            elif config.image_padding_embed == "pad_and_partial_pad":
                self.pad_embed = nn.Parameter(
                    torch.zeros((2, image_dim), device=config.init_device))
            else:
                raise ValueError(config.image_padding_embed)

    def reset_with_pretrained_weights(self):
        super().reset_parameters()  # resets the connector
        if self.config.vit_load_path:
            vit_load_path = Path(self.config.vit_load_path)
            state_dict_path = resource_path(
                vit_load_path.parent, vit_load_path.name,
                local_cache=vit_load_path.parent,
            )
            assert state_dict_path.is_file(), f"Model file {str(state_dict_path)} not found"
            state_dict = torch.load(state_dict_path, map_location="cpu")
            self.image_vit.load_state_dict(state_dict)
        else:
            self.image_vit.reset_parameters()
        if self.config.use_cls_feature:
            nn.init.xavier_uniform_(self.cls_projector.weight)
        if self.pad_embed is not None:
            nn.init.zeros_(self.pad_embed)

    def reset_parameters(self):
        super().reset_parameters()
        self.image_vit.reset_parameters()
        if self.config.use_cls_feature:
            nn.init.xavier_uniform_(self.cls_projector.weight)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        : param images: (batch_size, num_crops, num_patch, n_pixels)
        """
        cfg = self.config
        v_cfg = self.config.vision_backbone
        B, T, N, D = images.shape

        mask = torch.all(images.view(B * T, N, D) != -1, dim=(1, 2), keepdim=True)

        # Output all hidden states
        # n_layers x (batch_num_crops, (1+)n_tokens, image_emb_dim)
        images = images.view(B * T, N, D)
        image_features = self.image_vit(images)

        if cfg.vit_layers is not None:
            features = []
            for layer in cfg.vit_layers:
                features.append(image_features[layer])
            image_features = torch.cat(features, dim=-1)
        else:
            image_features = image_features[-1]

        cls_embed: torch.Tensor = None
        if self.num_prefix_tokens > 0:
            cls_embed = image_features[:, 0]
            image_features = image_features[:, 1:]
        
        image_features = image_features * mask
        image_features = image_features.view(B, T, N, -1)

        cls_embed = cls_embed.view(B, T, -1) if cls_embed is not None else None

        return image_features, cls_embed
    
    def forward(self, images: torch.Tensor, image_masks: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        cfg = self.config

        # image_features: (batch_size, num_crops(=num_image), num_patch, nximage_emb_dim)
        batch_size, num_image = images.shape[:2]
        image_features, cls_embed = self.encode_image(images)

        og_dtype = image_features.dtype
        if cfg.image_padding_embed:
            assert image_masks is not None
            if cfg.image_padding_embed == "pad_embed":
                all_pad = (image_masks == 0).to(dtype=torch.float32)
                pad_embed = self.pad_embed[None, None, None, :]
                image_features = image_features + pad_embed * torch.unsqueeze(all_pad, -1)
            elif cfg.image_padding_embed == "regress":
                pad_embed = self.pad_embed[None, None, None, :]
                image_features = image_features + pad_embed * torch.unsqueeze(torch.maximum(image_masks, torch.zeros_like(image_masks)), -1)
            elif cfg.image_padding_embed == "pad_and_partial_pad":
                og_dtype = image_features.dtype
                pad_embed = self.pad_embed[:, None, None, None, :]
                all_pad = image_masks == 0
                partial_pad = torch.logical_and(image_masks < 1, torch.logical_not(all_pad)).to(dtype=torch.float32)
                all_pad = all_pad.to(dtype=torch.float32)
                image_features = image_features + pad_embed[0] * torch.unsqueeze(all_pad, -1)
                image_features = image_features + pad_embed[1] * torch.unsqueeze(partial_pad, -1)
            else:
                raise ValueError(cfg.image_padding_embed)

        image_features = image_features.to(og_dtype)
        image_features = self.image_feature_dropout(image_features)
        if cls_embed is not None:
            cls_embed = self.image_feature_dropout(cls_embed)
        
        image_features = image_features.reshape(
            (batch_size, num_image) + cfg.vision_backbone.image_num_patch + (-1,),
        )

        if cfg.vision_backbone.image_num_patch[0] % cfg.image_pooling_h == 1:
            # Pad so we can still pool 2x2 patches
            image_features = F.pad(
                image_features,
                (0, 0, 0, 1, 0, 1, 0, 0, 0, 0),
            )
        
        # image pooling
        image_features = einops.rearrange(
            image_features,
            'b n (h dh) (w dw) c -> (b n h w) (dh dw) c',
            dh=cfg.image_pooling_h,
            dw=cfg.image_pooling_w,
        )

        if cfg.image_pooling_2d == ImagePooling2DType.attention_meanq:
            query = image_features.mean(-2, keepdim=True)
            image_features = self.image_pooling_2d(query, image_features)
        elif cfg.image_pooling_2d == ImagePooling2DType.attention_v2:
            image_features = self.image_pooling_2d(image_features)
        elif cfg.image_pooling_2d not in {ImagePooling2DType.none, ImagePooling2DType.stack}:
            image_features = self.image_pooling_2d(image_features[:, :1, :], image_features)

        h, w = cfg.llm_patches_per_crop
        image_features = image_features.reshape(batch_size, num_image, h * w, -1)

        # MLP layer to map the feature.
        if cfg.image_projector == ImageProjectType.mlpx2:
            for module in self.image_projector:
                image_features = module(image_features)
        else:
            image_features = self.image_projector(image_features)
        
        if self.config.use_cls_feature:
            cls_embed = self.cls_projector(cls_embed)
            if cfg.image_projector == ImageProjectType.mlpx2:
                for module in self.image_projector:
                    cls_embed = module(cls_embed)
            else:
                cls_embed = self.image_projector(cls_embed)
        
        # image_features: (batch_size, num_image, num_patch, d_model)
        # cls_embed: (batch_size, num_image, d_model)
        return image_features, cls_embed


class MolmoPretrainedModel(PreTrainedModel):
    config_class = MolmoConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MolmoDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        if self.vision_backbone is not None:
            self.vision_backbone.reset_parameters()
        self.reset_non_vision_parameters()


class MolmoModel(MolmoPretrainedModel):
    def __init__(
        self, 
        config: MolmoConfig, 
        init_params: bool = True
    ):
        super().__init__(config)
        self.config = config
        self.__cache = BufferCache()

        # Validate config.
        if self.config.embedding_size is not None and self.config.embedding_size != self.config.vocab_size:
            if self.config.embedding_size < self.config.vocab_size:
                raise OLMoConfigurationError("embedding size should be at least as big as vocab size")
            elif self.config.embedding_size % 128 != 0:
                import warnings

                warnings.warn(
                    "Embedding size is not a multiple of 128! This could hurt throughput performance.", UserWarning
                )

        if not (
            0 < self.config.block_group_size <= self.config.n_layers
            and self.config.n_layers % self.config.block_group_size == 0
        ):
            raise OLMoConfigurationError("n layers must be divisible by block group size")

        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)  # this is super slow so make sure torch won't use it

        wte = None
        if self.config.additional_vocab_size is not None:
            wte = Embedding(
                config.embedding_size or config.vocab_size,
                config.additional_vocab_size,
                config.d_model,
                device=config.init_device,
                initializer_range=config.initializer_range,
                new_embed_initializer_range=config.new_embedding_init_range
            )
        else:
            wte=nn.Embedding(
                config.embedding_size or config.vocab_size, config.d_model, device=config.init_device
            )

        self.transformer = nn.ModuleDict(
            dict(
                wte=wte,
                emb_drop=Dropout(config.embedding_dropout),
                ln_f=RMSLayerNorm(
                    config, 
                    size=config.d_model,
                    eps=config.layer_norm_eps),
            )
        )

        layers = [
            MolmoDecoderLayer(i, config, self.__cache) \
                for i in range(config.n_layers)
        ]
        self.transformer.update({"blocks": nn.ModuleList(layers)})
 
        self.vision_backbone: Optional[MolmoVisionBackbone] = None
        if config.vision_backbone is not None:
            self.vision_backbone = MolmoVisionBackbone.build(config)

        if self.vision_backbone is not None:
            self.vision_backbone.reset_with_pretrained_weights()


    @property
    def device(self) -> torch.device:
        device: torch.device = self.transformer.wte.weight.device  # type: ignore
        if device.type == "meta":
            return _non_meta_init_device(self.config)
        else:
            return device

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        response_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_masks: Optional[torch.Tensor] = None,
        image_input_idx: Optional[torch.Tensor] = None,
        subsegment_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        output_hidden_states: Optional[bool] = None,
        append_last_valid_logits: Optional[torch.Tensor] = None,
    ) -> MolmoOutput:
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param input_embeddings: A tensor of shape `(batch_size, seq_len, d_model)` with input
            embeddings. When provided, it is treated as the output of the input embedding layer.
        :param attention_mask: A tensor of shape `(batch_size, seq_len)` that indicates
            which input IDs are masked. A `1` value in the mask means that
            the corresponding input ID should *not* be ignored. A `0` means
            that the corresponding input ID is masked.

            This has the same meaning as the `attention_mask` in HuggingFace's `transformers`
            library.
        :param attention_bias: A tensor of shape `(batch_size, 1, seq_len, seq_len)`,
            `(1, 1, seq_len, seq_len)`, or `(seq_len, seq_len)`. This is used
            to introduce causal or other biases.

            If the tensor is a bool or byte tensor, a `True` or `1` at `attention_bias[:, :, i, j]`
            indicates that the i-th element in the sequence is allowed to attend to the j-th
            element in the sequence.

            If the tensor is a float tensor, it will just be added to the attention
            scores before the softmax.

            The default is causal, which corresponds to a lower-diagonal byte matrix of ones.
        :param response_mask: A tensor of shape `(batch_size, seq_len)` that indicates
            the response mask. A `1` value in the mask means that the corresponding token
            is a response token. A `0` means that the corresponding token is not
            a response token.
        :param past_key_values: Pre-computed keys and values for each attention block.
            Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        :param use_cache: If `True`, return key and value tensors for each block.
        :param last_logits_only: If `True`, only compute the logits for the last token of each sequence.
            This can speed up decoding when you only care about the next token.
        """
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        if past_key_values:
            assert len(past_key_values) == self.config.n_layers

        has_image = images is not None

        assert not (has_image and input_embeddings is not None), "Cannot provide both images and input embeddings."
        assert not (has_image and past_key_values is not None), "Cached key and values should not be used with images."

        batch_size, seq_len = input_ids.size() if input_embeddings is None else input_embeddings.size()[:2]
        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(-2)

        if self.config.unconditioned and input_embeddings is None:
            images = None
            image_input_idx = None

        if self.config.use_position_ids and attention_mask is None:
            attention_mask = input_ids != -1

        # print(f"image_input_idx: {image_input_idx}")

        # print(f"attention_mask: {attention_mask}")
        # print(f"input_ids: {input_ids}")
        # exit()
        if subsegment_ids is not None:
            assert not use_cache, "Subsegment_ids cannot be used with cache."
            subsegment_mask = subsegment_ids.unsqueeze(2) <= subsegment_ids.unsqueeze(1)
            attention_mask = (
                subsegment_mask.to(attention_mask.dtype) *
                attention_mask.unsqueeze(2) *
                attention_mask.unsqueeze(1))
            if position_ids is None:
                raise ValueError(f"Positioned ids must be given if using subsegment_ids")
        else:
            if self.config.use_position_ids and position_ids is None:
                position_ids = torch.clamp(
                    torch.cumsum(attention_mask.to(torch.int32), dim=-1) - 1,
                    min=0,
                ).broadcast_to((batch_size, attention_mask.shape[-1]))

        # Get embeddings of input.
        # shape: (batch_size, seq_len, d_model)
        if input_ids is not None:
            input_ids = input_ids * (input_ids != -1).to(input_ids.dtype)
        x = self.transformer.wte(input_ids) if input_embeddings is None else input_embeddings  # type: ignore

        num_image: Optional[int] = None
        image_token_mask = None
        if images is not None:
            # shape: (batch_size, num_image, num_patch, d_model)
            # cls_embed: (batch_size, num_image, d_model)
            image_features, cls_embed = self.vision_backbone(images, image_masks)
            num_image, num_patch = image_features.shape[1:3]
            assert image_input_idx.shape == (batch_size, num_image, num_patch)
            
            # print(f"batch_size, num_image, num_patch: {batch_size, num_image, num_patch}")
            
            # inster the image feature into the embedding.
            image_features = image_features.view(batch_size, num_image * num_patch, -1)
            image_input_idx = image_input_idx.view(batch_size, num_image * num_patch)
            # print(f"num_image * num_patch, {num_image, num_patch}")
            valid = image_input_idx >= 0
            batch_idx = torch.arange(batch_size, device=x.device)
            batch_idx = torch.tile(batch_idx[:, None], [1, image_features.shape[1]])

            # For hf demo/endpoint
            image_features = image_features.to(x.device)
            # print(f"valid: {valid}")
            # print(f"valid: {valid.size()}")
            # print(f"x: {x.size()}")
            # exit()

            x[batch_idx[valid], image_input_idx[valid]] += image_features[valid]
            ### ✨✨✨ label the image tokens
            image_token_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=x.device)
            image_token_mask[batch_idx[valid], image_input_idx[valid]] = True
            # print(f"x: {x.size()}")
            # print(f"image_token_mask: {image_token_mask.size()}")
            # print(f"image_token_mask: {image_token_mask.float().mean()}")
            # print()
            
            if self.config.use_cls_feature:
                ### ✨✨✨ label image tokens with the consideration of cls token. 
                # New sequence length after CLS insertion
                new_seq_len = x.shape[1]
                # Create new mask with same shape as new x
                new_image_token_mask = torch.zeros((batch_size, new_seq_len), dtype=torch.bool, device=x.device)
                
                # Insert: [CLS] tokens come after position 0, shift everything after
                new_image_token_mask[:, :1] = image_token_mask[:, :1]  # BOS stays
                new_image_token_mask[:, 1 + num_image:] = image_token_mask[:, 1:-num_image]  # shift rest
                # CLS tokens are not image tokens, so leave their positions as 0
                image_token_mask = new_image_token_mask  # update final mask
                # print(f"x: {x.size()}")

                x = torch.cat([x[:, :1], cls_embed, x[:, 1:-num_image]], dim=1)
                
                valid_images = torch.any(
                    (image_input_idx >= 0).view(batch_size, num_image, num_patch), dim=-1
                )
                valid_images = valid_images.to(attention_mask.dtype)
                attention_mask = torch.cat(
                    [attention_mask[:, :1], valid_images, attention_mask[:, 1:-num_image]],
                    dim=1,
                )
                position_ids = torch.clamp(
                    torch.cumsum(attention_mask, dim=-1) - 1,
                    min=0,
                ).broadcast_to((batch_size, attention_mask.shape[-1]))

        # Add input + positional embeddings and apply dropout.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.emb_drop(x)  # type: ignore

        # normalized
        if self.config.normalize_input_embeds:
            x = x * (self.config.d_model ** 0.5)
        # print(f"x: {x.size()}")
        # exit()
        # Transform the attention mask into what the blocks expect.
        if attention_mask is not None:
            # shape: (batch_size, 1, 1, seq_len)
            if len(attention_mask.shape) == 2:
                attention_mask = attention_mask[:, :past_length + seq_len]
                attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
            else:
                attention_mask = attention_mask.unsqueeze(1).to(dtype=torch.float)
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min
        # print(f"attention_mask: {attention_mask.size()}")
        # print(f"attention_mask: {attention_mask}")

        # print(f"attention_bias: {attention_bias}")
        # print(f"attention_bias: {attention_bias.size()}")

        # Merge attention mask with attention bias.
        if (
            attention_bias is not None
            or attention_mask is not None
            # NOTE (epwalsh): we need to initialize the attn bias in order for attn to work properly
            # with key+value cache. Otherwise `F.scaled_dot_product_attention()` doesn't seem to compute
            # scores correctly.
            or past_key_values is not None
        ):
            # print(f"attention_bias: {attention_bias}")

            if attention_bias is None:
                attention_bias = get_causal_attention_bias(self.__cache, past_length + seq_len, x.device)
            elif attention_bias.dtype in (torch.int8, torch.bool):
                attention_bias = attention_bias.to(dtype=torch.float)
                attention_bias.masked_fill_(attention_bias == 0.0, torch.finfo(attention_bias.dtype).min)

            # print(f"attention_bias: {attention_bias}")
            # print(f"attention_bias: {attention_bias.size()}")

            # Transform to the right shape and data type.
            mask_len = seq_len
            if attention_mask is not None:
                mask_len = attention_mask.shape[-1]
            elif past_key_values is not None:
                mask_len = past_key_values[0][0].shape[-2] + seq_len
            attention_bias = attention_bias[:, :, :mask_len, :mask_len].to(dtype=torch.float)

            # Add in the masking bias.
            if attention_mask is not None:
                attention_bias = attention_bias + attention_mask
                # Might get -infs after adding attention mask, since dtype.min + dtype.min = -inf.
                # `F.scaled_dot_product_attention()` doesn't handle -inf like you'd expect, instead
                # it can produce NaNs.
                ensure_finite_(attention_bias, check_neg_inf=True, check_pos_inf=False)

        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None

        # decoder layers
        all_hidden_states = []

        # Apply blocks one-by-one.
        for block_idx, layer in enumerate(self.transformer.blocks):
            if output_hidden_states:
                # add hidden states
                all_hidden_states.append(x)

            layer_past = None if past_key_values is None else past_key_values[block_idx]            
            x, cache = layer(
                             x, 
                             attention_bias=attention_bias, 
                             position_ids=position_ids, 
                             drop_mask=response_mask, 
                             layer_past=layer_past,
                             image_token_mask=image_token_mask, # ✨✨✨
                             use_cache=use_cache, 
                             )

            if attn_key_values is not None:
                assert cache is not None
                attn_key_values.append(cache)
        
        if images is not None and self.config.use_cls_feature:
            assert num_image is not None
            x = torch.cat(
                [x[:, :1], x[:, num_image+1:], torch.zeros_like(x[:, :num_image])],
                dim=1,
            )

        if last_logits_only:
            # shape: (batch_size, 1, d_model)
            if append_last_valid_logits is not None:
                last_valid_output = x[
                    torch.arange(x.shape[0], device=x.device), append_last_valid_logits.to(x.device)]
                x = last_valid_output.unsqueeze(1)
            else:
                x = x[:, -1, :].unsqueeze(1)

        # Apply final layer norm.
        # shape: (batch_size, seq_len or 1, d_model)
        x = self.transformer.ln_f(x)  # type: ignore
        if output_hidden_states:
            # add final hidden state post-final-layernorm, following HuggingFace's convention
            all_hidden_states.append(x)
        
        # Get logits.
        # shape: (batch_size, seq_len or 1, vocab_size)
        return MolmoOutput(
            last_hidden_states=x,
            attn_key_values=attn_key_values,
            hidden_states=tuple(all_hidden_states) \
                if output_hidden_states else None
            )

    def num_params(self, include_embedding: bool = True) -> int:
        """
        Get the total number of parameters.
        """
        params = (np for np in self.named_parameters())
        if not include_embedding:
            params = filter(  # type: ignore
                lambda np: ".wte." not in np[0] and ".wpe." not in np[0],
                params,
            )
        return sum(p.numel() for _, p in params)

    @classmethod
    def from_checkpoint(
        cls, checkpoint_dir: PathOrStr, device: str = "cpu",
        checkpoint_type: Optional[CheckpointType] = None
    ) -> OLMo:
        """
        Load an OLMo model from a checkpoint.
        """
        raise NotImplementedError("This method is not implemented yet.")

    def _make_state_dict_compatible(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Set[str]]]:
        """
        Handles some cases where the state dict is valid yet may need to be transformed in order to
        be loaded.

        This modifies the state dict in-place and also returns it, along with a mapping of original key
        names to new key names in cases where the keys were simply renamed. That mapping can be used
        to make a corresponding optimizer state dict compatible as well.
        """
        import re
        from fnmatch import fnmatch

        new_keys_to_og_keys: Dict[str, str] = {}

        # Remove "_fsdp_wrapped_module." prefix from all keys. We don't want this prefix when the model is
        # not wrapped in FSDP. And when the model is wrapped in FSDP, loading this state dict will still work
        # fine without the prefixes. This also simplifies the other steps below.
        for key in list(state_dict.keys()):
            state_dict[(new_key := key.replace("_fsdp_wrapped_module.", ""))] = state_dict.pop(key)
            new_keys_to_og_keys[new_key] = key

        # For backwards compatibility prior to fixing https://github.com/allenai/LLM/issues/222
        if self.config.block_type == BlockType.sequential:
            for key in list(state_dict.keys()):
                if fnmatch(key, "transformer.*.norm.weight"):
                    tensor = state_dict.pop(key)
                    state_dict[(new_key := key.replace("norm.weight", "attn_norm.weight"))] = tensor
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    state_dict[(new_key := key.replace("norm.weight", "ff_norm.weight"))] = tensor.clone()
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    del new_keys_to_og_keys[key]
                elif fnmatch(key, "transformer.*.norm.bias"):
                    tensor = state_dict.pop(key)
                    state_dict[(new_key := key.replace("norm.bias", "attn_norm.bias"))] = tensor
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    state_dict[(new_key := key.replace("norm.bias", "ff_norm.bias"))] = tensor.clone()
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    del new_keys_to_og_keys[key]

        # For loading a state dict that was saved with a different `block_group_size`.
        if "transformer.block_groups.0.0.attn_out.weight" in state_dict.keys():
            state_dict_block_group_size = len(
                [k for k in state_dict.keys() if fnmatch(k, "transformer.block_groups.0.*.attn_out.weight")]
            )
        else:
            state_dict_block_group_size = 1
        if self.config.block_group_size != state_dict_block_group_size:
            log.info(
                f"Regrouping state dict blocks from group size {state_dict_block_group_size} to "
                f"group size {self.config.block_group_size}"
            )
            # For simplicity we're first going to flatten out the block groups in the state dict (if necessary)
            # and then (re-)group them into the right block sizes.
            if state_dict_block_group_size > 1:
                for key in list(state_dict.keys()):
                    if (m := re.match(r"transformer.block_groups\.(\d+)\.(\d+)\..*", key)) is not None:
                        group_idx, group_block_idx = int(m.group(1)), int(m.group(2))
                        block_idx = (group_idx * state_dict_block_group_size) + group_block_idx
                        state_dict[
                            (
                                new_key := key.replace(
                                    f"block_groups.{group_idx}.{group_block_idx}.", f"blocks.{block_idx}."
                                )
                            )
                        ] = state_dict.pop(key)
                        new_keys_to_og_keys[new_key] = new_keys_to_og_keys.pop(key)

            if self.config.block_group_size > 1:
                # Group the state dict blocks into the right block size.
                for key in list(state_dict.keys()):
                    if (m := re.match(r"transformer.blocks\.(\d+)\..*", key)) is not None:
                        block_idx = int(m.group(1))
                        group_idx, group_block_idx = (
                            block_idx // self.config.block_group_size,
                            block_idx % self.config.block_group_size,
                        )
                        state_dict[
                            (
                                new_key := key.replace(
                                    f"blocks.{block_idx}.", f"block_groups.{group_idx}.{group_block_idx}."
                                )
                            )
                        ] = state_dict.pop(key)
                        new_keys_to_og_keys[new_key] = new_keys_to_og_keys.pop(key)

        og_keys_to_new: Dict[str, Set[str]] = defaultdict(set)
        for new_key, og_key in new_keys_to_og_keys.items():
            og_keys_to_new[og_key].add(new_key)

        return state_dict, og_keys_to_new


class MolmoForCausalLM(PreTrainedModel, GenerationMixin):
    """
    Extremely barebones HF model wrapper.
    """
    config_class = MolmoConfig
    base_model_prefix = "model"
    _no_split_modules = ["MolmoDecoderLayer"]

    def __init__(
        self, 
        config: MolmoConfig
    ):
        super().__init__(config)
        # model_config = create_model_config_from_pretrained_config(config)
        # Initialize model (always on CPU to start with so we don't run out of GPU memory).
        config.init_device = "cpu"
        v_cfg = config.vision_backbone
        if v_cfg is not None:
            v_cfg = VisionBackboneConfig(**v_cfg)
            config.vision_backbone = v_cfg
        self.model = MolmoModel(config)

        if not config.weight_tying:
            self.lm_head = nn.Linear(
                config.d_model,
                config.embedding_size or config.vocab_size,
                bias=config.include_bias,
                device=config.init_device,
            )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        response_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_masks: Optional[torch.Tensor] = None,
        image_input_idx: Optional[torch.Tensor] = None,
        subsegment_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        loss_masks: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        last_logits_only: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        append_last_valid_logits: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[
            Cache
        ] = None,  # This is a hack mitigation of an issue in transformers `4.39.x` https://github.com/huggingface/transformers/issues/29426
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if use_cache is None:
            use_cache = self.config.use_cache

        if output_attentions:
            raise ValueError("output_attentions is not yet supported in OLMo")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            input_embeddings=inputs_embeds,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            response_mask=response_mask,
            images=images,
            image_masks=image_masks,
            image_input_idx=image_input_idx,
            subsegment_ids=subsegment_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            last_logits_only=last_logits_only,
            output_hidden_states=output_hidden_states,
            append_last_valid_logits=append_last_valid_logits,
        )
        
        x = outputs.last_hidden_states
        if self.config.weight_tying:
            logits = F.linear(x, self.model.transformer.wte.weight, None)  # type: ignore
        else:
            logits = self.lm_head(x)  # type: ignore

        if self.config.scale_logits:
            logits.mul_(1 / math.sqrt(self.config.d_model))
        
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping
        
        if not last_logits_only and append_last_valid_logits is not None:
            last_valid_logit = logits[
                torch.arange(logits.shape[0], device=logits.device), append_last_valid_logits]
            logits = torch.cat([logits[:, :-1], last_valid_logit[:, None]], dim=1)

        loss = None
        if labels is not None:
            if loss_masks is not None:
                loss_masks = loss_masks * (loss_masks > 0)
                batch_size_in_tokens = max(loss_masks.sum().item(), 1)
                labels = labels.long()
                labels.masked_fill_(~(loss_masks > 0), -100)
                labels = labels.view(-1)
                logits_for_loss = logits.to(torch.float32).view(-1, logits.size(-1))
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
                loss = loss_fct(logits_for_loss, labels)
                loss = loss.view(input_ids.shape[0], -1)
                loss = loss * loss_masks
                loss = loss.sum() / batch_size_in_tokens
                use_zloss = getattr(self.config, "softmax_auxiliary_loss", False)
                if use_zloss:
                    z_squared = logits_for_loss.logsumexp(-1).pow(2)
                    z_loss = self.config.softmax_auxiliary_loss_scale * z_squared
                    z_loss = z_loss.view(input_ids.shape[0], -1)
                    z_loss = z_loss * loss_masks
                    z_loss = z_loss.sum() / batch_size_in_tokens
                    loss += z_loss
            else:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = torch.nn.CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.embedding_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.attn_key_values,
            hidden_states=outputs.hidden_states,
        )

    def can_generate(self) -> bool:
        return True

    @torch.no_grad()
    def generate_from_batch(
        self,
        batch: Dict[str, Any],
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ):
        if generation_config is not None:
            assert generation_config.use_cache
        
        images = batch.get("images")
        image_masks = batch.get("image_masks")
        image_input_idx = batch.get("image_input_idx")
         
        # Validate inputs.
        input_ids = batch["input_ids"]
        batch_size, seq_len = input_ids.shape
        attention_mask = batch.get("attention_mask", None)
        max_new_tokens = generation_config.max_new_tokens
        assert max_new_tokens is not None
        mask_len = seq_len + max_new_tokens if self.config.use_position_ids else seq_len
        position_ids: Optional[torch.Tensor] = None
        append_last_valid_logits: Optional[torch.Tensor] = None
        if self.config.use_position_ids and attention_mask is None:
            attention_mask = input_ids != -1
            position_ids = torch.clamp(
                torch.cumsum(attention_mask.to(torch.int32), dim=-1) - 1,
                min=0
            )
            append_last_valid_logits = attention_mask.long().sum(dim=-1) - 1
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((batch_size, max_new_tokens))],
                dim=1,
            )
        if attention_mask is not None:
            assert attention_mask.shape == (batch_size, mask_len)
        
        # print(str(super().generate))
        # exit()
        out = super().generate(
            batch["input_ids"],
            generation_config,
            attention_mask=attention_mask,
            images=images,
            image_masks=image_masks,
            image_input_idx=image_input_idx,
            position_ids=position_ids,
            append_last_valid_logits=append_last_valid_logits,
            **kwargs,
        )

        return out
        
    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, past_key_values: Optional[List[Tuple]] = None, **kwargs
    ):
        if past_key_values:
            # This is because we want the model to only process the last generated token.
            input_ids = input_ids[:, -1:]

        if self.config.use_position_ids:
            attention_mask = kwargs.get("attention_mask")
            images = kwargs.get("images")
            image_masks = kwargs.get("image_masks")
            image_input_idx = kwargs.get("image_input_idx")
            position_ids = kwargs.get("position_ids")
            append_last_valid_logits = kwargs.get("append_last_valid_logits")
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": True,
                "last_logits_only": True,
            }
            if past_key_values is None:
                model_inputs["images"] = images
                model_inputs["image_masks"] = image_masks
                model_inputs["image_input_idx"] = image_input_idx
                model_inputs["append_last_valid_logits"] = append_last_valid_logits
        else:    
            model_inputs = {"input_ids": input_ids, "past_key_values": past_key_values}

            model_inputs.update(kwargs)
            model_inputs["use_cache"] = kwargs.pop("use_cache", self.config.use_cache)
        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        if self.config.use_position_ids:
            model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
            if "append_last_valid_logits" in model_kwargs:
                del model_kwargs["append_last_valid_logits"]
            if "images" in model_kwargs:
                del model_kwargs["images"]
                del model_kwargs["image_masks"]
                del model_kwargs["image_input_idx"]
        cache_name, cache = super()._extract_past_from_model_output(outputs)
        model_kwargs[cache_name] = cache
        model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
        return model_kwargs

    # TODO: these are required to make the implementation complete.
    # def resize_position_embeddings(self, new_num_position_embeddings: int):
    #     pass
    #
    # def get_position_embeddings(self) -> Union[nn.Embedding, Tuple[nn.Embedding]]:
    #     pass
    #
    # def _reorder_cache(self, past_key_values, beam_idx):
    #     pass

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.transformer.wte

    def set_input_embeddings(self, value: torch.nn.Module):
        self.model.transformer.wte = value

    def get_output_embeddings(self):
        if self.config.weight_tying:
            return self.model.transformer.wte
        else:
            return self.lm_head

    def set_output_embeddings(self, value: torch.nn.Module):
        if self.config.weight_tying:
            self.model.transformer.wte = value
        else:
            self.lm_head = value

    def tie_weights(self):
        """
        This function is intentionally left as a no-op.

        Weight tying is handled as follows:
        - When the model is initialized, the `lm_head` layer is conditionally defined based on the `weight_tying` configuration.
        See: `if not config.weight_tying: self.transformer.update(...)` in `olmo/model.py`.
        - When computing logits, the `wte` weights are used directly if `weight_tying` is enabled.
        See: `if self.config.weight_tying: logits = F.linear(x, self.transformer.wte.weight, None)` in the `forward` method.

        Therefore, there is no need to explicitly tie the weights in this function.
        """
        pass

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> torch.nn.Embedding:
        """
        Resizes input token embeddings matrix of the model if `new_num_tokens != config.embedding_size`.

        Takes care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:
            new_num_tokens (`int`, *optional*):
                The new number of tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just
                returns a pointer to the input tokens `torch.nn.Embedding` module of the model without doing anything.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the embedding matrix to a multiple of the provided value. If `new_num_tokens` is set to
                `None` will just pad the embedding to a multiple of `pad_to_multiple_of`.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128. For more
                details about this, or help on choosing the correct value for resizing, refer to this guide:
                https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc

        Return:
            `torch.nn.Embedding`: Pointer to the input tokens Embeddings Module of the model.

        Note:
            This method differs from the base class implementation by resizing the `embedding_size` attribute of the
            model configuration instead of the `vocab_size`. It also includes a warning if the resized `embedding_size`
            is less than the `vocab_size`. In OLMo, `embedding_size` refers to the dimensionality of the model's token
            embeddings, while `vocab_size` refers to the number of unique tokens in the vocabulary.
        """
        model_embeds = self._resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        if new_num_tokens is None and pad_to_multiple_of is None:
            return model_embeds

        # Update base model and current model config
        self.config.embedding_size = model_embeds.weight.shape[0]
        self.model.config.embedding_size = model_embeds.weight.shape[0]

        # Check if the embedding size is less than the vocab size
        if self.config.embedding_size < self.config.vocab_size:
            warning_message = (
                f"Resizing token embeddings to size {self.config.embedding_size}, which is less than the vocab size "
                f"{self.config.vocab_size} defined in the model configuration. Make sure your tokenizer's vocabulary "
                "size is less than or equal to the new token embedding size."
            )
            log.warning(warning_message)

        # Tie weights again if needed
        self.tie_weights()

        return model_embeds
