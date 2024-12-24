"""MistralMoDExattn model configuration"""
from typing import Union

from transformers import PretrainedConfig, MistralConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class MistralMoDExAttnConfig(PretrainedConfig):
    model_type = "mistral_mod_exattn"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
            self,
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=14336,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_act="silu",
            max_position_embeddings=4096 * 32,
            initializer_range=0.02,
            rms_norm_eps=1e-5,
            use_cache=True,
            pad_token_id=None,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            sliding_window=4096,
            attention_dropout=0.0,
            # 🔍
            is_mod: list = None,
            mod_capacity: Union[float, list] = 0.5,
            mod_loss_coefficient: float = 0.1,
            mod_loss_type: str = "self",  # self cos cos-global
            rescale_hidden_states: bool = True,
            scale_factor: float = 1.0,
            scale_gap: float = 1.0,
            eval_use_topk: bool = False,
            gate_init_method: str = "zero",  # zero random
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout

        # 🔍
        self.is_mod = [False for _ in range(num_hidden_layers)] if is_mod is None else is_mod
        self.mod_capacity = mod_capacity
        self.mod_loss_coefficient = mod_loss_coefficient
        self.mod_loss_type = mod_loss_type
        self.rescale_hidden_states = rescale_hidden_states
        self.scale_factor = scale_factor
        self.scale_gap = scale_gap
        self.eval_use_topk = eval_use_topk
        self.gate_init_method = gate_init_method

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    # 🔍
    def from_mistral_config(
            config: MistralConfig,
            is_mod: list = None,
            mod_capacity: float = 0.5,
            mod_loss_coefficient: float = 0.1,
            mod_loss_type: str = "self",
            rescale_hidden_states: bool = True,
            scale_factor: float = 1.0,
            scale_gap: float = 1.0,
            eval_use_topk: bool = False,
            gate_init_method: str = "zero",
    ):
        return MistralMoDExAttnConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            hidden_act=config.hidden_act,
            max_position_embeddings=config.max_position_embeddings,
            initializer_range=config.initializer_range,
            rms_norm_eps=config.rms_norm_eps,
            use_cache=config.use_cache,
            pad_token_id=config.pad_token_id,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
            tie_word_embeddings=config.tie_word_embeddings,
            rope_theta=config.rope_theta,
            attention_dropout=config.attention_dropout,
            # 🔍
            is_mod=is_mod,
            mod_capacity=mod_capacity,
            mod_loss_coefficient=mod_loss_coefficient,
            mod_loss_type=mod_loss_type,
            rescale_hidden_states=rescale_hidden_states,
            scale_factor=scale_factor,
            scale_gap=scale_gap,
            eval_use_topk=eval_use_topk,
            gate_init_method=gate_init_method,
        )
