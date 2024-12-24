from transformers.configuration_utils import PretrainedConfig

from lm_eval.models_extra.llama_moe.configuration_llama_moe import LlamaMoEConfig


class LlamaMoEMergedConfig(PretrainedConfig):
    model_type = "llama_moe_merged"  # 🔍
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
            self,
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=None,
            hidden_act="silu",
            max_position_embeddings=2048,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            pretraining_tp=1,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            rope_scaling=None,
            attention_bias=False,
            attention_dropout=0.0,
            # MoE Expert Configs
            num_experts=16,
            num_selects=4,
            size_experts=None,
            # MoE Gate Configs
            original_num_experts=16,  # 🔍
            merge_layers=None,  # 🔍
            gate_type="mapping",  # 🔍
            mapping_dicts=None,  # 🔍
            anomaly_handle="random",  # 🔍
            gate_network="mlp",
            gate_use_softmax=True,
            gate_use_balance=True,
            gate_balance_loss_weight=1e-2,
            gate_add_noise=True,
            gate_noise_epsilon=1e-2,
            # MoE Calculator Configs
            multiply_gate_scores=True,
            score_scale_factor=1.0,
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        self.num_experts = num_experts
        self.num_selects = num_selects
        self.size_experts = size_experts

        #################################### 🔍
        self.original_num_experts = original_num_experts
        self.merge_layers = merge_layers
        self.gate_type = gate_type
        self.mapping_dicts = mapping_dicts

        # # 🔍 convert the keys in mapping_dicts into str format
        # if mapping_dicts is None:
        #     mapping_dicts = [None for _ in range(self.num_hidden_layers)]
        #
        # self.mapping_dicts = []
        # for i, mapping_dict in enumerate(mapping_dicts):
        #     if mapping_dict is not None:
        #         mapping_dict_with_str_keys = {}
        #         for key, value in mapping_dict.items():
        #             mapping_dict_with_str_keys[str(key)] = value
        #         self.mapping_dicts.append(mapping_dict_with_str_keys)
        #     else:
        #         self.mapping_dicts.append(None)

        self.anomaly_handle = anomaly_handle
        #################################### 🔍

        self.gate_network = gate_network
        self.gate_use_softmax = gate_use_softmax
        self.gate_use_balance = False  # 🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍
        self.gate_balance_loss_weight = gate_balance_loss_weight
        self.gate_add_noise = gate_add_noise
        self.gate_noise_epsilon = gate_noise_epsilon

        self.multiply_gate_scores = multiply_gate_scores
        self.score_scale_factor = score_scale_factor

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `name` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s name field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if (
                rope_scaling_factor is None
                or not isinstance(rope_scaling_factor, float)
                or rope_scaling_factor <= 1.0
        ):
            raise ValueError(
                f"`rope_scaling`'s factor field must be an float > 1, got {rope_scaling_factor}"
            )

    # 🔍
    def from_llama_moe_config(
            config: LlamaMoEConfig,
            merge_layers=None,
            num_experts=None,
            gate_type="mapping",
            mapping_dicts=None,
            anomaly_handle="random",
            **kwargs
    ):
        if merge_layers is None:
            merge_layers = list(range(config.num_layers))

        if num_experts is None:
            num_experts = config.num_experts
            intermediate_size = config.intermediate_size
            size_experts = None
        else:  # num_experts are changed layer-wisely, need to update the intermediate_size accordingly
            intermediate_size = [int(config.intermediate_size * num / config.num_experts) for num in num_experts]
            size_experts = None

        original_num_experts = config.num_experts  # this is used to correctly initialize the gate network

        return LlamaMoEMergedConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
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
            pretraining_tp=config.pretraining_tp,
            tie_word_embeddings=config.tie_word_embeddings,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            attention_bias=config.attention_bias,
            attention_dropout=config.attention_dropout,
            # MoE Expert Configs
            num_experts=num_experts,
            num_selects=config.num_selects,
            size_experts=size_experts,
            # MoE Gate Configs
            original_num_experts=original_num_experts,
            merge_layers=merge_layers,
            gate_type=gate_type,
            mapping_dicts=mapping_dicts,
            anomaly_handle=anomaly_handle,
            gate_network=config.gate_network,
            gate_use_softmax=config.gate_use_softmax,
            gate_use_balance=config.gate_use_balance,
            gate_balance_loss_weight=config.gate_balance_loss_weight,
            gate_add_noise=config.gate_add_noise,
            gate_noise_epsilon=config.gate_noise_epsilon,
            # MoE Calculator Configs
            multiply_gate_scores=config.multiply_gate_scores,
            score_scale_factor=config.score_scale_factor,
            **kwargs,
        )
