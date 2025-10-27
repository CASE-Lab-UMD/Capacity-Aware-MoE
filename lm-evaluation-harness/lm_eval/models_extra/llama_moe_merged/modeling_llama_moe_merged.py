import random
import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers import PreTrainedModel
from transformers.utils import logging

from .configuration_llama_moe_merged import LlamaMoEMergedConfig
from lm_eval.models_extra.llama_moe.modeling_llama_moe import LlamaMoEForCausalLM, LlamaMoEModel, LlamaMoEDecoderLayer, LinearGLUMoELayer, LlamaAttention, LlamaRMSNorm, UniversalCalculator, TopKBalancedNoisyGate, LinearGLUExperts

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaMoEMergedConfig"


class IdenticalMappingGate(TopKBalancedNoisyGate):
    """Note that the gate is non-differentiable, as the scores are created using the static mapping!"""

    def __init__(self, input_size, num_experts, num_selects, mapping_dict, anomaly_handle="random", gate_network="mlp"):
        super(TopKBalancedNoisyGate, self).__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.num_selects = num_selects

        self.mapping_dict = mapping_dict  # 🔍
        self.anomaly_handle = anomaly_handle  # 🔍

        self.gate_network_type = gate_network
        self.gate_network = self.get_gate_network(gate_network, input_size, num_experts)

    def forward(self, x):
        logits = self.gate_network(x)

        ########################################### 🔍
        # here the network is only used to calculate the selected expert pair for mapping, non-differentiable
        _, top_k_indices_from_gate = logits.topk(self.num_selects, dim=1)
        top_k_indices_from_gate, _ = torch.sort(top_k_indices_from_gate, dim=1, descending=False)  # sort first to get the correct score order

        top_k_indices = []

        for pair in torch.split(top_k_indices_from_gate, 1, dim=0):
            pair = str(tuple(pair.reshape(-1).tolist()))
            if pair in self.mapping_dict:
                top_k_indices.append(self.mapping_dict[pair])
            else:  # anomaly
                if self.anomaly_handle == "frequency":
                    top_k_indices.append(self.mapping_dict["anomaly"])
                elif self.anomaly_handle == "random":
                    top_k_indices.append(random.randint(0, self.num_experts - 1))
                else:
                    raise NotImplementedError

        top_k_indices = torch.tensor(top_k_indices, device=x.device).reshape(-1, 1)
        top_k_scores = torch.ones_like(top_k_indices)
        # print(top_k_indices)
        ########################################### 🔍

        importance = torch.bincount(top_k_indices.flatten(), minlength=self.num_experts)
        load = importance.clone()
        balance_loss = None
        # balance_loss = torch.tensor(-100.0, device=x.device)

        return {
            "topK_indices": top_k_indices,
            "topK_scores": top_k_scores,
            "balance_loss": balance_loss,
            "load": load,
            "importance": importance,
        }


class LinearGLUMoELayerMerged(LinearGLUMoELayer):
    def __init__(
            self,
            input_size,
            hidden_size,
            output_size,
            hidden_act,
            num_experts,
            num_selects,
            size_experts=None,
            bias=True,
            **kwargs,
    ):
        super(LinearGLUMoELayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_act = hidden_act
        self.num_experts = num_experts
        self.num_selects = num_selects
        self.size_experts = size_experts
        self.bias = bias
        self.gate_type = kwargs.get("gate_type", "mapping")  # 🔍

        # expert networks
        experts = LinearGLUExperts(
            input_size,
            hidden_size,
            output_size,
            hidden_act,
            num_experts,
            size_experts=size_experts,
            bias=bias,
        )

        # 🔍 create gate
        if self.gate_type == "mapping":
            self.gate = IdenticalMappingGate(
                self.input_size,
                kwargs.get("original_num_experts", self.num_experts),  # 🔍
                self.num_selects,
                mapping_dict=kwargs.get("mapping_dict", None),
                anomaly_handle=kwargs.get("anomaly_handle", "random"),
                gate_network=kwargs.get("gate_network", "mlp"),
            )
        elif self.gate_type == "network":
            self.gate = TopKBalancedNoisyGate(
                self.input_size,
                self.num_experts,
                self.num_selects,
                gate_network=kwargs.get("gate_network", "mlp"),
                use_softmax=kwargs.get("gate_use_softmax", True),
                use_balance=kwargs.get("gate_use_balance", True),
                balance_loss_weight=kwargs.get("gate_balance_loss_weight", 1e-2),
                add_noise=kwargs.get("gate_add_noise", True),
                noise_epsilon=kwargs.get("gate_noise_epsilon", 1e-2),
            )
        else:
            raise NotImplementedError

        # create calculator
        self.calculator = UniversalCalculator(
            experts,
            multiply_gate_scores=kwargs.get("multiply_gate_scores", True),
            score_scale_factor=kwargs.get("score_scale_factor", 1.0),
        )


class LlamaMoEDecoderLayerMerged(LlamaMoEDecoderLayer):
    def __init__(self, config: LlamaMoEMergedConfig, layer_index):
        super(LlamaMoEDecoderLayer, self).__init__()

        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        gating_config = {
            "original_num_experts": config.original_num_experts,  # 🔍
            "gate_type": config.gate_type,  # 🔍
            "mapping_dict": config.mapping_dicts[layer_index],  # 🔍
            "anomaly_handle": config.anomaly_handle,  # 🔍
            "gate_network": config.gate_network,
            "gate_use_softmax": config.gate_use_softmax,
            "gate_use_balance": config.gate_use_balance,
            "gate_balance_loss_weight": config.gate_balance_loss_weight,
            "gate_add_noise": config.gate_add_noise,
            "gate_noise_epsilon": config.gate_noise_epsilon,
        }
        calculator_config = {
            "multiply_gate_scores": config.multiply_gate_scores,
            "score_scale_factor": (
                config.score_scale_factor[layer_index]
                if isinstance(config.score_scale_factor, list)
                else config.score_scale_factor
            ),
        }

        if layer_index in config.merge_layers:
            self.mlp = LinearGLUMoELayerMerged(  # 🔍
                input_size=self.hidden_size,
                hidden_size=config.intermediate_size[layer_index] if isinstance(config.intermediate_size, list) else config.intermediate_size,  # 🔍
                output_size=self.hidden_size,
                hidden_act=config.hidden_act,
                num_experts=config.num_experts[layer_index] if isinstance(config.num_experts, list) else config.num_experts,  # 🔍
                num_selects=config.num_selects,
                size_experts=(
                    config.size_experts[layer_index]
                    if config.size_experts is not None
                    else None
                ),
                bias=False,
                **gating_config,
                **calculator_config,
            )
        else:
            self.mlp = LinearGLUMoELayer(
                input_size=self.hidden_size,
                hidden_size=config.intermediate_size[layer_index] if isinstance(config.intermediate_size, list) else config.intermediate_size,  # 🔍
                output_size=self.hidden_size,
                hidden_act=config.hidden_act,
                num_experts=config.num_experts[layer_index] if isinstance(config.num_experts, list) else config.num_experts,  # 🔍
                num_selects=config.num_selects,
                size_experts=(
                    config.size_experts[layer_index]
                    if config.size_experts is not None
                    else None
                ),
                bias=False,
                **gating_config,
                **calculator_config,
            )


class LlamaMoEPreTrainedModelMerged(PreTrainedModel):
    config_class = LlamaMoEMergedConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaMoEDecoderLayerMerged"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlamaMoEModel):
            module.gradient_checkpointing = value


class LlamaMoEModelMerged(LlamaMoEPreTrainedModelMerged, LlamaMoEModel):
    def __init__(self, config: LlamaMoEMergedConfig):
        LlamaMoEPreTrainedModelMerged.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [LlamaMoEDecoderLayerMerged(config, i) for i in range(config.num_hidden_layers)]  # 🔍
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()


class LlamaMoEForCausalLMMerged(LlamaMoEPreTrainedModelMerged, LlamaMoEForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: LlamaMoEMergedConfig):
        LlamaMoEPreTrainedModelMerged.__init__(self, config)
        self.model = LlamaMoEModelMerged(config)  # 🔍
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
