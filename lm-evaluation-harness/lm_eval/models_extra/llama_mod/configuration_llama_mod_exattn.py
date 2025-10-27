"""LLaMA-MoD model configuration"""
"""Modified from configuration_llama.py"""

from transformers.utils import logging

from .configuration_llama_mod import LlamaMoDConfig

logger = logging.get_logger(__name__)


class LlamaMoDExAttnConfig(LlamaMoDConfig):
    model_type = "llama_mod_exattn"
