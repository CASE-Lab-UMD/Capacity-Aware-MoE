from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AdapterArguments:
    """
    The subset of arguments related to adapter training.
    """
    train_adapter: bool = field(default=False, metadata={"help": "Train an adapter instead of the full model."})
    load_adapter: Optional[str] = field(
        default="", metadata={"help": "Pre-trained adapter module to be loaded from Hub."}
    )
    adapter_config: Optional[str] = field(
        default="pfeiffer", metadata={"help": "Adapter configuration. Either an identifier or a path to a file."}
    )
    adapter_non_linearity: Optional[str] = field(
        default=None, metadata={"help": "Override the non-linearity of the adapter configuration."}
    )
    adapter_reduction_factor: Optional[float] = field(
        default=None, metadata={"help": "Override the reduction factor of the adapter configuration."}
    )
    language: Optional[str] = field(default=None, metadata={"help": "The training language, e.g. 'en' for English."})
    adapter_dir: Optional[str] = field(
        default='/user/sunsiqi/hs/MoE/adapter-transformers-master/adapters',
        metadata={"help": "The local path to store adapters."}
    )
    gpu_id: Optional[int] = field(
        default=-1,
    )
    model_name: Optional[str] = field(
        default="mlpnet",
    )
    n_epochs: Optional[int] = field(
        default=10,
    )
    save_result_file: Optional[str] = field(
        default="sample.csv",
    )
    sweep_name: Optional[str] = field(
        default='exp_sample',
    )
    exact: Optional[bool] = field(
        default=True,
    )
    correction: Optional[bool] = field(
        default=True,
    )
    ground_metric: Optional[str] = field(
        default='euclidean',
    )
    weight_stats: Optional[bool] = field(
        default=True,
    )
    activation_histograms: Optional[bool] = field(
        default=True,
    )
    activation_mode: Optional[str] = field(
        default='raw',
    )
    geom_ensemble_type: Optional[str] = field(
        default='wts',
    )
    sweep_id: Optional[int] = field(
        default=21,
    )
    act_num_samples: Optional[int] = field(
        default=1,
    )
    ground_metric_normalize: Optional[str] = field(
        default='none',
    )
    activation_seed: Optional[int] = field(
        default=21,
    )
    prelu_acts: Optional[bool] = field(
        default=True,
    )
    recheck_acc: Optional[bool] = field(
        default=False,
    )
    ckpt_type: Optional[str] = field(
        default='final',
    )
    past_correction: Optional[bool] = field(
        default=True,
    )
    eval_aligned: Optional[bool] = field(
        default=False,
    )
    clip_gm: Optional[bool] = field(
        default=False,
    )
    clip_max: Optional[int] = field(
        default=5,
    )
    clip_min: Optional[int] = field(
        default=0,
    )
    reg: Optional[float] = field(
        default=1e-2,
    )
    importance: Optional[str] = field(
        default="l1",
    )
    unbalanced: Optional[bool] = field(
        default=True,
    )
    proper_marginals: Optional[bool] = field(
        default=True,
    )
    ensemble_step: Optional[float] = field(
        default=0.5,
    )
    skip_last_layer: Optional[bool] = field(
        default=False,
    )
    handle_skips: Optional[bool] = field(
        default=True,
    )
    update_acts: Optional[bool] = field(
        default=True,
    )
    gromov: Optional[bool] = field(
        default=False,
    )



@dataclass
class MultiLingAdapterArguments(AdapterArguments):
    """
    Arguemnts related to adapter training, extended by arguments for multilingual setups.
    """

    load_lang_adapter: Optional[str] = field(
        default=None, metadata={"help": "Pre-trained language adapter module to be loaded from Hub."}
    )
    lang_adapter_config: Optional[str] = field(
        default=None, metadata={"help": "Language adapter configuration. Either an identifier or a path to a file."}
    )
    lang_adapter_non_linearity: Optional[str] = field(
        default=None, metadata={"help": "Override the non-linearity of the language adapter configuration."}
    )
    lang_adapter_reduction_factor: Optional[int] = field(
        default=None, metadata={"help": "Override the reduction factor of the language adapter configuration."}
    )
    adapter_dir: Optional[str] = field(
        default='/user/sunsiqi/hs/MoE/adapter-transformers-master/adapters',
        metadata={"help": "The local path to store adapters."}
    )




