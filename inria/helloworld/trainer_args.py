from dataclasses import asdict, dataclass, field
from typing import Dict, List, Union

import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import LightningLoggerBase


@dataclass
class TrainerArgs:
    """Curated parameters for Pytorch Lightning Trainer
    See https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html
    for more details and other options to add here.
    """

    # required params
    log_every_n_steps: int  # set the logging frequency
    max_epochs: int  # number of epochs

    # optional params
    accelerator: str = "auto"  # autoselect between CPUs, GPUs or TPUs
    auto_select_gpus: bool = (
        True if torch.cuda.is_available() else False
    )  # specifies all available GPUs (if only one GPU is not occupied, uses one gpu)
    gpus: Union[int, str, List[int], None] = 1 if torch.cuda.is_available() else None  # use one gpu

    deterministic: bool = False  # keep it deterministic, set it to False to increase speed
    auto_scale_batch_size: str = "binsearch"  # run batch size scaling, result overrides hparams.batch_size
    benchmark: bool = True
    accumulate_grad_batches: int = 3
    gradient_clip_val: float = 0.5
    gradient_clip_algorithm: str = "value"
    auto_lr_find: bool = True  # run learning rate finder, results override hparams.learning_rate
    stochastic_weight_avg: bool = True

    callbacks: Union[List[Callback], Callback] = field(default_factory=lambda: [])
    logger: Union[List[LightningLoggerBase], LightningLoggerBase] = field(default_factory=lambda: [])

    def to_dict(self) -> Dict:
        return asdict(self)

    def __repr__(self) -> str:
        d = self.to_dict()
        return "\n".join([f"{key} = {d[key]}" for key in d])
