# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""This is the main entry point for the code."""

import hydra

from mtrl.app.run import run
from mtrl.utils import config as config_utils
from mtrl.utils.types import ConfigType

import wandb

def init_wandb(config):
    wandb.init(
        entity='robertmccarthy11',
        project="llm-curriculum",
        group=config.experiment.save.wandb.group,
        name=config.experiment.save.wandb.name,
        job_type='mtrl',
        config=config,
        )

@hydra.main(config_path="config", config_name="config")
def launch(config: ConfigType) -> None:
    config = config_utils.process_config(config)
    if config.experiment.save.wandb.use_wandb:
        init_wandb(config)
    return run(config)


if __name__ == "__main__":
    launch()
