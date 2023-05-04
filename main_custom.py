# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""This is the main entry point for the code."""

import hydra

from mtrl.app.run import run
from mtrl.utils import config as config_utils
from mtrl.utils.types import ConfigType

import wandb


@hydra.main(config_path="config", config_name="config_fetch")
def launch(config: ConfigType) -> None:
    config = config_utils.process_config(config)
    if config.wandb:
        wandb.init(project="your_project_name", config=config)
    return run(config)


if __name__ == "__main__":
    launch()