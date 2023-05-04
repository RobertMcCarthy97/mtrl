# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import mtenv
from gym.vector.async_vector_env import AsyncVectorEnv

from mtrl.env.vec_env import MetaWorldVecEnv, VecEnv, LLMVecEnv
from mtrl.utils.types import ConfigType


def build_dmcontrol_vec_env(
    domain_name: str,
    task_name: str,
    prefix: str,
    make_kwargs: ConfigType,
    env_id_list: List[int],
    seed_list: List[int],
    mode_list: List[str],
) -> VecEnv:
    def get_func_to_make_envs(seed: int, initial_task_state: int):
        def _func() -> mtenv.MTEnv:
            kwargs = deepcopy(make_kwargs)
            kwargs["seed"] += seed
            kwargs["initial_task_state"] = initial_task_state
            return mtenv.make(
                f"MT-HiPBMDP-{domain_name.capitalize()}-{task_name.capitalize()}-vary-{prefix.replace('_', '-')}-v0",
                **kwargs,
            )

        return _func

    funcs_to_make_envs = [
        get_func_to_make_envs(seed=seed, initial_task_state=task_state)
        for (seed, task_state) in zip(seed_list, env_id_list)
    ]

    env_metadata = {"ids": env_id_list, "mode": mode_list}

    env = VecEnv(env_metadata=env_metadata, env_fns=funcs_to_make_envs, context="spawn")

    return env


def build_metaworld_vec_env(
    config: ConfigType,
    benchmark: "metaworld.Benchmark",  # type: ignore[name-defined] # noqa: F821
    mode: str,
    env_id_to_task_map: Optional[Dict[str, "metaworld.Task"]],  # type: ignore[name-defined] # noqa: F821
) -> Tuple[AsyncVectorEnv, Optional[Dict[str, Any]]]:
    from mtenv.envs.metaworld.env import (
        get_list_of_func_to_make_envs as get_list_of_func_to_make_metaworld_envs,
    )
    benchmark_name = config.env.benchmark._target_.replace("metaworld.", "")
    num_tasks = int(benchmark_name.replace("MT", ""))
    make_kwargs = {
        "benchmark": benchmark,
        "benchmark_name": benchmark_name,
        "env_id_to_task_map": env_id_to_task_map,
        "num_copies_per_env": 1,
        "should_perform_reward_normalization": True,
    }

    funcs_to_make_envs, env_id_to_task_map = get_list_of_func_to_make_metaworld_envs(
        **make_kwargs
    )
    env_metadata = {
        "ids": list(range(num_tasks)),
        "mode": [mode for _ in range(num_tasks)],
    }
    env = MetaWorldVecEnv(
        env_metadata=env_metadata,
        env_fns=funcs_to_make_envs,
        context="spawn",
        shared_memory=False,
    )
    
    return env, env_id_to_task_map

'''
(Pdb) env_metadata
{'ids': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'mode': ['train', 'train', 'train', 'train', 'train', 'train', 'train', 'train', 'train', 'train']}

(Pdb) env_id_to_task_map
{'reach-v1': Task(env_name='reach-v1', data=......


'''


def build_llm_vec_env(
    config: ConfigType,
    mode: str,
) -> Tuple[AsyncVectorEnv, Optional[Dict[str, Any]]]:
    
    from mtenv.envs.metaworld.wrappers.normalized_env import NormalizedEnvWrapper
    from llm_curriculum_algo.env_wrappers import make_env
    
    def get_func_to_make_envs(single_task_name: str, mtenv_task_idx: int):
        
        def _make_env():
            env = make_env(
                use_language_goals=config.env.use_language_goals,
                single_task_names=[single_task_name],
                high_level_task_names=config.env.high_level_task_names,
                mtenv_wrapper=True,
                mtenv_task_idx=mtenv_task_idx,
                )
            env = NormalizedEnvWrapper(env, normalize_reward=True)
            # TODO: make sure norm env is correct
            return env

        return _make_env
    
    num_tasks = len(config.env.single_task_names)
    funcs_to_make_envs = [get_func_to_make_envs(single_task_name, i) for i, single_task_name in enumerate(config.env.single_task_names)]
    
    env_metadata = {
        "ids": list(range(num_tasks)),
        "mode": [mode for _ in range(num_tasks)],
    }
    env = LLMVecEnv(
        env_metadata=env_metadata,
        env_fns=funcs_to_make_envs,
        context="spawn",
        shared_memory=False,
    )
    
    assert env.num_envs == num_tasks
    
    return env