#!/bin/bash

# chmod +x run_custom_exp.sh
# ./run_custom_exp.sh


export PYTHONPATH=.


# #######################################
# # Multi-task SAC - testing

# python3 -u main_custom.py \
# setup=fetch_custom \
# env=fetch_custom \
# agent=state_sac \
# experiment.num_eval_episodes=1 \
# experiment.num_train_steps=2000 \
# setup.seed=1 \
# replay_buffer.batch_size=128 \
# agent.multitask.should_use_disentangled_alpha=True \
# agent.encoder.type_to_select=identity \
# agent.multitask.should_use_multi_head_policy=False \
# agent.multitask.actor_cfg.should_condition_model_on_task_info=False \
# agent.multitask.actor_cfg.should_condition_encoder_on_task_info=True \
# agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=True \
# env.num_envs=2 \
# env.single_task_names=['pick_up_cube', 'lift_cube'] \
# env.high_level_task_names=['move_cube_to_target'] \
# env.use_language_goals=False \

# ########################################


#######################################
# Multi-task SAC - full

python3 -u main_custom.py \
setup=fetch_custom \
env=fetch_custom \
agent=state_sac \
experiment.num_eval_episodes=1 \
experiment.num_train_steps=200000 \
setup.seed=1 \
replay_buffer.batch_size=$((128)) \
agent.multitask.should_use_disentangled_alpha=True \
agent.encoder.type_to_select=identity \
agent.multitask.should_use_multi_head_policy=False \
agent.multitask.should_use_task_encoder=False \
agent.multitask.actor_cfg.should_condition_model_on_task_info=False \
agent.multitask.actor_cfg.should_condition_encoder_on_task_info=False \
agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=False \
agent.multitask.task_encoder_cfg.model_cfg.pretrained_embedding_cfg.should_use=False \
agent.multitask.num_envs=1 \
env.num_envs=1 \
env.single_task_names="['lift_cube']" \
env.high_level_task_names="['move_cube_to_target']" \
env.use_language_goals=False \
experiment.should_resume=False \
experiment.save.wandb.use_wandb=True \
experiment.save.wandb.group=temp \
experiment.save.wandb.name=temp \

########################################