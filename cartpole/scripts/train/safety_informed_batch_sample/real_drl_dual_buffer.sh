#!/bin/bash

# Train
ID="Real-DRL-Dual-Buffer"
MODE='train'
CHECKPOINT="results/models/Pretrain-DRL"
TEACHER_ENABLE=true
TEACHER_CORRECT=true
WITH_FRICTION=true
FRICTION_CART=10
ACTUATOR_NOISE=true
TRAIN_RANDOM_RESET=false
EVAL_RANDOM_RESET=false

TEST_DUAL_BUFFER=true
USE_DUAL_REPLAY_BUFFER=true

SEED=1
APPLY_UNKNOWN_DISTRIBUTION=true

#TRAIN_RANDOM_RESET=false
#EVAL_RANDOM_RESET=false

#TRAINING_BY_STEPS=true
#MAX_TRAINING_EPISODES=1e3
TRAINING_BY_STEPS=false
MAX_TRAINING_EPISODES=150
GAMMA=0

LY_REWARD=12
INIT_CONDITION="[0.3618298350176420, 0.055707845883927, -0.3568100254495141, 0.09013432339178134, false]"


python main.py \
  general.id=${ID} \
  general.mode=${MODE} \
  general.seed=${SEED} \
  general.checkpoint=${CHECKPOINT} \
  general.test_dual_buffer=${TEST_DUAL_BUFFER} \
  general.training_by_steps=${TRAINING_BY_STEPS} \
  general.max_training_episodes=${MAX_TRAINING_EPISODES} \
  cartpole.rewards.lyapunov=${LY_REWARD} \
  cartpole.initial_condition="${INIT_CONDITION}" \
  cartpole.with_friction=${WITH_FRICTION} \
  cartpole.friction_cart=${FRICTION_CART} \
  cartpole.random_reset.train=${TRAIN_RANDOM_RESET} \
  cartpole.random_reset.eval=${EVAL_RANDOM_RESET} \
  cartpole.domain_random.actuator.apply=${ACTUATOR_NOISE} \
  phy_teacher.teacher_enable=${TEACHER_ENABLE} \
  phy_teacher.teacher_correct=${TEACHER_CORRECT} \
  drl_student.phydrl.gamma=${GAMMA} \
  drl_student.phydrl.use_dual_replay_buffer=${USE_DUAL_REPLAY_BUFFER} \
  drl_student.agents.unknown_distribution.apply=${APPLY_UNKNOWN_DISTRIBUTION}
