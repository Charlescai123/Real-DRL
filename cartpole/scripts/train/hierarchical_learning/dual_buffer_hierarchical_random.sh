#!/bin/bash

# Train
ID="Dual-Buffer-Hierarchical-Random"
MODE='train'
CHECKPOINT="results/models/Real-DRL"
TEACHER_ENABLE=true
TEACHER_CORRECT=true
WITH_FRICTION=true
FRICTION_CART=10
ACTUATOR_NOISE=true
TRAIN_RANDOM_RESET=true
EVAL_RANDOM_RESET=true

TEST_DUAL_BUFFER=false
USE_DUAL_REPLAY_BUFFER=true

SEED=1
APPLY_UNKNOWN_DISTRIBUTION=false

#TRAIN_RANDOM_RESET=false
#EVAL_RANDOM_RESET=false

#TRAINING_BY_STEPS=true
#MAX_TRAINING_EPISODES=1e3
TRAINING_BY_STEPS=false
MAX_TRAINING_EPISODES=100
GAMMA=0


python main.py \
  general.id=${ID} \
  general.mode=${MODE} \
  general.seed=${SEED} \
  general.checkpoint=${CHECKPOINT} \
  general.test_dual_buffer=${TEST_DUAL_BUFFER} \
  general.training_by_steps=${TRAINING_BY_STEPS} \
  general.max_training_episodes=${MAX_TRAINING_EPISODES} \
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
