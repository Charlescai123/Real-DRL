#!/bin/bash

# Test
ID="Teaching-wo-learn-4"
MODE='test'
CHECKPOINT="results/models/Teaching-wo-learn-4_30"
TEACHER_ENABLE=false
TEACHER_CORRECT=false
WITH_FRICTION=true
FRICTION_CART=20
ACTUATOR_NOISE=true
TRAIN_RANDOM_RESET=false
EVAL_RANDOM_RESET=false

TEST_DUAL_BUFFER=false
USE_DUAL_REPLAY_BUFFER=false

SEED=1
APPLY_UNKNOWN_DISTRIBUTION=false

#TRAIN_RANDOM_RESET=false
#EVAL_RANDOM_RESET=false

#TRAINING_BY_STEPS=true
#MAX_TRAINING_EPISODES=1e3
TRAINING_BY_STEPS=false
MAX_TRAINING_EPISODES=10
GAMMA=0

INIT_CONDITION="[-0.3899750175043573, 0.050707845883927, 0.3513748143538569, 0.09213432339178134, false]"

python main.py \
  general.id=${ID} \
  general.mode=${MODE} \
  general.seed=${SEED} \
  general.checkpoint=${CHECKPOINT} \
  general.test_dual_buffer=${TEST_DUAL_BUFFER} \
  general.training_by_steps=${TRAINING_BY_STEPS} \
  general.max_training_episodes=${MAX_TRAINING_EPISODES} \
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
