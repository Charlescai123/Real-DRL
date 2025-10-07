#!/bin/bash

# Train
ID="Pretrain-DRL"
MODE='train'
CHECKPOINT=null
TEACHER_ENABLE=false
TEACHER_CORRECT=false
DOMAIN_RANDOM_FRICTION_CART=false

APPLY_UNKNOWN_DISTRIBUTION=false
WITH_FRICTION=false
FRICTION_CART=3
FRICTION_POLE=0

SEED=0
TRAINING_BY_STEPS=false
MAX_TRAINING_STEPS=1e4
MAX_TRAINING_EPISODES=5000
TRAIN_RANDOM_RESET=true
EVAL_RANDOM_RESET=true
GAMMA=0

python main.py \
  general.id=${ID} \
  general.mode=${MODE} \
  general.seed=${SEED} \
  general.checkpoint=${CHECKPOINT} \
  general.training_by_steps=${TRAINING_BY_STEPS} \
  general.max_training_steps=${MAX_TRAINING_STEPS} \
  general.max_training_episodes=${MAX_TRAINING_EPISODES} \
  cartpole.with_friction=${WITH_FRICTION} \
  cartpole.friction_cart=${FRICTION_CART} \
  cartpole.friction_pole=${FRICTION_POLE} \
  cartpole.random_reset.train=${TRAIN_RANDOM_RESET} \
  cartpole.random_reset.eval=${EVAL_RANDOM_RESET} \
  cartpole.domain_random.friction_cart.apply=${DOMAIN_RANDOM_FRICTION_CART} \
  phy_teacher.teacher_enable=${TEACHER_ENABLE} \
  phy_teacher.teacher_correct=${TEACHER_CORRECT} \
  drl_student.phydrl.gamma=${GAMMA} \
  drl_student.agents.unknown_distribution.apply=${APPLY_UNKNOWN_DISTRIBUTION}

