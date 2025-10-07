#!/bin/bash

# Train
ID="Pretrain"
MODE='test'
CHECKPOINT="results/models/Pretrain"
TEACHER_ENABLE=false
TEACHER_CORRECT=false

APPLY_UNKNOWN_DISTRIBUTION=false
WITH_FRICTION=true
FRICTION_CART=3
FRICTION_POLE=0

RANDOM_SEED=0
TRAINING_BY_STEPS=true
MAX_TRAINING_STEPS=1e5
EVAL_RANDOM_RESET=false
GAMMA=0.8

python main.py \
  general.id=${ID} \
  general.mode=${MODE} \
  general.checkpoint=${CHECKPOINT} \
  general.training_by_steps=${TRAINING_BY_STEPS} \
  general.max_training_steps=${MAX_TRAINING_STEPS} \
  cartpole.with_friction=${WITH_FRICTION} \
  cartpole.friction_cart=${FRICTION_CART} \
  cartpole.friction_pole=${FRICTION_POLE} \
  cartpole.random_reset.seed=${RANDOM_SEED} \
  cartpole.random_reset.eval=${EVAL_RANDOM_RESET} \
  phy_teacher.teacher_enable=${TEACHER_ENABLE} \
  phy_teacher.teacher_correct=${TEACHER_CORRECT} \
  drl_student.phydrl.gamma=${GAMMA} \
  drl_student.agents.unknown_distribution.apply=${APPLY_UNKNOWN_DISTRIBUTION}

