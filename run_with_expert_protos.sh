#!/usr/bin/bash

EXP_NAME=ta_expert_protos2/with

for run in $(seq 1 5); do
  if ! test -e experiments/$EXP_NAME/run-$run/$ds/results.json; then
    ./main.py --experiment_name $EXP_NAME/run-$run --proto_len 0.3 --no_encoder_pretraining --protos_per_class 2 --skip_scaling --epochs 140 --verbose --target_protos_dir ./target_protos2/ --l2_target_protos_coeff 1e-4 --penalize_class_0
  fi
done
