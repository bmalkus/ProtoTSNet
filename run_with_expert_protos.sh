#!/usr/bin/bash

# EXP_NAME=new_ta_expert_protos/with_expert
ds=Libras
EXP_NAME=libras_expert_protos/with
MAX_RUNS=5

for run in $(seq 1 $MAX_RUNS); do
  if [[ ! -e experiments/$EXP_NAME/run-$run/$ds/results.json ]] || [[ "$1" == "--overwrite" ]]; then
    echo " ### Run $run/$MAX_RUNS ###"
    # ./main.py --experiment_name $EXP_NAME/run-$run --proto_len 0.5 --no_encoder_pretraining --protos_per_class 4 --skip_scaling --epochs 200 --l1_coeff 1e-2 --target_protos_dir ./target_protos_new/
    ./main.py --uea_dataset $ds --experiment_name $EXP_NAME/run-$run --proto_len 0.5 --protos_per_class 2 --epochs 200 --no_permuting_encoder --target_protos_dir ./target_protos_libras/
  fi
done
