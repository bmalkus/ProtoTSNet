#!/usr/bin/fish

set datasets (cat ../DataDimensionsPipe.csv | tail -n+2 | cut -d\| -f1)

for ds in $datasets
  for run in (seq 1 5)
    if ! test -e experiments/RegularEncBest/run-$run/$ds
      python ./main.py --dataset $ds --experiment_name RegularEncBest/run-$run --no_permuting_encoder --no_encoder_pretraining
    end
  end
end
