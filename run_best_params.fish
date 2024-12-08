#!/usr/bin/fish

set datasets (cat ../DataDimensionsPipe.csv | tail -n+2 | cut -d\| -f1)

for run in (seq 1 5)
  for ds in $datasets
    if ! test -e experiments/HyperparamBest/run-$run/$ds
      python ./main.py --dataset $ds --experiment_name HyperparamBest/run-$run
    end
  end
end
