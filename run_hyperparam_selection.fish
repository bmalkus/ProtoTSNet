#!/usr/bin/fish

set datasets (cat ../DataDimensionsPipe.csv | tail -n+2 | cut -d\| -f1)

for ds in $datasets
  if ! test -e experiments/HyperparamSelect/$ds
    python ./main.py --dataset $ds --experiment_name HyperparamSelect --param_selection
  end
end
