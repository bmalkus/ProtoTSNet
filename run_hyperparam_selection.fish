#!/usr/bin/fish

set datasets (cat ../DataDimensionsPipe.csv | tail -n+2 | cut -d\| -f1)

set EXP_NAME HyperparamSelect

for ds in $datasets
  if ! test -e experiments/$EXP_NAME/$ds
    python ./main.py --uea_dataset $ds --experiment_name $EXP_NAME --param_selection
  end
end
