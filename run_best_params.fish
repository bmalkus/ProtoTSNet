#!/usr/bin/fish

set datasets (cat ../DataDimensionsPipe.csv | tail -n+2 | cut -d\| -f1)

set EXP_NAME HyperparamBestDiffrentScaling

for ds in $datasets
  for run in (seq 1 5)
    if ! test -e experiments/$EXP_NAME/run-$run/$ds
      python ./main.py --uea_dataset $ds --experiment_name $EXP_NAME/run-$run
    end
  end
end
