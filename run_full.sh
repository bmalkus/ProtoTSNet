#!/usr/bin/bash

datasets="ArticularyWordRecognition AtrialFibrillation BasicMotions CharacterTrajectories Cricket DuckDuckGeese EigenWorms Epilepsy EthanolConcentration ERing FaceDetection FingerMovements HandMovementDirection Handwriting Heartbeat InsectWingbeat JapaneseVowels Libras LSST MotorImagery NATOPS PenDigits PEMS-SF PhonemeSpectra RacketSports SelfRegulationSCP1 SelfRegulationSCP2 SpokenArabicDigits StandWalkJump UWaveGestureLibrary"

EXP_NAME=GroupingEncPretraining

for ds in $datasets; do
  for run in $(seq 1 5); do
    if ! test -e experiments/$EXP_NAME/run-$run/$ds/results.json; then
      python ./main.py --uea_dataset $ds --experiment_name $EXP_NAME/run-$run
    fi
  done
done
