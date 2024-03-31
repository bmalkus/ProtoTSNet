#!/usr/bin/bash

datasets="ArticularyWordRecognition AtrialFibrillation BasicMotions CharacterTrajectories Cricket DuckDuckGeese EigenWorms Epilepsy EthanolConcentration ERing FaceDetection FingerMovements HandMovementDirection Handwriting Heartbeat InsectWingbeat JapaneseVowels Libras LSST MotorImagery NATOPS PenDigits PEMS-SF Phoneme RacketSports SelfRegulationSCP1 SelfRegulationSCP2 SpokenArabicDigits StandWalkJump UWaveGestureLibrary"

EXP_NAME=HyperparamSelect

for ds in $datasets; do
  if ! test -e experiments/$EXP_NAME/$ds; then
    # cross-validation handled inside main.py
    python ./main.py --uea_dataset $ds --experiment_name $EXP_NAME --param_selection
  fi
done
