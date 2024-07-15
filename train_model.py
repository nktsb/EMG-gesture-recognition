"""
 -----------------------------------------------------------------------------
 Project       : EMG-gesture-recognition
 File          : train_model.py
 Author        : nktsb
 Created on    : 08.05.2024
 GitHub        : https://github.com/nktsb/EMG-gesture-recognition
 -----------------------------------------------------------------------------
 Copyright (c) 2024 nktsb
 All rights reserved.
 -----------------------------------------------------------------------------
"""

import os
import sys
import time
import json

import numpy as np

from emg_python.emg_ml import EMG_ds_handler, EMG_model_trainer
from emg_python.emg_files import EMG_file
from emg_python.emg_preprocessing import EMG_preprocessor
from emg_python.emg_features import EMG_feature_extractor

from run_emg import DEVICE_INFO, DATASET_FOLDER, FEATURES_FOLDER, MODELS_FOLDER, TRAIN_INFO

SAMPLE_RATE = DEVICE_INFO['sample_rate']

NORMALIZE_METHOD = TRAIN_INFO['normalize_method']

EMG_PREPROC_INFO = TRAIN_INFO['emg_preprocessor_info']
ACCEL_PREPROC_INFO = TRAIN_INFO['accel_preprocessor_info']
GYRO_PREPROC_INFO = TRAIN_INFO['gyro_preprocessor_info']

EMG_FEATURES_INFO = TRAIN_INFO['emg_features_info']
ACCEL_FEATURES_INFO = TRAIN_INFO['accel_features_info']
GYRO_FEATURES_INFO = TRAIN_INFO['gyro_features_info']

ML_INFO = TRAIN_INFO['ml_info']

emg_preprocessor = EMG_preprocessor(normalize_method=NORMALIZE_METHOD, \
                                    preprocessor_info=EMG_PREPROC_INFO)
accel_preprocessor = EMG_preprocessor(normalize_method=NORMALIZE_METHOD, \
                                      preprocessor_info=ACCEL_PREPROC_INFO)
gyro_preprocessor = EMG_preprocessor(normalize_method=NORMALIZE_METHOD, \
                                     preprocessor_info=GYRO_PREPROC_INFO)

emg_feature_extractor = EMG_feature_extractor(EMG_FEATURES_INFO)
accel_feature_extractor = EMG_feature_extractor(ACCEL_FEATURES_INFO)
gyro_feature_extractor = EMG_feature_extractor(GYRO_FEATURES_INFO)

if __name__ == '__main__':
    trainer = EMG_model_trainer(DATASET_FOLDER, FEATURES_FOLDER, emg_preprocessor, \
                             accel_preprocessor, gyro_preprocessor, emg_feature_extractor, \
                             accel_feature_extractor, gyro_feature_extractor, ML_INFO, \
                             MODELS_FOLDER)

    trainer.train_models()
