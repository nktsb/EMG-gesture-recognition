"""
 -----------------------------------------------------------------------------
 Project       : EMG-gesture-recognition
 File          : show_dataset.py
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

from emg_python.emg_dataset import EMG_ds_shower
from emg_python.emg_preprocessing import EMG_preprocessor
from emg_python.emg_features import EMG_feature_extractor

from run_emg import DATASET_FOLDER
from run_emg import TRAIN_INFO

NORMALIZE_METHOD = TRAIN_INFO['normalize_method']
EMG_PREPROC_INFO = TRAIN_INFO['emg_preprocessor_info']
ACCEL_PREPROC_INFO = TRAIN_INFO['accel_preprocessor_info']
GYRO_PREPROC_INFO = TRAIN_INFO['gyro_preprocessor_info']

def EMG_show_dataset_task(fdir, emg_preprocessor, accel_preprocessor, gyro_preprocessor):
    try:
        ds_shower = EMG_ds_shower(emg_preprocessor, accel_preprocessor, gyro_preprocessor)
        ds_shower.show_dataset_files(fdir)
    finally:
        quit()

if __name__ == '__main__':
    try:
        emg_preprocessor = EMG_preprocessor(normalize_method=NORMALIZE_METHOD, \
                                            preprocessor_info=EMG_PREPROC_INFO)
        accel_preprocessor = EMG_preprocessor(normalize_method=NORMALIZE_METHOD, \
                                            preprocessor_info=ACCEL_PREPROC_INFO)
        gyro_preprocessor = EMG_preprocessor(normalize_method=NORMALIZE_METHOD, \
                                            preprocessor_info=GYRO_PREPROC_INFO)
        while True:
            while True:
                user_name = input("\r\nEnter username and press Enter, or enter 'quit' to close\r\n")
                if user_name:
                    if(user_name =='quit'):
                        quit()
                    else:
                        usr_fdir = f"./{DATASET_FOLDER}/{user_name}"
                        if os.path.isdir(usr_fdir):
                            EMG_show_dataset_task(usr_fdir, emg_preprocessor, accel_preprocessor, gyro_preprocessor)
                            break
                        else:
                            print(f"There's no dataset for user {user_name}")
                else:
                    print("Empty username!")
            while True:
                a = input("Show other user dataset? y/n\r\n")
                if a == 'n':
                    quit()
                elif a == 'y':
                    break
                else:
                    print("Wrong input!")
    finally:
        quit()