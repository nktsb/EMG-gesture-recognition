"""
 -----------------------------------------------------------------------------
 Project       : EMG-gesture-recognition
 File          : time_test.py
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

from threading import Thread
from threading import Event

import numpy as np
import pandas as pd

from emg_python.emg_device import EMG_device
from emg_python.emg_plot import Plotter
from emg_python.emg_files import EMG_file
from emg_python.emg_files import get_date_time
from emg_python.emg_preprocessing import EMG_preprocessor
from emg_python.emg_features import EMG_feature_extractor
from emg_python.emg_ml import EMG_classifier

from run_emg import BAUDRATE, PING_ANSWER, EMG_SNS_INFO, MODEL_FPATH, DATASET_FOLDER, \
        MODE_FILE_NAME, SAMPLE_RATE, DEVICE_INFO, EMG_dev, emg_dev_stop, \
        EMG_online_plotting_task, EMG_predict, EMG_recognition_init, EMG_dev_is_sample_ready \
         
FILES_NUM_LIMIT = 25

def emg_dev_update_data(emg_dev: EMG_device, emg_data: dict, accel_data: dict, gyro_data: dict):
    for ch in emg_dev.emg_data:
        emg_dev.emg_data[ch][:] = emg_data[ch][:]
    for ch in emg_dev.accel_data:
        emg_dev.accel_data[ch][:] = accel_data[ch][:]
    for ch in emg_dev.gyro_data:
        emg_dev.gyro_data[ch][:] = gyro_data[ch][:]

def emulate_emg_dev(emg_classificator: EMG_classifier,
                    emg_preprocessor: EMG_preprocessor, \
                    accel_preprocessor: EMG_preprocessor, \
                    gyro_preprocessor: EMG_preprocessor, \
                    emg_feature_extractor: EMG_feature_extractor, \
                    accel_feature_extractor: EMG_feature_extractor, \
                    gyro_feature_extractor: EMG_feature_extractor, \
                    emg_dev: EMG_device, ds_folder: str, user: list, \
                    stop_event: Event):

    fdir_base = f"{ds_folder}/{user}"

    gestures = [gesture for gesture in os.listdir(fdir_base) if not gesture.startswith(".")]
    if not gestures:
        print("There is no dataset files")
        return False
    gestures.sort()

    gestures_times = dict()

    for gesture in gestures:
        gestures_times[gesture] = dict()
        gestures_times[gesture]["preprocessing"] = list()
        gestures_times[gesture]["feature_extraction"] = list()
        gestures_times[gesture]["classification"] = list()
    
        fdir = f"{fdir_base}/{gesture}"
        if os.path.isdir(fdir):
            files_in_dir = os.listdir(fdir)
            if not files_in_dir:
                continue
            gesture_files_in_dir = [
                gest_file for gest_file in files_in_dir if gest_file.startswith(gesture)
            ]
            gesture_files_in_dir.sort()
            for i, gesture_file in enumerate(gesture_files_in_dir):
                if i == FILES_NUM_LIMIT:
                    break
                fpath = f"{fdir}/{gesture_file}"
                current_file = EMG_file(fpath)

                emg_dev_update_data(emg_dev, current_file.json["sns_data"]["emg_data"], \
                                    current_file.json["sns_data"]["accel_data"], \
                                    current_file.json["sns_data"]["gyro_data"])
                del current_file

                res, preproc_time, extraction_time, classif_time = EMG_predict(emg_classificator, \
                            emg_preprocessor, \
                            accel_preprocessor, \
                            gyro_preprocessor, \
                            emg_feature_extractor, \
                            accel_feature_extractor, \
                            gyro_feature_extractor)

                gestures_times[gesture]["preprocessing"].append(preproc_time)
                gestures_times[gesture]["feature_extraction"].append(extraction_time)
                gestures_times[gesture]["classification"].append(classif_time)

            gestures_times[gesture]["preprocessing"] = np.mean(gestures_times[gesture]["preprocessing"])
            gestures_times[gesture]["feature_extraction"] = np.mean(gestures_times[gesture]["feature_extraction"])
            gestures_times[gesture]["classification"] = np.mean(gestures_times[gesture]["classification"])
            print(gestures_times[gesture])
            if stop_event.is_set():
                return

    report_frame = pd.DataFrame(gestures_times).transpose().round(2)
    report_frame.to_csv("time_test.csv", mode = 'a', index=True)
    stop_event.set()


if __name__ == '__main__':

    if MODE_FILE_NAME:
        emg_classificator, emg_preprocessor, emg_feature_extractor, accel_preprocessor, accel_feature_extractor, \
            gyro_preprocessor, gyro_feature_extractor = EMG_recognition_init(EMG_dev, MODEL_FPATH)

    try:
        emg_dev_thread = Thread(target=emulate_emg_dev, args=(emg_classificator, \
                                                                emg_preprocessor, \
                                                                accel_preprocessor, \
                                                                gyro_preprocessor, \
                                                                emg_feature_extractor, \
                                                                accel_feature_extractor, \
                                                                gyro_feature_extractor, \
                                                                EMG_dev, DATASET_FOLDER, \
                                                                'nktsb3', emg_dev_stop))
        emg_dev_thread.start()
        while not EMG_dev_is_sample_ready(EMG_dev):
            time.sleep(0.02)

        ### Plotting thread
        plot_legend  = EMG_dev.emg_channels + EMG_dev.accel_axes + EMG_dev.gyro_axes

        plt_emg_data = list(EMG_dev.emg_data.values())
        plt_accel_data = list(EMG_dev.accel_data.values())
        plt_gyro_data = list(EMG_dev.gyro_data.values())

        emg_min_max = DEVICE_INFO['emg_min_max']
        accel_min_max = DEVICE_INFO['accel_min_max']
        gyro_min_max = DEVICE_INFO['gyro_min_max']

        plot_data  = plt_emg_data + plt_accel_data + plt_gyro_data
        subplots = [len(plt_emg_data), len(plt_accel_data), len(plt_gyro_data)]
        max_mins = [emg_min_max, accel_min_max, gyro_min_max]

        plotter = Plotter(plot_data, plot_legend, max_mins, \
                            show_legend=True, title="Raw data", sublots=subplots)

        @staticmethod
        def EMG_online_plotting_quit_clbk(event):
            plotter.stop_animate()
            plotter.close()

        buttons = {
            'Quit': EMG_online_plotting_quit_clbk
        }

        EMG_online_plotting_task(plotter, buttons=buttons)

    finally:
        emg_dev_stop.set()
        quit()
