"""
 -----------------------------------------------------------------------------
 Project       : EMG-gesture-recognition
 File          : run_emg.py
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

from emg_python.emg_device import EMG_device
from emg_python.emg_plot import Plotter
from emg_python.emg_preprocessing import EMG_preprocessor
from emg_python.emg_features import EMG_feature_extractor
from emg_python.emg_ml import EMG_classifier
from emg_python.emg_files import get_date_time

try:
    with open(sys.argv[1], 'r') as fp:
        params = json.load(fp)
        fp.close()
except:
    print("Error: there is no parameters file!")
    quit()

# Project info
PROJECT_INFO = params['project_info']
DATASET_FOLDER = PROJECT_INFO['dataset_folder']
FEATURES_FOLDER = PROJECT_INFO['features_folder']
MODELS_FOLDER = PROJECT_INFO['models_folder']

# Training info
TRAIN_INFO = params['training_info']
GESTURES_SET = params['gestures_list']

# Device info
DEVICE_INFO = params['device_info']

BAUDRATE = DEVICE_INFO['dev_baudrate']
PING_ANSWER = DEVICE_INFO['dev_ping_ans']
SAMPLE_RATE = DEVICE_INFO['sample_rate']
EMG_SNS_INFO = {
    'emg_channels': [f"{DEVICE_INFO['dev_emg_preamble']}{i}" for i in range(DEVICE_INFO['emg_channels_num'])],
    'accel_axes': [f"{DEVICE_INFO['dev_accel_preamble']}{i}" for i in DEVICE_INFO['accel_axes']],
    'gyro_axes': [f"{DEVICE_INFO['dev_gyro_preamble']}{i}" for i in DEVICE_INFO['gyro_axes']],
    'sample_size': DEVICE_INFO['sample_size'],
    'sample_rate': DEVICE_INFO['sample_rate']
}

# Online recognition info
MODE_FILE_NAME = params['online_rec_model_file']
MODEL_FPATH = f"{MODELS_FOLDER}/{MODE_FILE_NAME}"

emg_dev_stop = Event()

EMG_dev = EMG_device(BAUDRATE, PING_ANSWER, EMG_SNS_INFO)


def EMG_recognition_init(emg_dev: EMG_device, model_fpath: str):

    emg_data_ptr = emg_dev.emg_data
    accel_data_ptr = emg_dev.accel_data
    gyro_data_ptr = emg_dev.gyro_data

    emg_classificator = EMG_classifier(model_fpath)
    if not emg_classificator:
        return False
    features_map = emg_classificator.get_features_map()

    emg_features = dict()
    accel_features = dict()
    gyro_features = dict()

    for ch in emg_data_ptr:
        emg_features[ch] = list()
    for ch in accel_data_ptr:
        accel_features[ch] = list()
    for ch in gyro_data_ptr:
        gyro_features[ch] = list()

    for channel_feature in features_map:

        channel = channel_feature['channel']
        feature = channel_feature['feature']

        if channel in emg_features:
            emg_features[channel].append(feature)
        elif channel in accel_features:
            accel_features[channel].append(feature)
        elif channel in gyro_features:
            gyro_features[channel].append(feature)

    preprocessor_info = emg_classificator.get_preprocessing_info()
    features_info = emg_classificator.get_features_extraction_info()

    emg_preprocessor = EMG_preprocessor(SAMPLE_RATE, preprocessor_info['emg_normalize_method'], \
                                        preprocessor_info['emg_preprocessor_info'])
    accel_preprocessor = EMG_preprocessor(SAMPLE_RATE, preprocessor_info['accel_normalize_method'], \
                                          preprocessor_info['accel_preprocessor_info'])
    gyro_preprocessor = EMG_preprocessor(SAMPLE_RATE, preprocessor_info['gyro_normalize_method'], \
                                         preprocessor_info['gyro_preprocessor_info'])

    emg_feature_extractor = EMG_feature_extractor(features_info['emg_features_info'], SAMPLE_RATE)
    accel_feature_extractor = EMG_feature_extractor(features_info['accel_features_info'], SAMPLE_RATE)
    gyro_feature_extractor = EMG_feature_extractor(features_info['gyro_features_info'], SAMPLE_RATE)

    emg_preprocessor.set_data_ptr(emg_data_ptr)
    accel_preprocessor.set_data_ptr(accel_data_ptr)
    gyro_preprocessor.set_data_ptr(gyro_data_ptr)
    emg_feature_extractor.set_data_ptr_and_features(emg_preprocessor.res_data, \
                                                    is_all_features=False, \
                                                    selected_features=emg_features)
    accel_feature_extractor.set_data_ptr_and_features(accel_preprocessor.res_data, \
                                                      is_all_features=False, \
                                                      selected_features=accel_features)
    gyro_feature_extractor.set_data_ptr_and_features(gyro_preprocessor.res_data, \
                                                     is_all_features=False, \
                                                     selected_features=gyro_features)

    return emg_classificator, emg_preprocessor, emg_feature_extractor, \
        accel_preprocessor, accel_feature_extractor, \
        gyro_preprocessor, gyro_feature_extractor

def EMG_predict(emg_classificator: EMG_classifier,
                emg_preprocessor: EMG_preprocessor, \
                accel_preprocessor: EMG_preprocessor, \
                gyro_preprocessor: EMG_preprocessor, \
                emg_feature_extractor: EMG_feature_extractor, \
                accel_feature_extractor: EMG_feature_extractor, \
                gyro_feature_extractor: EMG_feature_extractor
                ):
    print(f"\r\n{get_date_time()}:\t Start preprocessing...")
    start_preproc = time.time()
    emg_preprocessor.preprocess_data()
    accel_preprocessor.preprocess_data()
    gyro_preprocessor.preprocess_data()
    preproc_time = (time.time() - start_preproc) * 1000

    print(f"{get_date_time()}:\t Start extracting features...")
    start_extraction = time.time()
    emg_feature_extractor.extract_features()
    accel_feature_extractor.extract_features()
    gyro_feature_extractor.extract_features()
    extraction_time = (time.time() - start_extraction) * 1000

    print(f"{get_date_time()}:\t Start recognition...")
    start_classif = time.time()
    X = list()
    for emg_ch in emg_feature_extractor.features_vals:
        for emg_feat in emg_feature_extractor.features_vals[emg_ch]:
            X.append(emg_feature_extractor.features_vals[emg_ch][emg_feat])

    for accel_ch in accel_feature_extractor.features_vals:
        for accel_feat in accel_feature_extractor.features_vals[accel_ch]:
            X.append(accel_feature_extractor.features_vals[accel_ch][accel_feat])

    for gyro_ch in gyro_feature_extractor.features_vals:
        for gyro_feat in gyro_feature_extractor.features_vals[gyro_ch]:
            X.append(gyro_feature_extractor.features_vals[gyro_ch][gyro_feat])

    res = emg_classificator.predict(X)
    classif_time = (time.time() - start_classif) * 1000

    print(f"{get_date_time()}:\t {res[0]}: {res[1]}")
    
    return res, preproc_time, extraction_time, classif_time


def EMG_recognition_task(emg_classificator: EMG_classifier,
                         emg_preprocessor: EMG_preprocessor, \
                         accel_preprocessor: EMG_preprocessor, \
                         gyro_preprocessor: EMG_preprocessor, \
                         emg_feature_extractor: EMG_feature_extractor, \
                         accel_feature_extractor: EMG_feature_extractor, \
                         gyro_feature_extractor: EMG_feature_extractor, \
                         stop_event: Event):

    while not stop_event.is_set():
        EMG_predict(emg_classificator, \
                    emg_preprocessor, \
                    accel_preprocessor, \
                    gyro_preprocessor, \
                    emg_feature_extractor, \
                    accel_feature_extractor, \
                    gyro_feature_extractor)
        time.sleep(0.5)

def EMG_online_plotting_task(plotter: Plotter, buttons: dict=None):
    try:
        if buttons:
            plotter.add_buttons(buttons)
        plotter.animate()
    finally:
        print("Online plotting stopped")
        return

def EMG_dev_task(emg_dev: EMG_device, stop_event: Event, device_port: str):
    try:
        emg_dev.run_measure()
        while not stop_event.is_set():
            emg_dev.read_data()
    finally:
        emg_dev.stop_measure()
        emg_dev.disconnect()
        return

def EMG_dev_is_sample_ready(emg_dev: EMG_device):
    data = [emg_dev.emg_data, emg_dev.accel_data, emg_dev.gyro_data]
    common_buff = dict()
    for sample in data:
        common_buff.update(sample)
        
    return all(all(item is not None for item in value) for value in common_buff.values())

if __name__ == '__main__':

    if MODE_FILE_NAME:
        emg_classificator, emg_preprocessor, emg_feature_extractor, accel_preprocessor, accel_feature_extractor, \
            gyro_preprocessor, gyro_feature_extractor = EMG_recognition_init(EMG_dev, MODEL_FPATH)

    emg_port = EMG_dev.find_port()
    if emg_port:
        try:
            ### EMG device thread
            EMG_dev.connect(emg_port)
            emg_dev_thread = Thread(target=EMG_dev_task, args=(EMG_dev, emg_dev_stop, emg_port))
            emg_dev_thread.start()

            while not EMG_dev_is_sample_ready(EMG_dev):
                time.sleep(0.02)

            if MODE_FILE_NAME:
                ### EMG recognition thread
                emg_recognition_thread = Thread(target=EMG_recognition_task, \
                                            args=(emg_classificator, \
                                            emg_preprocessor, \
                                            accel_preprocessor, \
                                            gyro_preprocessor, \
                                            emg_feature_extractor, \
                                            accel_feature_extractor, \
                                            gyro_feature_extractor, \
                                            emg_dev_stop))
                emg_recognition_thread.start()

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
