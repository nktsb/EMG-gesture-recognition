"""
 -----------------------------------------------------------------------------
 Project       : EMG-gesture-recognition
 File          : emg_dataset.py
 Author        : nktsb
 Created on    : 08.05.2024
 GitHub        : https://github.com/nktsb/EMG-gesture-recognition
 -----------------------------------------------------------------------------
 Copyright (c) 2024 nktsb
 All rights reserved.
 -----------------------------------------------------------------------------
"""

import sys
import os
import re
import time
import json

from threading import Event

from emg_python.emg_files import EMG_file
from emg_python.emg_plot import Plotter
from emg_python.emg_device import EMG_device
from emg_python.emg_preprocessing import EMG_preprocessor
from emg_python.emg_features import EMG_feature_extractor
from emg_python.emg_preprocessing import EMG_preprocessor


def get_sequential_fpath(fdir: str, fname_base: str, fname_extension: str):

    if not os.path.isdir(fdir):
        os.makedirs(fdir)

    files_in_dir = os.listdir(fdir)
    based_files_in_dir = [file for file in files_in_dir if file.startswith(fname_base)]

    num = 0
    if based_files_in_dir:
        based_files_nums = [int(re.search(r'\d+', fname).group()) for fname in based_files_in_dir]
        num = max(based_files_nums) + 1
    res = f"{fdir}/{fname_base}{num}{fname_extension}"
    return res, num


class EMG_ds_writer():
    def __init__(
            self, user_name: str, stop_event: Event, \
            emg_dev: EMG_device, plotter: Plotter):

        self.emg_dev = emg_dev

        self.save_event = Event()
        self.back_event = Event()
        self.remove_event = Event()
        self.next_event = Event()
        self.quit_event = Event()

        self.stop_event = stop_event
        self.user_name = user_name

        self.plotter = plotter

    def save_clbk(self, event):
        self.save_event.set()

    def remove_clbk(self, event):
        self.remove_event.set()

    def back_clbk(self, event):
        self.back_event.set()

    def next_clbk(self, event):
        self.next_event.set()

    def quit_clbk(self, event):
        self.plotter.stop_animate()
        self.plotter.close()
        self.quit_event.set()

    @staticmethod
    def save_file(self, fpath: str, gesture: str):
        file = EMG_file(fpath, user_name=self.user_name, gesture_name=gesture)
        file.write_data('sns_info', self.emg_dev.sns_info)
        file.write_data('sns_data', self.emg_dev.data)
        del file

    @staticmethod
    def remove_file(self, fpath):
        file = EMG_file(fpath)
        file.remove()
        del file

    def write_dataset(self, directory: str, gestures_set: list):
        gest_idx = 0
        while gest_idx < len(gestures_set):
            gesture = gestures_set[gest_idx]
            file_counter = 0
            file_dir = f"{directory}/{self.user_name}/{gesture}"
            fpath_base = f"{file_dir}/{gesture}"

            if not os.path.isdir(file_dir):
                print(f"There's no user dataset dir: '{file_dir}' creating...")
                os.makedirs(file_dir)

            while True:
                fpath, file_counter = get_sequential_fpath(file_dir, gesture, ".json")

                print(f"\r\nPress [Save] to save gesture {gesture} sample in file: {fpath}")

                while not self.stop_event.is_set() \
                        and not self.next_event.is_set() \
                        and not self.back_event.is_set() \
                        and not self.remove_event.is_set() \
                        and not self.quit_event.is_set() \
                        and not self.save_event.is_set():
                    time.sleep(0.01)

                if self.quit_event.is_set():
                    return

                if self.save_event.is_set():
                    self.save_event.clear()
                    self.save_file(self, fpath, gesture)
                    print(f"Gesture {gesture} saved in file {fpath}")

                if self.next_event.is_set():
                    self.next_event.clear()
                    gest_idx += 1
                    gest_idx %= len(gestures_set)
                    break
                
                if self.back_event.is_set():
                    self.back_event.clear()
                    if gest_idx > 0:
                        gest_idx -= 1
                    else:
                        gest_idx = len(gestures_set) - 1
                    break

                if self.remove_event.is_set():
                    self.remove_event.clear()
                    if file_counter > 0:
                        file_counter -= 1
                        fpath = f"./{fpath_base}{file_counter}.json"
                        self.remove_file(self, fpath)
                        print(f"File {fpath} removed!")
                        break
        return False

class EMG_ds_shower():
    def __init__(self, emg_preprocessor: EMG_preprocessor, accel_preprocessor: EMG_preprocessor, \
                 gyro_preprocessor: EMG_preprocessor):

        self.quit_event = Event()
        self.next_g_event = Event()
        self.prev_g_event = Event()
        self.emg_preprocessor = emg_preprocessor
        self.accel_preprocessor = accel_preprocessor
        self.gyro_preprocessor = gyro_preprocessor

    def show_plot(self, file: EMG_file):
        sns_info = file.read_data('sns_info')
        emg_channels = sns_info['emg_channels']
        acc_axes = sns_info['accel_axes']
        gyr_axes = sns_info['gyro_axes']
        sample_rate = sns_info['sample_rate']

        sns_data = file.read_data('sns_data')
        emg_data = sns_data['emg_data']
        accel_data = sns_data['accel_data']
        gyro_data = sns_data['gyro_data']

        self.emg_preprocessor.set_sample_rate(sample_rate)
        self.accel_preprocessor.set_sample_rate(sample_rate)
        self.gyro_preprocessor.set_sample_rate(sample_rate)

        self.emg_preprocessor.set_data_ptr(emg_data)
        self.accel_preprocessor.set_data_ptr(accel_data)
        self.gyro_preprocessor.set_data_ptr(gyro_data)
        self.emg_preprocessor.preprocess_data()
        self.accel_preprocessor.preprocess_data()
        self.gyro_preprocessor.preprocess_data()

        plot_legend = [emg_ch for emg_ch in emg_channels]
        plot_legend += [ax for ax in acc_axes]
        plot_legend += [ax for ax in gyr_axes]

        plt_emg_data = list(self.emg_preprocessor.res_data.values())
        plt_accel_data = list(self.accel_preprocessor.res_data.values())
        plt_gyro_data = list(self.gyro_preprocessor.res_data.values())

        emg_max = max(max(lst) for lst in plt_emg_data)
        emg_min = min(min(lst) for lst in plt_emg_data)
        accel_max = max(max(lst) for lst in plt_accel_data)
        accel_min = min(min(lst) for lst in plt_accel_data)
        gyro_max = max(max(lst) for lst in plt_gyro_data)
        gyro_min = min(min(lst) for lst in plt_gyro_data)

        plot_data  = plt_emg_data + plt_accel_data + plt_gyro_data

        subplots = [len(plt_emg_data), len(plt_accel_data), len(plt_gyro_data)]
        max_mins = [[emg_min, emg_max], [accel_min, accel_max], [gyro_min, gyro_max]]

        self.plot = Plotter(plot_data, plot_legend, max_mins, title=file.fpath, \
                            show_legend=True, sublots=subplots)

        buttons = {
            'Next\nfile': self.next_clbk,
            'Next\ngesture': self.next_g_clbk,
            'Previous\ngesture': self.prev_g_clbk,
            'Quit': self.quit_clbk
        }
        
        self.plot.add_buttons(buttons)
        self.plot.show_once()
        del self.plot

    def quit_clbk(self, event):
        self.quit_event.set()
        self.plot.close()

    def next_g_clbk(self, event):
        self.next_g_event.set()
        self.plot.close()

    def prev_g_clbk(self, event):
        self.prev_g_event.set()
        self.plot.close()

    def next_clbk(self, event):
        self.plot.close()

    def show_dataset_files(self, fdir_base:str):
        gest_idx = 0
        gestures = [gesture for gesture in os.listdir(fdir_base) if not gesture.startswith(".")]
        if not gestures:
            print("There is no dataset files")
            return False
        gestures.sort()

        while gest_idx < len(gestures):
            gesture = gestures[gest_idx]
            fdir = f"{fdir_base}/{gesture}"
            if os.path.isdir(fdir):
                files_in_dir = os.listdir(fdir)
                if not files_in_dir:
                    gest_idx += 1
                    gest_idx %= len(gestures)
                    continue
                gesture_files_in_dir = [
                    gest_file for gest_file in files_in_dir if gest_file.startswith(gesture)
                ]
                gesture_files_in_dir.sort()
                for gesture_file in gesture_files_in_dir:
                    fpath = f"{fdir}/{gesture_file}"
                    current_file = EMG_file(fpath)
                    self.show_plot(current_file)
                    del current_file

                    if(self.next_g_event.is_set()):
                        self.next_g_event.clear()
                        gest_idx += 1
                        gest_idx %= len(gestures)
                        break
                    if(self.prev_g_event.is_set()):
                        self.prev_g_event.clear()
                        if gest_idx > 0:
                            gest_idx -= 1
                        else:
                            gest_idx = len(gestures) - 1
                        break
        
                    if(self.quit_event.is_set()):
                        return
            else:
                gest_idx = 0
