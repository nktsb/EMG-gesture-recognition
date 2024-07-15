"""
 -----------------------------------------------------------------------------
 Project       : EMG-gesture-recognition
 File          : write_dataset.py
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

from threading import Thread
from threading import Event

from emg_python.emg_dataset import EMG_ds_writer
from emg_python.emg_plot import Plotter

from run_emg import DATASET_FOLDER, GESTURES_SET
from run_emg import DEVICE_INFO, EMG_dev, EMG_dev_task, emg_dev_stop, EMG_dev_is_sample_ready
from run_emg import EMG_online_plotting_task

emg_write_ds_stop = Event()

def EMG_write_dataset_task(ds: EMG_ds_writer):
    try:
        while not ds.stop_event.is_set():
            ds.write_dataset(DATASET_FOLDER, GESTURES_SET)
    finally:
        print("Writing dataset stopped")
        return

if __name__ == '__main__':
    emg_port = EMG_dev.find_port()
    if emg_port:
        try:
            while True:
                user_name = input("\r\nEnter username and press Enter, or enter 'quit' to close\r\n")
                if user_name:
                    if user_name == "quit":
                        quit()
                    else:
                        break
                else:
                    print("Empty username!")

            ### EMG device thread
            EMG_dev.connect(emg_port)
            emg_dev_thread = Thread(target=EMG_dev_task, args=(EMG_dev, emg_dev_stop, emg_port))
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

            ### Dataset writer thread
            emg_ds_writer = EMG_ds_writer( user_name, emg_write_ds_stop, EMG_dev, plotter)
            emg_write_ds_thread = Thread(target=EMG_write_dataset_task, args=[emg_ds_writer])
            emg_write_ds_thread.start()

            buttons = {
                'Save': emg_ds_writer.save_clbk,
                'Next\ngesture': emg_ds_writer.next_clbk,
                'Remove\nlast': emg_ds_writer.remove_clbk,
                'Prev\ngesture': emg_ds_writer.back_clbk,
                'Quit': emg_ds_writer.quit_clbk
            }

            EMG_online_plotting_task(plotter, buttons=buttons)

        finally:
            emg_dev_stop.set()
            if emg_write_ds_stop:
                emg_write_ds_stop.set()
            quit()
