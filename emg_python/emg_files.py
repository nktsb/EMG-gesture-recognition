"""
 -----------------------------------------------------------------------------
 Project       : EMG-gesture-recognition
 File          : emg_files.py
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
import json
import datetime

def get_date_time():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')[:-3]

class EMG_file():
    def __init__(
            self, fpath: str, user_name: str="", \
            gesture_name: str=""):

        self.fpath = fpath
        self.is_exist = bool(os.path.isfile(self.fpath) and os.path.getsize(self.fpath))

        self.json = {
            'general_info': dict()
        }

        if self.is_exist:
            self.read_json(self)
        if user_name and gesture_name:
            self.create_file(self, user_name, gesture_name)

    @staticmethod
    def read_json(self):
        with open(self.fpath, 'r') as fp:
                self.json = json.load(fp)
                fp.close()

    @staticmethod
    def write_json(self):
        date_time = get_date_time()
        self.json['general_info'].update({'last_change_time': date_time})

        with open(self.fpath, 'w') as fp:
            json.dump(self.json, fp, indent = 4, separators = (', ', ': '), ensure_ascii=False)
            fp.close()

    @staticmethod
    def create_file(self, user_name: str, gesture_name: str):
        if self.is_exist:
            print(f"File {self.fpath} already exists")
            return False
        else:
            print(f"File {self.fpath} creating...")
            date_time = get_date_time()
            self.json['general_info'].update(
                {
                    'file': self.fpath,
                    'create_time': date_time,
                    'last_change_time': None,
                    'user_name': user_name,
                    'gesture_name': gesture_name
                }
            )
            self.write_json(self)
            self.is_exist = True

    def remove(self):
        os.remove(self.fpath)

    def write_data(self, key: str, data: dict):

        print(f"Updating file {self.fpath} data...")
        self.json[key] = data
        self.write_json(self)
        print("Data updated")

    def read_data(self, key):
        if key in self.json:
            return self.json[key]
        return None

    def append_data(self, json_data: dict):
        self.json.update(json_data)
        self.write_json(self)


if __name__ == "__main__":

    test = EMG_file("./dataset/test_file.json", user_name="test_user", \
                    gesture_name="test_gesture")

    emg_test = {
        'EMG0': [1000, 2000, 3000, 4000],
        'EMG1': [500, 1500, 2500, 3500],
        'EMG2': [250, 750, 1250, 1750]
    }
    accel_test = {
        'ACCX': [100, 200, 300, 400],
        'ACCY': [100, 250, 400, 550],
        'ACCZ': [100, 300, 500, 700]
    }
    gyro_test = {
        'ACCX': [-100, -200, -300, -400],
        'ACCY': [-100, -250, -400, -550],
        'ACCZ': [-100, -300, -500, -700]
    }
    test_data = {
        'emg_data': emg_test,
        'accel_data': accel_test,
        'gyro_data': gyro_test
    }
    
    sns_info = {
        'emg_channels': [key for key in emg_test],
        'accel_axes': [key for key in accel_test],
        'gyro_axes': [key for key in gyro_test],
        'sample_size': 4,
        'sample_rate': 1
    }

    test.write_data('sns_info', sns_info)
    test.write_data('sns_data', test_data)
    print(test.read_data('sns_info'))
    print(test.read_data('sns_data'))
    
    del test
    quit()