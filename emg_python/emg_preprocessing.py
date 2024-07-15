"""
 -----------------------------------------------------------------------------
 Project       : EMG-gesture-recognition
 File          : emg_preprocessing.py
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
import numpy as np

from scipy.signal import medfilt, butter, lfilter, filtfilt

class Normalizer():
    def __init__(self, method: str, min_val=0, max_val=1, offset=0):
        normalize_method_list = {
            '': self.empty,
            'Z': self.z,
            'MAX_MIN': self.max_min
        }

        if method in normalize_method_list:
            self.normalize = normalize_method_list[method]
        else:
            print(f"Error: normalize method doesn't exist: {method}")
            return

        self.min_val = min_val + offset
        self.max_val = max_val + offset

    def max_min(self, data: list):
        max_val = self.max_val
        min_val = self.min_val
        normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
        normalized_data = np.clip(normalized_data, 0.0, 1.0)
        return normalized_data

    def z(self, data: list):
        mean = np.mean(data)
        std_dev = np.std(data)
        if std_dev == 0:
            std_dev = 1
        normalized_data = [(x - mean) / std_dev for x in data]
        return normalized_data

    def empty(self, data: list):
        return data

class Filter():
    def __init__(
            self, \
            method: str, \
            sample_rate: int, \
            filter_param):

        filter_method_list = {
            '': self.empty,
            'RMS': self.rms_envelope,
            'MOV_AV': self.mov_average,
            'LOW_PASS': self.low_pass_butterword,
            'HIGH_PASS': self.high_pass_butterword,
            'BAND_PASS': self.band_pass_butterword,
            'MEDIAN': self.median
        }
        if method in filter_method_list:
            self.filter = filter_method_list[method]
        else:
            print(f"Error: filter doesn't exist: {method}")
            return

        self.sample_rate = sample_rate

        self.filter_param = filter_param
        self.window_size = filter_param['window_size']
        self.offset = filter_param['offset']
        self.order = filter_param['order']
        self.low_cutoff_freq = filter_param['low_cutoff_freq']
        self.high_cutoff_freq = filter_param['high_cutoff_freq']

    def mov_average(self, data: list):
        offset_data = [x + self.offset for x in data]
        filtered_data = list()

        for i in range(len(offset_data)):
            start_index = max(0, i - self.window_size)
            end_index = i + 1

            window_values = offset_data[start_index:end_index]
            average_value = sum(window_values) / len(window_values)
            filtered_data.append(average_value)

        return filtered_data
    
    def rms_envelope(self, data: list):
        offset_data = [x + self.offset for x in data]
        squared_signal = np.power(offset_data, 2)
        
        window = np.ones(self.window_size) / self.window_size
        smoothed_squared_signal = np.convolve(squared_signal, window, mode='same')
        
        rms_envelope = np.sqrt(smoothed_squared_signal)

        return rms_envelope

    def high_pass_butterword(self, data: list):
        offset_data = [x + self.offset for x in data]
        nyquist_frequency = 0.5 * self.sample_rate
        normalized_cutoff = self.low_cutoff_freq / nyquist_frequency
        b, a = butter(self.order, normalized_cutoff, btype='high', analog=False)
        filtered_data = filtfilt(b, a, offset_data)

        return filtered_data

    def low_pass_butterword(self, data: list):
        offset_data = [x + self.offset for x in data]
        nyquist_frequency = 0.5 * self.sample_rate
        normal_cutoff = self.high_cutoff_freq / nyquist_frequency
        b, a = butter(self.order, normal_cutoff, btype='low', analog=False)
        filtered_data = lfilter(b, a, offset_data)
        return filtered_data

    def band_pass_butterword(self, data: list):
        offset_data = [x + self.offset for x in data]
        nyquist_frequency = 0.5 * self.sample_rate
        low = self.low_cutoff_freq / nyquist_frequency
        high = self.high_cutoff_freq / nyquist_frequency
        b, a = butter(self.order, [low, high], btype='band')
        filtered_data = lfilter(b, a, offset_data)
        return filtered_data

    def median(self, data: list):
        offset_data = [x + self.offset for x in data]
        filtered_data = medfilt(offset_data, kernel_size=self.window_size)
        return filtered_data

    def empty(self, data: list):
        offset_data = [x + self.offset for x in data]
        return offset_data

class EMG_preprocessor():
    def __init__(
            self, \
            sample_rate: int=1000, \
            normalize_method: str="", \
            preprocessor_info: dict=None):

        if preprocessor_info is None:
            preprocessor_info = {'max': 1, 'min': 0, 'window_size': 1}

        self.sample_rate = sample_rate

        self.filt_data = dict()
        self.res_data = dict()

        self.normalizer = Normalizer(normalize_method, \
                                     min_val=preprocessor_info['min'], \
                                     max_val=preprocessor_info['max'], \
                                     offset=preprocessor_info['offset'])

        filter_method = preprocessor_info['filter_method']

        self.filter = Filter(filter_method, \
                             self.sample_rate, \
                             filter_param=preprocessor_info)
        
        self.preprocessor_info = preprocessor_info
        self.normalize_method = normalize_method

    def set_sample_rate(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.filter.sample_rate = sample_rate

    def set_data_ptr(self, data: dict):
        self.raw_data = data
        for ch in self.raw_data:
            self.filt_data[ch] = [0 for _ in range(len(self.raw_data[ch]))]
            self.res_data[ch] = [0 for _ in range(len(self.raw_data[ch]))]


    def preprocess_data(self):
        for ch in self.raw_data:
            self.filt_data[ch][:] = self.filter.filter(self.raw_data[ch][:])
            self.res_data[ch][:] = self.normalizer.normalize(self.filt_data[ch])
