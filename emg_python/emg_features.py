"""
 -----------------------------------------------------------------------------
 Project       : EMG-gesture-recognition
 File          : emg_features.py
 Author        : nktsb
 Created on    : 08.05.2024
 GitHub        : https://github.com/nktsb/EMG-gesture-recognition
 -----------------------------------------------------------------------------
 Copyright (c) 2024 nktsb
 All rights reserved.
 -----------------------------------------------------------------------------
"""

import sys
import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis

class EMG_feature_extractor():
    def __init__(
            self, features_info: dict, sample_rate: int=1000):

        self.features_methods_list = {
            ## TIME DOMAIN
            'EN': self.en, #energy
            'MAV': self.mav, #mean absolute value
            'MAD': self.mad, #mean absolute deviation
            'WL': self.wl, #waveform length
            'STD': self.std, #standart deviation 
            'SSC': self.ssc, #slope sign change
            'ZC': self.zc, #zero crossing
            'RMS': self.rms, #root mean square
            'NP': self.np, #number of peaks
            'SKEW': self.skew, #skewness
            'KURT': self.kurt, #kurtosis
            'VAR': self.var, #variance
            'WA': self.wa, #Wilson amplitude
            'PERC': self.perc, #Percentile
            'IAV': self.iav, #Integral absolute value

            ## FREQUENCY DOMAIN
            'MNF': self.mnf, #mean frequency
            'MDF': self.mdf, #median frequency
            'MNP': self.mnp, #mean power
            'TTP': self.ttp, #total power
            'PKF': self.pkf, #peak frequency
            'SE': self.se, #spectral entropy
            'FR': self.fr, #frequency ratio
            'PSR': self.psr #power spectrum ratio
        }

        self.features_info = features_info
        self.threshold = features_info['threshold']
        self.ssc_lvl = features_info['ssc_lvl']
        self.percentile = features_info['percentile']
        self.fr_low_range = features_info['fr_low_range']
        self.fr_high_range = features_info['fr_high_range']
        self.psr_range = features_info['psr_range']

        self.sample_rate = sample_rate

    def set_data_ptr_and_features(self, data: dict, is_all_features: bool=False, selected_features: dict=None):
        self.raw_data = data
        self.features_vals = dict()
        self.features_methods = dict()
        for ch in self.raw_data:
            self.features_vals[ch] = dict()
            self.features_methods[ch] = dict()

            for feature in self.features_methods_list:
                if is_all_features or feature in selected_features[ch]:
                    self.features_methods[ch][feature] = self.features_methods_list[feature]
                    self.features_vals[ch][feature] = None

    def set_sample_rate(self, sample_rate):
        self.sample_rate = sample_rate

    def wa(self, sample: list):
        wa = 0
        for i in range(1, len(sample)):
            if abs(sample[i] - sample[i-1]) > self.threshold:
                wa += 1
        return wa
    
    def perc(self, sample: list):
        return np.percentile(sample, self.percentile)

    def var(self, sample: list):
        signal = np.array(sample)
        return np.var(signal)

    def skew(self, sample: list):
        signal = np.array(sample)
        return skew(signal)

    def kurt(self, sample: list):
        signal = np.array(sample)
        return kurtosis(signal)

    def en(self, sample: list):
        signal = np.array(sample)
        return np.sum(np.square(signal))

    def mav(self, sample: list): # mean absolute value
        signal = np.array(sample)
        return np.mean(np.abs(signal))

    def mad(self, sample: list):
        signal = np.array(sample)
        mean_value = np.mean(signal)
        mad = np.mean(np.abs(signal - mean_value))
        return mad

    def wl(self, sample: list): # waveform length
        signal = np.array(sample)
        return np.sum(np.abs(np.diff(signal)))

    def std(self, sample: list): # standart deviations
        signal = np.array(sample)
        return np.std(signal)

    def ssc(self, sample: list): # slope sign change
        ssc = 0
        for i in range (1, (len(sample) - 1)):
            if ((sample[i] - sample[i + 1])*(sample[i] - sample[i - 1])) >= self.ssc_lvl:
                ssc += 1
        return ssc

    def zc(self, sample: list): # zero crossing
        signal = np.array(sample)
        zc = 0
        for i in range(1, len(signal)):
            if (signal[i-1] > self.threshold and signal[i] < self.threshold) \
                or (signal[i-1] < self.threshold and signal[i] > self.threshold):
                zc += 1
        return zc

    def rms(self, sample: list): # root mean square
        signal = np.array(sample)
        rms = np.sqrt(np.mean(np.square(signal)))
        return rms

    def np(self, sample: list): # number of peaks
        npeaks = 0
        rms = self.rms(sample)
        for i in range (0, len(sample)):
            if sample[i] > rms:
                npeaks += 1
        return npeaks
    
    def iav(self, sample: list):
        signal = np.array(sample)
        iav = np.sum(signal)
        return iav

    def mnf(self, sample: list):
        freqs, psd = welch(sample, fs=self.sample_rate)
        if np.sum(psd) == 0:
            return 0
        mean_frequency = np.sum(freqs * psd) / np.sum(psd)
        return mean_frequency

    def mdf(self, sample: list):
        freqs, psd = welch(sample, fs=self.sample_rate)
        if np.sum(psd) == 0:
            return 0
        cumulative_power = np.cumsum(psd)
        median_frequency = freqs[np.searchsorted(cumulative_power, cumulative_power[-1] / 2)]
        return median_frequency

    def pkf(self, sample: list):
        freqs, psd = welch(sample, fs=self.sample_rate)
        if np.sum(psd) == 0:
            return 0
        peak_freq_index = np.argmax(psd)
        peak_freq = freqs[peak_freq_index]
        return peak_freq
    
    def se(self, sample: list):
        freqs, psd = welch(sample, fs=self.sample_rate)
        if np.sum(psd) == 0:
            return 0
        psd_norm = psd / np.sum(psd)
        spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + np.finfo(float).eps))
        return spectral_entropy
    
    def mnp(self, sample: list):
        freqs, psd = welch(sample, fs=self.sample_rate)
        return np.mean(psd)
    
    def ttp(self, sample: list):
        freqs, psd = welch(sample, fs=self.sample_rate)
        return np.sum(psd)
    
    def fr(self, sample: list):
        freqs, psd = welch(sample, fs=self.sample_rate)
    
        low_freq_power = np.trapz(psd[(freqs >= self.fr_low_range[0]) & (freqs < self.fr_low_range[1])], \
                                  freqs[(freqs >= self.fr_low_range[0]) & (freqs < self.fr_low_range[1])])
        high_freq_power = np.trapz(psd[(freqs >= self.fr_high_range[0]) & (freqs < self.fr_high_range[1])], \
                                   freqs[(freqs >= self.fr_high_range[0]) & (freqs < self.fr_high_range[1])])

        fr = low_freq_power / high_freq_power
        return fr
    
    def psr(self, sample: list):
        signal = np.array(sample)
        freqs, psd = welch(signal, fs=self.sample_rate)
        
        band_power = np.trapz(psd[(freqs >= self.psr_range[0]) & (freqs <= self.psr_range[1])], \
                              freqs[(freqs >= self.psr_range[0]) & (freqs <= self.psr_range[1])])
        total_power = np.trapz(psd, freqs)

        psr = band_power / total_power
        return psr

    def extract_features(self):
        for ch in self.raw_data:
            for feat in self.features_methods[ch]:
                self.features_vals[ch][feat] = self.features_methods[ch][feat](self.raw_data[ch][:])


if __name__ == '__main__':
    test_features = {
        'features': ['MAV', 'WL', 'STD', 'SSC', 'ZC', 'RMS', 'NP'],
        'threshold': 0.5
    }
    test_arr = {
        'EMG0': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'EMG1': [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0],
        'ACCX': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    }
    
    test_features = EMG_feature_extractor(test_arr, 10, test_features)
    test_features.extract_features()
    for ch in test_features.features_vals:
        print(f"\r\n{ch}: {test_features.data[ch]}")
        for feat in test_features.features_vals[ch]:
            print(feat, test_features.features_vals[ch][feat])
    quit()
