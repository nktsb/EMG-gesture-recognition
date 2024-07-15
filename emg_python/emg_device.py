"""
 -----------------------------------------------------------------------------
 Project       : EMG-gesture-recognition
 File          : emg_device.py
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
import serial
import serial.tools.list_ports

BUFF_SIZE = 8192
DEF_TMO = 1

class EMG_device():

    def __init__(
            self, baudrate: int, ping_answer: str, sns_info: dict) -> None:

        self.port = None
        self.baudrate = baudrate

        self.ping_answer = ping_answer

        self.sns_info = sns_info

        self.emg_channels = sns_info['emg_channels']
        self.accel_axes = sns_info['accel_axes']
        self.gyro_axes = sns_info['gyro_axes']
        self.sample_size = sns_info['sample_size']
        self.sample_rate = sns_info['sample_rate']

        self.emg_data = dict()
        for ch in self.emg_channels:
            self.emg_data[ch] = [None for _ in range(self.sample_size)]
    
        self.accel_data = dict()
        for ax in self.accel_axes:
            self.accel_data[ax] = [None for _ in range(self.sample_size)]

        self.gyro_data = dict()
        for ax in self.gyro_axes:
            self.gyro_data[ax] = [None for _ in range(self.sample_size)]

        self.data = {
            'emg_data': self.emg_data,
            'accel_data': self.accel_data,
            'gyro_data': self.gyro_data
        }

        self.buffer = [None for _ in range(BUFF_SIZE)]

    def connect(self, port):
        try:
            self.port = serial.Serial(port=port, baudrate=self.baudrate, timeout=DEF_TMO)
            print("Device connected")
        except:
            print("Can't connect device")
        
    def disconnect(self):
        try:
            self.port.close()
            print("Device disconnected")
        except:
            print("Can't disconnect device")
    

    def run_measure(self):
        try:
            self.port.write('r'.encode("ascii"))
            print("Measuring started")
        except:
            print("Starting measuring failed")

    def stop_measure(self):
        try:
            self.port.write('s'.encode("ascii"))
            print("Measuring stopped")
        except:
            print("Stopping measuring failed")

    def find_port(self):
        ports = serial.tools.list_ports.comports()

        for port in sorted(ports):
            try:
                ser = serial.Serial(port=f"/dev/{port.name}", baudrate=self.baudrate, timeout=DEF_TMO)
                ser.write('p'.encode("ascii"))
                ping_ans = str(ser.readline(), 'ascii')
                ser.close()

                if ping_ans == self.ping_answer:
                    print(f"Device {port.name} pinged successfully")
                    return f"/dev/{port.name}"
                else:
                    print(f"Device {port.name} wrong")
            except:
                pass
        
        print(f"Device not found")
        return False
    
    def read_data(self):
        string = str(self.port.readline(), 'ascii')

        preamble = string[:string.index(':')]
        data_list_ptr = list()

        if preamble in self.emg_data:
            data_list_ptr = self.emg_data[preamble]
        elif preamble in self.accel_data:
            data_list_ptr = self.accel_data[preamble]
        elif preamble in self.gyro_data:
            data_list_ptr = self.gyro_data[preamble]
        else:
            return False
        
        values_str = string[string.index(':') + 1:]
        values = list(map(int, values_str.split(',')))

        for _ in range(len(values)):
            data_list_ptr.pop(0)
    
        data_list_ptr += values

        return True
