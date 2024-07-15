"""
 -----------------------------------------------------------------------------
 Project       : EMG-gesture-recognition
 File          : emg_plot.py
 Author        : nktsb
 Created on    : 08.05.2024
 GitHub        : https://github.com/nktsb/EMG-gesture-recognition
 -----------------------------------------------------------------------------
 Copyright (c) 2024 nktsb
 All rights reserved.
 -----------------------------------------------------------------------------
"""

import sys
import time
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from matplotlib.widgets import CheckButtons

# color definition
black = '#1A1A1A'
gray = '#EEEEEE'

red = '#fe0000'
green = '#00f61b'
blue = '#20E1FF'
pink = '#F94EBC'
yellow = '#fff801'
orange = '#ffaa3b'
violet = '#fa3bff'
brown = '#c9954c'
cyan = '#b3ecf1'
limegreen = '#b9ff66'

color_set = [red, green, blue, pink, yellow, orange, violet, brown, cyan, limegreen]

class Plotter:
    def __init__(self, data: list, labels: list, max_mins: list, \
                 show_legend: bool=False, title: str=None, sublots: list=None, ):

        self.data = data
        window_size = min([len(x) for x in data])
        ch_num = len(data)

        if not sublots:
            sublots = [ch_num]

        self.labels = labels
        self.plot = [[] for i in range(len(data))]

        self.x = np.arange(window_size)
        self.intensity_plot_init(self, window_size, ch_num, max_mins, title, \
                                 show_legend, sublots)

        self.buttons = list()
        self.anim = None

    def add_buttons(self, buttons: dict):
        for button in buttons:
            idx = list(buttons.keys()).index(button)
            axbut = self.fig.add_axes([0.7 - (idx * 0.12), 0.05, 0.1, 0.075])
            butt = Button(axbut, button)
            butt.on_clicked(buttons[button])
            self.buttons.append(butt)
            pass

    def show_once(self):
        self.upd_plot(0)
        plt.show()

    def animate(self):
        self.anim = animation.FuncAnimation(
            self.fig, self.upd_plot, interval=32, frames=1)
        plt.show()

    def stop_animate(self):
        self.anim.event_source.stop()
    
    def resume_animate(self):
        self.anim.event_source.start()
    
    def close(self):
        plt.close(self.fig)

    @staticmethod
    def intensity_plot_init(self, window_size, ch_num, max_mins, title, show_legend, subplots):

        self.fig, self.ax = plt.subplots(len(subplots), 1)
        plt.subplots_adjust(hspace=0.3)

        self.fig.subplots_adjust(bottom=0.2, right=0.8)

        for i, ax in enumerate(self.ax):
            ax.set_facecolor(black)
            ax.set_xlim([0, window_size])
            ax.set_ylim(max_mins[i])

        if title:
            self.ax[0].set_title(title, loc='center')

        ch_idx = 0
        for i, ax in enumerate(self.ax):
            for _ in range(subplots[i]):
                self.plot[ch_idx], = ax.plot([], [], color=color_set[ch_idx], label=self.labels[ch_idx])
                ch_idx += 1

        plt.xticks([i * window_size/10 for i in range(10 + 1)])
        plt.grid(True)
        plt.axhline(y=0, color=gray)


        if show_legend:
            visibility = [True for _ in range(ch_num)]
            rax = self.fig.add_axes([0.8, 0.5 - ((0.05 * ch_num) / 2), 0.15, 0.05 * ch_num])
            self.check = CheckButtons(rax, self.labels, visibility)
            for rect, color in zip(self.check.rectangles, color_set):
                rect.set_facecolor(color)
            self.check.on_clicked(self.legend_clbk)

    def legend_clbk(self, label):
        for plot in self.plot:
            if plot.get_label() == label:
                plot.set_visible(not plot.get_visible())
        plt.draw()

    def upd_plot(self, frame):
        for i, plot in enumerate(self.plot):
            plot.set_data(self.x, self.data[i])
