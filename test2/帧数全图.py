# -*- coding:utf-8 -*-
__author__: 'Rvelamen'
__data__ = '2021/3/21 12:09'

from matplotlib import pyplot as plt
import numpy as np
import wave
import scipy.signal as signal

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def choose_windows(type, wlen):
    """
    type
        1、 汉明窗
        2、 海宁窗
        3、 矩形窗
        4、 三角窗
    :param type:
    :param wlen:
    :return:
    """
    if type == 1:
        window = np.array([0.54 - 0.46 * np.cos(2 * np.pi * n / (wlen - 1)) for n in range(wlen)])
    elif type == 2:
        window = np.array([0.5 * (1 - np.cos(2 * np.pi * n / (wlen - 1))) for n in range(wlen)])
    elif type == 3:
        window = np.ones(wlen)
    elif type == 4:
        window = signal.triang(wlen)
    return window

wlen = 512
inc = 128
f = wave.open(r"materials/11.wav", "rb")
param = f.getparams()
nchannels, sampwidth, framerate, nframes = param[:4]    # 输出声道数、 量化位数、 采样频率、 采样点数
str_data = f.readframes(nframes)                        # 读取音频数
wave_data = np.fromstring(str_data, dtype=np.short)     # 转为一维短整型
wave_data = wave_data * 1.0 / max(abs(wave_data))       # 数据归一化

time = np.arange(0, len(wave_data)) * 1.0 / framerate             # 计算一帧的时间区域


plt.figure(figsize=(10, 4))
plt.plot(time, wave_data)
plt.show()