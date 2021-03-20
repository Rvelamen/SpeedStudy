# -*- coding:utf-8 -*-
__author__: 'Rvelamen'
__data__ = '2021/3/19 15:04'

from matplotlib import pyplot as plt
import numpy as np

wlen = 64  # 帧长
inc = 32  # 帧移
N = 1000  # 采样点个数
t = np.linspace(0, 1, N)  # 得到各个点采样时间(1秒)
time = np.arange(0, wlen) * (1.0 / N)  # 一帧的时间长度

y = np.sin(2 * np.pi * 250 * t) + 2 * np.sin(2 * np.pi * 185.5 * t)  # 函数

nf = int(np.ceil((1.0 * N - wlen + inc) / inc))  # 帧数
pad_length = int((nf - 1) * inc + wlen)  # 加上超出部分
zeros = np.zeros((pad_length - N,))  # 补零
pad_signal = np.concatenate((y, zeros))  # 填补后的信号

indices = np.tile(np.arange(0, wlen), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc), (wlen, 1)).T  # 各帧的采样点
indices = np.array(indices, dtype=np.int32)
frames = pad_signal[indices]
a = frames[2:3]
plt.ion()
plt.figure(figsize=(10, 4))
plt.title("The Original waveform")
plt.plot(time, a[0])
plt.show()

# 合成第2帧的情况
plt.figure(figsize=(10, 4))
plt.plot(np.concatenate((time, time + wlen * 1.0 / N)), np.concatenate((a[0], a[0])))
plt.ioff()
plt.show()
