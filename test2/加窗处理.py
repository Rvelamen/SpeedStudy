# -*- coding:utf-8 -*-
__author__: 'Rvelamen'
__data__ = '2021/3/20 23:44'

from matplotlib import pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

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

wlen = 64                          # 帧长
inc = 32                           # 帧移
N = 1000                            # 采样点个数
t = np.linspace(0, 1, N)            # 得到各个点采样时间(1秒)
time = np.arange(0, wlen) * (1.0/N)      # 一帧的时间长度

y = np.sin(2 * np.pi * 250 * t) + 2 * np.sin(2 * np.pi * 185.5 * t)  # 函数

nf = int(np.ceil((1.0 * N - wlen + inc) / inc)) # 帧数
pad_length = int((nf - 1) * inc + wlen)     # 加上超出部分
zeros = np.zeros((pad_length - N,))            # 补零
pad_signal = np.concatenate((y, zeros))     # 填补后的信号

indices = np.tile(np.arange(0, wlen), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc),(wlen, 1)).T   # 各帧的采样点
indices = np.array(indices, dtype=np.int32)
frames = pad_signal[indices]
a = frames[2:3]

# 加窗前
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.title("未加窗振幅图")
plt.xlabel("振幅值")
plt.ylabel("频率")
amp = np.arange(wlen)
x = np.fft.fft(a[0])
abs_y = np.abs(x)
plt.plot(amp[range(int(wlen/2))], abs_y[range(int(wlen/2))])

# 添加汉明窗
plt.subplot(122)
plt.title("填加汉明窗振幅图")
plt.xlabel("振幅值")
plt.ylabel("频率")
window = choose_windows(1, wlen)
b = a[0]*window
x = np.fft.fft(b)
abs_y = np.abs(x)
plt.plot(amp[range(int(wlen/2))], abs_y[range(int(wlen/2))])



plt.show()