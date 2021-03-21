# -*- coding:utf-8 -*-
__author__: 'Rvelamen'
__data__ = '2021/3/21 9:58'

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
time = np.arange(0, wlen) * 1.0 / framerate             # 计算一帧的时间区域
signal_length = len(wave_data)

if signal_length < wlen:
    nf = 1
else:
    nf = int(np.ceil((1.0 * signal_length - wlen + inc) / inc)) # 计算帧数，向上取

pad_length = int((nf-1) * inc + wlen)                   # 计算nf帧展开后的长度
zeros = np.zeros(pad_length - signal_length)            # 补零
pad_signal = np.concatenate((wave_data, zeros))         # 补零填充

indices = np.tile(np.arange(0, wlen), (nf, 1)) + np.tile(np.arange(0, nf*inc, inc), (wlen, 1)).T    # 划分数据每一帧的区域
indices = np.array(indices, dtype=np.int32)             # indices转为32位整形
frames = pad_signal[indices]                            # 切割pad_signal, 得到每帧的信号

a = frames[80:81]

plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.title("第81帧时域图")
plt.xlabel("时间")
plt.ylabel("归一化振幅")
plt.plot(time, a[0])
plt.grid()


window = choose_windows(1, wlen)
plt.subplot(122)
plt.title("第81帧短时能量图")
plt.xlabel("时间")
plt.ylabel("归一化振幅")
b = a[0] * window
c = np.square(b)
plt.plot(time, c)
plt.grid()
plt.show()

















# wlen = 64  # 帧长
# inc = 32  # 帧移
# N = 1000  # 采样点个数
# t = np.linspace(0, 1, N)  # 得到各个点采样时间(1秒)
# time = np.arange(0, wlen) * (1.0 / N)  # 一帧的时间长度
#
# y = np.sin(2 * np.pi * 250 * t) + 2 * np.sin(2 * np.pi * 185.5 * t)  # 函数
#
# nf = int(np.ceil((1.0 * N - wlen + inc) / inc))  # 帧数
# pad_length = int((nf - 1) * inc + wlen)  # 加上超出部分
# zeros = np.zeros((pad_length - N,))  # 补零
# pad_signal = np.concatenate((y, zeros))  # 填补后的信号
# pad_signal = pad_signal/np.max(pad_signal)  # 归一化
#
# indices = np.tile(np.arange(0, wlen), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc), (wlen, 1)).T  # 各帧的采样点
# indices = np.array(indices, dtype=np.int32)
# frames = pad_signal[indices]
# a = frames[2:3]
#
# window = choose_windows(1, wlen)
# b = a[0] * window
# c = np.square(b)
# plt.figure(figsize=(10, 4))
# plt.title("添加汉明窗归一化短时能量")
# plt.xlabel("时间/s")
# plt.ylabel("归一化振幅")
# plt.plot(time, c, c="g")
# plt.grid()
# plt.show()


