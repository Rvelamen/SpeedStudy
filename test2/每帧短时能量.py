# -*- coding:utf-8 -*-
__author__: 'Rvelamen'
__data__ = '2021/3/21 23:09'

from matplotlib import pyplot as plt
import numpy as np
import wave
import scipy.signal as signal

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 窗型选择
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

# 计算短时能量
def calEnergy(frames, nf, window):
    energy = np.zeros(nf)
    for i in range(0, nf):
        a = frames[i:i+1]
        b = a[0] * window
        c = np.square(b)
        energy[i] = np.sum(c)
    energy = energy * 1.0 / max(abs(energy))
    return energy

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

energy = calEnergy(frames, nf, window=choose_windows(1, wlen))  # 添加汉明窗后的短时能量

plt.figure(figsize=(6, 4))
plt.title("短时能量图")
plt.xlabel("帧数")
plt.ylabel("归一化能量")
plt.plot(np.arange(0, nf), energy)
plt.show()


