# -*- coding:utf-8 -*-
__author__: 'Rvelamen'
__data__ = '2021/3/19 15:04'

from matplotlib import pyplot as plt
import numpy as np

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
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.title("The Original waveform")
plt.xlabel("time(/s)")
plt.ylabel("Amplitude")
plt.legend()
plt.plot(time, a[0])

plt.subplot(122)
amp = np.arange(wlen)
x = np.fft.fft(a[0])
abs_y = np.abs(x)
plt.plot(amp[range(int(wlen/2))], abs_y[range(int(wlen/2))])

plt.show()
# 合成第2帧的情况
print(time)
print(time+wlen*1.0/N)
print(np.concatenate((time, time+wlen*1.0/N)))
plt.plot(np.concatenate((time, time+wlen*1.0/N)), np.concatenate((a[0], a[0])))
plt.show()
