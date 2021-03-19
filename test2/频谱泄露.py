# -*- coding:utf-8 -*-
__author__: 'Rvelamen'
__data__ = '2021/3/19 15:04'

from matplotlib import pyplot as plt
import numpy as np

N = 1000                    # 采样点个数
t = np.linspace(0, 1, N)    # 得到各个点采样时间

y = np.cos(2 * np.pi * 150 * t) + np.cos(2 * np.pi * 320 * t)  # 构造振幅为0.7的50HZ正弦波 和 振幅为1 的120HZ的正弦波
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.title("The Original waveform")
plt.xlabel("time(/s)")
plt.ylabel("Amplitude")
plt.legend()
plt.plot(t[:100], y[:100])
plt.show()
