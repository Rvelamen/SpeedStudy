# -*- coding:utf-8 -*-
__author__: 'Rvelamen'
__data__ = '2021/3/16 13:43'

import numpy as np
import wave
import matplotlib.pyplot as plt
import scipy.signal as signal

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

wlen = 512  # wlen为帧长，inc为帧移，重叠部分为overlap，overlap=wlen - inc
inc = 128
f = wave.open(r"materials/11.wav", "rb")
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]  # 输出声道数，量化位数，采样频率，采样点数
str_data = f.readframes(nframes)  # 读取音频数据
wave_data = np.fromstring(str_data, dtype=np.short)  # 转为一维short类型的数组
wave_data = wave_data * 1.0 / (max(abs(wave_data)))  # 归一化处理
time = np.arange(0, wlen) * (1.0 / framerate)  # 一帧的时间
signal_length = len(wave_data)  # 信号总长度

if signal_length <= wlen:  # 若信号长度小于一个帧的长度，则帧数定义为1
    nf = 1
else:  # 否则，计算信号帧数
    nf = int(np.ceil((1.0 * signal_length - wlen + inc) / inc))

pad_length = int((nf - 1) * inc + wlen)  # 所有帧加起来总的铺平后的长度
zeros = np.zeros((pad_length - signal_length,))  # 不够的长度使用0填补，类似于FFT中的扩充数组操作
print(signal_length)
print(pad_length)
pad_signal = np.concatenate((wave_data, zeros))  # 填补后的信号记为pad_signal

indices = np.tile(np.arange(0, wlen), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc),
                                                         (wlen, 1)).T  # 相当于对所有帧的采样点采样，得到nf*wlen长度的矩阵
indices = np.array(indices, dtype=np.int32)  # 将indices转化为矩阵
frames = pad_signal[indices]  # 得到帧信号

a = frames[50:51]

# print(a[0])
# plt.figure(figsize=(10, 4))
plt.plot(time, a[0], c="g")
plt.grid()
plt.show()





# 傅里叶变换加窗前
fft_signal2 = np.fft.fft(a[0])
fft_signal2 = abs(fft_signal2)
plt.plot(time, fft_signal2, c='b', label="")

# 加窗汉明窗
windown = choose_windows(1, wlen)        # 调用汉明窗
b = a[0] * windown
# 傅里叶变换
fft_signal = np.fft.fft(b)  # 语音信号FFT变换
fft_signal = abs(fft_signal)  # 取变换结果的模
plt.plot(time, fft_signal, c='y')

# 加窗海宁窗
# windown = choose_windows(2, wlen)        # 调用汉明窗
# b = a[0] * windown
# # 傅里叶变换
# fft_signal = np.fft.fft(b)  # 语音信号FFT变换
# fft_signal = abs(fft_signal)  # 取变换结果的模
# plt.plot(time, fft_signal, c='r')


# 加窗矩形窗
# windown = choose_windows(3, wlen)        # 调用汉明窗
# b = a[0] * windown
# # 傅里叶变换
# fft_signal = np.fft.fft(b)  # 语音信号FFT变换
# fft_signal = abs(fft_signal)  # 取变换结果的模
# plt.plot(time, fft_signal, c='g')

# 加窗三角窗
# windown = choose_windows(4, wlen)        # 调用汉明窗
# b = a[0] * windown
# # 傅里叶变换
# fft_signal = np.fft.fft(b)  # 语音信号FFT变换
# fft_signal = abs(fft_signal)  # 取变换结果的模
# plt.plot(time, fft_signal, c='k')

# 短时能量
# c = np.square(b)
plt.legend()
plt.grid()
plt.show()


