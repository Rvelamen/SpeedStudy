import numpy as np
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


wlen = 512
inc = 128

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.xlabel('nframes')
plt.ylabel('Amplitude')
plt.title('Window funcition')
plt.plot(choose_windows(1, wlen), c='b', label='hanming')
plt.plot(choose_windows(2, wlen), c='g', label='haining')
plt.plot(choose_windows(3, wlen), c='r', label='rectangle')
plt.plot(choose_windows(4, wlen), c='k', label='triangle')
plt.legend()
plt.show()
