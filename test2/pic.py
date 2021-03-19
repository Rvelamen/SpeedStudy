from matplotlib import pyplot as plt
import numpy as np

x = np.linspace(0, 2*np.pi, 1400)                               # X的范围位于【0，2π】
print(x)
y = np.sin(np.pi * x * 500)

plt.subplot(121)
plt.plot(x[:30], y[:30], ls='-', lw=2, label='xxx', color='g')  # ls是线条的风格，lw是线条的宽度，label为标签文本

plt.subplot(122)
fft_signal2 = np.fft.fft(y)
fft_signal2 = abs(fft_signal2)
plt.plot(np.arange(len(fft_signal2)), fft_signal2, c='b', label="")
plt.legend()
plt.show()  # 展示图象
