from matplotlib import pyplot as plt
import numpy as np

N = 1000  # 采样点个数
t = np.linspace(0, 1, N)  # 得到各个点采样时间
y = 0.7 * np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)  # 构造振幅为0.7的50HZ正弦波 和 振幅为1 的120HZ的正弦波
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.title("The Original waveform")
plt.xlabel("time(/s)")
plt.ylabel("Amplitude")
plt.legend()
plt.plot(t[:100], y[:100])

# 快速傅里叶
x = np.fft.fft(y)
amp = np.arange(N)  # 频率

# print(len(x))
# print(x[:5])
# abs_y = np.abs(x)
# plt.subplot(122)
# plt.title("amplitude spectrum")     # 双边振幅谱(未归一化)
# plt.xlabel("Amplitude")
# plt.ylabel("A * N/2")
# plt.legend()
# plt.plot(amp, abs_y)
# plt.show()

# plt.subplot(122)
# angle_y = np.angle(x)
# plt.title("phase position")  # 双边相位谱(未归一化)
# plt.xlabel("Amplitude")
# plt.ylabel("phase postion")
# plt.legend()
# plt.plot(amp, angle_y)

abs_y = np.abs(x)
print(np.max(abs_y))
normalization = abs_y/np.max(abs_y)
plt.subplot(122)
plt.title("amplitude spectrum")     # 单边振幅谱(归一化)
plt.xlabel("Amplitude")
plt.ylabel("A * N/2")
plt.legend()
plt.plot(amp[range(int(N/2))], normalization[range(int(N/2))])
plt.show()

