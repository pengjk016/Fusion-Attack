# import numpy as np
# import matplotlib.pyplot as plt
# from numpy import fft
# from mpl_toolkits.mplot3d import Axes3D
#
# # 雷达参数设置
# maxR = 200
# rangeRes = 1
# maxV = 70
# fc = 77e9
# c = 3e8
# r0 = 100
# v0 = 70
#
# B = c/(2*rangeRes)
# Tchirp = 5.5*2*maxR/c
# endle_time = 6.3e-6
# slope = B/Tchirp
# f_IFmax = (slope*2*maxR)/c
# f_IF = (slope*2*r0)/c
#
# Nd = 128
# Nr = 1024
# vres = (c/fc)/(2*Nd*(Tchirp+endle_time))
# Fs = Nr/Tchirp
#
# # 时间采样
# t = np.linspace(0, Nd*Tchirp, Nr*Nd)  # 发射和接收信号的采样时间
# angle_freq = fc*t + (slope*t*t)/2  # 角频率
# freq = fc + slope*t  # 频率
# Tx = np.cos(2*np.pi*angle_freq)  # 发射波形
#
# # 绘制发射信号
# plt.subplot(4,2,1)
# plt.plot(t[0:1024], Tx[0:1024])
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.title('Tx Signal')
#
# # 绘制发射信号频率
# plt.subplot(4,2,3)
# plt.plot(t[0:1024], freq[0:1024])
# plt.xlabel('Time')
# plt.ylabel('Frequency')
# plt.title('Tx F-T')
#
# # 目标距离随时间变化（运动目标）
# r0 = r0 + v0*t
#
# # 接收信号参数
# td = 2*r0/c  # 延迟时间
# Rx = np.cos(2*np.pi*(fc*(t-td) + (slope*(t-td)**2)/2))  # 接收波形
#
# # 绘制接收信号
# plt.subplot(4,2,2)
# plt.plot(t[0:1024], Rx[0:1024])
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.title('Rx Signal')
#
# # 绘制接收信号频率
# plt.subplot(4,2,4)  # 修正：原代码此处重复使用了subplot(4,2,3)，导致图形覆盖
# plt.plot(t[0:1024]+td[0:1024], freq[0:1024])  # 原变量名freqRx与定义不符，修正为freq
# plt.xlabel('Time')
# plt.ylabel('Frequency')
# plt.title('Chirp F-T')
#
# # 中频信号(IF)
# IF_angle_freq = fc*t + (slope*t*t)/2 - (fc*(t-td) + (slope*(t-td)**2)/2)
# freqIF = slope*td
# IFx = np.cos(2*np.pi*IF_angle_freq)  # 简化中频信号计算
#
# # 绘制中频信号
# plt.subplot(4,2,5)
# plt.plot(t[0:1024], IFx[0:1024])
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.title('IFx Signal')
#
# # 距离FFT
# doppler = 10*np.log10(np.abs(np.fft.fft(IFx[0:1024])))
# frequency = np.fft.fftfreq(1024, 1/Fs)
# range_ = frequency*c/(2*slope)  # 避免变量名与内置函数冲突，将range改为range_
# plt.subplot(4,2,6)
# plt.plot(range_[0:512], doppler[0:512])
# plt.xlabel('Distance')
# plt.ylabel('Amplitude (dB)')
# plt.title('Range FFT')
#
# # 频谱图（修正警告：将NFFT作为关键字参数传递）
# plt.subplot(4,2,7)
# # 原代码：plt.specgram(IFx,1024,Fs)
# plt.specgram(IFx, NFFT=1024, Fs=Fs)  # 关键修改：使用NFFT=1024替代位置参数
# plt.xlabel('Time')
# plt.ylabel('Frequency')
# plt.title('Spectrogram')
#
# plt.tight_layout(pad=3, w_pad=0.05, h_pad=0.05)
# plt.show()
#
# # 速度维处理
# chirpamp = []
# for chirpnum in range(1, Nd+1):
#     # 提取每个chirp的数据（修正索引计算）
#     start_idx = (chirpnum-1)*Nr
#     end_idx = chirpnum*Nr
#     chirpamp.append(np.mean(IFx[start_idx:end_idx]))  # 原代码只取单个点，改为取均值更合理
#
# # 速度FFT
# doppler_fft = 10*np.log10(np.abs(np.fft.fft(chirpamp)))
# FFTfrequency = np.fft.fftfreq(Nd, 1/Fs)
# velocity = (FFTfrequency * c) / (2 * fc)  # 修正速度计算公式（基于多普勒频移）
#
# plt.figure()
# plt.plot(velocity[:int(Nd/2)], doppler_fft[:int(Nd/2)])
# plt.xlabel('Velocity (m/s)')
# plt.ylabel('Amplitude (dB)')
# plt.title('Doppler FFT')
# plt.show()
#
# # 2D FFT (距离-速度谱)
# mat2D = np.zeros((Nd, Nr))
# for i in range(Nd):
#     mat2D[i, :] = IFx[i*Nr : (i+1)*Nr]
#
# # 二维FFT
# Z_fft2 = abs(np.fft.fft2(mat2D))
# # 取前半部分（消除对称冗余）
# Data_fft2 = Z_fft2[:int(Nd/2), :int(Nr/2)]
#
# plt.figure()
# plt.imshow(Data_fft2, aspect='auto', cmap='jet')
# plt.xlabel("Range Bin")
# plt.ylabel("Velocity Bin")
# plt.title('Range-Velocity 2D FFT')
# plt.colorbar(label='Amplitude')
# plt.show()


#
# # 单个目标增加状态打印语句
# import numpy as np
# import matplotlib.pyplot as plt
# from numpy import fft
# from mpl_toolkits.mplot3d import Axes3D
#
# # 雷达参数设置
# maxR = 200
# rangeRes = 1
# maxV = 70
# fc = 77e9
# c = 3e8
# r0 = 111  # 初始目标距离（设定值）
# v0 = 20  # 初始目标速度（设定值）
#
# # -------------------------- 新增：打印初始设定的目标距离和速度 --------------------------
# print(f"=== 初始设定参数 ===")
# print(f"初始目标距离 r0: {r0} m")
# print(f"初始目标速度 v0: {v0} m/s")
# print(f"雷达最大探测距离 maxR: {maxR} m")
# print(f"雷达最大探测速度 maxV: {maxV} m/s\n")
#
# # 衍生雷达参数计算
# B = c/(2*rangeRes)
# Tchirp = 5.5*2*maxR/c
# endle_time = 6.3e-6
# slope = B/Tchirp
# f_IFmax = (slope*2*maxR)/c
# f_IF = (slope*2*r0)/c
#
# Nd = 128  # 每帧Chirp数量
# Nr = 1024 # 每个Chirp的采样点数
# vres = (c/fc)/(2*Nd*(Tchirp+endle_time))  # 速度分辨率
# Fs = Nr/Tchirp  # 采样频率
#
# # 时间采样（覆盖所有Chirp的总时间）
# t = np.linspace(0, Nd*Tchirp, Nr*Nd)  # 发射和接收信号的采样时间
# angle_freq = fc*t + (slope*t*t)/2  # 角频率（用于生成发射信号）
# freq = fc + slope*t  # 发射信号的实时频率
# Tx = np.cos(2*np.pi*angle_freq)  # 发射波形
#
# # 绘制发射信号
# plt.subplot(4,2,1)
# plt.plot(t[0:1024], Tx[0:1024])
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('Tx Signal (1st Chirp)')
#
# # 绘制发射信号频率
# plt.subplot(4,2,3)
# plt.plot(t[0:1024], freq[0:1024]/1e9)  # 转换为GHz，便于显示
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (GHz)')
# plt.title('Tx Signal Frequency-Time')
#
# # 目标距离随时间变化（运动目标：距离=初始距离+速度×时间）
# r_dynamic = r0 + v0*t  # 动态更新的实时目标距离
# v_dynamic = np.ones_like(t) * v0  # 目标速度恒定（此处设定为匀速）
#
# # -------------------------- 新增：打印动态更新后的目标距离（关键时间点） --------------------------
# # 选择3个关键时间点：初始时刻、中间时刻、最后时刻
# t_points = [0, int(len(t)/2), -1]
# print(f"=== 动态更新的目标距离（匀速运动） ===")
# for idx in t_points:
#     print(f"时间 t = {t[idx]:.6f} s 时，目标距离 = {r_dynamic[idx]:.2f} m，目标速度 = {v_dynamic[idx]:.2f} m/s")
# print()
#
# # 接收信号参数（考虑目标距离延迟）
# td = 2 * r_dynamic / c  # 信号往返目标的延迟时间（随距离动态变化）
# # 接收波形（修正延迟后的发射信号）
# Rx = np.cos(2*np.pi*(fc*(t-td) + (slope*(t-td)**2)/2))
#
# # 绘制接收信号
# plt.subplot(4,2,2)
# plt.plot(t[0:1024], Rx[0:1024])
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('Rx Signal (1st Chirp)')
#
# # 绘制接收信号频率（修正时间延迟）
# plt.subplot(4,2,4)
# plt.plot(t[0:1024]+td[0:1024], freq[0:1024]/1e9)  # 时间轴叠加延迟，频率轴转GHz
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (GHz)')
# plt.title('Rx Signal Frequency-Time')
#
# # 中频信号(IF)：发射信号与接收信号的差频（包含距离和速度信息）
# IF_angle_freq = fc*t + (slope*t*t)/2 - (fc*(t-td) + (slope*(t-td)**2)/2)
# IFx = np.cos(2*np.pi*IF_angle_freq)  # 简化后的中频信号
#
# # 绘制中频信号
# plt.subplot(4,2,5)
# plt.plot(t[0:1024], IFx[0:1024])
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('IF Signal (1st Chirp)')
#
# # -------------------------- 距离FFT：计算并打印检测到的目标距离 --------------------------
# # 对第一个Chirp的中频信号做距离FFT
# ifft_data = IFx[0:Nr]  # 取第一个Chirp的数据（Nr=1024点）
# range_fft = np.fft.fft(ifft_data)
# range_amp = 10*np.log10(np.abs(range_fft))  # 幅度转dB
# frequency = np.fft.fftfreq(Nr, 1/Fs)  # FFT频率轴
# range_axis = frequency * c / (2*slope)  # 频率→实际距离（核心换算）
#
# # 找到距离FFT的峰值（对应检测到的目标距离）
# valid_range_idx = np.where((range_axis > 0) & (range_axis < maxR))[0]  # 过滤无效距离（0~maxR）
# if len(valid_range_idx) > 0:
#     peak_range_idx = valid_range_idx[np.argmax(range_amp[valid_range_idx])]
#     detected_range = range_axis[peak_range_idx]
# else:
#     detected_range = -1  # 未检测到目标
#
# # 绘制距离FFT结果
# plt.subplot(4,2,6)
# plt.plot(range_axis[0:int(Nr/2)], range_amp[0:int(Nr/2)])  # 取前半段（正距离）
# plt.scatter(detected_range, range_amp[peak_range_idx], color='red', s=50, label=f'Detected: {detected_range:.2f}m')
# plt.xlabel('Distance (m)')
# plt.ylabel('Amplitude (dB)')
# plt.title('Range FFT Result')
# plt.legend()
#
# # -------------------------- 新增：打印距离FFT检测结果 --------------------------
# print(f"=== 距离FFT检测结果 ===")
# if detected_range > 0:
#     print(f"检测到的目标距离: {detected_range:.2f} m")
#     print(f"设定的实时目标距离（第一个Chirp时刻）: {r_dynamic[0]:.2f} m")
#     print(f"距离检测误差: {abs(detected_range - r_dynamic[0]):.2f} m\n")
# else:
#     print("未检测到有效目标距离\n")
#
# # 频谱图（中频信号的时频分析）
# plt.subplot(4,2,7)
# plt.specgram(IFx, NFFT=1024, Fs=Fs, cmap='jet')
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (Hz)')
# plt.title('IF Signal Spectrogram')
#
# plt.tight_layout(pad=3, w_pad=0.05, h_pad=0.05)
# plt.show()
#
# # -------------------------- 速度FFT：计算并打印检测到的目标速度 --------------------------
# # 提取每个Chirp的中频信号幅度（用于速度FFT）
# chirp_amps = []
# for chirp_idx in range(Nd):
#     start = chirp_idx * Nr
#     end = (chirp_idx + 1) * Nr
#     chirp_amp = np.mean(IFx[start:end])  # 取每个Chirp的平均幅度（简化）
#     chirp_amps.append(chirp_amp)
#
# # 速度FFT计算
# vel_fft = np.fft.fft(chirp_amps)
# vel_amp = 10*np.log10(np.abs(vel_fft))  # 幅度转dB
# vel_freq = np.fft.fftfreq(Nd, Tchirp + endle_time)  # 多普勒频移轴（Chirp间隔为Tchirp+endle_time）
# vel_axis = vel_freq * c / (2*fc)  # 频移→实际速度（核心换算）
#
# # 找到速度FFT的峰值（对应检测到的目标速度）
# valid_vel_idx = np.where(np.abs(vel_axis) < maxV)[0]  # 过滤无效速度（-maxV~maxV）
# if len(valid_vel_idx) > 0:
#     peak_vel_idx = valid_vel_idx[np.argmax(vel_amp[valid_vel_idx])]
#     detected_vel = vel_axis[peak_vel_idx]
# else:
#     detected_vel = -1  # 未检测到目标
#
# # 绘制速度FFT结果
# plt.figure()
# plt.plot(vel_axis[:int(Nd/2)], vel_amp[:int(Nd/2)])  # 取前半段（正速度，远离雷达）
# plt.scatter(detected_vel, vel_amp[peak_vel_idx], color='red', s=50, label=f'Detected: {detected_vel:.2f}m/s')
# plt.xlabel('Velocity (m/s)')
# plt.ylabel('Amplitude (dB)')
# plt.title('Doppler FFT Result')
# plt.legend()
# plt.show()
#
# # -------------------------- 新增：打印速度FFT检测结果 --------------------------
# print(f"=== 速度FFT检测结果 ===")
# if detected_vel > 0:
#     print(f"检测到的目标速度: {detected_vel:.2f} m/s")
#     print(f"设定的目标速度: {v0:.2f} m/s")
#     print(f"速度检测误差: {abs(detected_vel - v0):.2f} m/s\n")
# else:
#     print("未检测到有效目标速度\n")
#
# # -------------------------- 2D FFT（距离-速度谱）：标注检测峰值 --------------------------
# # 构建距离-速度数据矩阵（Nd个Chirp，每个Chirp Nr个采样点）
# range_vel_mat = np.zeros((Nd, Nr))
# for i in range(Nd):
#     range_vel_mat[i, :] = IFx[i*Nr : (i+1)*Nr]
#
# # 二维FFT计算
# range_vel_fft = abs(np.fft.fft2(range_vel_mat))
# # 取前半段（消除FFT对称冗余，只保留正距离和正速度）
# range_vel_fft = range_vel_fft[:int(Nd/2), :int(Nr/2)]
#
# # 绘制距离-速度谱
# plt.figure()
# im = plt.imshow(range_vel_fft, aspect='auto', cmap='jet',
#                 extent=[0, maxR, 0, maxV])  # 坐标轴映射为实际距离和速度
# # 标注检测到的目标位置（距离-速度峰值）
# if detected_range > 0 and detected_vel > 0:
#     plt.scatter(detected_range, detected_vel, color='white', s=100, marker='x',
#                 label=f'Target: ({detected_range:.2f}m, {detected_vel:.2f}m/s)')
# plt.xlabel("Distance (m)")
# plt.ylabel("Velocity (m/s)")
# plt.title('Range-Velocity 2D FFT Spectrum')
# plt.colorbar(im, label='Amplitude')
# plt.legend()
# plt.show()
#
# # -------------------------- 新增：打印最终距离-速度检测结果 --------------------------
# print(f"=== 最终目标检测结果（距离-速度谱） ===")
# if detected_range > 0 and detected_vel > 0:
#     print(f"目标距离: {detected_range:.2f} m")
#     print(f"目标速度: {detected_vel:.2f} m/s")
# else:
#     print("未检测到有效目标")
#
#
#
#
#
# # 多目标
# import numpy as np
# import matplotlib.pyplot as plt
# from numpy import fft
# from scipy.signal import find_peaks
# from mpl_toolkits.mplot3d import Axes3D
#
# # 多目标配置：[目标1, 目标2, ...]（初始距离m, 速度m/s, 信号强度系数）
# targets_config = [
#     (100, 70, 1.0),    # 目标1：100m, 70m/s, 强信号
#     (150, 30, 0.6),    # 目标2：150m, 30m/s, 弱信号
# ]
# num_targets = len(targets_config)
#
# # 雷达基础参数（不变）
# maxR = 200
# rangeRes = 1
# maxV = 70
# fc = 77e9
# c = 3e8
# B = c/(2*rangeRes)
# Tchirp = 5.5*2*maxR/c
# endle_time = 6.3e-6
# slope = B/Tchirp
#
# Nd = 128  # 每帧Chirp数量
# Nr = 1024 # 每个Chirp采样点数
# vres = (c/fc)/(2*Nd*(Tchirp+endle_time))
# Fs = Nr/Tchirp
#
# # 时间采样（不变）
# t = np.linspace(0, Nd*Tchirp, Nr*Nd)
#
# # 打印多目标初始设定
# print(f"=== 多目标初始设定参数 ===")
# for i, (r0, v0, power) in enumerate(targets_config, 1):
#     print(f"目标{i}: 初始距离={r0}m, 速度={v0}m/s, 信号强度系数={power}")
# print(f"雷达最大探测距离={maxR}m, 最大探测速度={maxV}m/s\n")
#
# # 多目标动态距离计算（不变）
# targets_r_dynamic = []
# targets_v = []
# for r0, v0, _ in targets_config:
#     r_dynamic = r0 + v0 * t
#     targets_r_dynamic.append(r_dynamic)
#     targets_v.append(np.ones_like(t) * v0)
#
# # 多目标回波信号叠加（不变）
# Rx = np.zeros_like(t, dtype=np.float64)
# for i in range(num_targets):
#     r_dynamic = targets_r_dynamic[i]
#     power = targets_config[i][2]
#     td = 2 * r_dynamic / c
#     target_echo = power * np.cos(2*np.pi*(fc*(t-td) + (slope*(t-td)**2)/2))
#     Rx += target_echo
# # 添加噪声
# Rx += 0.1 * (np.random.randn(*Rx.shape) + 1j * np.random.randn(*Rx.shape)).real
#
# # 发射信号生成（不变）
# angle_freq = fc*t + (slope*t*t)/2
# freq = fc + slope*t
# Tx = np.cos(2*np.pi*angle_freq)
#
# # -------------------------- 修复1：距离FFT图（判断是否有目标再添加图例） --------------------------
# plt.figure(figsize=(16, 12))
#
# # 1. 发射信号
# plt.subplot(4,2,1)
# plt.plot(t[0:Nr], Tx[0:Nr])
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('Tx Signal (1st Chirp)')
#
# # 2. 接收信号
# plt.subplot(4,2,2)
# plt.plot(t[0:Nr], Rx[0:Nr])
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('Rx Signal (1st Chirp, Multi-Target Overlay)')
#
# # 3. 发射信号频率
# plt.subplot(4,2,3)
# plt.plot(t[0:Nr], freq[0:Nr]/1e9)
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (GHz)')
# plt.title('Tx Signal Frequency-Time')
#
# # 4. 接收信号频率
# plt.subplot(4,2,4)
# td_ref = 2 * targets_r_dynamic[0][0] / c
# plt.plot(t[0:Nr]+td_ref, freq[0:Nr]/1e9)
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (GHz)')
# plt.title('Rx Signal Frequency-Time (Ref Delay)')
#
# # 5. 中频信号
# IF_angle_freq = fc*t + (slope*t*t)/2 - (fc*(t-td_ref) + (slope*(t-td_ref)**2)/2)
# IFx = np.cos(2*np.pi*IF_angle_freq)
# plt.subplot(4,2,5)
# plt.plot(t[0:Nr], IFx[0:Nr])
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('IF Signal (1st Chirp)')
#
# # 6. 距离FFT（核心修复：先判断是否有检测到的目标）
# plt.subplot(4,2,6)
# ifft_data = IFx[0:Nr]
# range_fft = np.fft.fft(ifft_data)
# range_amp = 10*np.log10(np.abs(range_fft) + 1e-6)
# frequency = np.fft.fftfreq(Nr, 1/Fs)
# range_axis = frequency * c / (2*slope)
#
# # 多峰值检测
# peak_threshold = np.mean(range_amp) + 2 * np.std(range_amp)
# valid_range_mask = (range_axis > 0) & (range_axis < maxR)
# valid_range_amp = range_amp[valid_range_mask]
# valid_range_axis = range_axis[valid_range_mask]
# peaks, _ = find_peaks(valid_range_amp, height=peak_threshold, distance=10)
# detected_ranges = valid_range_axis[peaks] if len(peaks) > 0 else []
# detected_range_amps = valid_range_amp[peaks] if len(peaks) > 0 else []
#
# # 绘制距离FFT曲线
# plt.plot(range_axis[0:int(Nr/2)], range_amp[0:int(Nr/2)], label='Range FFT Baseline')
# # 若检测到目标，添加标注和标签；否则提示无目标
# if len(detected_ranges) > 0:
#     for i, (r, amp) in enumerate(zip(detected_ranges, detected_range_amps)):
#         plt.scatter(r, amp, color=f'C{i}', s=60, label=f'Target {i+1}: {r:.2f}m')
#     plt.legend()  # 只有检测到目标时才调用legend()
# else:
#     plt.text(maxR/2, np.max(range_amp)/2, 'No Target Detected', ha='center', fontsize=10)
# plt.xlabel('Distance (m)')
# plt.ylabel('Amplitude (dB)')
# plt.title('Range FFT (Multi-Target Peaks)')
#
# # 7. 频谱图（无需图例，无警告）
# plt.subplot(4,2,7)
# plt.specgram(IFx, NFFT=1024, Fs=Fs, cmap='jet')
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (Hz)')
# plt.title('IF Signal Spectrogram')
#
# plt.tight_layout(pad=3, w_pad=0.05, h_pad=0.05)
# plt.show()
#
# # -------------------------- 修复2：速度FFT图（同理判断目标） --------------------------
# plt.figure(figsize=(10, 6))
#
# # 提取Chirp幅度（不变）
# chirp_amps = []
# for chirp_idx in range(Nd):
#     start = chirp_idx * Nr
#     end = (chirp_idx + 1) * Nr
#     chirp_amps.append(np.mean(IFx[start:end]))
#
# # 速度FFT计算
# vel_fft = np.fft.fft(chirp_amps)
# vel_amp = 10*np.log10(np.abs(vel_fft) + 1e-6)
# vel_freq = np.fft.fftfreq(Nd, Tchirp + endle_time)
# vel_axis = vel_freq * c / (2*fc)
#
# # 多峰值检测
# valid_vel_mask = np.abs(vel_axis) < maxV
# valid_vel_amp = vel_amp[valid_vel_mask]
# valid_vel_axis = vel_axis[valid_vel_mask]
# vel_peaks, _ = find_peaks(valid_vel_amp, height=peak_threshold-5, distance=5)
# # 过滤正速度（与目标设定一致）
# detected_vels = [valid_vel_axis[p] for p in vel_peaks if valid_vel_axis[p] > 0] if len(vel_peaks) > 0 else []
# detected_vel_amps = [valid_vel_amp[p] for p in vel_peaks if valid_vel_axis[p] > 0] if len(vel_peaks) > 0 else []
#
# # 绘制速度FFT曲线
# plt.plot(vel_axis[:int(Nd/2)], vel_amp[:int(Nd/2)], label='Doppler FFT Baseline')
# # 若检测到目标，添加标注和图例；否则提示
# if len(detected_vels) > 0:
#     for i, (v, amp) in enumerate(zip(detected_vels, detected_vel_amps)):
#         plt.scatter(v, amp, color=f'C{i}', s=60, label=f'Target {i+1}: {v:.2f}m/s')
#     plt.legend()
# else:
#     plt.text(maxV/2, np.max(vel_amp)/2, 'No Target Detected', ha='center', fontsize=10)
# plt.xlabel('Velocity (m/s)')
# plt.ylabel('Amplitude (dB)')
# plt.title('Doppler FFT (Multi-Target Peaks)')
# plt.show()
#
# # -------------------------- 修复3：2D距离-速度谱（同理判断目标） --------------------------
# plt.figure(figsize=(12, 8))
#
# # 构建2D矩阵并FFT（不变）
# range_vel_mat = np.zeros((Nd, Nr))
# for i in range(Nd):
#     range_vel_mat[i, :] = IFx[i*Nr : (i+1)*Nr]
# range_vel_fft = abs(np.fft.fft2(range_vel_mat))
# range_vel_fft = range_vel_fft[:int(Nd/2), :int(Nr/2)]
#
# # 绘制2D谱
# extent = [0, maxR, 0, maxV]
# im = plt.imshow(range_vel_fft, aspect='auto', cmap='jet', extent=extent)
#
# # 匹配距离-速度对并标注
# print(f"\n=== 多目标检测结果汇总 ===")
# print(f"{'目标':<4} {'设定距离':<10} {'检测距离':<10} {'距离误差':<10} {'设定速度':<10} {'检测速度':<10} {'速度误差':<10}")
# print("-"*80)
# detected_pairs = []  # 存储（检测距离，检测速度）对
# if len(detected_ranges) > 0 and len(detected_vels) > 0:
#     # 按目标数量匹配（取最少的检测数）
#     match_count = min(num_targets, len(detected_ranges), len(detected_vels))
#     for i in range(match_count):
#         set_r = targets_r_dynamic[i][0]
#         set_v = targets_config[i][1]
#         det_r = detected_ranges[i]
#         det_v = detected_vels[i]
#         r_err = abs(det_r - set_r)
#         v_err = abs(det_v - set_v)
#         detected_pairs.append((det_r, det_v))
#         # 标注目标
#         plt.scatter(det_r, det_v, color=f'C{i}', s=100, marker='x',
#                     label=f'Target {i+1}: ({det_r:.2f}m, {det_v:.2f}m/s)')
#         # 打印结果
#         print(f"{i+1:<4} {set_r:<10.2f} {det_r:<10.2f} {r_err:<10.2f} {set_v:<10.2f} {det_v:<10.2f} {v_err:<10.2f}")
#     # 有目标时添加图例
#     plt.legend()
# else:
#     # 无目标时提示
#     plt.text(maxR/2, maxV/2, 'No Target Detected', ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
#     print("未检测到任何有效目标")
#
# plt.xlabel("Distance (m)")
# plt.ylabel("Velocity (m/s)")
# plt.title('Range-Velocity 2D FFT Spectrum (Multi-Target)')
# plt.colorbar(im, label='Amplitude')
# plt.show()
#






# 多目标 FMCW 雷达仿真（支持 N 个目标）+ 全流程打印,没有对角度进行模拟
import numpy as np
import matplotlib.pyplot as plt
from numpy import fft
from mpl_toolkits.mplot3d import Axes3D  # 未直接使用，仅保留以兼容你的原始结构

# ========================== 雷达与目标参数 ==========================
maxR = 200
rangeRes = 1
maxV = 70
fc = 77e9
c = 3e8

# --- 在这里配置多个目标：r0(m), v0(m/s), amp(幅度/等效RCS权重) ---
targets = [
    {"r0": 111, "v0": 20,  "amp": 1.00},
    {"r0": 60,  "v0": -15, "amp": 0.90},
    # 继续添加即可：
    {"r0": 150, "v0": 8, "amp": 0.8},
]

print("=== 初始设定参数 ===")
print(f"目标数量: {len(targets)}")
for i, tg in enumerate(targets, 1):
    print(f"[目标 {i}] r0: {tg['r0']} m, v0: {tg['v0']} m/s, amp: {tg['amp']}")
print(f"雷达最大探测距离 maxR: {maxR} m")
print(f"雷达最大探测速度 maxV: {maxV} m/s\n")

# ========================== 衍生参数 ==========================
B = c/(2*rangeRes)
Tchirp = 5.5*2*maxR/c
endle_time = 6.3e-6  # 与你原代码保持同名
slope = B/Tchirp
f_IFmax = (slope*2*maxR)/c

Nd = 128   # 每帧 Chirp 数
Nr = 1024  # 每个 Chirp 的采样点
Fs = Nr/Tchirp
PRI = Tchirp + endle_time
vres = (c/fc)/(2*Nd*PRI)

print("=== 雷达派生参数 ===")
print(f"带宽 B: {B/1e6:.1f} MHz, 斜率: {slope/1e12:.2f} THz/s")
print(f"Tchirp: {Tchirp*1e6:.2f} us, 间歇: {endle_time*1e6:.2f} us, PRI: {PRI*1e6:.2f} us")
print(f"采样 Fs: {Fs/1e6:.2f} MHz, Nr: {Nr}, Nd: {Nd}")
print(f"速度分辨率 vres: {vres:.3f} m/s\n")

# ========================== 时间轴 & 发射信号 ==========================
t = np.linspace(0, Nd*Tchirp, Nr*Nd, endpoint=False)  # 仅在 chirp 内采样
angle_tx = fc*t + 0.5*slope*t*t
freq = fc + slope*t
Tx = np.cos(2*np.pi*angle_tx)

# ========================== 多目标回波 & IF 基带 ==========================
Rx = np.zeros_like(t)
IFx = np.zeros_like(t)

print("=== 动态更新的目标距离（匀速运动） ===")
t_points = [0, len(t)//2, -1]
for m, tg in enumerate(targets, 1):
    r_dyn = tg["r0"] + tg["v0"]*t
    v_dyn = np.ones_like(t) * tg["v0"]
    td = 2*r_dyn / c
    angle_rx = fc*(t - td) + 0.5*slope*(t - td)**2
    Rx  += tg["amp"] * np.cos(2*np.pi*angle_rx)
    IFx += tg["amp"] * np.cos(2*np.pi*(angle_tx - angle_rx))  # 只保留差频，相当于已低通

    print(f"[目标 {m}] r0={tg['r0']} m, v0={tg['v0']} m/s")
    for idx in t_points:
        print(f"  时间 t = {t[idx]:.6f} s, 距离 = {r_dyn[idx]:.2f} m, 速度 = {v_dyn[idx]:.2f} m/s")
    print()

# ========================== 基本可视化 ==========================
plt.figure(figsize=(12,10))
plt.subplot(4,2,1)
plt.plot(t[:1024], Tx[:1024])
plt.xlabel('Time (s)'); plt.ylabel('Amplitude')
plt.title('Tx Signal (1st Chirp)')

plt.subplot(4,2,3)
plt.plot(t[:1024], freq[:1024]/1e9)
plt.xlabel('Time (s)'); plt.ylabel('Frequency (GHz)')
plt.title('Tx Signal Frequency-Time')

plt.subplot(4,2,2)
plt.plot(t[:1024], Rx[:1024])
plt.xlabel('Time (s)'); plt.ylabel('Amplitude')
plt.title('Rx Signal (Sum of Targets, 1st Chirp)')

plt.subplot(4,2,5)
plt.plot(t[:1024], IFx[:1024])
plt.xlabel('Time (s)'); plt.ylabel('Amplitude')
plt.title('IF Signal (1st Chirp)')

# 先占位，后面放 Range FFT 结果
plt.subplot(4,2,6); plt.title('Range FFT Result (1st Chirp)')
plt.subplot(4,2,7)
plt.specgram(IFx, NFFT=1024, Fs=Fs, cmap='jet')
plt.xlabel('Time (s)'); plt.ylabel('Frequency (Hz)')
plt.title('IF Signal Spectrogram')
plt.tight_layout(pad=3, w_pad=0.05, h_pad=0.05)
plt.show()

# ========================== 距离FFT：多峰检测（首个 Chirp） ==========================
ifft_data = IFx[0:Nr] * np.hanning(Nr)
range_fft = np.fft.rfft(ifft_data)                 # 只取正频谱
range_amp = 20*np.log10(np.abs(range_fft) + 1e-12)
fr = np.fft.rfftfreq(Nr, 1/Fs)
range_axis = fr * c / (2*slope)

valid_r = (range_axis > 0) & (range_axis < maxR)
ra = range_amp[valid_r]
rx = range_axis[valid_r]

# 寻找与目标数相同的最强峰（简单NMS）
def top_k_peaks_1d(y, k, suppr_bins=3):
    y = y.copy()
    peaks = []
    for _ in range(k):
        idx = np.argmax(y)
        val = y[idx]
        if not np.isfinite(val) or val <= -1e9:
            break
        peaks.append((idx, val))
        l = max(0, idx - suppr_bins); r = min(len(y), idx + suppr_bins + 1)
        y[l:r] = -1e12  # 抑制邻域
    return peaks

peaks_1d = top_k_peaks_1d(ra, k=len(targets), suppr_bins=4)

print("=== 距离FFT检测结果（首个 Chirp，多目标） ===")
if len(peaks_1d) == 0:
    print("未检测到有效目标距离\n")
else:
    for i, (idx, val) in enumerate(peaks_1d, 1):
        print(f"[距离峰 {i}] 距离 ≈ {rx[idx]:.2f} m, 幅度 = {val:.2f} dB")
    print()

plt.figure(figsize=(7,4))
plt.plot(rx, ra, label='Range FFT (1st Chirp)')
for (idx, _) in peaks_1d:
    plt.scatter(rx[idx], ra[idx], s=50, c='red')
plt.xlabel('Distance (m)'); plt.ylabel('Amplitude (dB)')
plt.title('Range FFT (Top Peaks)')
plt.grid(True); plt.legend()
plt.tight_layout()
plt.show()

# ========================== 2D FFT：Range-Doppler 处理 ==========================
IF_mat = IFx.reshape(Nd, Nr)  # [Nd, Nr]

# 加窗
win_r = np.hanning(Nr)
win_d = np.hanning(Nd)
Xr = np.fft.rfft(IF_mat * win_r[np.newaxis, :], n=Nr, axis=1)   # [Nd, Nr//2+1]
fr = np.fft.rfftfreq(Nr, d=1/Fs)
range_axis = fr * c / (2*slope)

Xd = np.fft.fftshift(np.fft.fft(Xr * win_d[:, np.newaxis], n=Nd, axis=0), axes=0)  # [Nd, Nr//2+1]
fd = np.fft.fftshift(np.fft.fftfreq(Nd, d=PRI))
vel_axis = fd * c / (2*fc)

valid_r = (range_axis >= 0) & (range_axis <= maxR)
valid_v = (vel_axis >= -maxV) & (vel_axis <= maxV)
RD = np.abs(Xd[np.ix_(valid_v, valid_r)])  # [Nv, Nr_valid]
RD_db = 20*np.log10(RD + 1e-12)

# 可视化 Range-Doppler
plt.figure(figsize=(7.5,5.5))
extent = [range_axis[valid_r][0], range_axis[valid_r][-1],
          vel_axis[valid_v][0],   vel_axis[valid_v][-1]]
im = plt.imshow(RD_db, aspect='auto', origin='lower', extent=extent, cmap='jet')
plt.xlabel('Distance (m)'); plt.ylabel('Velocity (m/s)')
plt.title('Range-Doppler Map (dB)')
plt.colorbar(im, label='dB')
plt.tight_layout()
plt.show()

# ========================== 2D 峰值检测（取 N 个目标） ==========================
def find_top_k_peaks_2d(mat, k, suppr_v=2, suppr_r=6):
    work = mat.copy()
    peaks = []
    for _ in range(k):
        idx = np.argmax(work)
        if not np.isfinite(work.flat[idx]):
            break
        v_i, r_i = np.unravel_index(idx, work.shape)
        val = work[v_i, r_i]
        peaks.append((v_i, r_i, val))
        v0 = max(0, v_i - suppr_v); v1 = min(work.shape[0], v_i + suppr_v + 1)
        r0 = max(0, r_i - suppr_r); r1 = min(work.shape[1], r_i + suppr_r + 1)
        work[v0:v1, r0:r1] = -np.inf
    return peaks

peaks_2d = find_top_k_peaks_2d(RD, k=len(targets), suppr_v=2, suppr_r=6)

detections = []
for (vi, ri, val) in peaks_2d:
    detections.append({
        "R": range_axis[valid_r][ri],
        "V": vel_axis[valid_v][vi],
        "P_dB": 20*np.log10(val + 1e-12)
    })

print("=== 2D FFT（Range-Doppler）检测结果 ===")
if len(detections) == 0:
    print("未检测到有效目标\n")
else:
    for k, det in enumerate(detections, 1):
        print(f"[检测峰 {k}] 距离 = {det['R']:.2f} m, 速度 = {det['V']:.2f} m/s, 功率 = {det['P_dB']:.2f} dB")

    # 与真值做最近邻匹配（使用 t=0 的 r0, v0）
    truths = [{"R": tg["r0"], "V": tg["v0"]} for tg in targets]
    used = set()
    print("\n--- 与真值匹配及误差 ---")
    for i, gt in enumerate(truths, 1):
        best_j, best_d = None, None
        for j, det in enumerate(detections):
            if j in used:
                continue
            d = np.hypot(det["R"] - gt["R"], det["V"] - gt["V"])
            if (best_d is None) or (d < best_d):
                best_d, best_j = d, j
        if best_j is not None:
            used.add(best_j)
            det = detections[best_j]
            print(f"[目标 {i}] 真值: R={gt['R']:.2f} m, V={gt['V']:.2f} m/s | "
                  f"检测: R={det['R']:.2f} m, V={det['V']:.2f} m/s | "
                  f"误差: ΔR={abs(det['R']-gt['R']):.2f} m, ΔV={abs(det['V']-gt['V']):.2f} m/s")
    print()

# 在 Range-Doppler 图上标注检测峰
plt.figure(figsize=(7.5,5.5))
im = plt.imshow(RD_db, aspect='auto', origin='lower', extent=extent, cmap='jet')
for det in detections:
    plt.scatter(det["R"], det["V"], s=80, c='white', marker='x')
plt.xlabel('Distance (m)'); plt.ylabel('Velocity (m/s)')
plt.title('Range-Doppler Map with Detections')
plt.colorbar(im, label='dB')
plt.tight_layout()
plt.show()








# 多目标 FMCW 雷达仿真（支持 N 个目标）+ 全流程打印,对角度进行模拟
# --- 目标配置：r0(m), v0(m/s), amp, ang(°) ---
targets = [
    {"r0": 111, "v0": 20,  "amp": 1.00, "ang":  5},
    {"r0":  60, "v0": -15, "amp": 0.90, "ang": -3},
    {"r0": 150, "v0": 8,   "amp": 0.80, "ang":  0},
]
# ========================== 阵列参数（新增） ==========================
M = 8                               # 阵列天线数（>=2）
lamb = c / fc                       # 波长
d = lamb / 2                        # 元间距（常用 λ/2）
Na = 128                            # 角度FFT长度（可 > M 进行零填充）
assert M >= 2, "角度分辨至少需要2根天线"

# ========================== 多天线 IF 基带（复数）生成（替代原 IFx） ==========================
IFx_ant = np.zeros((Nd*Nr, M), dtype=np.complex128)  # [Nd*Nr, M] 复数基带
Rx = np.zeros_like(t)                                 # 仅保留单通道实数 Rx 可选

print("=== 动态更新的目标距离（匀速运动） ===")
t_points = [0, len(t)//2, -1]
for m_idx, tg in enumerate(targets, 1):
    r_dyn = tg["r0"] + tg["v0"]*t
    v_dyn = np.ones_like(t) * tg["v0"]
    td = 2*r_dyn / c
    angle_rx = fc*(t - td) + 0.5*slope*(t - td)**2

    # 复数基带（混频后差频）——用复指数模型，便于叠加相位
    base_bb = tg["amp"] * np.exp(1j * 2*np.pi * (angle_tx - angle_rx))

    # 按角度为每根天线加相位（ULA 线阵）
    theta = np.deg2rad(tg.get("ang", 0.0))
    phase_per_elem = 2*np.pi * d * np.sin(theta) / lamb  # 每个阵元相位步进
    for m in range(M):
        IFx_ant[:, m] += base_bb * np.exp(1j * phase_per_elem * m)

    # 可选：保留一个“总和”的实数 Rx 仅用于观测
    Rx  += tg["amp"] * np.cos(2*np.pi*angle_rx)

    print(f"[目标 {m_idx}] r0={tg['r0']} m, v0={tg['v0']} m/s, ang={tg['ang']}°")
    for idx in t_points:
        print(f"  时间 t = {t[idx]:.6f} s, 距离 = {r_dyn[idx]:.2f} m, 速度 = {v_dyn[idx]:.2f} m/s")
    print()
# ========================== R-D-θ 立方体 ==========================
# 先把 IFx_ant reshape 为 [Nd, Nr, M]
IF_cube = IFx_ant.reshape(Nd, Nr, M)

# 窗函数
win_r = np.hanning(Nr)
win_d = np.hanning(Nd)

# 距离 FFT（对 Nr 维，取正频，逐天线）
Xr = np.fft.rfft(IF_cube * win_r[np.newaxis, :, np.newaxis], n=Nr, axis=1)  # [Nd, Nr//2+1, M]
fr = np.fft.rfftfreq(Nr, d=1/Fs)
range_axis = fr * c / (2*slope)

# 多普勒 FFT（对 Nd 维）
Xd = np.fft.fftshift(np.fft.fft(Xr * win_d[:, np.newaxis, np.newaxis], n=Nd, axis=0), axes=0)  # [Nd, Nr//2+1, M]
fd = np.fft.fftshift(np.fft.fftfreq(Nd, d=PRI))
vel_axis = fd * c / (2*fc)

# 角度 FFT（对 M 维，零填充到 Na；再 fftshift）
# 先将最后一维零填充到 Na
Xp = np.zeros((Nd, Xd.shape[1], Na), dtype=np.complex128)
Xp[:, :, :M] = Xd
Xa = np.fft.fftshift(np.fft.fft(Xp, n=Na, axis=2), axes=2)  # [Nd, Nr//2+1, Na]

# 合法轴裁剪
valid_r = (range_axis >= 0) & (range_axis <= maxR)
valid_v = (vel_axis >= -maxV) & (vel_axis <= maxV)
RDA = Xa[np.ix_(np.where(valid_v)[0], np.where(valid_r)[0], np.arange(Na))]  # [Nv, Nr_valid, Na]
P_RDA = np.abs(RDA)  # 幅度谱
P_RDA_db = 20*np.log10(P_RDA + 1e-12)

# 角度轴映射（由阵列空间频率 f_s → sinθ）
# Xa 经过fftshift后，角度bin k ∈ [-Na/2, Na/2)
k = np.arange(-Na//2, Na//2)
fs_spatial = k / Na                        # 归一化空间频率 ∈ [-0.5, 0.5)
sin_theta = np.clip(fs_spatial * (lamb / d), -1.0, 1.0)
angle_axis = np.rad2deg(np.arcsin(sin_theta))  # 角度轴（度）


def topk_peaks_3d(power_cube, k, suppr_v=2, suppr_r=6, suppr_a=3):
  """在 [Nv, Nr, Na] 的立方体上做贪心Top-K峰值搜索，并在三维邻域做非极大值抑制。"""
  work = power_cube.copy()
  peaks = []
  Nv, Nr, Na = work.shape
  for _ in range(k):
    flat_idx = np.argmax(work)
    peak_val = work.flat[flat_idx]
    if not np.isfinite(peak_val) or peak_val <= -1e8:
      break
    vi, ri, ai = np.unravel_index(flat_idx, work.shape)
    peaks.append((vi, ri, ai, peak_val))
    v0, v1 = max(0, vi - suppr_v), min(Nv, vi + suppr_v + 1)
    r0, r1 = max(0, ri - suppr_r), min(Nr, ri + suppr_r + 1)
    a0, a1 = max(0, ai - suppr_a), min(Na, ai + suppr_a + 1)
    work[v0:v1, r0:r1, a0:a1] = -np.inf
  return peaks


# Top-K（用目标数或16取小）
K = min(len(targets), 16)
# 在“线性幅度”域做峰值更稳，这里用 P_RDA（若想用dB则先转回线性）
peaks3d = topk_peaks_3d(P_RDA, k=K, suppr_v=2, suppr_r=6, suppr_a=3)

detections = []
Nv = np.sum(valid_v);
Nr_valid = np.sum(valid_r)

for (vi, ri, ai, val) in peaks3d:
  R = range_axis[valid_r][ri]
  V = vel_axis[valid_v][vi]
  ang_deg = angle_axis[ai]
  detections.append({
    "R": float(R),
    "V": float(V),
    "ang_deg": float(ang_deg),
    "P_lin": float(val),
    "P_dB": 20 * np.log10(val + 1e-12),
  })

print("=== R-V-θ 3D 峰值检测结果 ===")
for i, det in enumerate(detections, 1):
  print(f"[{i}] R={det['R']:.2f} m, V={det['V']:.2f} m/s, θ={det['ang_deg']:.2f}°, P={det['P_dB']:.2f} dB")
print()
