import cereal.messaging as messaging

from opendbc.can.packer import CANPacker
from opendbc.can.parser import CANParser
from openpilot.common.params import Params
from openpilot.selfdrive.boardd.boardd_api_impl import can_list_to_can_capnp
from openpilot.selfdrive.car import crc8_pedal
from openpilot.tools.sim.lib.common import SimulatorState
from panda.python import Panda
import math




# 多目标 FMCW 雷达仿真（支持 N 个目标）+ 全流程打印
import numpy as np
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

# ========================== 距离FFT：多峰检测（首个 Chirp） ==========================
ifft_data = IFx[0:Nr] * np.hanning(Nr)
range_fft = np.fft.rfft(ifft_data)                 # 只取正频谱
range_amp = 20*np.log10(np.abs(range_fft) + 1e-12)
fr = np.fft.rfftfreq(Nr, 1/Fs)
range_axis = fr * c / (2*slope)

valid_r = (range_axis > 0) & (range_axis < maxR)
ra = range_amp[valid_r]
rx = range_axis[valid_r]

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

peaks_2d = find_top_k_peaks_2d(RD, k=len(targets), suppr_v=2, suppr_r=6)#

detections = []#
for (vi, ri, val) in peaks_2d:#
    detections.append({#
        "R": range_axis[valid_r][ri],#
        "V": vel_axis[valid_v][vi],#
        "P_dB": 20*np.log10(val + 1e-12)#
    })#

# print("=== 2D FFT（Range-Doppler）检测结果 ===")
# if len(detections) == 0:
#     print("未检测到有效目标\n")
# else:
#     for k, det in enumerate(detections, 1):
#         print(f"[检测峰 {k}] 距离 = {det['R']:.2f} m, 速度 = {det['V']:.2f} m/s, 功率 = {det['P_dB']:.2f} dB")
#
#     # 与真值做最近邻匹配（使用 t=0 的 r0, v0）
#     truths = [{"R": tg["r0"], "V": tg["v0"]} for tg in targets]
#     used = set()
#     print("\n--- 与真值匹配及误差 ---")
#     for i, gt in enumerate(truths, 1):
#         best_j, best_d = None, None
#         for j, det in enumerate(detections):
#             if j in used:
#                 continue
#             d = np.hypot(det["R"] - gt["R"], det["V"] - gt["V"])
#             if (best_d is None) or (d < best_d):
#                 best_d, best_j = d, j
#         if best_j is not None:
#             used.add(best_j)
#             det = detections[best_j]








class SimulatedCar:
  """Simulates a honda civic 2016 (panda state + can messages) to OpenPilot"""
  packer = CANPacker("honda_civic_touring_2016_can_generated")
  rpacker = CANPacker("acura_ilx_2016_nidec")

  def __init__(self):
    self.pm = messaging.PubMaster(['can', 'pandaStates'])
    self.sm = messaging.SubMaster(['carControl', 'controlsState', 'carParams'])
    self.cp = self.get_car_can_parser()
    self.idx = 0
    self.params = Params()
    self.obd_multiplexing = False

  @staticmethod
  def get_car_can_parser():
    dbc_f = 'honda_civic_touring_2016_can_generated'
    checks = [
      (0xe4, 100),
      (0x1fa, 50),
      (0x200, 50),
    ]
    return CANParser(dbc_f, checks, 0)

  def send_can_messages(self, simulator_state: SimulatorState):
    if not simulator_state.valid:
      return
    # print(f'simulator_state.valid:{simulator_state.valid}')
    msg = []

    # *** powertrain bus ***

    speed = simulator_state.speed * 3.6 # convert m/s to kph
    msg.append(self.packer.make_can_msg("ENGINE_DATA", 0, {"XMISSION_SPEED": speed}))
    msg.append(self.packer.make_can_msg("WHEEL_SPEEDS", 0, {
      "WHEEL_SPEED_FL": speed,
      "WHEEL_SPEED_FR": speed,
      "WHEEL_SPEED_RL": speed,
      "WHEEL_SPEED_RR": speed
    }))

    msg.append(self.packer.make_can_msg("SCM_BUTTONS", 0, {"CRUISE_BUTTONS": simulator_state.cruise_button}))

    values = {
      "COUNTER_PEDAL": self.idx & 0xF,
      "INTERCEPTOR_GAS": simulator_state.user_gas * 2**12,
      "INTERCEPTOR_GAS2": simulator_state.user_gas * 2**12,
    }
    checksum = crc8_pedal(self.packer.make_can_msg("GAS_SENSOR", 0, values)[2][:-1])
    values["CHECKSUM_PEDAL"] = checksum
    msg.append(self.packer.make_can_msg("GAS_SENSOR", 0, values))

    msg.append(self.packer.make_can_msg("GEARBOX", 0, {"GEAR": 4, "GEAR_SHIFTER": 8}))
    msg.append(self.packer.make_can_msg("GAS_PEDAL_2", 0, {}))
    msg.append(self.packer.make_can_msg("SEATBELT_STATUS", 0, {"SEATBELT_DRIVER_LATCHED": 1}))
    msg.append(self.packer.make_can_msg("STEER_STATUS", 0, {"STEER_TORQUE_SENSOR": simulator_state.user_torque}))
    msg.append(self.packer.make_can_msg("STEERING_SENSORS", 0, {"STEER_ANGLE": simulator_state.steering_angle}))
    msg.append(self.packer.make_can_msg("VSA_STATUS", 0, {}))
    msg.append(self.packer.make_can_msg("STANDSTILL", 0, {"WHEELS_MOVING": 1 if simulator_state.speed >= 1.0 else 0}))
    msg.append(self.packer.make_can_msg("STEER_MOTOR_TORQUE", 0, {}))
    msg.append(self.packer.make_can_msg("EPB_STATUS", 0, {}))
    msg.append(self.packer.make_can_msg("DOORS_STATUS", 0, {}))
    msg.append(self.packer.make_can_msg("CRUISE_PARAMS", 0, {}))
    msg.append(self.packer.make_can_msg("CRUISE", 0, {}))
    msg.append(self.packer.make_can_msg("SCM_FEEDBACK", 0,
                                    {
                                      "MAIN_ON": 1,
                                      "LEFT_BLINKER": simulator_state.left_blinker,
                                      "RIGHT_BLINKER": simulator_state.right_blinker
                                    }))
    msg.append(self.packer.make_can_msg("POWERTRAIN_DATA", 0,
                                    {
                                    "ACC_STATUS": int(simulator_state.is_engaged),
                                    "PEDAL_GAS": simulator_state.user_gas,
                                    "BRAKE_PRESSED": simulator_state.user_brake > 0
                                    }))
    msg.append(self.packer.make_can_msg("HUD_SETTING", 0, {}))
    msg.append(self.packer.make_can_msg("CAR_SPEED", 0, {}))

    # *** cam bus ***
    msg.append(self.packer.make_can_msg("STEERING_CONTROL", 2, {}))
    msg.append(self.packer.make_can_msg("ACC_HUD", 2, {}))
    msg.append(self.packer.make_can_msg("LKAS_HUD", 2, {}))
    msg.append(self.packer.make_can_msg("BRAKE_COMMAND", 2, {}))

    # # *** radar bus ***
    # if self.idx % 5 == 0:
    #   msg.append(self.rpacker.make_can_msg("RADAR_DIAGNOSTIC", 1, {"RADAR_STATE": 0x79}))
    #   for i in range(16):
    #     msg.append(self.rpacker.make_can_msg("TRACK_%d" % i, 1, {
    #         "LONG_DIST": 215.5,  # 距离
    #          "LAT_DIST": -30,  # 横向偏移
    #          "REL_SPEED": -30,  # 相对速度
    #       }))

    if self.idx % 5 == 0:
      # radar 状态消息
      msg.append(self.rpacker.make_can_msg("RADAR_DIAGNOSTIC", 1, {"RADAR_STATE": 0x79}))

      # 最多 16 个 track
      max_tracks = 16
      for i in range(max_tracks):
        if i < len(detections):
          det = detections[i]
          msg.append(self.rpacker.make_can_msg("TRACK_%d" % i, 1, {
            "LONG_DIST": float(det["R"]),  # 距离
            "LAT_DIST": 0.0,  # 横向位置（这里先设为0）
            "REL_SPEED": float(det["V"]),  # 相对速度
          }))
        else:
          # 没有目标时填充无效数据
          msg.append(self.rpacker.make_can_msg("TRACK_%d" % i, 1, {
            "LONG_DIST": 255.5,  # 无效距离
            "LAT_DIST": 0.0,
            "REL_SPEED": 0.0,
          }))


    self.pm.send('can', can_list_to_can_capnp(msg))

  def send_panda_state(self, simulator_state):
    self.sm.update(0)

    if self.params.get_bool("ObdMultiplexingEnabled") != self.obd_multiplexing:
      self.obd_multiplexing = not self.obd_multiplexing
      self.params.put_bool("ObdMultiplexingChanged", True)

    dat = messaging.new_message('pandaStates', 1)
    dat.valid = True
    dat.pandaStates[0] = {
      'ignitionLine': simulator_state.ignition,
      'pandaType': "blackPanda",
      'controlsAllowed': True,
      'safetyModel': 'hondaNidec',
      'alternativeExperience': self.sm["carParams"].alternativeExperience,
      'safetyParam': Panda.FLAG_HONDA_GAS_INTERCEPTOR
    }
    self.pm.send('pandaStates', dat)

  def update(self, simulator_state: SimulatorState):
    self.send_can_messages(simulator_state)

    if self.idx % 50 == 0: # only send panda states at 2hz
      self.send_panda_state(simulator_state)

    self.idx += 1
