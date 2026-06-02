#模拟快速接近的目标
import cereal.messaging as messaging

from opendbc.can.packer import CANPacker
from opendbc.can.parser import CANParser
from openpilot.common.params import Params
from openpilot.selfdrive.boardd.boardd_api_impl import can_list_to_can_capnp
from openpilot.selfdrive.car import crc8_pedal
from openpilot.tools.sim.lib.common import SimulatorState
from panda.python import Panda
import math
import numpy as np
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
    # 雷达参数初始化
    self.maxR = 200
    self.rangeRes = 1
    self.maxV = 70
    self.fc = 77e9
    self.c = 3e8
    self.dt = 0.05 # 模拟时间步 (s)
    self.a_brake = -8 # 急刹加速度 (m/s²)
    self.r_max = 100 # 初始远距离 (m)
    self.r_min = 10 # 目标近距离 (m)
    self.v0_init = -20 # 初始相对速度 (m/s, 负为接近)
    self.targets = [
      {"r": self.r_max, "v": self.v0_init, "amp": 1.00}, # 动态目标
    ]
    # 衍生参数
    self.B = self.c / (2 * self.rangeRes)
    self.Tchirp = 5.5 * 2 * self.maxR / self.c
    self.endle_time = 6.3e-6
    self.slope = self.B / self.Tchirp
    self.f_IFmax = (self.slope * 2 * self.maxR) / self.c
    self.Nd = 128
    self.Nr = 1024
    self.Fs = self.Nr / self.Tchirp
    self.PRI = self.Tchirp + self.endle_time
    self.vres = (self.c / self.fc) / (2 * self.Nd * self.PRI)
    # 预计算时间轴等（但在update中重新计算动态部分）
    self.t = np.linspace(0, self.Nd * self.Tchirp, self.Nr * self.Nd, endpoint=False)
  @staticmethod
  def get_car_can_parser():
    dbc_f = 'honda_civic_touring_2016_can_generated'
    checks = [
      (0xe4, 100),
      (0x1fa, 50),
      (0x200, 50),
    ]
    return CANParser(dbc_f, checks, 0)
  def compute_radar(self):
    # 更新第一个目标：物理急刹模拟
    self.targets[0]["r"] += self.targets[0]["v"] * self.dt + 0.5 * self.a_brake * self.dt**2
    self.targets[0]["v"] += self.a_brake * self.dt
    # 如果距离 <= r_min，重置
    if self.targets[0]["r"] <= self.r_min:
      self.targets[0]["r"] = self.r_max
      self.targets[0]["v"] = self.v0_init
    print(f"Updated target 1 r: {self.targets[0]['r']:.2f} m, v: {self.targets[0]['v']:.2f} m/s (idx: {self.idx})")
    # 发射信号
    angle_tx = self.fc * self.t + 0.5 * self.slope * self.t**2
    Tx = np.cos(2 * np.pi * angle_tx)
    # 多目标回波 & IF
    Rx = np.zeros_like(self.t)
    IFx = np.zeros_like(self.t)
    for i, tg in enumerate(self.targets):
      if i == 0:
        r0, v0 = tg["r"], tg["v"] # 使用动态 r, v
      else:
        r0, v0 = tg["r0"], tg["v0"] # 其他固定
      r_dyn = r0 + v0 * self.t
      td = 2 * r_dyn / self.c
      angle_rx = self.fc * (self.t - td) + 0.5 * self.slope * (self.t - td)**2
      Rx += tg["amp"] * np.cos(2 * np.pi * angle_rx)
      IFx += tg["amp"] * np.cos(2 * np.pi * (angle_tx - angle_rx))
    # Range-Doppler 处理
    IF_mat = IFx.reshape(self.Nd, self.Nr)
    win_r = np.hanning(self.Nr)
    win_d = np.hanning(self.Nd)
    Xr = np.fft.rfft(IF_mat * win_r[np.newaxis, :], n=self.Nr, axis=1)
    fr = np.fft.rfftfreq(self.Nr, d=1/self.Fs)
    range_axis = fr * self.c / (2 * self.slope)
    Xd = np.fft.fftshift(np.fft.fft(Xr * win_d[:, np.newaxis], n=self.Nd, axis=0), axes=0)
    fd = np.fft.fftshift(np.fft.fftfreq(self.Nd, d=self.PRI))
    vel_axis = fd * self.c / (2 * self.fc)
    valid_r = (range_axis >= 0) & (range_axis <= self.maxR)
    valid_v = (vel_axis >= -self.maxV) & (vel_axis <= self.maxV)
    RD = np.abs(Xd[np.ix_(valid_v, valid_r)])
    RD_db = 20 * np.log10(RD + 1e-12)
    # 峰值检测
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
    peaks_2d = find_top_k_peaks_2d(RD, k=len(self.targets), suppr_v=2, suppr_r=6)
    detections = []
    for (vi, ri, val) in peaks_2d:
      detections.append({
        "R": range_axis[valid_r][ri],
        "V": vel_axis[valid_v][vi],
        "P_dB": 20 * np.log10(val + 1e-12)
      })
    return detections
  def send_can_messages(self, simulator_state: SimulatorState):
    if not simulator_state.valid:
      return
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
    # *** radar bus ***
    if self.idx % 5 == 0:
      detections = self.compute_radar() # 动态计算雷达
      msg.append(self.rpacker.make_can_msg("RADAR_DIAGNOSTIC", 1, {"RADAR_STATE": 0x79}))
      for i in range(16):
        if i < len(detections):
          det = detections[i]
          msg.append(self.rpacker.make_can_msg("TRACK_%d" % i, 1, {
              "LONG_DIST": det["R"],
              "LAT_DIST": 0.0,
              "REL_SPEED": det["V"],
          }))
        else:
          msg.append(self.rpacker.make_can_msg("TRACK_%d" % i, 1, {
              "LONG_DIST": 255.5,
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
