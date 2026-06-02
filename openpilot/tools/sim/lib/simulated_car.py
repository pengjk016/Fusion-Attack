#根据相机模型输出，radar配合攻击
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
import os

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
    v0 = max(0, v_i - suppr_v);
    v1 = min(work.shape[0], v_i + suppr_v + 1)
    r0 = max(0, r_i - suppr_r);
    r1 = min(work.shape[1], r_i + suppr_r + 1)
    work[v0:v1, r0:r1] = -np.inf
  return peaks


def append_dect(surrounding_info):
  c = 3e8  # 光速 (m/s)

  # ========================== 雷达基础参数  ==========================
  fc = 1.5e9  # fc (中心频率) = 1.5 GHz
  B = 25e6  # B (带宽) = 25.00 MHz
  slope = 0.05e12  # S (调频斜率) = 0.05 MHz/us = 5e10 Hz/s
  Tchirp = 501.12e-6  # Tchirp (脉冲宽度) = 501.12 us
  Nd = 256  # Nchirps (每帧 Chirp 数) = 256

  rangeRes = 6.09  # d_res (距离分辨率) = 6.09 m
  maxR = 1558.92  # d_max (最大探测距离) = 1558.92 m
  vRes = 0.78  # v_res (速度分辨率) = 0.78 m/s
  maxV = 99.71  # v_max (最大速度) = 99.71 m/s

  # ========================== 雷达派生参数计算 ==========================
  # 1. 脉冲重复间隔 (PRI): 反推自 maxV = c / (4 * fc * PRI)
  PRI = c / (4 * fc * maxV)

  # 2. 采样率 (Fs): 反推自 maxR = Fs * c / (2 * slope)
  Fs = maxR * 2 * slope / c

  # 3. 每个 Chirp 的采样点 (Nr): Fs * Tchirp
  Nr = int(np.round(Fs * Tchirp))

  # 4. 计算 FFT 矩阵中每个格子（Bin）对应的真实物理尺寸
  range_bin_size = c / (2 * slope * Tchirp)  # 距离 Bin 大小 (约 5.98m)
  vel_bin_size = c / (2 * fc * Nd * PRI)  # 速度 Bin 大小 (约 0.78m/s)

  # 利用 rangeRes 和 vRes 动态计算峰值检测的抑制窗口
  # 物理意义：目标能量会扩散到至少 1 个分辨率大小的区域，按比例抑制可防止虚警
  suppr_r_dynamic = max(1, int(np.ceil(rangeRes / range_bin_size)))
  suppr_v_dynamic = max(1, int(np.ceil(vRes / vel_bin_size)))

  if not surrounding_info or not isinstance(surrounding_info, list):
    return []

  targets = []
  for obj in surrounding_info:
    if not isinstance(obj, dict) or 'relative_position' not in obj or 'relative_velocity' not in obj:
      continue
    pos = obj['relative_position']
    vel = obj['relative_velocity']
    print(f'pos:{pos},vel:{vel}')

    if len(pos) < 1 or len(vel) < 1:
      continue

    r0 = float(pos[0])
    v0_kmh = float(vel[0])
    v0 = v0_kmh / 3.6

    if 0 < r0 < maxR and -maxV < v0 < maxV:
      targets.append({
        "r0": r0,
        "v0": v0,
        "amp": 1.00,
        "relative_position": pos
      })
    else:
      print(f"目标参数超出雷达范围，跳过：距离{r0}m，速度{v0_kmh}km/h（{v0}m/s）")

  if not targets:
    return []

  # ========================== 时间轴 & 发射信号 ==========================
  t = np.linspace(0, Nd * Tchirp, Nr * Nd, endpoint=False)
  angle_tx = fc * t + 0.5 * slope * t * t
  Tx = np.cos(2 * np.pi * angle_tx)

  # ========================== 多目标回波 & IF 基带 ==========================
  IF_mat = np.zeros((Nd, Nr))
  lambda_ = c / fc

  # ✅ 修复漏洞：改为遍历 targets 数组，累加所有目标的 IF 信号
  for tgt in targets:
    r_t = tgt["r0"]
    v_t = tgt["v0"]
    amp = tgt["amp"]

    for d in range(Nd):
      fb = 2 * slope * r_t / c
      phase = 4 * np.pi * (r_t + v_t * d * PRI) / lambda_
      t_chirp = np.linspace(0, Tchirp, Nr)
      IF_chirp = amp * np.cos(2 * np.pi * (fb * t_chirp + phase / (2 * np.pi)))
      IF_mat[d] += IF_chirp  # 叠加多目标回波

  # DSP 处理
  win_r = np.hanning(Nr)
  win_d = np.hanning(Nd)
  Xr = np.fft.rfft(IF_mat * win_r[np.newaxis, :], n=Nr, axis=1)
  fr = np.fft.rfftfreq(Nr, d=1 / Fs)
  range_axis = fr * c / (2 * slope)
  Xd = np.fft.fftshift(np.fft.fft(Xr * win_d[:, np.newaxis], n=Nd, axis=0), axes=0)
  fd = np.fft.fftshift(np.fft.fftfreq(Nd, d=PRI))
  vel_axis = fd * c / (2 * fc)

  # 有效范围过滤
  valid_r = (range_axis >= 0) & (range_axis <= maxR)
  valid_v = (vel_axis >= -maxV) & (vel_axis <= maxV)
  RD = np.abs(Xd[np.ix_(valid_v, valid_r)])

  # ========================== 峰值检测 ==========================
  # ✅ 传入用距离/速度分辨率动态计算出来的 suppr 窗口
  peaks_2d = find_top_k_peaks_2d(RD, k=len(targets), suppr_v=suppr_v_dynamic, suppr_r=suppr_r_dynamic)

  detections = []
  for (vi, ri, val) in peaks_2d:
        # ✅ 新增：记录横向位置（若有），供CAN消息使用
        lat_pos = targets[len(detections)]['relative_position'][1] if (len(targets) > len(detections) and len(targets[len(detections)]['relative_position'])>1) else 0.0
        detections.append({
            "R": range_axis[valid_r][ri],  # 纵向距离（m）
            "V": vel_axis[valid_v][vi],    # 纵向相对速度（m/s）
            "P_dB": 20 * np.log10(val + 1e-12),
            "Y": lat_pos  # 横向位置（m）
        })
  return detections


class SimulatedCar:
    """Simulates a honda civic 2016 (panda state + can messages) to OpenPilot"""
    packer = CANPacker("honda_civic_touring_2016_can_generated")
    rpacker = CANPacker("acura_ilx_2016_nidec")

    def __init__(self):
        self.pm = messaging.PubMaster(['can', 'pandaStates'])
        self.sm = messaging.SubMaster(['carControl', 'controlsState', 'carParams','modelV2'])
        self.cp = self.get_car_can_parser()
        self.idx = 0
        self.params = Params()
        self.obd_multiplexing = False

    @staticmethod
    def get_car_can_parser():
        dbc_f = 'honda_civic_touring_2016_can_generated'
        checks = [(0xe4, 100), (0x1fa, 50), (0x200, 50)]
        return CANParser(dbc_f, checks, 0)

    def send_can_messages(self, simulator_state: SimulatorState):
        if not simulator_state.valid:
            return
        msg = []

        # 获取周边目标信息
        leads = self.sm['modelV2'].leadsV3
        print(f"modelV2 leads: {len(leads)} 个前车")

        #e.g.(openpilot-py3.11) pjk@pjkcomputer:~/PycharmProjects/openpilot0.9.6/openpilot$ touch /tmp/adversarial_patch_enabled
        attack_enabled = os.path.exists("/tmp/adversarial_patch_enabled")

        # 获取周边目标信息
        if attack_enabled:
          # 攻击模式：优先使用 modelV2 (被攻击后的视觉) 的输出来欺骗雷达
          leads = self.sm['modelV2'].leadsV3
          print(f"[攻击模式] modelV2 leads: {len(leads)} 个前车")
          surrounding_info = []
          for i, lead in enumerate(leads):
            dRel = lead.x[0] - 1.52-13# 相对距离
            vRel = lead.v[0] - simulator_state.speed # 相对速度(m/s)
            print(f"[攻击模式] Lead {i}: dRel={dRel:.2f}m, vRel={vRel:.2f}m/s")
            surrounding_info = [{'relative_position': [dRel, 0], 'relative_velocity': [vRel * 3.6, 0]}]

          if len(leads) == 0:
            # 视觉没拿到信息时，回退到模拟器原始数据
            surrounding_info = getattr(simulator_state, "surrounding_info", None)
        else:
          # 正常模式：直接使用模拟器的真实数据
          print(f"[正常模式] 使用模拟器原始雷达数据")
          leads = self.sm['modelV2'].leadsV3
          surrounding_info = []
          for i, lead in enumerate(leads):
            dRel = lead.x[0] - 1.52  # 相对距离
            vRel = lead.v[0] - simulator_state.speed  # 相对速度(m/s)
            print(f"[正常模式] Lead {i}: dRel={dRel:.2f}m, vRel={vRel:.2f}m/s")
            surrounding_info = [{'relative_position': [dRel, 0], 'relative_velocity': [vRel * 3.6, 0]}]
          if len(leads) == 0:
            # 视觉没拿到信息时，回退到模拟器原始数据
            surrounding_info = getattr(simulator_state, "surrounding_info", None)

        # 接收append_dect的返回值，定义detections变量
        detections = append_dect(surrounding_info)


        # *** powertrain bus ***（与原代码一致，无修改）
        speed = simulator_state.speed * 3.6  # convert m/s to kph
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
        msg.append(self.packer.make_can_msg("SCM_FEEDBACK", 0, {
            "MAIN_ON": 1,
            "LEFT_BLINKER": simulator_state.left_blinker,
            "RIGHT_BLINKER": simulator_state.right_blinker
        }))
        msg.append(self.packer.make_can_msg("POWERTRAIN_DATA", 0, {
            "ACC_STATUS": int(simulator_state.is_engaged),
            "PEDAL_GAS": simulator_state.user_gas,
            "BRAKE_PRESSED": simulator_state.user_brake > 0
        }))
        msg.append(self.packer.make_can_msg("HUD_SETTING", 0, {}))
        msg.append(self.packer.make_can_msg("CAR_SPEED", 0, {}))

        # *** cam bus ***（与原代码一致，无修改）
        msg.append(self.packer.make_can_msg("STEERING_CONTROL", 2, {}))
        msg.append(self.packer.make_can_msg("ACC_HUD", 2, {}))
        msg.append(self.packer.make_can_msg("LKAS_HUD", 2, {}))
        msg.append(self.packer.make_can_msg("BRAKE_COMMAND", 2, {}))

        # *** radar bus ***（✅ 关键修改：使用有效detections，适配横向位置）
        if self.idx % 5 == 0:
            # radar 状态消息
            msg.append(self.rpacker.make_can_msg("RADAR_DIAGNOSTIC", 1, {"RADAR_STATE": 0x79}))
            # 最多 16 个 track
            max_tracks = 16
            for i in range(max_tracks):
                if i < len(detections):
                    det = detections[i]
                    msg.append(self.rpacker.make_can_msg("TRACK_%d" % i, 1, {
                        "LONG_DIST": float(det["R"]),  # 纵向距离（m）
                        "LAT_DIST": float(det.get("Y", 0.0)),  
                        "REL_SPEED": float(det["V"]),  # 纵向相对速度（m/s）
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

    def update(self, simulator_state):
        self.send_can_messages(simulator_state)
        if self.idx % 50 == 0:  # only send panda states at 2hz
            self.send_panda_state(simulator_state)
        self.idx += 1
