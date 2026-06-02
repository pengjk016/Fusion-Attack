#TBD：加上三个参数assignment th。。。，确认融合逻辑（是否在matlab里面没有），试试效果，跟matlab更像，六维状态矩阵
#!/usr/bin/env python3
import importlib
from collections import deque
from typing import Optional, Dict, Any, List
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

import capnp
from cereal import messaging, log, car
from openpilot.common.numpy_fast import interp
from openpilot.common.params import Params
from openpilot.common.realtime import DT_CTRL, Ratekeeper, Priority, config_realtime_process

# =========================================================
# 与 MATLAB Radar-Camera Fusion 一致的参数与建模假设
# =========================================================

_LEAD_ACCEL_TAU = 1.5

RADAR_TO_CENTER = 2.7
RADAR_TO_CAMERA = 1.52


# =========================================================
# Detection 数据结构（保持不变）
# =========================================================

class Detection:
    def __init__(self, sensor_type: str, x: float, y: float, vx: float, noise_cov: np.ndarray):
        self.sensor_type = sensor_type  # 'radar' or 'vision'
        self.x = x
        self.y = y
        self.vx = vx
        self.noise_cov = noise_cov


# =========================================================
# 与 MATLAB 一致的 Linear Kalman Filter
# 状态: [x, vx, ax, y, vy, ay]
# =========================================================

def init_linear_kf(dt: float) -> KalmanFilter:
    kf = KalmanFilter(dim_x=6, dim_z=4)  # z: [x, vx, y, vy]

    # 初始状态: [x=0, vx=0, ax=0, y=0, vy=0, ay=0]
    kf.x = np.zeros(6)

    # 状态转移矩阵 F (2D Constant Acceleration)
    kf.F = np.array([
        [1, dt, dt**2/2, 0, 0, 0],
        [0, 1, dt, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, dt, dt**2/2],
        [0, 0, 0, 0, 1, dt],
        [0, 0, 0, 0, 0, 1]
    ])

    # 缺省测量矩阵 H: 测量 [x, vx, y, vy] = H * state
    kf.H = np.array([
        [1, 0, 0, 0, 0, 0],  # x
        [0, 1, 0, 0, 0, 0],  # vx
        [0, 0, 0, 1, 0, 0],  # y
        [0, 0, 0, 0, 1, 0]   # vy
    ])

    # 状态协方差 P
    kf.P = np.diag([10.0, 5.0, 1.0, 10.0, 5.0, 1.0])  # vy/ay 初始不确定小

    # 过程噪声 Q: x维度标准, y维度 vy/ay sigma极小 (使vy/ay近似常量)
    sigma_x = 1.0  # x方向加速度噪声
    Q1d_x = np.array([
        [dt**4/4, dt**3/2, dt**2/2],
        [dt**3/2, dt**2, dt],
        [dt**2/2, dt, 1]
    ]) * sigma_x**2

    sigma_y = 0.001  # y方向极小噪声，使vy/ay近似常量
    Q1d_y = np.array([
        [dt**4/4, dt**3/2, dt**2/2],
        [dt**3/2, dt**2, dt],
        [dt**2/2, dt, 1]
    ]) * sigma_y**2

    kf.Q = np.block([
        [Q1d_x, np.zeros((3,3))],
        [np.zeros((3,3)), Q1d_y]
    ])

    return kf


def update_kf_with_detection(kf: KalmanFilter, det: Detection):
  """
  与 MATLAB Radar-Camera Fusion 行为一致：
  - 状态: [x, vx, ax, y, vy, ay]
  - radar: 观测 x, vx, y; vy=0 + 极大噪声（忽略）
  - vision: 观测 x, vx(噪声大), y; vy=0 + 极大噪声
  """

  if det.sensor_type == 'radar':
    # Radar: 全量测但vy未知
    z = np.array([det.x, det.vx, det.y, 0.0])
    R = np.diag([2.0, 2.0, 2.0, 1e6])  # vy 极大噪声

  else:
    # Vision: vx 噪声较大, vy未知
    z = np.array([det.x, det.vx, det.y, 0.0])
    R = np.diag([3.0, 100.0, 3.0, 1e6])  # vx噪声较大, vy极大

  kf.update(z, R=R)


# =========================================================
# 多目标融合跟踪器（与 MATLAB 结构一致）
# =========================================================

class MultiObjectTracker:
    def __init__(self, dt: float):
        self.dt = dt
        self.tracks: List[KalmanFilter] = []

    def predict(self):
        for kf in self.tracks:
            kf.predict()

    def update(self, detections: List[Detection]):
        if not self.tracks:
            for det in detections:
                kf = init_linear_kf(self.dt)
                kf.x = np.array([det.x, det.vx, 0.0, det.y, 0.0, 0.0])  # ax=0, vy=0, ay=0
                self.tracks.append(kf)
            return

        if not detections:
            return

        # 只用位置做关联（MATLAB 也是这么做的）
        track_pos = np.array([[kf.x[0], kf.x[3]] for kf in self.tracks])
        det_pos = np.array([[d.x, d.y] for d in detections])

        cost = np.linalg.norm(track_pos[:, None] - det_pos[None, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(cost)

        # 更新匹配到的轨迹
        for r, c in zip(row_ind, col_ind):
            update_kf_with_detection(self.tracks[r], detections[c])

        # 新增未匹配的检测
        matched = set(col_ind)
        for i, det in enumerate(detections):
            if i not in matched:
                kf = init_linear_kf(self.dt)
                kf.x = np.array([det.x, det.vx, 0.0, det.y, 0.0, 0.0])  # ax=0, vy=0, ay=0
                self.tracks.append(kf)

    def get_tracks(self):
        return self.tracks


# =========================================================
# RadarD：融合 radar + vision（对齐 MATLAB）
# =========================================================

class RadarD:
    def __init__(self, radar_ts: float, delay: int = 0):
        self.tracker = MultiObjectTracker(radar_ts)
        self.v_ego = 0.0
        self.v_ego_hist = deque([0.0], maxlen=delay + 1)
        self.last_v_ego_frame = -1

        self.radar_state: Optional[capnp._DynamicStructBuilder] = None
        self.radar_state_valid = False

    def update(self, sm: messaging.SubMaster, rr: Optional[car.RadarData]):
        radar_points = rr.points if rr else []
        radar_errors = rr.errors if rr else []

        if sm.recv_frame['carState'] != self.last_v_ego_frame:
            self.v_ego = sm['carState'].vEgo
            self.v_ego_hist.append(self.v_ego)
            self.last_v_ego_frame = sm.recv_frame['carState']

        detections: List[Detection] = []

        # Radar detections（已是 Cartesian）
        for pt in radar_points:
            detections.append(
                Detection(
                    'radar',
                    pt.dRel,
                    pt.yRel,
                    pt.vRel,
                    np.eye(3) * 2.0
                )
            )

        # Vision detections（无侧向速度）
        for vl in sm['modelV2'].leadsV3:
            detections.append(
                Detection(
                    'vision',
                    vl.x[0],
                    -vl.y[0],
                    vl.v[0],
                    np.eye(2) * 3.0
                )
            )

        self.tracker.predict()
        self.tracker.update(detections)

        # 发布 radarState
        self.radar_state_valid = sm.all_checks() and len(radar_errors) == 0
        self.radar_state = log.RadarState.new_message()
        self.radar_state.mdMonoTime = sm.logMonoTime['modelV2']
        self.radar_state.carStateMonoTime = sm.logMonoTime['carState']
        self.radar_state.radarErrors = list(radar_errors)

        tracks = self.tracker.get_tracks()
        for i, kf in enumerate(tracks[:2]):
            lead_key = "leadOne" if i == 0 else "leadTwo"
            setattr(self.radar_state, lead_key, {
                "dRel": float(kf.x[0]),
                "yRel": float(kf.x[3]),
                "vRel": float(kf.x[1] - self.v_ego),
                "vLead": float(kf.x[1]),
                "status": True,
                "radar": True,
                "radarTrackId": i
            })

    def publish(self, pm: messaging.PubMaster, lag_ms: float):
        msg = messaging.new_message("radarState")
        msg.valid = self.radar_state_valid
        msg.radarState = self.radar_state
        msg.radarState.cumLagMs = lag_ms
        pm.send("radarState", msg)


# =========================================================
# main（保持 openpilot 原结构）
# =========================================================

def main():
    config_realtime_process(5, Priority.CTRL_LOW)
    with car.CarParams.from_bytes(Params().get("CarParams", block=True)) as msg:
        CP = msg

    RadarInterface = importlib.import_module(
        f'selfdrive.car.{CP.carName}.radar_interface'
    ).RadarInterface

    can_sock = messaging.sub_sock('can')
    sm = messaging.SubMaster(['modelV2', 'carState'], frequency=int(1. / DT_CTRL))
    pm = messaging.PubMaster(['radarState'])

    RI = RadarInterface(CP)
    rk = Ratekeeper(1.0 / CP.radarTimeStep, print_delay_threshold=None)
    RD = RadarD(CP.radarTimeStep, RI.delay)

    while True:
        can_strings = messaging.drain_sock_raw(can_sock, wait_for_one=True)
        rr = RI.update(can_strings)
        sm.update(0)
        if rr is None:
            continue

        RD.update(sm, rr)
        RD.publish(pm, -rk.remaining * 1000.0)
        rk.monitor_time()


if __name__ == "__main__":
    main()

