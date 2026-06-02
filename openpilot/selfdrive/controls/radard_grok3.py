#欧氏距离，能跑通的最后一版

import importlib
from collections import deque
from typing import Optional, List
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

import capnp
from cereal import messaging, log, car
from openpilot.common.params import Params
from openpilot.common.realtime import DT_CTRL, Ratekeeper, Priority, config_realtime_process


# =========================================================
# Detection 数据结构
# =========================================================
_LEAD_ACCEL_TAU = 1.5

RADAR_TO_CENTER = 2.7
RADAR_TO_CAMERA = 1.52
class Detection:
    def __init__(self, sensor_type: str, x: float, y: float, vx: float):
        self.sensor_type = sensor_type  # 'radar' or 'vision'
        self.x = x
        self.y = y
        self.vx = vx


# =========================================================
# Linear Kalman Filter (6维 CA，与 MATLAB 一致)
# =========================================================

def init_linear_kf(dt: float) -> KalmanFilter:
    kf = KalmanFilter(dim_x=6, dim_z=4)

    kf.x = np.zeros(6)

    kf.F = np.array([
        [1, dt, dt**2/2, 0, 0, 0],
        [0, 1, dt,      0, 0, 0],
        [0, 0, 1,       0, 0, 0],
        [0, 0, 0,       1, dt, dt**2/2],
        [0, 0, 0,       0, 1, dt],
        [0, 0, 0,       0, 0, 1]
    ])

    kf.H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0]
    ])

    kf.P = np.diag([10.0, 5.0, 1.0, 10.0, 5.0, 1.0])

    # 过程噪声（与 MATLAB sigma=1 一致，y 方向极小以实现“常量”）
    sigma_x = 1.0
    sigma_y = 0.001
    Q1d_x = sigma_x**2 * np.array([[dt**4/4, dt**3/2, dt**2/2],
                                   [dt**3/2, dt**2,   dt],
                                   [dt**2/2, dt,      1]])
    Q1d_y = sigma_y**2 * np.array([[dt**4/4, dt**3/2, dt**2/2],
                                   [dt**3/2, dt**2,   dt],
                                   [dt**2/2, dt,      1]])
    kf.Q = np.block([[Q1d_x, np.zeros((3,3))],
                     [np.zeros((3,3)), Q1d_y]])

    return kf


def update_kf_with_detection(kf: KalmanFilter, det: Detection):
    if det.sensor_type == 'radar':
        z = np.array([det.x, det.vx, det.y, 0.0])
        R = np.diag([2.0, 2.0, 2.0, 1e6])   # vy 忽略
    else:
        z = np.array([det.x, det.vx, det.y, 0.0])
        R = np.diag([3.0, 100.0, 3.0, 1e6]) # vx 噪声大，vy 忽略

    kf.update(z, R=R)


# =========================================================
# Track 类：实现 Confirmation / Deletion 逻辑
# =========================================================

class Track:
    def __init__(self, kf: KalmanFilter):
        self.kf = kf
        self.is_confirmed = False
        self.hit_streak = 0          # 当前连续 hit 次数
        self.history = deque(maxlen=3)  # 最近 3 次的 hit/miss (1/0)
        self.misses_in_a_row = 0     # 连续 miss 次数，用于 DeletionThreshold=5

    def predict(self):
        self.kf.predict()
        # 预测时默认 miss
        self.history.append(0)
        self.misses_in_a_row += 1

    def update(self, det: Detection):
        update_kf_with_detection(self.kf, det)
        self.history[-1] = 1
        self.hit_streak += 1
        self.misses_in_a_row = 0

        # ConfirmationThreshold [2 3]: 最近 3 次中至少 2 次 hit
        if sum(self.history) >= 2:
            self.is_confirmed = True


# =========================================================
# MultiObjectTracker：完全复制 MATLAB multiObjectTracker 逻辑
# =========================================================

class MultiObjectTracker:
    def __init__(self, dt: float):
        self.dt = dt
        self.tracks: List[Track] = []
        self.assignment_threshold = 35.0   # 与 MATLAB 完全一致

    def predict(self):
        for track in self.tracks:
            track.predict()

    def update(self, detections: List[Detection]):
        # 预测所有轨迹
        self.predict()

        if not self.tracks:
            # 无轨迹：所有检测初始化为新轨迹（Tentative）
            for det in detections:
                kf = init_linear_kf(self.dt)
                kf.x = np.array([det.x, det.vx, 0.0, det.y, 0.0, 0.0])
                self.tracks.append(Track(kf))
            return

        if not detections:
            # 无检测：所有轨迹 miss，检查删除
            self.tracks = [t for t in self.tracks if t.misses_in_a_row < 5]
            return

        # 计算位置距离代价矩阵
        track_pos = np.array([[t.kf.x[0], t.kf.x[3]] for t in self.tracks])
        det_pos   = np.array([[d.x, d.y] for d in detections])
        cost = np.linalg.norm(track_pos[:, None] - det_pos[None, :], axis=2)

        row_ind, col_ind = linear_sum_assignment(cost)

        assigned_tracks = set()
        assigned_dets   = set()

        # 仅当代价 <= AssignmentThreshold 时才关联
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] <= self.assignment_threshold:
                self.tracks[r].update(detections[c])
                assigned_tracks.add(r)
                assigned_dets.add(c)

        # 未被关联的轨迹已在上一步 predict 中标记为 miss
        # 删除连续 5 次 miss 的轨迹（DeletionThreshold = 5）
        self.tracks = [t for i, t in enumerate(self.tracks)
                       if i in assigned_tracks or t.misses_in_a_row < 5]

        # 未被关联的检测初始化为新轨迹（Tentative）
        for i, det in enumerate(detections):
            if i not in assigned_dets:
                kf = init_linear_kf(self.dt)
                kf.x = np.array([det.x, det.vx, 0.0, det.y, 0.0, 0.0])
                self.tracks.append(Track(kf))

    def get_confirmed_tracks(self):
        """返回已确认的轨迹，按 dRel（x）升序排序（最近的在前）"""
        confirmed = [t.kf for t in self.tracks if t.is_confirmed]
        confirmed.sort(key=lambda kf: kf.x[0])
        return confirmed


# =========================================================
# RadarD 主类
# =========================================================

class RadarD:
    def __init__(self, radar_ts: float, delay: int = 0):
        self.tracker = MultiObjectTracker(radar_ts)
        self.v_ego = 0.0
        self.v_ego_hist = deque([0.0], maxlen=delay + 1)
        self.last_v_ego_frame = -1
        self.radar_state_valid = False

    def update(self, sm: messaging.SubMaster, rr: Optional[car.RadarData]):
        radar_points = rr.points if rr else []
        radar_errors = rr.errors if rr else []

        if sm.recv_frame['carState'] != self.last_v_ego_frame:
            self.v_ego = sm['carState'].vEgo
            self.v_ego_hist.append(self.v_ego)
            self.last_v_ego_frame = sm.recv_frame['carState']

        detections: List[Detection] = []

        for pt in radar_points:
            detections.append(Detection('radar', pt.dRel, pt.yRel, pt.vRel))

        for vl in sm['modelV2'].leadsV3:
            detections.append(Detection('vision', vl.x[0], -vl.y[0], vl.v[0]))
        print(f'detections:{detections}')
        self.tracker.update(detections)

        # 发布 radarState
        self.radar_state_valid = sm.all_checks() and len(radar_errors) == 0
        self.radar_state = log.RadarState.new_message()
        self.radar_state.mdMonoTime = sm.logMonoTime['modelV2']
        self.radar_state.carStateMonoTime = sm.logMonoTime['carState']
        self.radar_state.radarErrors = list(radar_errors)

        tracks = self.tracker.get_confirmed_tracks()
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
        print(f'发送了radar信息，内容是{msg.radarState}')


# =========================================================
# main
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
