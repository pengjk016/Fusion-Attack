#!/usr/bin/env python3
import importlib
import math
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
from openpilot.common.swaglog import cloudlog

# Default lead acceleration decay set to 50% at 1s
_LEAD_ACCEL_TAU = 1.5

# radar tracks
SPEED, ACCEL = 0, 1     # Kalman filter states enum

# stationary qualification parameters
V_EGO_STATIONARY = 4.   # no stationary object flag below this speed

RADAR_TO_CENTER = 2.7   # (deprecated) RADAR is ~ 2.7m ahead from center of car
RADAR_TO_CAMERA = 1.52  # RADAR is ~ 1.5m ahead from center of mesh frame


# ==============================
# 新增融合逻辑
# ==============================
class Detection:
    def __init__(self, sensor_type: str, x: float, y: float, vx: float, vy: float, noise_cov: np.ndarray):
        self.sensor_type = sensor_type  # 'radar' or 'vision'
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.noise_cov = noise_cov

def init_kf(initial_x: float, initial_vx: float, dt: float = 0.05):
    kf = KalmanFilter(dim_x=4, dim_z=4)
    kf.x = np.array([initial_x, initial_vx, 0, 0])  # [x, vx, y, vy]
    kf.F = np.array([[1, dt, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, dt],
                     [0, 0, 0, 1]])
    kf.H = np.eye(4)
    kf.Q = np.eye(4) * 0.1
    kf.R = np.eye(4) * 1.0
    return kf

class MultiObjectTracker:
    def __init__(self, dt: float = 0.05):
        self.dt = dt
        self.tracks: List[KalmanFilter] = []

    def predict(self):
        for kf in self.tracks:
            kf.predict()

    def update(self, detections: List[Detection]):
        if not self.tracks or not detections:
            for det in detections:
                kf = init_kf(det.x, det.vx, self.dt)
                kf.update(np.array([det.x, det.vx, det.y, det.vy]), R=det.noise_cov)
                self.tracks.append(kf)
            return

        # 匈牙利算法匹配
        track_states = np.array([kf.x[[0, 2]] for kf in self.tracks])
        det_positions = np.array([[det.x, det.y] for det in detections])
        cost_matrix = np.linalg.norm(track_states[:, None] - det_positions[None, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 更新匹配到的轨迹
        for r, c in zip(row_ind, col_ind):
            self.tracks[r].update(np.array([detections[c].x, detections[c].vx,
                                            detections[c].y, detections[c].vy]),
                                  R=detections[c].noise_cov)

        # 新增未匹配的检测
        matched_dets = set(col_ind)
        for i, det in enumerate(detections):
            if i not in matched_dets:
                kf = init_kf(det.x, det.vx, self.dt)
                kf.update(np.array([det.x, det.vx, det.y, det.vy]), R=det.noise_cov)
                self.tracks.append(kf)

    def get_tracks(self) -> List[np.ndarray]:
        return [kf.x for kf in self.tracks]


# ==============================
# 原 Track 类保持不变
# ==============================
class Track:
    def __init__(self, identifier: int, v_lead: float, kalman_params):
        self.identifier = identifier
        self.cnt = 0
        self.aLeadTau = _LEAD_ACCEL_TAU
        self.K_A = kalman_params.A
        self.K_C = kalman_params.C
        self.K_K = kalman_params.K
        self.kf = KF1D([[v_lead], [0.0]], self.K_A, self.K_C, self.K_K)

    def update(self, d_rel: float, y_rel: float, v_rel: float, v_lead: float, measured: float):
        self.dRel = d_rel
        self.yRel = y_rel
        self.vRel = v_rel
        self.vLead = v_lead
        self.measured = measured

        if self.cnt > 0:
            self.kf.update(self.vLead)

        self.vLeadK = float(self.kf.x[SPEED][0])
        self.aLeadK = float(self.kf.x[ACCEL][0])

        if abs(self.aLeadK) < 0.5:
            self.aLeadTau = _LEAD_ACCEL_TAU
        else:
            self.aLeadTau *= 0.9

        self.cnt += 1

    def get_RadarState(self, model_prob: float = 0.0):
        return {
            "dRel": float(self.dRel),
            "yRel": float(self.yRel),
            "vRel": float(self.vRel),
            "vLead": float(self.vLead),
            "vLeadK": float(self.vLeadK),
            "aLeadK": float(self.aLeadK),
            "aLeadTau": float(self.aLeadTau),
            "status": True,
            "fcw": self.is_potential_fcw(model_prob),
            "modelProb": model_prob,
            "radar": True,
            "radarTrackId": self.identifier,
        }

    def is_potential_fcw(self, model_prob: float):
        return model_prob > .9


# ==============================
# 原 KalmanParams 保持不变
# ==============================
class KalmanParams:
    def __init__(self, dt: float):
        assert dt > .01 and dt < .2
        self.A = [[1.0, dt], [0.0, 1.0]]
        self.C = [1.0, 0.0]
        dts = [dt * 0.01 for dt in range(1, 21)]
        K0 = [0.12287673, 0.14556536, 0.16522756, 0.18281627, 0.1988689,  0.21372394,
              0.22761098, 0.24069424, 0.253096,   0.26491023, 0.27621103, 0.29750003,
              0.30757767, 0.31732515, 0.32677158, 0.33594201, 0.34485814, 0.35353899, 0.36200124]
        K1 = [0.29666309, 0.29330885, 0.29042818, 0.28787125, 0.28555364, 0.28342219,
              0.28144091, 0.27958406, 0.27783249, 0.27617149, 0.27458948, 0.27307714,
              0.27162685, 0.27023228, 0.26888809, 0.26758976, 0.26633338, 0.26511557, 0.26393339, 0.26278425]
        self.K = [[interp(dt, dts, K0)], [interp(dt, dts, K1)]]


# ==============================
# 原 RadarD 类替换融合逻辑
# ==============================
class RadarD:
    def __init__(self, radar_ts: float, delay: int = 0):
        self.current_time = 0.0
        self.tracks: Dict[int, Track] = {}
        self.kalman_params = KalmanParams(radar_ts)
        self.v_ego = 0.0
        self.v_ego_hist = deque([0.0], maxlen=delay+1)
        self.last_v_ego_frame = -1
        self.radar_state: Optional[capnp._DynamicStructBuilder] = None
        self.radar_state_valid = False
        self.ready = False

        # 新增融合跟踪器
        self.fusion_tracker = MultiObjectTracker(radar_ts)

    def update(self, sm: messaging.SubMaster, rr: Optional[car.RadarData]):
        self.ready = sm.seen['modelV2']
        self.current_time = 1e-9*max(sm.logMonoTime.values())

        radar_points = rr.points if rr else []
        radar_errors = rr.errors if rr else []

        if sm.recv_frame['carState'] != self.last_v_ego_frame:
            self.v_ego = sm['carState'].vEgo
            self.v_ego_hist.append(self.v_ego)
            self.last_v_ego_frame = sm.recv_frame['carState']

        # 转成 Detection
        detections = []
        for pt in radar_points:
            detections.append(Detection('radar', pt.dRel, pt.yRel, pt.vRel, 0, np.eye(4)*2.0))
        for vl in sm['modelV2'].leadsV3:
            detections.append(Detection('vision', vl.x[0], -vl.y[0], vl.v[0], 0, np.eye(4)*3.0))

        # 融合预测更新
        self.fusion_tracker.predict()
        self.fusion_tracker.update(detections)

        # 发布融合结果
        self.radar_state_valid = sm.all_checks() and len(radar_errors) == 0
        self.radar_state = log.RadarState.new_message()
        self.radar_state.mdMonoTime = sm.logMonoTime['modelV2']
        self.radar_state.radarErrors = list(radar_errors)
        self.radar_state.carStateMonoTime = sm.logMonoTime['carState']

        if len(sm['modelV2'].temporalPose.trans):
            model_v_ego = sm['modelV2'].temporalPose.trans[0]
        else:
            model_v_ego = self.v_ego

        # 输出融合轨迹
        fused_tracks = self.fusion_tracker.get_tracks()
        for i, track in enumerate(fused_tracks[:2]):  # 只输出前两个
            lead_key = f"leadOne" if i == 0 else f"leadTwo"
            setattr(self.radar_state, lead_key, {
                "dRel": float(track[0]),
                "yRel": float(track[2]),
                "vRel": float(track[1] - self.v_ego),
                "vLead": float(track[1]),
                "status": True,
                "radar": True,
                "radarTrackId": i
            })

    def publish(self, pm: messaging.PubMaster, lag_ms: float):
        radar_msg = messaging.new_message("radarState")
        radar_msg.valid = self.radar_state_valid
        radar_msg.radarState = self.radar_state
        radar_msg.radarState.cumLagMs = lag_ms
        pm.send("radarState", radar_msg)


# ==============================
# 主函数保持不变
# ==============================
def main():
    config_realtime_process(5, Priority.CTRL_LOW)
    with car.CarParams.from_bytes(Params().get("CarParams", block=True)) as msg:
        CP = msg
    RadarInterface = importlib.import_module(f'selfdrive.car.{CP.carName}.radar_interface').RadarInterface

    can_sock = messaging.sub_sock('can')
    sm = messaging.SubMaster(['modelV2', 'carState'], frequency=int(1./DT_CTRL))
    pm = messaging.PubMaster(['radarState', 'liveTracks'])

    RI = RadarInterface(CP)
    rk = Ratekeeper(1.0 / CP.radarTimeStep, print_delay_threshold=None)
    RD = RadarD(CP.radarTimeStep, RI.delay)

    while 1:
        can_strings = messaging.drain_sock_raw(can_sock, wait_for_one=True)
        rr = RI.update(can_strings)
        sm.update(0)
        if rr is None:
            continue
        RD.update(sm, rr)
        RD.publish(pm, -rk.remaining*1000.0)
        rk.monitor_time()


if __name__ == "__main__":
    main()
