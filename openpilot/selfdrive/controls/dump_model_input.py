import time
import numpy as np
import cv2
from cereal.visionipc import VisionIpcClient, VisionStreamType
import importlib
import math
from collections import deque
from typing import Optional, Dict, Any

import capnp
from cereal import messaging, log, car
from openpilot.common.numpy_fast import interp
from openpilot.common.params import Params
from openpilot.common.realtime import DT_CTRL
from openpilot.common.swaglog import cloudlog

from openpilot.common.simple_kalman import KF1D

import csv
import os

SAVE_DIR = "../../attack_picture"
CSV_DIR = "../../attack_picture"
START_IDX = 1
INTERVAL = 0.05
NUM_FRAMES = 100
MODEL_FRAME_TIMEOUT_MS = 1500
MODEL_FRAME_POLL_MS = 100

_LEAD_ACCEL_TAU = 1.5
SPEED, ACCEL = 0, 1
V_EGO_STATIONARY = 4.
RADAR_TO_CENTER = 2.7
RADAR_TO_CAMERA = 1.52


class KalmanParams:
  def __init__(self, dt: float):
    assert dt > .01 and dt < .2
    self.A = [[1.0, dt], [0.0, 1.0]]
    self.C = [1.0, 0.0]
    dts = [dt * 0.01 for dt in range(1, 21)]
    K0 = [0.12287673, 0.14556536, 0.16522756, 0.18281627, 0.1988689, 0.21372394,
          0.22761098, 0.24069424, 0.253096, 0.26491023, 0.27621103, 0.28705801,
          0.29750003, 0.30757767, 0.31732515, 0.32677158, 0.33594201, 0.34485814,
          0.35353899, 0.36200124]
    K1 = [0.29666309, 0.29330885, 0.29042818, 0.28787125, 0.28555364, 0.28342219,
          0.28144091, 0.27958406, 0.27783249, 0.27617149, 0.27458948, 0.27307714,
          0.27162685, 0.27023228, 0.26888809, 0.26758976, 0.26633338, 0.26511557,
          0.26393339, 0.26278425]
    self.K = [[interp(dt, dts, K0)], [interp(dt, dts, K1)]]


class Track:
  def __init__(self, identifier: int, v_lead: float, kalman_params: KalmanParams):
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

  def potential_low_speed_lead(self, v_ego: float):
    return abs(self.yRel) < 1.0 and (v_ego < V_EGO_STATIONARY) and (0.75 < self.dRel < 25)

  def is_potential_fcw(self, model_prob: float):
    return model_prob > .9


def laplacian_pdf(x: float, mu: float, b: float):
  b = max(b, 1e-4)
  return math.exp(-abs(x - mu) / b)


def match_vision_to_track(v_ego: float, lead: capnp._DynamicStructReader, tracks: Dict[int, Track]):
  offset_vision_dist = lead.x[0] - RADAR_TO_CAMERA

  def prob(c):
    prob_d = laplacian_pdf(c.dRel, offset_vision_dist, lead.xStd[0])
    prob_y = laplacian_pdf(c.yRel, -lead.y[0], lead.yStd[0])
    prob_v = laplacian_pdf(c.vRel + v_ego, lead.v[0], lead.vStd[0])
    return prob_d * prob_y * prob_v

  track = max(tracks.values(), key=prob)

  dist_sane = abs(track.dRel - offset_vision_dist) < max([(offset_vision_dist) * .25, 5.0])
  vel_sane = (abs(track.vRel + v_ego - lead.v[0]) < 10) or (v_ego + track.vRel > 3)
  return track if dist_sane and vel_sane else None


def get_RadarState_from_vision(lead_msg: capnp._DynamicStructReader, v_ego: float, model_v_ego: float):
  lead_v_rel_pred = lead_msg.v[0] - model_v_ego
  return {
    "dRel": float(lead_msg.x[0] - RADAR_TO_CAMERA),
    "yRel": float(-lead_msg.y[0]),
    "vRel": float(lead_v_rel_pred),
    "vLead": float(v_ego + lead_v_rel_pred),
    "vLeadK": float(v_ego + lead_v_rel_pred),
    "aLeadK": 0.0,
    "aLeadTau": 0.3,
    "fcw": False,
    "modelProb": float(lead_msg.prob),
    "status": True,
    "radar": False,
    "radarTrackId": -1,
  }


def get_lead(v_ego: float, ready: bool, tracks: Dict[int, Track], lead_msg: capnp._DynamicStructReader,
             model_v_ego: float, low_speed_override: bool = True) -> Dict[str, Any]:
  if len(tracks) > 0 and ready and lead_msg.prob > .5:
    track = match_vision_to_track(v_ego, lead_msg, tracks)
  else:
    track = None

  lead_dict = {'status': False}
  if track is not None:
    lead_dict = track.get_RadarState(lead_msg.prob)
  elif track is None and ready and lead_msg.prob > .5:
    lead_dict = get_RadarState_from_vision(lead_msg, v_ego, model_v_ego)

  if low_speed_override:
    low_speed_tracks = [c for c in tracks.values() if c.potential_low_speed_lead(v_ego)]
    if low_speed_tracks:
      closest_track = min(low_speed_tracks, key=lambda c: c.dRel)
      if (not lead_dict['status']) or (closest_track.dRel < lead_dict['dRel']):
        lead_dict = closest_track.get_RadarState()

  return lead_dict


class RadarD:
  def __init__(self, radar_ts: float, delay: int = 0):
    self.current_time = 0.0
    self.tracks: Dict[int, Track] = {}
    self.kalman_params = KalmanParams(radar_ts)
    self.v_ego = 0.0
    self.v_ego_hist = deque([0.0], maxlen=delay + 1)
    self.last_v_ego_frame = -1
    self.radar_state: Optional[capnp._DynamicStructBuilder] = None
    self.radar_state_valid = False
    self.ready = False

  def update(self, sm: messaging.SubMaster, rr: Optional[car.RadarData]):
    self.ready = sm.seen['modelV2']
    self.current_time = 1e-9 * max(sm.logMonoTime.values())

    radar_points = rr.points if rr is not None else []
    radar_errors = rr.errors if rr is not None else []

    if sm.recv_frame['carState'] != self.last_v_ego_frame:
      self.v_ego = sm['carState'].vEgo
      self.v_ego_hist.append(self.v_ego)
      self.last_v_ego_frame = sm.recv_frame['carState']

    ar_pts = {pt.trackId: [pt.dRel, pt.yRel, pt.vRel, pt.measured] for pt in radar_points}

    for ids in list(self.tracks.keys()):
      if ids not in ar_pts:
        self.tracks.pop(ids, None)

    for ids, rpt in ar_pts.items():
      v_lead = rpt[2] + self.v_ego_hist[0]
      if ids not in self.tracks:
        self.tracks[ids] = Track(ids, v_lead, self.kalman_params)
      self.tracks[ids].update(rpt[0], rpt[1], rpt[2], v_lead, rpt[3])

    self.radar_state_valid = sm.all_checks() and len(radar_errors) == 0
    self.radar_state = log.RadarState.new_message()
    self.radar_state.mdMonoTime = sm.logMonoTime['modelV2']
    self.radar_state.radarErrors = list(radar_errors)
    self.radar_state.carStateMonoTime = sm.logMonoTime['carState']

    model_v_ego = sm['modelV2'].temporalPose.trans[0] if len(sm['modelV2'].temporalPose.trans) else self.v_ego
    leads_v3 = sm['modelV2'].leadsV3
    if len(leads_v3) > 1:
      self.radar_state.leadOne = get_lead(self.v_ego, self.ready, self.tracks, leads_v3[0], model_v_ego, True)
      self.radar_state.leadTwo = get_lead(self.v_ego, self.ready, self.tracks, leads_v3[1], model_v_ego, False)


def wait_for_model_frame(sm: messaging.SubMaster, target_frame_id: int,
                         timeout_ms: int = MODEL_FRAME_TIMEOUT_MS) -> bool:
  if sm.seen['modelV2']:
    current_frame_id = sm['modelV2'].frameId
    if current_frame_id == target_frame_id:
      return True
    if current_frame_id > target_frame_id:
      return False

  deadline = time.monotonic() + timeout_ms / 1000.0
  while time.monotonic() < deadline:
    remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
    sm.update(min(MODEL_FRAME_POLL_MS, remaining_ms))
    if not sm.updated['modelV2']:
      continue

    current_frame_id = sm['modelV2'].frameId
    if current_frame_id < target_frame_id:
      continue
    return current_frame_id == target_frame_id

  return False


def save_frame(frame_data: np.ndarray, height: int, width: int, fname: str):
  yuv = np.asarray(frame_data)
  if yuv.ndim == 1:
    yuv = yuv.reshape((height + height // 2, width))
  rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_NV12)
  cv2.imwrite(fname, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def main():
  client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, True)
  client.connect(True)
  time.sleep(2.0)

  idx = START_IDX

  can_sock = messaging.sub_sock('can')
  sm = messaging.SubMaster(['modelV2', 'carState'], frequency=int(1. / DT_CTRL))

  with car.CarParams.from_bytes(Params().get("CarParams", block=True)) as msg:
    CP = msg
  RadarInterface = importlib.import_module(f'selfdrive.car.{CP.carName}.radar_interface').RadarInterface
  RI = RadarInterface(CP)

  RD = RadarD(CP.radarTimeStep, RI.delay)

  # ==================== Initialize CSV DOC  ====================
  os.makedirs(CSV_DIR, exist_ok=True)
  CSV_FILE = f"{CSV_DIR}/leadOne.csv"


  fieldnames = ['frame_idx', 'vision_frame_id', 'model_frame_id', 'image_path',
                'dRel', 'yRel', 'vRel', 'vLead', 'vLeadK',
                'aLeadK', 'aLeadTau', 'status', 'fcw', 'modelProb', 'radar', 'radarTrackId',
                'raw_vision_dRel', 'vEgo']

  with open(CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
  print(f"CSV doc has been created: {CSV_FILE}")
  # ===================================================

  captured_frames = 0
  while captured_frames < NUM_FRAMES:
    buf = client.recv()
    if buf is None:
      continue

    vision_frame_id = client.frame_id
    frame_data = np.array(buf.data, copy=True)
    frame_height, frame_width = buf.height, buf.width

    if not wait_for_model_frame(sm, vision_frame_id):
      latest_model_frame_id = sm['modelV2'].frameId if sm.seen['modelV2'] else -1
      print(f"Skipping vision frame {vision_frame_id}: latest modelV2 frame is {latest_model_frame_id}")
      continue

    model_frame_id = sm['modelV2'].frameId
    frame_idx = idx
    fname = f"{SAVE_DIR}/{frame_idx}.png"
    save_frame(frame_data, frame_height, frame_width, fname)
    idx += 1
    captured_frames += 1

    can_strings = messaging.drain_sock_raw(can_sock, wait_for_one=True)
    rr = RI.update(can_strings)

    if rr is None:
      continue

    RD.update(sm, rr)


    raw_vision_dRel = 0.0
    leads_v3 = sm['modelV2'].leadsV3
    if len(leads_v3) > 0:
 
      raw_vision_dRel = float(leads_v3[0].x[0] - RADAR_TO_CAMERA)

    current_v_ego = float(sm['carState'].vEgo) if sm.updated['carState'] else RD.v_ego
    # ===========================================================


    lead_one = RD.radar_state.leadOne
    image_path = f"{SAVE_DIR}/{frame_idx}.png"

    if lead_one.status:
      row = {
        'frame_idx': frame_idx,
        'vision_frame_id': vision_frame_id,
        'model_frame_id': model_frame_id,
        'image_path': image_path,
        'dRel': float(lead_one.dRel),
        'yRel': float(lead_one.yRel),
        'vRel': float(lead_one.vRel),
        'vLead': float(lead_one.vLead),
        'vLeadK': float(lead_one.vLeadK),
        'aLeadK': float(lead_one.aLeadK),
        'aLeadTau': float(lead_one.aLeadTau),
        'status': bool(lead_one.status),
        'fcw': bool(lead_one.fcw),
        'modelProb': float(lead_one.modelProb),
        'radar': bool(lead_one.radar),
        'radarTrackId': int(lead_one.radarTrackId),
        'raw_vision_dRel': raw_vision_dRel,
        'vEgo': current_v_ego
      }
    else:
      row = {
        'frame_idx': frame_idx,
        'vision_frame_id': vision_frame_id,
        'model_frame_id': model_frame_id,
        'image_path': image_path,
        'dRel': None,
        'yRel': None,
        'vRel': None,
        'vLead': None,
        'vLeadK': None,
        'aLeadK': None,
        'aLeadTau': None,
        'status': False,
        'fcw': False,
        'modelProb': 0.0,
        'radar': False,
        'radarTrackId': -1,
        'raw_vision_dRel': raw_vision_dRel,
        'vEgo': current_v_ego
      }

    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as csvfile:
      writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
      writer.writerow(row)

    print(f"Saved frame {frame_idx} | dRel: {raw_vision_dRel:.2f}m | vEgo: {current_v_ego:.2f}m/s")
    # ===========================================================

    time.sleep(INTERVAL)

  print("Done.")


if __name__ == "__main__":
  time.sleep(0)
  main()
