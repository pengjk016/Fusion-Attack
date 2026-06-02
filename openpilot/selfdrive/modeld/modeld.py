#!/usr/bin/env python3
import os
import time
import pickle
import numpy as np
import cereal.messaging as messaging
from cereal import car, log
from pathlib import Path
from typing import Dict, Optional
from setproctitle import setproctitle
from cereal.messaging import PubMaster, SubMaster
from cereal.visionipc import VisionIpcClient, VisionStreamType, VisionBuf
from openpilot.common.swaglog import cloudlog
from openpilot.common.params import Params
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import config_realtime_process
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.selfdrive import sentry
from openpilot.selfdrive.car.car_helpers import get_demo_car_params
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper
from openpilot.selfdrive.modeld.runners import ModelRunner, Runtime
from openpilot.selfdrive.modeld.parse_model_outputs import Parser
from openpilot.selfdrive.modeld.fill_model_msg import fill_model_msg, fill_pose_msg, PublishState
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.models.commonmodel_pyx import ModelFrame, CLContext

import cv2
import zmq
from PIL import Image

PROCESS_NAME = "selfdrive.modeld.modeld"
SEND_RAW_PRED = os.getenv('SEND_RAW_PRED')

MODEL_PATHS = {
  ModelRunner.THNEED: Path(__file__).parent / 'models/supercombo.thneed',
  ModelRunner.ONNX: Path(__file__).parent / 'models/supercombo.onnx'}

METADATA_PATH = Path(__file__).parent / 'models/supercombo_metadata.pkl'


# ENABLE_PATCH = True # 改成True就开启攻击
PATCH_MODE_FILE = "/tmp/adversarial_patch_mode"
DEFAULT_PATCH_MODE = "add"


def sync_buffer_to_device(buf: VisionBuf) -> None:
  err = buf.sync_to_device()
  if err != 0:
    cloudlog.warning(f"failed to sync patched frame to device: {err}")


def get_patch_mode() -> str:
  try:
    mode = Path(PATCH_MODE_FILE).read_text(encoding='utf-8').strip().lower()
  except OSError:
    mode = DEFAULT_PATCH_MODE

  return mode if mode in ("replace", "add") else DEFAULT_PATCH_MODE


def apply_patch_to_roi(roi: np.ndarray, patch: np.ndarray, patch_mode: str) -> np.ndarray:
  if patch_mode == "add":
    return np.clip(roi.astype(np.float32) + patch, 0, 255).astype(np.uint8)
  return np.clip(patch, 0, 255).astype(np.uint8)

class FrameMeta:
  frame_id: int = 0
  timestamp_sof: int = 0
  timestamp_eof: int = 0

  def __init__(self, vipc=None):
    if vipc is not None:
      self.frame_id, self.timestamp_sof, self.timestamp_eof = vipc.frame_id, vipc.timestamp_sof, vipc.timestamp_eof


class ModelState:
  frame: ModelFrame
  wide_frame: ModelFrame
  inputs: Dict[str, np.ndarray]
  output: np.ndarray
  prev_desire: np.ndarray
  model: ModelRunner

  def __init__(self, context: CLContext):
    self.frame = ModelFrame(context)
    self.wide_frame = ModelFrame(context)
    self.prev_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
    self.inputs = {
      'desire': np.zeros(ModelConstants.DESIRE_LEN * (ModelConstants.HISTORY_BUFFER_LEN + 1), dtype=np.float32),
      'traffic_convention': np.zeros(ModelConstants.TRAFFIC_CONVENTION_LEN, dtype=np.float32),
      'lateral_control_params': np.zeros(ModelConstants.LATERAL_CONTROL_PARAMS_LEN, dtype=np.float32),
      'prev_desired_curv': np.zeros(ModelConstants.PREV_DESIRED_CURV_LEN * (ModelConstants.HISTORY_BUFFER_LEN + 1),
                                    dtype=np.float32),
      'nav_features': np.zeros(ModelConstants.NAV_FEATURE_LEN, dtype=np.float32),
      'nav_instructions': np.zeros(ModelConstants.NAV_INSTRUCTION_LEN, dtype=np.float32),
      'features_buffer': np.zeros(ModelConstants.HISTORY_BUFFER_LEN * ModelConstants.FEATURE_LEN, dtype=np.float32),
    }

    with open(METADATA_PATH, 'rb') as f:
      model_metadata = pickle.load(f)

    self.output_slices = model_metadata['output_slices']
    net_output_size = model_metadata['output_shapes']['outputs'][1]
    self.output = np.zeros(net_output_size, dtype=np.float32)
    self.parser = Parser()

    self.model = ModelRunner(MODEL_PATHS, self.output, Runtime.GPU, False, context)
    self.model.addInput("input_imgs", None)
    self.model.addInput("big_input_imgs", None)
    for k, v in self.inputs.items():
      self.model.addInput(k, v)

  def slice_outputs(self, model_outputs: np.ndarray) -> Dict[str, np.ndarray]:
    parsed_model_outputs = {k: model_outputs[np.newaxis, v] for k, v in self.output_slices.items()}
    if SEND_RAW_PRED:
      parsed_model_outputs['raw_pred'] = model_outputs.copy()
    return parsed_model_outputs

  def run(self, buf: VisionBuf, wbuf: VisionBuf, transform: np.ndarray, transform_wide: np.ndarray,
          inputs: Dict[str, np.ndarray], prepare_only: bool) -> Optional[Dict[str, np.ndarray]]:
    inputs['desire'][0] = 0
    self.inputs['desire'][:-ModelConstants.DESIRE_LEN] = self.inputs['desire'][ModelConstants.DESIRE_LEN:]
    self.inputs['desire'][-ModelConstants.DESIRE_LEN:] = np.where(inputs['desire'] - self.prev_desire > .99,
                                                                  inputs['desire'], 0)
    self.prev_desire[:] = inputs['desire']

    self.inputs['traffic_convention'][:] = inputs['traffic_convention']
    self.inputs['lateral_control_params'][:] = inputs['lateral_control_params']
    self.inputs['nav_features'][:] = inputs['nav_features']
    self.inputs['nav_instructions'][:] = inputs['nav_instructions']

    self.model.setInputBuffer("input_imgs",
                              self.frame.prepare(buf, transform.flatten(), self.model.getCLBuffer("input_imgs")))
    if wbuf is not None:
      self.model.setInputBuffer("big_input_imgs", self.wide_frame.prepare(wbuf, transform_wide.flatten(),
                                                                          self.model.getCLBuffer("big_input_imgs")))

    if prepare_only:
      return None

    self.model.execute()
    outputs = self.parser.parse_outputs(self.slice_outputs(self.output))

    self.inputs['features_buffer'][:-ModelConstants.FEATURE_LEN] = self.inputs['features_buffer'][
      ModelConstants.FEATURE_LEN:]
    self.inputs['features_buffer'][-ModelConstants.FEATURE_LEN:] = outputs['hidden_state'][0, :]
    self.inputs['prev_desired_curv'][:-ModelConstants.PREV_DESIRED_CURV_LEN] = self.inputs['prev_desired_curv'][
      ModelConstants.PREV_DESIRED_CURV_LEN:]
    self.inputs['prev_desired_curv'][-ModelConstants.PREV_DESIRED_CURV_LEN:] = outputs['desired_curvature'][0, :]
    return outputs


def main(demo=False):
  cloudlog.warning("modeld init")

  sentry.set_tag("daemon", PROCESS_NAME)
  cloudlog.bind(daemon=PROCESS_NAME)
  setproctitle(PROCESS_NAME)
  config_realtime_process(7, 54)

  cloudlog.warning("setting up CL context")
  cl_context = CLContext()
  cloudlog.warning("CL context ready; loading model")
  model = ModelState(cl_context)
  cloudlog.warning("models loaded, modeld starting")

  zmq_context = zmq.Context()
  zmq_socket = zmq_context.socket(zmq.SUB)
  zmq_socket.connect("tcp://127.0.0.1:5555")
  zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")
  print("ZMQ 客户端已启动，等待接收真实前车 BBox...")

  # ==========================================
  # 预处理对抗补丁：保持真实扰动数值
  # ==========================================
  opt_patch = np.load(Path(__file__).parent / 'optimpatch.npy').astype(np.float32)

  if opt_patch.ndim == 3 and opt_patch.shape[0] in [1, 3, 4]:
    opt_patch = np.transpose(opt_patch, (1, 2, 0))

  if opt_patch.ndim == 3 and opt_patch.shape[-1] == 3:
    opt_patch = opt_patch[:, :, ::-1]  # RGB -> BGR
  elif opt_patch.ndim == 3 and opt_patch.shape[-1] == 4:
    opt_patch = opt_patch[:, :, :3][:, :, ::-1]
  elif opt_patch.ndim == 2:
    opt_patch = np.repeat(opt_patch[:, :, np.newaxis], 3, axis=2)
  # ==========================================

  while True:
    available_streams = VisionIpcClient.available_streams("camerad", block=False)
    if available_streams:
      use_extra_client = VisionStreamType.VISION_STREAM_WIDE_ROAD in available_streams and VisionStreamType.VISION_STREAM_ROAD in available_streams
      main_wide_camera = VisionStreamType.VISION_STREAM_ROAD not in available_streams
      break
    time.sleep(.1)

  vipc_client_main_stream = VisionStreamType.VISION_STREAM_WIDE_ROAD if main_wide_camera else VisionStreamType.VISION_STREAM_ROAD
  vipc_client_main = VisionIpcClient("camerad", vipc_client_main_stream, True, cl_context)

  vipc_client_extra = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, False, cl_context)

  while not vipc_client_main.connect(False):
    time.sleep(0.1)
  while use_extra_client and not vipc_client_extra.connect(False):
    time.sleep(0.1)

  pm = PubMaster(["modelV2", "cameraOdometry"])
  sm = SubMaster(
    ["carState", "roadCameraState", "liveCalibration", "driverMonitoringState", "navModel", "navInstruction",
     "carControl"])

  publish_state = PublishState()
  params = Params()

  frame_dropped_filter = FirstOrderFilter(0., 10., 1. / ModelConstants.MODEL_FREQ)
  frame_id = 0
  last_vipc_frame_id = 0
  run_count = 0

  model_transform_main = np.zeros((3, 3), dtype=np.float32)
  model_transform_extra = np.zeros((3, 3), dtype=np.float32)
  live_calib_seen = False
  nav_features = np.zeros(ModelConstants.NAV_FEATURE_LEN, dtype=np.float32)
  nav_instructions = np.zeros(ModelConstants.NAV_INSTRUCTION_LEN, dtype=np.float32)
  buf_main, buf_extra = None, None
  meta_main = FrameMeta()
  meta_extra = FrameMeta()

  if demo:
    CP = get_demo_car_params()
  else:
    with car.CarParams.from_bytes(params.get("CarParams", block=True)) as msg:
      CP = msg

  steer_delay = CP.steerActuatorDelay + .2
  DH = DesireHelper()

  target_box = None

  while True:
    while meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
      buf_main = vipc_client_main.recv()
      meta_main = FrameMeta(vipc_client_main)
      if buf_main is None:
        break

    if buf_main is None:
      continue

    try:
      while True:
        msg = zmq_socket.recv_string(flags=zmq.NOBLOCK)
        if msg != "None":
          u_min, v_min, u_max, v_max = map(int, msg.split(','))
          target_box = (u_min, v_min, u_max, v_max)
        else:
          target_box = None
    except zmq.Again:
      pass

    yuv_data = buf_main.data
    img_width = vipc_client_main.width
    img_height = vipc_client_main.height
    # 简洁版：如果没设置过，默认是 False

    PATCH_SWITCH_FILE = "/tmp/adversarial_patch_enabled"

    # 检查文件是否存在来决定是否开启
    enable_patch_runtime = os.path.exists(PATCH_SWITCH_FILE)
    patch_mode = get_patch_mode()

    if enable_patch_runtime and target_box is not None:
      yuv_nv12 = np.frombuffer(yuv_data, dtype=np.uint8).reshape((img_height + img_height // 2, img_width))
      bgr_img = cv2.cvtColor(yuv_nv12, cv2.COLOR_YUV2BGR_NV12)

      x1, y1, x2, y2 = target_box

      # =========================================================
      # 第一步：全距离自适应视距与物理框校准（MetaDrive 3D 框 -> YOLO 视觉框）
      # 原理：3D物理碰撞盒的固定高度误差，投影到画面上的像素偏差与物体像素高度严格成正比。
      # 在34.5m处(最佳Y_OFFSET=30)，orig_h约115像素。比例系数 = 30 / 115 ≈ 0.26
      # =========================================================
      orig_w = x2 - x1
      orig_h = y2 - y1

      DYNAMIC_Y_OFFSET = orig_h * 0.26 - 5
      X_OFFSET = 0
      SCALE_W = 1.0
      SCALE_H = 1.0

      cx = (x1 + x2) / 2.0 + X_OFFSET
      cy = (y1 + y2) / 2.0 + DYNAMIC_Y_OFFSET

      yolo_w = orig_w * SCALE_W
      yolo_h = orig_h * SCALE_H

      yolo_x1 = cx - yolo_w / 2.0
      yolo_x2 = cx + yolo_w / 2.0
      yolo_y1 = cy - yolo_h / 2.0
      yolo_y2 = cy + yolo_h / 2.0

      # =========================================================
      # 第二步：离线训练裁剪对齐（YOLO 视觉框 -> 最终补丁贴图框）
      # 在校准好的视觉框基础上，严格执行离线训练时的裁剪比例，切掉底盘和轮胎。
      # =========================================================
      PATCH_CROP_RATIO = [0, 0.72, 0, 1]  # [Top, Bottom, Left, Right]

      new_x1 = int(yolo_x1 + yolo_w * PATCH_CROP_RATIO[2])
      new_y1 = int(yolo_y1 + yolo_h * PATCH_CROP_RATIO[0])
      new_x2 = int(yolo_x1 + yolo_w * PATCH_CROP_RATIO[3])
      new_y2 = int(yolo_y1 + yolo_h * PATCH_CROP_RATIO[1])

      box_w = new_x2 - new_x1
      box_h = new_y2 - new_y1

      x1, y1, x2, y2 = new_x1, new_y1, new_x2, new_y2
      # =========================================================

      if box_h > 0 and box_w > 0:
        resized_patch = cv2.resize(opt_patch, (box_w, box_h), interpolation=cv2.INTER_NEAREST)

        # 安全验证：如果越界则丢弃，防止 Numpy 切片广播报错
        if x1 >= 0 and y1 >= 0 and x2 <= img_width and y2 <= img_height:
          roi = bgr_img[y1:y2, x1:x2]
          bgr_img[y1:y2, x1:x2] = apply_patch_to_roi(roi, resized_patch, patch_mode)

          yuv_i420 = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420)
          yuv_flat = yuv_i420.flatten()
          y_size = img_height * img_width
          uv_size = (img_height // 2) * (img_width // 2)

          Y = yuv_flat[:y_size].reshape(img_height, img_width)
          U = yuv_flat[y_size: y_size + uv_size].reshape(img_height // 2, img_width // 2)
          V = yuv_flat[y_size + uv_size: y_size + 2 * uv_size].reshape(img_height // 2, img_width // 2)

          uv_nv12 = np.zeros((img_height // 2, img_width), dtype=np.uint8)
          uv_nv12[:, 0::2] = U
          uv_nv12[:, 1::2] = V

          yuv_nv12_with_patch = np.vstack([Y, uv_nv12])

          nv12_flat = yuv_nv12_with_patch.flatten().astype(np.uint8)
          np_buf = np.frombuffer(buf_main.data, dtype=np.uint8)
          np_buf[:len(nv12_flat)] = nv12_flat
          sync_buffer_to_device(buf_main)

    if use_extra_client:
      while True:
        buf_extra = vipc_client_extra.recv()
        meta_extra = FrameMeta(vipc_client_extra)
        if buf_extra is None or meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
          break
      if buf_extra is None:
        continue
    else:
      buf_extra = buf_main
      meta_extra = meta_main

    sm.update(0)
    desire = DH.desire
    is_rhd = sm["driverMonitoringState"].isRHD
    frame_id = sm["roadCameraState"].frameId
    lateral_control_params = np.array([sm["carState"].vEgo, steer_delay], dtype=np.float32)

    if sm.updated["liveCalibration"]:
      device_from_calib_euler = np.array(sm["liveCalibration"].rpyCalib, dtype=np.float32)
      model_transform_main = get_warp_matrix(device_from_calib_euler, main_wide_camera, False).astype(np.float32)
      model_transform_extra = get_warp_matrix(device_from_calib_euler, True, True).astype(np.float32)
      live_calib_seen = True

    traffic_convention = np.zeros(2)
    traffic_convention[int(is_rhd)] = 1

    vec_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
    if desire >= 0 and desire < ModelConstants.DESIRE_LEN:
      vec_desire[desire] = 1

    timestamp_llk = sm["navModel"].locationMonoTime
    nav_valid = sm.valid["navModel"]
    nav_enabled = nav_valid and params.get_bool("ExperimentalMode")

    if not nav_enabled:
      nav_features[:] = 0
      nav_instructions[:] = 0

    if nav_enabled and sm.updated["navModel"]:
      nav_features = np.array(sm["navModel"].features)

    if nav_enabled and sm.updated["navInstruction"]:
      nav_instructions[:] = 0
      for maneuver in sm["navInstruction"].allManeuvers:
        distance_idx = 25 + int(maneuver.distance / 20)
        direction_idx = 0
        if maneuver.modifier in ("left", "slight left", "sharp left"):
          direction_idx = 1
        if maneuver.modifier in ("right", "slight right", "sharp right"):
          direction_idx = 2
        if 0 <= distance_idx < 50:
          nav_instructions[distance_idx * 3 + direction_idx] = 1

    vipc_dropped_frames = max(0, meta_main.frame_id - last_vipc_frame_id - 1)
    frames_dropped = frame_dropped_filter.update(min(vipc_dropped_frames, 10))
    if run_count < 10:
      frame_dropped_filter.x = 0.
      frames_dropped = 0.
    run_count = run_count + 1

    frame_drop_ratio = frames_dropped / (1 + frames_dropped)
    prepare_only = vipc_dropped_frames > 0

    inputs: Dict[str, np.ndarray] = {
      'desire': vec_desire,
      'traffic_convention': traffic_convention,
      'lateral_control_params': lateral_control_params,
      'nav_features': nav_features,
      'nav_instructions': nav_instructions}

    mt1 = time.perf_counter()
    model_output = model.run(buf_main, buf_extra, model_transform_main, model_transform_extra, inputs, prepare_only)
    mt2 = time.perf_counter()
    model_execution_time = mt2 - mt1

    if model_output is not None:
      modelv2_send = messaging.new_message('modelV2')
      posenet_send = messaging.new_message('cameraOdometry')
      fill_model_msg(modelv2_send, model_output, publish_state, meta_main.frame_id, meta_extra.frame_id, frame_id,
                     frame_drop_ratio,
                     meta_main.timestamp_eof, timestamp_llk, model_execution_time, nav_enabled, live_calib_seen)

      desire_state = modelv2_send.modelV2.meta.desireState
      l_lane_change_prob = desire_state[log.Desire.laneChangeLeft]
      r_lane_change_prob = desire_state[log.Desire.laneChangeRight]
      lane_change_prob = l_lane_change_prob + r_lane_change_prob
      DH.update(sm['carState'], sm['carControl'].latActive, lane_change_prob)
      modelv2_send.modelV2.meta.laneChangeState = DH.lane_change_state
      modelv2_send.modelV2.meta.laneChangeDirection = DH.lane_change_direction

      fill_pose_msg(posenet_send, model_output, meta_main.frame_id, vipc_dropped_frames, meta_main.timestamp_eof,
                    live_calib_seen)
      pm.send('modelV2', modelv2_send)
      pm.send('cameraOdometry', posenet_send)

    last_vipc_frame_id = meta_main.frame_id


if __name__ == "__main__":
  try:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='A boolean for demo mode.')
    args = parser.parse_args()
    main(demo=args.demo)
  except KeyboardInterrupt:
    cloudlog.warning(f"child {PROCESS_NAME} got SIGINT")
  except Exception:
    sentry.capture_exception()
    raise
