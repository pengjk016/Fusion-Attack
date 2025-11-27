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

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# def save_model_input(img_cl_buf, width, height, fname):
#     # 把 CL buffer 拷贝到 numpy
#     img = np.frombuffer(img_cl_buf.get_host(), dtype=np.uint8)
#     img = img.reshape(height, width, 3)  # 模型输入是 HWC 格式
#     Image.fromarray(img).save(fname, "JPEG")
#
#

PROCESS_NAME = "selfdrive.modeld.modeld"
SEND_RAW_PRED = os.getenv('SEND_RAW_PRED')

MODEL_PATHS = {
  ModelRunner.THNEED: Path(__file__).parent / 'models/supercombo.thneed',
  ModelRunner.ONNX: Path(__file__).parent / 'models/supercombo.onnx'}

METADATA_PATH = Path(__file__).parent / 'models/supercombo_metadata.pkl'

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
  prev_desire: np.ndarray  # for tracking the rising edge of the pulse
  model: ModelRunner

  def __init__(self, context: CLContext):
    self.frame = ModelFrame(context)
    self.wide_frame = ModelFrame(context)
    self.prev_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
    self.inputs = {
      'desire': np.zeros(ModelConstants.DESIRE_LEN * (ModelConstants.HISTORY_BUFFER_LEN+1), dtype=np.float32),
      'traffic_convention': np.zeros(ModelConstants.TRAFFIC_CONVENTION_LEN, dtype=np.float32),
      'lateral_control_params': np.zeros(ModelConstants.LATERAL_CONTROL_PARAMS_LEN, dtype=np.float32),
      'prev_desired_curv': np.zeros(ModelConstants.PREV_DESIRED_CURV_LEN * (ModelConstants.HISTORY_BUFFER_LEN+1), dtype=np.float32),
      'nav_features': np.zeros(ModelConstants.NAV_FEATURE_LEN, dtype=np.float32),
      'nav_instructions': np.zeros(ModelConstants.NAV_INSTRUCTION_LEN, dtype=np.float32),
      'features_buffer': np.zeros(ModelConstants.HISTORY_BUFFER_LEN * ModelConstants.FEATURE_LEN, dtype=np.float32),
    }

    with open(METADATA_PATH, 'rb') as f:
      model_metadata = pickle.load(f)

    self.output_slices = model_metadata['output_slices']
    net_output_size = model_metadata['output_shapes']['outputs'][1]#6504
    self.output = np.zeros(net_output_size, dtype=np.float32)
    self.parser = Parser()

    self.model = ModelRunner(MODEL_PATHS, self.output, Runtime.GPU, False, context)
    self.model.addInput("input_imgs", None)
    self.model.addInput("big_input_imgs", None)
    for k,v in self.inputs.items():
      self.model.addInput(k, v)

  def slice_outputs(self, model_outputs: np.ndarray) -> Dict[str, np.ndarray]:
    parsed_model_outputs = {k: model_outputs[np.newaxis, v] for k,v in self.output_slices.items()}
    if SEND_RAW_PRED:
      parsed_model_outputs['raw_pred'] = model_outputs.copy()
    return parsed_model_outputs

  def run(self, buf: VisionBuf, wbuf: VisionBuf, transform: np.ndarray, transform_wide: np.ndarray,
                inputs: Dict[str, np.ndarray], prepare_only: bool) -> Optional[Dict[str, np.ndarray]]:
    # Model decides when action is completed, so desire input is just a pulse triggered on rising edge
    inputs['desire'][0] = 0
    self.inputs['desire'][:-ModelConstants.DESIRE_LEN] = self.inputs['desire'][ModelConstants.DESIRE_LEN:]
    self.inputs['desire'][-ModelConstants.DESIRE_LEN:] = np.where(inputs['desire'] - self.prev_desire > .99, inputs['desire'], 0)
    self.prev_desire[:] = inputs['desire']

    self.inputs['traffic_convention'][:] = inputs['traffic_convention']
    self.inputs['lateral_control_params'][:] = inputs['lateral_control_params']
    self.inputs['nav_features'][:] = inputs['nav_features']
    self.inputs['nav_instructions'][:] = inputs['nav_instructions']

    # print('traffic_convention:',inputs['traffic_convention'])
    # print('lateral_control_params:',inputs['lateral_control_params'])
    # print('nav_features:',inputs['nav_features'])
    # print('nav_instructions:',inputs['nav_instructions'])

    # if getCLBuffer is not None, frame will be None
    self.model.setInputBuffer("input_imgs", self.frame.prepare(buf, transform.flatten(), self.model.getCLBuffer("input_imgs")))
    if wbuf is not None:
      self.model.setInputBuffer("big_input_imgs", self.wide_frame.prepare(wbuf, transform_wide.flatten(), self.model.getCLBuffer("big_input_imgs")))

    if prepare_only:
      return None

    self.model.execute()
    outputs = self.parser.parse_outputs(self.slice_outputs(self.output))


    self.inputs['features_buffer'][:-ModelConstants.FEATURE_LEN] = self.inputs['features_buffer'][ModelConstants.FEATURE_LEN:]
    self.inputs['features_buffer'][-ModelConstants.FEATURE_LEN:] = outputs['hidden_state'][0, :]
    self.inputs['prev_desired_curv'][:-ModelConstants.PREV_DESIRED_CURV_LEN] = self.inputs['prev_desired_curv'][ModelConstants.PREV_DESIRED_CURV_LEN:]
    self.inputs['prev_desired_curv'][-ModelConstants.PREV_DESIRED_CURV_LEN:] = outputs['desired_curvature'][0, :]
    print('lead:',outputs['lead'],'len prob:',outputs['lead_prob'])
    return outputs

#
# def main(demo=False):
#
#   cloudlog.warning("modeld init")
#
#   sentry.set_tag("daemon", PROCESS_NAME)
#   cloudlog.bind(daemon=PROCESS_NAME)
#   setproctitle(PROCESS_NAME)
#   config_realtime_process(7, 54)
#
#   cloudlog.warning("setting up CL context")
#   cl_context = CLContext()
#   cloudlog.warning("CL context ready; loading model")
#   model = ModelState(cl_context)
#   cloudlog.warning("models loaded, modeld starting")
#   yolo_model = YOLO("yolov8n.pt")
#   # visionipc clients
#   while True:
#     available_streams = VisionIpcClient.available_streams("camerad", block=False)
#     if available_streams:
#       use_extra_client = VisionStreamType.VISION_STREAM_WIDE_ROAD in available_streams and VisionStreamType.VISION_STREAM_ROAD in available_streams
#       main_wide_camera = VisionStreamType.VISION_STREAM_ROAD not in available_streams
#       break
#     time.sleep(.1)
#
#   vipc_client_main_stream = VisionStreamType.VISION_STREAM_WIDE_ROAD if main_wide_camera else VisionStreamType.VISION_STREAM_ROAD
#   vipc_client_main = VisionIpcClient("camerad", vipc_client_main_stream, True, cl_context)
#   print(f'\n\n\n\n\ntype:{type(vipc_client_main)}\n\n\n\n\n')
#
#
#   vipc_client_extra = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, False, cl_context)
#   cloudlog.warning(f"vision stream set up, main_wide_camera: {main_wide_camera}, use_extra_client: {use_extra_client}")
#
#   while not vipc_client_main.connect(False):
#     time.sleep(0.1)
#   while use_extra_client and not vipc_client_extra.connect(False):
#     time.sleep(0.1)
#
#   cloudlog.warning(f"connected main cam with buffer size: {vipc_client_main.buffer_len} ({vipc_client_main.width} x {vipc_client_main.height})")
#   if use_extra_client:
#     cloudlog.warning(f"connected extra cam with buffer size: {vipc_client_extra.buffer_len} ({vipc_client_extra.width} x {vipc_client_extra.height})")
#
#   # messaging
#   pm = PubMaster(["modelV2", "cameraOdometry"])
#   sm = SubMaster(["carState", "roadCameraState", "liveCalibration", "driverMonitoringState", "navModel", "navInstruction", "carControl"])
#
#   publish_state = PublishState()
#   params = Params()
#
#   # setup filter to track dropped frames
#   frame_dropped_filter = FirstOrderFilter(0., 10., 1. / ModelConstants.MODEL_FREQ)
#   frame_id = 0
#   last_vipc_frame_id = 0
#   run_count = 0
#
#   model_transform_main = np.zeros((3, 3), dtype=np.float32)
#   model_transform_extra = np.zeros((3, 3), dtype=np.float32)
#   live_calib_seen = False
#   nav_features = np.zeros(ModelConstants.NAV_FEATURE_LEN, dtype=np.float32)
#   nav_instructions = np.zeros(ModelConstants.NAV_INSTRUCTION_LEN, dtype=np.float32)
#   buf_main, buf_extra = None, None
#   meta_main = FrameMeta()
#   meta_extra = FrameMeta()
#
#
#   if demo:
#     CP = get_demo_car_params()
#   else:
#     with car.CarParams.from_bytes(params.get("CarParams", block=True)) as msg:
#       CP = msg
#   cloudlog.info("modeld got CarParams: %s", CP.carName)
#
#   # TODO this needs more thought, use .2s extra for now to estimate other delays
#   steer_delay = CP.steerActuatorDelay + .2
#
#   DH = DesireHelper()
#
#   while True:
#     # Keep receiving frames until we are at least 1 frame ahead of previous extra frame
#     while meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
#       buf_main = vipc_client_main.recv()
#       meta_main = FrameMeta(vipc_client_main)
#       if buf_main is None:
#         break
#
#     if buf_main is None:
#       cloudlog.error("vipc_client_main no frame")
#       continue
#
#
#     yuv_data = buf_main.data
#     img_width = vipc_client_main.width
#     img_height = vipc_client_main.height
#
#     yuv_nv12 = np.frombuffer(yuv_data, dtype=np.uint8).reshape((img_height + img_height // 2, img_width))
#
#     # NV12 → BGR（直接用对应转换常量，OpenCV自动处理U/V上采样）
#     bgr_img = cv2.cvtColor(yuv_nv12, cv2.COLOR_YUV2BGR_NV12)
#     # -----------------------------------------------------------------------------------
#
#     # -------------------------- YOLO 实时推理（保持不变） --------------------------
#     # 推理（设置 conf=0.3 过滤低置信度目标，verbose=False 关闭冗余输出）
#     results = yolo_model(bgr_img, conf=0.3, verbose=False)
#     car_detections = [box for box in results[0].boxes if box.cls == 2]  # 2 = 车辆（COCO 类别）
#     print(f"\nYOLO 识别结果：车辆 {len(car_detections)} 辆")
#     #
#     #
#     # #绘图
#     # flag = True
#     # if len(car_detections)==4 and flag:
#     #   flag = False
#     #   first_car = car_detections[0]
#     #   x1, y1, x2, y2 = map(int, first_car.xyxy[0].cpu().numpy())  # 边界框坐标（整数化）
#     #   conf = round(float(first_car.conf[0].cpu().numpy()), 2)  # 置信度
#     #
#     #   # 转换图像格式：OpenCV（BGR）→ matplotlib（RGB）
#     #   img_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
#     #
#     #   # 创建绘图对象
#     #   plt.figure(figsize=(10, 6))
#     #   plt.imshow(img_rgb)
#     #
#     #   # 绘制矩形bbox（红色，线宽2）
#     #   rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
#     #                        fill=False, color='red', linewidth=2)
#     #   plt.gca().add_patch(rect)
#     #
#     #   # 添加置信度标签（红色背景，白色文字）
#     #   plt.text(x1, y1 - 10, f"Car (conf: {conf})",
#     #            color='white', backgroundcolor='red',
#     #            fontsize=10, fontweight='bold')
#     #
#     #   # 隐藏坐标轴，让图片更清晰
#     #   plt.axis('off')
#     #   plt.title("YOLO Vehicle Detection (with Bounding Box)", fontsize=12)
#     #   plt.tight_layout()
#     #   plt.savefig("./vehicle_detection_with_bbox.png", dpi=150, bbox_inches='tight')
#     #   plt.show()  # 显示图片（阻塞程序，关闭窗口后继续运行）
#
#     for box in car_detections:
#       x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
#       conf = round(float(box.conf[0].cpu().numpy()), 2)
#       print(f"  车辆 - 置信度：{conf}，边界框：({x1},{y1})-({x2},{y2})")
#
#
#
#
#     if use_extra_client:
#       # Keep receiving extra frames until frame id matches main camera
#       while True:
#         buf_extra = vipc_client_extra.recv()
#         meta_extra = FrameMeta(vipc_client_extra)
#         if buf_extra is None or meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
#           break
#
#       if buf_extra is None:
#         cloudlog.error("vipc_client_extra no frame")
#         continue
#
#       if abs(meta_main.timestamp_sof - meta_extra.timestamp_sof) > 10000000:
#         cloudlog.error("frames out of sync! main: {} ({:.5f}), extra: {} ({:.5f})".format(
#           meta_main.frame_id, meta_main.timestamp_sof / 1e9,
#           meta_extra.frame_id, meta_extra.timestamp_sof / 1e9))
#
#     else:
#       # Use single camera
#       buf_extra = buf_main
#       meta_extra = meta_main
#
#     sm.update(0)
#     desire = DH.desire
#     is_rhd = sm["driverMonitoringState"].isRHD
#     frame_id = sm["roadCameraState"].frameId
#     lateral_control_params = np.array([sm["carState"].vEgo, steer_delay], dtype=np.float32)
#     if sm.updated["liveCalibration"]:
#       device_from_calib_euler = np.array(sm["liveCalibration"].rpyCalib, dtype=np.float32)
#       model_transform_main = get_warp_matrix(device_from_calib_euler, main_wide_camera, False).astype(np.float32)
#       model_transform_extra = get_warp_matrix(device_from_calib_euler, True, True).astype(np.float32)
#       live_calib_seen = True
#
#     traffic_convention = np.zeros(2)
#     traffic_convention[int(is_rhd)] = 1
#
#     vec_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
#     if desire >= 0 and desire < ModelConstants.DESIRE_LEN:
#       vec_desire[desire] = 1
#
#     # Enable/disable nav features
#     timestamp_llk = sm["navModel"].locationMonoTime
#     nav_valid = sm.valid["navModel"] # and (nanos_since_boot() - timestamp_llk < 1e9)
#     nav_enabled = nav_valid and params.get_bool("ExperimentalMode")
#
#     if not nav_enabled:
#       nav_features[:] = 0
#       nav_instructions[:] = 0
#
#     if nav_enabled and sm.updated["navModel"]:
#       nav_features = np.array(sm["navModel"].features)
#
#     if nav_enabled and sm.updated["navInstruction"]:
#       nav_instructions[:] = 0
#       for maneuver in sm["navInstruction"].allManeuvers:
#         distance_idx = 25 + int(maneuver.distance / 20)
#         direction_idx = 0
#         if maneuver.modifier in ("left", "slight left", "sharp left"):
#           direction_idx = 1
#         if maneuver.modifier in ("right", "slight right", "sharp right"):
#           direction_idx = 2
#         if 0 <= distance_idx < 50:
#           nav_instructions[distance_idx*3 + direction_idx] = 1
#
#     # tracked dropped frames
#     vipc_dropped_frames = max(0, meta_main.frame_id - last_vipc_frame_id - 1)
#     frames_dropped = frame_dropped_filter.update(min(vipc_dropped_frames, 10))
#     if run_count < 10: # let frame drops warm up
#       frame_dropped_filter.x = 0.
#       frames_dropped = 0.
#     run_count = run_count + 1
#
#     frame_drop_ratio = frames_dropped / (1 + frames_dropped)
#     prepare_only = vipc_dropped_frames > 0
#     if prepare_only:
#       cloudlog.error(f"skipping model eval. Dropped {vipc_dropped_frames} frames")
#
#     inputs:Dict[str, np.ndarray] = {
#       'desire': vec_desire,
#       'traffic_convention': traffic_convention,
#       'lateral_control_params': lateral_control_params,
#       'nav_features': nav_features,
#       'nav_instructions': nav_instructions}
#
#     mt1 = time.perf_counter()
#     model_output = model.run(buf_main, buf_extra, model_transform_main, model_transform_extra, inputs, prepare_only)
#     mt2 = time.perf_counter()
#     model_execution_time = mt2 - mt1
#
#     if model_output is not None:
#       modelv2_send = messaging.new_message('modelV2')
#       posenet_send = messaging.new_message('cameraOdometry')
#       fill_model_msg(modelv2_send, model_output, publish_state, meta_main.frame_id, meta_extra.frame_id, frame_id, frame_drop_ratio,
#                       meta_main.timestamp_eof, timestamp_llk, model_execution_time, nav_enabled, live_calib_seen)
#
#       desire_state = modelv2_send.modelV2.meta.desireState
#       l_lane_change_prob = desire_state[log.Desire.laneChangeLeft]
#       r_lane_change_prob = desire_state[log.Desire.laneChangeRight]
#       lane_change_prob = l_lane_change_prob + r_lane_change_prob
#       DH.update(sm['carState'], sm['carControl'].latActive, lane_change_prob)
#       modelv2_send.modelV2.meta.laneChangeState = DH.lane_change_state
#       modelv2_send.modelV2.meta.laneChangeDirection = DH.lane_change_direction
#
#       fill_pose_msg(posenet_send, model_output, meta_main.frame_id, vipc_dropped_frames, meta_main.timestamp_eof, live_calib_seen)
#       pm.send('modelV2', modelv2_send)
#       pm.send('cameraOdometry', posenet_send)
#
#     last_vipc_frame_id = meta_main.frame_id


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
  yolo_model = YOLO("yolov8n.pt")

  # -------------------------- 加载预优化好的 opt_patch --------------------------
  # 假设你的 opt_patch 已保存为 numpy 数组（如果是 torch 张量，需先转 numpy）
  # 若 opt_patch 是实时优化的，可在此处添加优化逻辑（参考你之前的有限差分代码）
  # 这里直接使用你提供的 opt_patch（转换为 numpy 数组）
  opt_patch = np.array([
    [13.041, 92.308, -5.5909, 18.461, 69.593, 83.897, 88.423, -14.179, 86.214, 41.372, 38.864, 59.82, 24.02, 47.698,
     64.686, 38.133, 24.204, 95.593, 26.735, 27.112, 91.057, 94.355, 27.739],
    [62.041, 89.393, 27.762, 100.64, 55.161, 75.135, 99.729, 50.51, 57.697, 55.852, 59.305, 30.097, 42.109, 43.056,
     71.612, 34.148, 24.362, 101.44, 21.334, 3.9434, 30.453, 64.399, 105.86],
    [38.104, 78.35, 92.295, 83.472, 105.65, 31.837, 27.238, 67.908, 8.681, 86.623, 33.784, 25.141, 51.9, 21.759, 21.937,
     58.866, 39.218, 42.279, 65.593, 19.061, 93.792, 7.283, 53.97],
    [74.168, 43.5, 21.183, 103.51, 102.8, 85.765, 91.181, 77.072, 63.957, 16.333, 28.352, 65.876, -3.7709, 63.752,
     8.9939, 49.561, 55.489, 74.419, 33.535, 77.386, 17.27, 89.562, 44.51],
    [83.073, 44.455, 72.518, 79.754, 59.668, 12.172, 66.722, 14.386, 90.393, 111.31, 81.1, 24.832, 59.764, 63.045,
     47.921, 78.555, 95.039, 17.227, 83.779, 17.736, 57.696, 54.956, 29.94],
    [40.442, 21.578, 70.526, 58.797, 43.037, 52.006, 40.137, 62.073, 98.589, 86.519, 94.95, 47.995, 85.061, 50.284,
     23.254, 28.681, 52.186, 41.995, 62.595, 61.852, 20.034, 39.558, 6.5472],
    [15.049, 16.407, 32.733, 90.399, 32.891, 23.174, 60.274, 27.72, 77.007, 67.296, 25.387, 57.793, 22.662, 6.948,
     10.059, 46.774, 76.621, 15.532, 56.36, 51.689, 76.376, 55.329, 31.601],
    [63.399, 66.998, 32.703, 7.7389, 64.985, 101.32, 64.183, 24.035, -0.35746, 70.967, -15.673, 84.601, 46.887, 33.475,
     37.658, 27.909, 34.791, 5.7715, 65.383, 66.417, 49.918, 27.694, 22.482],
    [44.208, 62.842, 48.222, 95.852, 18.906, 83.933, 13.729, 79.437, 27.924, 28.391, 74.238, 71.112, 83.121, 58.996,
     76.403, 82.071, 57.518, 99.634, 5.0734, 87.354, 77.381, 67.203, 37.622],
    [84.106, 72.779, 89.404, 95.852, 49.072, 13.771, 16.878, 60.797, 7.1996, 75.007, 66.933, 90.898, 55.696, 56.295,
     62.281, 1.0447, 47.194, 75.473, 2.6946, 69.838, 63.117, 7.0932, 49.281],
    [24.355, 71.914, 47.693, 66.022, 65.962, 8.1397, 11.221, 55.859, 50.024, 72.353, 17.257, 101.99, 111.53, 26.128,
     76.929, 32.988, 22.32, 38.469, 87.169, 33.031, 16.398, 85.065, 49.515],
    [1.1777, 48.165, 15.965, 26.632, 82.549, 39.948, -17.464, 102.84, 24.973, 85.128, 0.63034, 111.21, 29.786, 5.9087,
     35.44, 4.4716, -0.95903, 82.504, 50.122, 94.686, 29.354, 34.091, 6.3662],
    [18.441, 76.246, 40.924, 68.943, 30.889, 96.047, 46.224, 35.884, 28.239, 88.514, 44.664, 74.667, 36.312, 44.63,
     -1.1777, 45.398, 28.037, 46.538, 48.005, 78.628, 53.055, 13.645, 90.318],
    [83.325, 31.504, 58.458, 23.86, 35.397, 53.302, 4.8758, 65.192, 47.612, 53.824, -6.0655, 38.742, 36.319, 47.425,
     35.322, 1.1593, 15.362, 82.984, 49.894, 66.545, 9.993, 87.374, 55.184],
    [94.423, 29.375, 67.449, 1.1583, 65.119, 24.913, 4.4537, 61.556, 15.59, 73.986, 43.5, 51.809, 20.351, 70.184,
     38.684, 84.825, 23.202, 42.163, 91.585, 53.197, 43.583, 58.993, 34.895],
    [9.0714, 73.079, 52.137, 79.982, 35.349, 45.756, 128.58, 12.313, 42.168, 12.773, 17.815, 44.147, 81.129, 9.7169,
     53.673, 3.2985, 10.508, 25.977, 63.037, 10.136, 53.138, 14.971, 14.832],
    [53.387, 78.541, 86.136, 89.708, 59.911, 81.369, 21.93, 71.118, 55.748, 51.028, 26.788, 19.835, 23.963, 63.239,
     8.8009, 84.823, 65.774, 60.856, 48.707, 59.436, 39.527, 46.567, 85.943],
    [105.96, -9.5596, 45.481, 46.625, 12.84, 83.074, 23.569, 17.844, 40.363, 34.423, 32.309, -5.6871, 50.895, 73.673,
     11.091, 26.897, 94.84, 60.898, 40.447, 64.34, 9.3115, 89.242, 111.66],
    [122.83, -10.333, 12.565, 38.562, 13.57, 99.505, 44.894, 56.897, 3.2257, 41.834, 65.348, 53.895, 31.052, 112.83,
     102.85, 50.035, 23.789, 52.854, 28.767, -12.552, 61.151, 50.465, 32.339]
  ], dtype=np.float32)

  # -------------------------- 原有初始化逻辑（不变） --------------------------
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
  cloudlog.warning(f"vision stream set up, main_wide_camera: {main_wide_camera}, use_extra_client: {use_extra_client}")

  while not vipc_client_main.connect(False):
    time.sleep(0.1)
  while use_extra_client and not vipc_client_extra.connect(False):
    time.sleep(0.1)

  cloudlog.warning(
    f"connected main cam with buffer size: {vipc_client_main.buffer_len} ({vipc_client_main.width} x {vipc_client_main.height})")
  if use_extra_client:
    cloudlog.warning(
      f"connected extra cam with buffer size: {vipc_client_extra.buffer_len} ({vipc_client_extra.width} x {vipc_client_extra.height})")

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
  cloudlog.info("modeld got CarParams: %s", CP.carName)

  steer_delay = CP.steerActuatorDelay + .2
  DH = DesireHelper()

  # -------------------------- 主循环（添加 patch 叠加逻辑） --------------------------
  while True:
    # 接收相机帧（原有逻辑不变）
    while meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
      buf_main = vipc_client_main.recv()
      meta_main = FrameMeta(vipc_client_main)
      if buf_main is None:
        break

    if buf_main is None:
      cloudlog.error("vipc_client_main no frame")
      continue

    # -------------------------- 1. 提取相机图像并转换格式 --------------------------
    yuv_data = buf_main.data
    img_width = vipc_client_main.width
    img_height = vipc_client_main.height

    # NV12 → 2D numpy 数组（原有逻辑不变）
    yuv_nv12 = np.frombuffer(yuv_data, dtype=np.uint8).reshape((img_height + img_height // 2, img_width))

    # NV12 → BGR（用于 YOLO 检测和 patch 叠加）
    bgr_img = cv2.cvtColor(yuv_nv12, cv2.COLOR_YUV2BGR_NV12)

    # -------------------------- 2. YOLO 检测车辆框（原有逻辑优化） --------------------------
    results = yolo_model(bgr_img, conf=0.3, verbose=False)
    car_detections = [box for box in results[0].boxes if box.cls == 2]  # 筛选车辆（COCO 类别 2）
    target_box = None
    if car_detections:
      # 筛选最大的车辆框（默认是前车）
      max_area = 0
      for box in car_detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        area = (x2 - x1) * (y2 - y1)
        if area > max_area and area > 100:  # 过滤极小框（避免噪声）
          max_area = area
          target_box = (x1, y1, x2, y2)
      print(f"YOLO 检测到前车框：{target_box}（面积：{max_area}）")

    # -------------------------- 3. 叠加 opt_patch 到目标框（核心逻辑） --------------------------

    # -------------------------- 3. 叠加 opt_patch 到目标框（已修复维度+边界） --------------------------
    if target_box is not None:
      x1, y1, x2, y2 = target_box
      box_h = y2 - y1  # 目标框高度
      box_w = x2 - x1  # 目标框宽度

      # 1. 裁剪目标框，避免超出图像范围
      x1 = max(0, x1)
      y1 = max(0, y1)
      x2 = min(img_width, x2)
      y2 = min(img_height, y2)
      box_h = y2 - y1
      box_w = x2 - x1
      if box_h <= 0 or box_w <= 0:
        print("目标框超出图像范围，跳过 patch 叠加")
        continue

      # 2. 调整 patch 尺寸+维度（匹配 BGR 图像）
      resized_patch = cv2.resize(opt_patch, (box_w, box_h), interpolation=cv2.INTER_LINEAR)
      resized_patch = np.clip(resized_patch, 0, 255).astype(np.uint8)
      resized_patch_3d = np.repeat(resized_patch[:, :, np.newaxis], 3, axis=2)  # (H,W,3)

      # 3. 叠加 patch 到 BGR 图像（直接替换像素）
      bgr_img[y1:y2, x1:x2] = resized_patch_3d
      print(f"Patch 叠加完成：目标框 ({x1},{y1})-({x2},{y2})，patch 尺寸 ({box_w},{box_h})")

      # -------------------------- 4. 兼容低版本 OpenCV：BGR → NV12（无 bytes 转换） --------------------------
      # 步骤1：BGR → YUV420（平面格式）
      yuv420 = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420)

      # 步骤2：转为 2D 数组（H + H//2, W）
      yuv420_2d = yuv420.reshape((img_height + img_height // 2, img_width))

      # 步骤3：提取 Y 通道和 UV 平面，交错得到 NV12
      Y = yuv420_2d[:img_height, :]  # Y 通道：(H, W)
      uv_plane = yuv420_2d[img_height:, :]  # UV 平面：(H//2, W)
      UV = np.empty_like(uv_plane)
      UV[:, ::2] = uv_plane[:, :img_width // 2]  # U → 偶数列
      UV[:, 1::2] = uv_plane[:, img_width // 2:]  # V → 奇数列
      yuv_nv12_with_patch = np.concatenate([Y, UV], axis=0)  # NV12：(H + H//2, W)

      # -------------------------- 5. 直接写入 VisionBuf 缓冲区（核心修复：无 bytes 转换） --------------------------
      # 关键：将 NV12 的 numpy 数组直接写入 buf_main 的底层数据（避免 bytes 转换）
      # VisionBuf.data 是 buffer 类型，可通过 numpy 直接赋值
      nv12_flat = yuv_nv12_with_patch.flatten()  # 转为 1D 数组（匹配缓冲区存储格式）

      # 确保数组类型和长度匹配
      if nv12_flat.dtype != np.uint8:
        nv12_flat = nv12_flat.astype(np.uint8)
      if len(nv12_flat) > len(buf_main.data):
        nv12_flat = nv12_flat[:len(buf_main.data)]  # 截断过长部分

      # 直接通过 numpy 将数据写入缓冲区（避免类型错误）
      np_buf = np.frombuffer(buf_main.data, dtype=np.uint8)
      np_buf[:len(nv12_flat)] = nv12_flat
      print(f"带 patch 的 NV12 已写入缓冲区：长度 {len(nv12_flat)}/{len(buf_main.data)}")
    else:
      print("未检测到有效车辆框，跳过 patch 叠加")

    # -------------------------- 原有后续逻辑（不变） --------------------------
    if use_extra_client:
      while True:
        buf_extra = vipc_client_extra.recv()
        meta_extra = FrameMeta(vipc_client_extra)
        if buf_extra is None or meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
          break

      if buf_extra is None:
        cloudlog.error("vipc_client_extra no frame")
        continue

      if abs(meta_main.timestamp_sof - meta_extra.timestamp_sof) > 10000000:
        cloudlog.error("frames out of sync! main: {} ({:.5f}), extra: {} ({:.5f})".format(
          meta_main.frame_id, meta_main.timestamp_sof / 1e9,
          meta_extra.frame_id, meta_extra.timestamp_sof / 1e9))
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
    if prepare_only:
      cloudlog.error(f"skipping model eval. Dropped {vipc_dropped_frames} frames")

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
