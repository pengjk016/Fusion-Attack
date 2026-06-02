#!/usr/bin/env python3
import argparse
import csv
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from cereal import messaging
from cereal.visionipc import VisionIpcServer, VisionStreamType
from openpilot.common.realtime import Ratekeeper
from openpilot.tools.sim.lib.common import H, W


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
DEFAULT_IMAGE_DIR = Path("/home/pjk/PycharmProjects/openpilot0.9.6/CAP/data/imgs/test_optim_patch26314")
DEFAULT_CSV_OUT = Path("/home/pjk/PycharmProjects/openpilot0.9.6/CAP/offline_fusion_results/test_optim_patch26314_model_leads.csv")
DEFAULT_PATCH_PATH = Path("/home/pjk/PycharmProjects/openpilot0.9.6/openpilot/selfdrive/modeld/281.npy")
DEFAULT_YOLO_MODEL = Path("/home/pjk/PycharmProjects/openpilot0.9.6/openpilot/selfdrive/modeld/yolov8n.pt")
MODEL_PATCH_SWITCH_FILE = Path("/tmp/adversarial_patch_enabled")
RADAR_TO_CAMERA = 1.52
MODEL_FRAME_POLL_MS = 20
MODEL_LEAD_COUNT = 3
PATCH_CROP_RATIO = (0.0, 0.72, 0.0, 1.0)

CSV_FIELDNAMES = [
  "frame_idx",
  "frame_id",
  "image_path",
  "model_frame_id",
  "frame_match",
  "status",
  "model_timestamp_eof",
  "source_timestamp_ns",
  "model_execution_time",
  "gpu_execution_time",
  "lead_index",
  "prob",
  "prob_time",
  "x0",
  "dRel",
  "y0",
  "v0",
  "a0",
  "xStd0",
  "yStd0",
  "vStd0",
  "aStd0",
]

COMPARE_CSV_FIELDNAMES = [
  "frame_idx",
  "image_path",
  "patched_image_path",
  "patch_applied",
  "detection_conf",
  "detection_class",
  "det_x1",
  "det_y1",
  "det_x2",
  "det_y2",
  "patch_x1",
  "patch_y1",
  "patch_x2",
  "patch_y2",
  "clean_frame_id",
  "patched_frame_id",
  "clean_model_frame_id",
  "patched_model_frame_id",
  "clean_frame_match",
  "patched_frame_match",
  "clean_status",
  "patched_status",
  "lead_index",
  "clean_prob",
  "patched_prob",
  "clean_x0",
  "patched_x0",
  "clean_dRel",
  "patched_dRel",
  "delta_dRel",
  "clean_y0",
  "patched_y0",
  "clean_v0",
  "patched_v0",
  "clean_a0",
  "patched_a0",
  "clean_xStd0",
  "patched_xStd0",
  "clean_yStd0",
  "patched_yStd0",
  "clean_vStd0",
  "patched_vStd0",
  "clean_aStd0",
  "patched_aStd0",
]


def sort_key(path: Path):
  try:
    return (0, int(path.stem))
  except ValueError:
    return (1, path.name)


def list_images(image_dir: Path) -> List[Path]:
  paths = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
  return sorted(paths, key=sort_key)


def bgr_to_nv12_bytes(bgr: np.ndarray, width: int, height: int) -> bytes:
  if bgr.shape[:2] != (height, width):
    bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_LINEAR)

  yuv_i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
  yuv_flat = yuv_i420.reshape(-1)

  y_size = height * width
  uv_size = (height // 2) * (width // 2)

  y = yuv_flat[:y_size].reshape(height, width)
  u = yuv_flat[y_size:y_size + uv_size].reshape(height // 2, width // 2)
  v = yuv_flat[y_size + uv_size:y_size + 2 * uv_size].reshape(height // 2, width // 2)

  uv_nv12 = np.empty((height // 2, width), dtype=np.uint8)
  uv_nv12[:, 0::2] = u
  uv_nv12[:, 1::2] = v
  return np.vstack([y, uv_nv12]).tobytes()


def send_camera_state(pm: messaging.PubMaster, service: str, frame_id: int, timestamp_ns: int) -> None:
  dat = messaging.new_message(service, valid=True)
  msg = getattr(dat, service)
  msg.frameId = frame_id
  msg.timestampSof = timestamp_ns
  msg.timestampEof = timestamp_ns
  msg.transform = [
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
  ]
  pm.send(service, dat)


def send_live_calibration(pm: messaging.PubMaster) -> None:
  dat = messaging.new_message("liveCalibration", valid=True)
  dat.liveCalibration.validBlocks = 20
  dat.liveCalibration.rpyCalib = [0.0, 0.0, 0.0]
  pm.send("liveCalibration", dat)


def frame_stream(paths: List[Path], once: bool) -> Iterable[Path]:
  while True:
    yield from paths
    if once:
      return


def wait_for_model_frame(sm: messaging.SubMaster, target_frame_id: int, timeout_ms: int) -> bool:
  if sm.seen["modelV2"]:
    current_frame_id = sm["modelV2"].frameId
    if current_frame_id == target_frame_id:
      return True
    if current_frame_id > target_frame_id:
      return False

  deadline = time.monotonic() + timeout_ms / 1000.0
  while time.monotonic() < deadline:
    remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
    sm.update(min(MODEL_FRAME_POLL_MS, remaining_ms))
    if not sm.updated["modelV2"]:
      continue

    current_frame_id = sm["modelV2"].frameId
    if current_frame_id < target_frame_id:
      continue
    return current_frame_id == target_frame_id

  return False


def lead_value(values, default: Optional[float] = None) -> Optional[float]:
  return float(values[0]) if len(values) > 0 else default


def lead_csv_row(frame_idx: int, frame_id: int, image_path: Path, timestamp_ns: int,
                 model_msg, lead_index: int, frame_match: bool, status: str):
  base_row = {
    "frame_idx": frame_idx,
    "frame_id": frame_id,
    "image_path": str(image_path),
    "model_frame_id": int(model_msg.frameId) if frame_match else -1,
    "frame_match": frame_match,
    "status": status,
    "model_timestamp_eof": int(model_msg.timestampEof) if frame_match else "",
    "source_timestamp_ns": timestamp_ns,
    "model_execution_time": float(model_msg.modelExecutionTime) if frame_match else "",
    "gpu_execution_time": float(model_msg.gpuExecutionTime) if frame_match else "",
    "lead_index": lead_index,
    "prob": "",
    "prob_time": "",
    "x0": "",
    "dRel": "",
    "y0": "",
    "v0": "",
    "a0": "",
    "xStd0": "",
    "yStd0": "",
    "vStd0": "",
    "aStd0": "",
  }

  if not frame_match:
    return base_row

  leads = model_msg.leadsV3
  if lead_index < 0 or lead_index >= len(leads):
    base_row["status"] = "no_lead"
    return base_row

  lead = leads[lead_index]
  x0 = lead_value(lead.x)
  base_row.update({
    "status": "ok",
    "prob": float(lead.prob),
    "prob_time": float(lead.probTime),
    "x0": x0 if x0 is not None else "",
    "dRel": (x0 - RADAR_TO_CAMERA) if x0 is not None else "",
    "y0": lead_value(lead.y, ""),
    "v0": lead_value(lead.v, ""),
    "a0": lead_value(lead.a, ""),
    "xStd0": lead_value(lead.xStd, ""),
    "yStd0": lead_value(lead.yStd, ""),
    "vStd0": lead_value(lead.vStd, ""),
    "aStd0": lead_value(lead.aStd, ""),
  })
  return base_row


def load_patch(path: Path) -> np.ndarray:
  patch = np.load(path).astype(np.float32)
  if patch.ndim == 3 and patch.shape[0] in (1, 3, 4):
    patch = np.transpose(patch, (1, 2, 0))

  if patch.ndim == 3 and patch.shape[-1] == 3:
    patch = patch[:, :, ::-1]
  elif patch.ndim == 3 and patch.shape[-1] == 4:
    patch = patch[:, :, :3][:, :, ::-1]
  elif patch.ndim == 2:
    patch = np.repeat(patch[:, :, np.newaxis], 3, axis=2)

  if patch.ndim != 3 or patch.shape[-1] != 3:
    raise RuntimeError(f"Unsupported patch shape from {path}: {patch.shape}")
  return patch


def apply_patch_to_roi(roi: np.ndarray, patch: np.ndarray, patch_mode: str) -> np.ndarray:
  if patch_mode == "add":
    return np.clip(roi.astype(np.float32) + patch, 0, 255).astype(np.uint8)
  return np.clip(patch, 0, 255).astype(np.uint8)


def get_yolo_box(img_bgr: np.ndarray, yolo_model, conf: float) -> Tuple[Optional[Tuple[int, int, int, int]], float, int]:
  boxes = yolo_model(img_bgr, conf=conf, verbose=False)[0].boxes
  target_box = None
  target_conf = 0.0
  target_cls = -1
  max_area = 0
  img_h, img_w = img_bgr.shape[:2]

  for box in boxes:
    cls_id = int(box.cls.item())
    if cls_id not in (2, 3, 5, 7):
      continue

    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
    if y2 > img_h * 0.85:
      continue

    cx = (x1 + x2) / 2.0
    if cx < img_w * 0.3 or cx > img_w * 0.7:
      continue

    area = max(0, x2 - x1) * max(0, y2 - y1)
    if area > max_area:
      max_area = area
      target_box = (x1, y1, x2, y2)
      target_conf = float(box.conf.item())
      target_cls = cls_id

  return target_box, target_conf, target_cls


def patch_box_from_detection(box_xyxy: Optional[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
  if box_xyxy is None:
    return None

  x1, y1, x2, y2 = [float(v) for v in box_xyxy]
  orig_w = x2 - x1
  orig_h = y2 - y1
  if orig_w <= 0 or orig_h <= 0:
    return None

  dynamic_y_offset = orig_h * 0.26 - 5
  cx = (x1 + x2) / 2.0
  cy = (y1 + y2) / 2.0 + dynamic_y_offset
  yolo_x1 = cx - orig_w / 2.0
  yolo_y1 = cy - orig_h / 2.0

  top, bottom, left, right = PATCH_CROP_RATIO
  new_x1 = int(yolo_x1 + orig_w * left)
  new_y1 = int(yolo_y1 + orig_h * top)
  new_x2 = int(yolo_x1 + orig_w * right)
  new_y2 = int(yolo_y1 + orig_h * bottom)
  if new_x2 <= new_x1 or new_y2 <= new_y1:
    return None
  return new_x1, new_y1, new_x2, new_y2


def patch_metadata(detection_box, detection_conf: float, detection_cls: int,
                   patch_box: Optional[Tuple[int, int, int, int]], patch_applied: bool) -> Dict[str, object]:
  det = detection_box or ("", "", "", "")
  pbox = patch_box or ("", "", "", "")
  return {
    "patch_applied": patch_applied,
    "detection_conf": detection_conf if detection_box is not None else "",
    "detection_class": detection_cls if detection_box is not None else "",
    "det_x1": det[0],
    "det_y1": det[1],
    "det_x2": det[2],
    "det_y2": det[3],
    "patch_x1": pbox[0],
    "patch_y1": pbox[1],
    "patch_x2": pbox[2],
    "patch_y2": pbox[3],
  }


def apply_patch_to_front_car(img_bgr: np.ndarray, yolo_model, patch: np.ndarray,
                             conf: float, patch_mode: str) -> Tuple[np.ndarray, Dict[str, object]]:
  detection_box, detection_conf, detection_cls = get_yolo_box(img_bgr, yolo_model, conf)
  target_box = patch_box_from_detection(detection_box)
  if target_box is None:
    return img_bgr.copy(), patch_metadata(detection_box, detection_conf, detection_cls, target_box, False)

  x1, y1, x2, y2 = target_box
  box_w = x2 - x1
  box_h = y2 - y1
  img_h, img_w = img_bgr.shape[:2]
  if box_w <= 0 or box_h <= 0 or x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
    return img_bgr.copy(), patch_metadata(detection_box, detection_conf, detection_cls, target_box, False)

  out = img_bgr.copy()
  resized_patch = cv2.resize(patch, (box_w, box_h), interpolation=cv2.INTER_NEAREST)
  roi = out[y1:y2, x1:x2]
  out[y1:y2, x1:x2] = apply_patch_to_roi(roi, resized_patch, patch_mode)
  return out, patch_metadata(detection_box, detection_conf, detection_cls, target_box, True)


def lead_summary_from_model(model_msg, frame_match: bool, lead_index: int) -> Dict[str, object]:
  if not frame_match or model_msg is None:
    return {
      "status": "no_model_frame",
      "prob": "",
      "x0": "",
      "dRel": "",
      "y0": "",
      "v0": "",
      "a0": "",
      "xStd0": "",
      "yStd0": "",
      "vStd0": "",
      "aStd0": "",
    }

  leads = model_msg.leadsV3
  if lead_index < 0 or lead_index >= len(leads):
    return {
      "status": "no_lead",
      "prob": "",
      "x0": "",
      "dRel": "",
      "y0": "",
      "v0": "",
      "a0": "",
      "xStd0": "",
      "yStd0": "",
      "vStd0": "",
      "aStd0": "",
    }

  lead = leads[lead_index]
  x0 = lead_value(lead.x)
  return {
    "status": "ok",
    "prob": float(lead.prob),
    "x0": x0 if x0 is not None else "",
    "dRel": (x0 - RADAR_TO_CAMERA) if x0 is not None else "",
    "y0": lead_value(lead.y, ""),
    "v0": lead_value(lead.v, ""),
    "a0": lead_value(lead.a, ""),
    "xStd0": lead_value(lead.xStd, ""),
    "yStd0": lead_value(lead.yStd, ""),
    "vStd0": lead_value(lead.vStd, ""),
    "aStd0": lead_value(lead.aStd, ""),
  }


def collect_model_result(sm: messaging.SubMaster, frame_match: bool, lead_indexes: Iterable[int]) -> Dict[str, object]:
  model_msg = sm["modelV2"] if frame_match else None
  return {
    "model_frame_id": int(model_msg.frameId) if frame_match else -1,
    "frame_match": frame_match,
    "leads": {
      lead_index: lead_summary_from_model(model_msg, frame_match, lead_index)
      for lead_index in lead_indexes
    },
  }


def empty_model_result(lead_indexes: Iterable[int]) -> Dict[str, object]:
  return {
    "model_frame_id": -1,
    "frame_match": False,
    "leads": {
      lead_index: lead_summary_from_model(None, False, lead_index)
      for lead_index in lead_indexes
    },
  }


def compare_csv_rows(frame_idx: int, image_path: Path, patched_image_path: Path, patch_meta: Dict[str, object],
                     clean_frame_id: int, patched_frame_id: int, clean_result: Dict[str, object],
                     patched_result: Dict[str, object], lead_indexes: Iterable[int]) -> List[Dict[str, object]]:
  rows = []
  for lead_index in lead_indexes:
    clean = clean_result["leads"][lead_index]
    patched = patched_result["leads"][lead_index]
    clean_drel = clean["dRel"]
    patched_drel = patched["dRel"]
    delta_drel = ""
    if clean_drel != "" and patched_drel != "":
      delta_drel = float(patched_drel) - float(clean_drel)

    row = {
      "frame_idx": frame_idx,
      "image_path": str(image_path),
      "patched_image_path": str(patched_image_path),
      **patch_meta,
      "clean_frame_id": clean_frame_id,
      "patched_frame_id": patched_frame_id,
      "clean_model_frame_id": clean_result["model_frame_id"],
      "patched_model_frame_id": patched_result["model_frame_id"],
      "clean_frame_match": clean_result["frame_match"],
      "patched_frame_match": patched_result["frame_match"],
      "clean_status": clean["status"],
      "patched_status": patched["status"],
      "lead_index": lead_index,
      "clean_prob": clean["prob"],
      "patched_prob": patched["prob"],
      "clean_x0": clean["x0"],
      "patched_x0": patched["x0"],
      "clean_dRel": clean_drel,
      "patched_dRel": patched_drel,
      "delta_dRel": delta_drel,
      "clean_y0": clean["y0"],
      "patched_y0": patched["y0"],
      "clean_v0": clean["v0"],
      "patched_v0": patched["v0"],
      "clean_a0": clean["a0"],
      "patched_a0": patched["a0"],
      "clean_xStd0": clean["xStd0"],
      "patched_xStd0": patched["xStd0"],
      "clean_yStd0": clean["yStd0"],
      "patched_yStd0": patched["yStd0"],
      "clean_vStd0": clean["vStd0"],
      "patched_vStd0": patched["vStd0"],
      "clean_aStd0": clean["aStd0"],
      "patched_aStd0": patched["aStd0"],
    }
    rows.append(row)
  return rows


def publish_bgr_frame(vipc_server: VisionIpcServer, pm: messaging.PubMaster, bgr: np.ndarray,
                      frame_id: int, fps: float, width: int, height: int,
                      publish_wide: bool, publish_live_calibration: bool) -> int:
  nv12 = bgr_to_nv12_bytes(bgr, width, height)
  timestamp_ns = int((frame_id / fps) * 1e9)

  vipc_server.send(VisionStreamType.VISION_STREAM_ROAD, nv12, frame_id, timestamp_ns, timestamp_ns)
  send_camera_state(pm, "roadCameraState", frame_id, timestamp_ns)
  if publish_live_calibration:
    send_live_calibration(pm)

  if publish_wide:
    vipc_server.send(VisionStreamType.VISION_STREAM_WIDE_ROAD, nv12, frame_id, timestamp_ns, timestamp_ns)
    send_camera_state(pm, "wideRoadCameraState", frame_id, timestamp_ns)

  return timestamp_ns


def main() -> None:
  parser = argparse.ArgumentParser(description="Publish an image directory as openpilot's camerad VisionIPC stream.")
  parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
  parser.add_argument("--fps", type=float, default=20.0)
  parser.add_argument("--start-frame-id", type=int, default=1)
  parser.add_argument("--width", type=int, default=W)
  parser.add_argument("--height", type=int, default=H)
  parser.add_argument("--once", action="store_true", help="Stop after one pass through the image directory.")
  parser.add_argument("--wide", action="store_true", help="Also publish the same images as wideRoadCameraState.")
  parser.add_argument("--wait-for-readers", action="store_true")
  parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV_OUT,
                      help="CSV path for modelV2 lead predictions. Use --no-csv to disable.")
  parser.add_argument("--no-csv", action="store_true", help="Only publish frames; do not wait for or record modelV2.")
  parser.add_argument("--model-timeout-ms", type=int, default=1500)
  parser.add_argument("--lead-index", type=int, default=0, help="leadsV3 index to save when --all-leads is not set.")
  parser.add_argument("--all-leads", action="store_true", help="Write all leadsV3 rows instead of only --lead-index.")
  parser.add_argument("--start-modeld", action="store_true",
                      help="Start openpilot modeld from this script and stop it on exit.")
  parser.add_argument("--compare-patch", action="store_true",
                      help="For each source image, run clean and patched frames and save both dRel values.")
  parser.add_argument("--patch-path", type=Path, default=DEFAULT_PATCH_PATH)
  parser.add_argument("--patched-image-dir", type=Path, default=None,
                      help="Directory for patched images. Defaults to <image-dir>/patched.")
  parser.add_argument("--yolo-model", type=Path, default=DEFAULT_YOLO_MODEL)
  parser.add_argument("--yolo-conf", type=float, default=0.05)
  parser.add_argument("--patch-mode", choices=("add", "replace"), default="add")
  parser.add_argument("--keep-modeld-runtime-patch", action="store_true",
                      help="Do not remove /tmp/adversarial_patch_enabled before comparison.")
  parser.add_argument("--no-restart-between-passes", action="store_true",
                      help="Do not restart modeld between clean and patched passes. This disables independent comparison.")
  args = parser.parse_args()

  record_csv = not args.no_csv
  if args.compare_patch and record_csv and not args.start_modeld and not args.no_restart_between_passes:
    raise RuntimeError("--compare-patch needs --start-modeld for independent clean/patched passes. "
                       "Use --no-restart-between-passes only if you intentionally manage modeld yourself.")

  image_paths = list_images(args.image_dir)
  if not image_paths:
    raise RuntimeError(f"No images found in {args.image_dir}")

  patch = None
  yolo_model = None
  patched_image_dir = args.patched_image_dir or (args.image_dir / "patched")
  if args.compare_patch:
    try:
      from ultralytics import YOLO
    except ImportError as e:
      raise RuntimeError("ultralytics is required for --compare-patch; run this script with poetry run.") from e

    patch = load_patch(args.patch_path)
    yolo_model = YOLO(str(args.yolo_model))
    patched_image_dir.mkdir(parents=True, exist_ok=True)
    if not args.keep_modeld_runtime_patch and MODEL_PATCH_SWITCH_FILE.exists():
      MODEL_PATCH_SWITCH_FILE.unlink()
      print(f"Removed {MODEL_PATCH_SWITCH_FILE} to avoid applying the modeld runtime patch twice")

  pm_services = ["roadCameraState"]
  if args.wide:
    pm_services.append("wideRoadCameraState")
  if args.start_modeld:
    pm_services.append("liveCalibration")
  pm = messaging.PubMaster(pm_services)

  vipc_server = VisionIpcServer("camerad")
  vipc_server.create_buffers(VisionStreamType.VISION_STREAM_ROAD, 5, False, args.width, args.height)
  if args.wide:
    vipc_server.create_buffers(VisionStreamType.VISION_STREAM_WIDE_ROAD, 5, False, args.width, args.height)
  vipc_server.start_listener()

  if args.wait_for_readers and not args.start_modeld:
    pm.wait_for_readers_to_update("roadCameraState", 10)

  modeld_process = None
  if args.start_modeld:
    from openpilot.selfdrive.car.car_helpers import write_car_param
    from openpilot.selfdrive.manager.process_config import managed_processes

    write_car_param()
    modeld_process = managed_processes["modeld"]
    modeld_process.start()
    pm.wait_for_readers_to_update("roadCameraState", 10)
    send_live_calibration(pm)
    print("Started modeld for image replay")

  sm = messaging.SubMaster(["modelV2"]) if record_csv else None
  csv_file = None
  csv_writer = None
  if record_csv:
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    csv_file = args.csv_out.open("w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=COMPARE_CSV_FIELDNAMES if args.compare_patch else CSV_FIELDNAMES)
    csv_writer.writeheader()
    csv_file.flush()
    print(f"Writing model lead CSV to {args.csv_out}")

  rk = Ratekeeper(args.fps, print_delay_threshold=None)
  frame_id = args.start_frame_id
  frame_idx = 0

  print(f"Publishing {len(image_paths)} images from {args.image_dir} at {args.fps:g} Hz")
  try:
    if args.compare_patch:
      assert patch is not None and yolo_model is not None
      lead_indexes = list(range(MODEL_LEAD_COUNT)) if args.all_leads else [args.lead_index]
      source_records: List[Tuple[int, Path]] = []
      clean_results: Dict[int, Dict[str, object]] = {}
      clean_frame_ids: Dict[int, int] = {}

      print("Running clean pass for all source images")
      for image_path in image_paths:
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
          print(f"Skipping unreadable image: {image_path}")
          continue

        frame_idx += 1
        source_records.append((frame_idx, image_path))
        clean_frame_id = frame_id
        clean_frame_ids[frame_idx] = clean_frame_id

        publish_bgr_frame(vipc_server, pm, bgr, clean_frame_id, args.fps, args.width, args.height,
                          args.wide, args.start_modeld)
        clean_result = empty_model_result(lead_indexes)
        if record_csv and sm is not None:
          clean_frame_match = wait_for_model_frame(sm, clean_frame_id, args.model_timeout_ms)
          if not clean_frame_match:
            latest_model = sm["modelV2"].frameId if sm.seen["modelV2"] else -1
            print(f"No aligned clean modelV2 for frame_id={clean_frame_id}; latest modelV2 frame is {latest_model}")
          clean_result = collect_model_result(sm, clean_frame_match, lead_indexes)
        clean_results[frame_idx] = clean_result

        if frame_idx % int(max(args.fps, 1.0)) == 0:
          print(f"clean pass frame_idx={frame_idx} frame_id={clean_frame_id} image={image_path.name}")

        frame_id += 1
        rk.keep_time()

      if record_csv and args.start_modeld and not args.no_restart_between_passes:
        assert modeld_process is not None
        print("Restarting modeld before patched pass for independent comparison")
        modeld_process.stop()
        time.sleep(1.0)
        modeld_process.start()
        pm.wait_for_readers_to_update("roadCameraState", 10)
        send_live_calibration(pm)
        sm = messaging.SubMaster(["modelV2"])
        frame_id = args.start_frame_id
        rk = Ratekeeper(args.fps, print_delay_threshold=None)

      print("Running patched pass for all source images")
      for source_frame_idx, image_path in source_records:
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
          print(f"Skipping unreadable image during patched pass: {image_path}")
          continue

        patched_bgr, patch_meta = apply_patch_to_front_car(bgr, yolo_model, patch, args.yolo_conf, args.patch_mode)
        patched_image_path = patched_image_dir / image_path.name
        cv2.imwrite(str(patched_image_path), patched_bgr)

        patched_frame_id = frame_id
        publish_bgr_frame(vipc_server, pm, patched_bgr, patched_frame_id, args.fps, args.width, args.height,
                          args.wide, args.start_modeld)
        patched_result = empty_model_result(lead_indexes)
        if record_csv and sm is not None:
          patched_frame_match = wait_for_model_frame(sm, patched_frame_id, args.model_timeout_ms)
          if not patched_frame_match:
            latest_model = sm["modelV2"].frameId if sm.seen["modelV2"] else -1
            print(f"No aligned patched modelV2 for frame_id={patched_frame_id}; latest modelV2 frame is {latest_model}")
          patched_result = collect_model_result(sm, patched_frame_match, lead_indexes)

        if record_csv and csv_writer is not None and csv_file is not None:
          for row in compare_csv_rows(source_frame_idx, image_path, patched_image_path, patch_meta,
                                      clean_frame_ids[source_frame_idx], patched_frame_id,
                                      clean_results[source_frame_idx], patched_result, lead_indexes):
            csv_writer.writerow(row)
          csv_file.flush()

        if source_frame_idx % int(max(args.fps, 1.0)) == 0:
          print(f"patched pass frame_idx={source_frame_idx} frame_id={patched_frame_id} image={image_path.name}")

        frame_id += 1
        rk.keep_time()

      return

    for image_path in frame_stream(image_paths, args.once):
      bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
      if bgr is None:
        print(f"Skipping unreadable image: {image_path}")
        continue

      frame_idx += 1
      timestamp_ns = publish_bgr_frame(vipc_server, pm, bgr, frame_id, args.fps, args.width, args.height,
                                       args.wide, args.start_modeld)

      if record_csv and sm is not None and csv_writer is not None and csv_file is not None:
        frame_match = wait_for_model_frame(sm, frame_id, args.model_timeout_ms)
        if frame_match:
          model_msg = sm["modelV2"]
          lead_indexes = range(len(model_msg.leadsV3)) if args.all_leads else [args.lead_index]
          for lead_index in lead_indexes:
            csv_writer.writerow(lead_csv_row(frame_idx, frame_id, image_path, timestamp_ns,
                                             model_msg, lead_index, True, "ok"))
          csv_file.flush()
        else:
          latest_model = sm["modelV2"].frameId if sm.seen["modelV2"] else -1
          print(f"No aligned modelV2 for frame_id={frame_id}; latest modelV2 frame is {latest_model}")
          csv_writer.writerow(lead_csv_row(frame_idx, frame_id, image_path, timestamp_ns,
                                           None, args.lead_index, False, "no_model_frame"))
          csv_file.flush()

      if frame_id % int(max(args.fps, 1.0)) == 0:
        print(f"frame_id={frame_id} image={image_path.name}")

      frame_id += 1
      rk.keep_time()
  finally:
    if csv_file is not None:
      csv_file.close()
    if modeld_process is not None:
      modeld_process.stop()


if __name__ == "__main__":
  main()
