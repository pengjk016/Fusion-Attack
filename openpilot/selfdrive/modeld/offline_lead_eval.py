#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
  import cv2  # type: ignore
except ImportError:
  cv2 = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from openpilot.selfdrive.modeld.constants import ModelConstants

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
ALIGN_BY_FRAME_ID = "frame_id"
ALIGN_BY_INDEX = "index"
VideoStreamMeta = namedtuple("VideoStreamMeta", ["camera_state", "frame_sizes"])
TICI_ROAD_FRAME_SIZE = (1928, 1208)
TICI_WIDE_FRAME_SIZE = (1928, 1208)
ROAD_CAMERA_FRAME_SIZES = {"tici": TICI_ROAD_FRAME_SIZE, "tizi": TICI_ROAD_FRAME_SIZE}
WIDE_ROAD_CAMERA_FRAME_SIZES = {"tici": TICI_WIDE_FRAME_SIZE, "tizi": TICI_WIDE_FRAME_SIZE}
VIPC_STREAM_METADATA = [
  VideoStreamMeta("roadCameraState", ROAD_CAMERA_FRAME_SIZES),
  VideoStreamMeta("wideRoadCameraState", WIDE_ROAD_CAMERA_FRAME_SIZES),
]


class BaseFrameReader:
  def close(self) -> None:
    pass


def rgb24toyuv(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  yuv_from_rgb = np.array([
    [0.299, 0.587, 0.114],
    [-0.14714119, -0.28886916, 0.43601035],
    [0.61497538, -0.51496512, -0.10001026],
  ])
  img = np.dot(rgb.reshape(-1, 3), yuv_from_rgb.T).reshape(rgb.shape)

  ys = img[:, :, 0]
  us = (img[::2, ::2, 1] + img[1::2, ::2, 1] + img[::2, 1::2, 1] + img[1::2, 1::2, 1]) / 4 + 128
  vs = (img[::2, ::2, 2] + img[1::2, ::2, 2] + img[::2, 1::2, 2] + img[1::2, 1::2, 2]) / 4 + 128
  return ys, us, vs


def rgb24toyuv420(rgb: np.ndarray) -> np.ndarray:
  ys, us, vs = rgb24toyuv(rgb)

  y_len = rgb.shape[0] * rgb.shape[1]
  uv_len = y_len // 4

  yuv420 = np.empty(y_len + 2 * uv_len, dtype=rgb.dtype)
  yuv420[:y_len] = ys.reshape(-1)
  yuv420[y_len:y_len + uv_len] = us.reshape(-1)
  yuv420[y_len + uv_len:y_len + 2 * uv_len] = vs.reshape(-1)
  return yuv420.clip(0, 255).astype("uint8")


def rgb24tonv12(rgb: np.ndarray) -> np.ndarray:
  ys, us, vs = rgb24toyuv(rgb)

  y_len = rgb.shape[0] * rgb.shape[1]
  uv_len = y_len // 4

  nv12 = np.empty(y_len + 2 * uv_len, dtype=rgb.dtype)
  nv12[:y_len] = ys.reshape(-1)
  nv12[y_len::2] = us.reshape(-1)
  nv12[y_len + 1::2] = vs.reshape(-1)
  return nv12.clip(0, 255).astype("uint8")


@dataclass(frozen=True)
class StreamSpec:
  state: str
  video_path: Optional[Path]
  frames_dir: Optional[Path]
  frame_id_offset: Optional[int]


@dataclass(frozen=True)
class RunSpec:
  name: str
  log_path: Path
  stream_specs: Dict[str, StreamSpec]


@dataclass(frozen=True)
class LeadRecord:
  sequence_index: int
  frame_id: int
  frame_id_extra: int
  timestamp_eof: int
  lead_index: int
  prob: float
  prob_time: float
  t: Tuple[float, ...]
  x: Tuple[float, ...]
  y: Tuple[float, ...]
  v: Tuple[float, ...]
  a: Tuple[float, ...]
  x_std: Tuple[float, ...]
  y_std: Tuple[float, ...]
  v_std: Tuple[float, ...]
  a_std: Tuple[float, ...]

  def alignment_key(self) -> Tuple[int, int, int]:
    return (self.frame_id, self.frame_id_extra, self.lead_index)

  def frame_key(self) -> Tuple[int, int]:
    return (self.frame_id, self.frame_id_extra)

  def to_csv_row(self) -> Dict[str, Any]:
    row: Dict[str, Any] = {
      "sequence_index": self.sequence_index,
      "frame_id": self.frame_id,
      "frame_id_extra": self.frame_id_extra,
      "timestamp_eof": self.timestamp_eof,
      "lead_index": self.lead_index,
      "prob": self.prob,
      "prob_time": self.prob_time,
    }
    for i in range(len(self.t)):
      row[f"t_{i}"] = self.t[i]
      row[f"x_{i}"] = self.x[i]
      row[f"x_std_{i}"] = self.x_std[i]
      row[f"y_{i}"] = self.y[i]
      row[f"y_std_{i}"] = self.y_std[i]
      row[f"v_{i}"] = self.v[i]
      row[f"v_std_{i}"] = self.v_std[i]
      row[f"a_{i}"] = self.a[i]
      row[f"a_std_{i}"] = self.a_std[i]
    return row


class ImageDirectoryFrameReader(BaseFrameReader):
  def __init__(self, directory: Path, w: int, h: int, frame_id_offset: Optional[int]):
    self.directory = directory
    self.w = w
    self.h = h
    self.frame_type = "image_dir"
    self._frame_paths = sorted(
      [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES],
      key=lambda p: p.name,
    )
    if not self._frame_paths:
      raise ValueError(f"no image files found in {directory}")

    self._frame_map = self._build_frame_map(frame_id_offset)
    self.frame_count = max(self._frame_map) + 1

  def _build_frame_map(self, frame_id_offset: Optional[int]) -> Dict[int, Path]:
    frame_map: Dict[int, Path] = {}
    numeric_names = True
    for path in self._frame_paths:
      try:
        int(path.stem)
      except ValueError:
        numeric_names = False
        break

    if numeric_names:
      for path in self._frame_paths:
        frame_id = int(path.stem)
        if frame_id in frame_map:
          raise ValueError(f"duplicate frame id {frame_id} in {self.directory}")
        frame_map[frame_id] = path
      return frame_map

    start_frame_id = 0 if frame_id_offset is None else frame_id_offset
    for i, path in enumerate(self._frame_paths):
      frame_map[start_frame_id + i] = path
    return frame_map

  def _load_rgb(self, frame_id: int) -> np.ndarray:
    if frame_id not in self._frame_map:
      raise KeyError(f"frame id {frame_id} not found in {self.directory}")

    frame_path = self._frame_map[frame_id]
    if cv2 is not None:
      bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
      if bgr is None:
        raise ValueError(f"failed to read image {frame_path}")
      if bgr.shape[1] != self.w or bgr.shape[0] != self.h:
        bgr = cv2.resize(bgr, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
      return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    with Image.open(frame_path) as img:
      rgb = img.convert("RGB")
      if rgb.size != (self.w, self.h):
        rgb = rgb.resize((self.w, self.h), Image.Resampling.BILINEAR)
      return np.asarray(rgb, dtype=np.uint8)

  def get(self, idx: int, count: int = 1, pix_fmt: str = "yuv420p") -> List[np.ndarray]:
    if pix_fmt not in ("rgb24", "nv12", "yuv420p"):
      raise ValueError(f"unsupported pixel format {pix_fmt!r}")

    frames: List[np.ndarray] = []
    for frame_id in range(idx, idx + count):
      rgb = self._load_rgb(frame_id)
      if pix_fmt == "rgb24":
        frames.append(rgb)
      elif pix_fmt == "nv12":
        frames.append(rgb24tonv12(rgb))
      else:
        frames.append(rgb24toyuv420(rgb))
    return frames


def meta_from_camera_state(state: str) -> Optional[VideoStreamMeta]:
  return next((meta for meta in VIPC_STREAM_METADATA if meta.camera_state == state), None)


def available_streams(log_messages: Optional[Sequence[Any]] = None) -> List[VideoStreamMeta]:
  if log_messages is None:
    return list(VIPC_STREAM_METADATA)

  present_states = {msg.which() for msg in log_messages}
  return [meta for meta in VIPC_STREAM_METADATA if meta.camera_state in present_states]


def lead_csv_fieldnames(include_stds: bool) -> List[str]:
  fieldnames = [
    "sequence_index",
    "frame_id",
    "frame_id_extra",
    "timestamp_eof",
    "lead_index",
    "prob",
    "prob_time",
  ]
  for i in range(len(ModelConstants.LEAD_T_IDXS)):
    fieldnames.append(f"t_{i}")
    fieldnames.append(f"x_{i}")
    if include_stds:
      fieldnames.append(f"x_std_{i}")
    fieldnames.append(f"y_{i}")
    if include_stds:
      fieldnames.append(f"y_std_{i}")
    fieldnames.append(f"v_{i}")
    if include_stds:
      fieldnames.append(f"v_std_{i}")
    fieldnames.append(f"a_{i}")
    if include_stds:
      fieldnames.append(f"a_std_{i}")
  return fieldnames


def diff_csv_fieldnames() -> List[str]:
  fieldnames = [
    "sequence_index_a",
    "sequence_index_b",
    "frame_id",
    "frame_id_extra",
    "lead_index",
    "prob_a",
    "prob_b",
    "prob_delta",
    "prob_abs_delta",
  ]
  for i in range(len(ModelConstants.LEAD_T_IDXS)):
    fieldnames.append(f"t_{i}")
    for field in ("x", "y", "v", "a"):
      fieldnames.append(f"{field}_a_{i}")
      fieldnames.append(f"{field}_b_{i}")
      fieldnames.append(f"{field}_delta_{i}")
      fieldnames.append(f"{field}_abs_delta_{i}")
  return fieldnames


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Replay modeld on two offline inputs and compare modelV2.leadsV3 outputs.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""Examples:
  Compare two logged runs with original videos:
    python3 selfdrive/modeld/offline_lead_eval.py \\
      --a-log /tmp/baseline/rlog.bz2 \\
      --a-road-camera /tmp/baseline/fcamera.hevc \\
      --a-wide-road-camera /tmp/baseline/ecamera.hevc \\
      --b-log /tmp/perturbed/rlog.bz2 \\
      --b-road-camera /tmp/perturbed/fcamera.hevc \\
      --b-wide-road-camera /tmp/perturbed/ecamera.hevc \\
      --out-dir /tmp/lead_eval

  Compare one log template with two different road-frame directories:
    python3 selfdrive/modeld/offline_lead_eval.py \\
      --a-log /tmp/route/rlog.bz2 \\
      --a-road-frames-dir /tmp/baseline_frames \\
      --a-wide-road-camera /tmp/route/ecamera.hevc \\
      --b-log /tmp/route/rlog.bz2 \\
      --b-road-frames-dir /tmp/patched_frames \\
      --b-wide-road-camera /tmp/route/ecamera.hevc \\
      --out-dir /tmp/lead_eval
""",
  )
  parser.add_argument("--a-log", required=True, help="Run A rlog/qlog path used for metadata and replay inputs.")
  parser.add_argument("--b-log", help="Run B rlog/qlog path. Defaults to --a-log.")
  parser.add_argument("--a-road-camera", help="Run A road camera video path, usually fcamera.hevc.")
  parser.add_argument("--a-wide-road-camera", help="Run A wide road camera video path, usually ecamera.hevc.")
  parser.add_argument("--b-road-camera", help="Run B road camera video path, usually fcamera.hevc.")
  parser.add_argument("--b-wide-road-camera", help="Run B wide road camera video path, usually ecamera.hevc.")
  parser.add_argument("--a-road-frames-dir", help="Run A road-camera image directory.")
  parser.add_argument("--a-wide-road-frames-dir", help="Run A wide-road-camera image directory.")
  parser.add_argument("--b-road-frames-dir", help="Run B road-camera image directory.")
  parser.add_argument("--b-wide-road-frames-dir", help="Run B wide-road-camera image directory.")
  parser.add_argument("--a-road-frame-id-offset", type=int, help="Sequential frame-id offset for run A road images.")
  parser.add_argument("--a-wide-road-frame-id-offset", type=int, help="Sequential frame-id offset for run A wide-road images.")
  parser.add_argument("--b-road-frame-id-offset", type=int, help="Sequential frame-id offset for run B road images.")
  parser.add_argument("--b-wide-road-frame-id-offset", type=int, help="Sequential frame-id offset for run B wide-road images.")
  parser.add_argument("--align-by", choices=[ALIGN_BY_FRAME_ID, ALIGN_BY_INDEX], default=ALIGN_BY_FRAME_ID,
                      help="How to match run A and run B lead rows before diffing.")
  parser.add_argument("--lead-prob-threshold", type=float, default=0.5,
                      help="Threshold used for visibility/disagreement summary metrics.")
  parser.add_argument("--send-raw-pred", action="store_true",
                      help="Enable SEND_RAW_PRED while replaying modeld.")
  parser.add_argument("--disable-progress", action="store_true", help="Disable tqdm progress bars from process replay.")
  parser.add_argument("--out-dir", required=True, help="Directory to write CSV and JSON outputs into.")
  return parser.parse_args()


def resolve_existing_path(path_str: str) -> Path:
  path = Path(path_str).expanduser().resolve()
  if not path.exists():
    raise FileNotFoundError(path)
  return path


def load_log_messages(path: Path) -> List[Any]:
  from openpilot.tools.lib.logreader import LogReader

  return list(LogReader(str(path)))


def get_device_type(log_messages: Sequence[Any]) -> str:
  device_type = next((str(msg.initData.deviceType) for msg in log_messages if msg.which() == "initData"), None)
  if device_type is None:
    raise ValueError("initData missing from log; cannot infer camera frame sizes")
  return device_type


def get_first_frame_id(log_messages: Sequence[Any], state: str) -> int:
  for msg in log_messages:
    if msg.which() == state:
      return int(getattr(msg, state).frameId)
  raise ValueError(f"{state} missing from log")


def build_run_spec(args: argparse.Namespace, prefix: str) -> RunSpec:
  log_path = resolve_existing_path(getattr(args, f"{prefix}_log"))
  stream_specs = {
    "roadCameraState": StreamSpec(
      state="roadCameraState",
      video_path=resolve_existing_path(getattr(args, f"{prefix}_road_camera")) if getattr(args, f"{prefix}_road_camera") else None,
      frames_dir=resolve_existing_path(getattr(args, f"{prefix}_road_frames_dir")) if getattr(args, f"{prefix}_road_frames_dir") else None,
      frame_id_offset=getattr(args, f"{prefix}_road_frame_id_offset"),
    ),
    "wideRoadCameraState": StreamSpec(
      state="wideRoadCameraState",
      video_path=resolve_existing_path(getattr(args, f"{prefix}_wide_road_camera")) if getattr(args, f"{prefix}_wide_road_camera") else None,
      frames_dir=resolve_existing_path(getattr(args, f"{prefix}_wide_road_frames_dir")) if getattr(args, f"{prefix}_wide_road_frames_dir") else None,
      frame_id_offset=getattr(args, f"{prefix}_wide_road_frame_id_offset"),
    ),
  }
  return RunSpec(name=prefix, log_path=log_path, stream_specs=stream_specs)


def validate_stream_spec(stream_spec: StreamSpec) -> None:
  if stream_spec.video_path is not None and stream_spec.frames_dir is not None:
    raise ValueError(f"{stream_spec.state} cannot use both a video path and a frames directory")


def build_frame_readers(run_spec: RunSpec, log_messages: Sequence[Any]) -> Dict[str, BaseFrameReader]:
  from openpilot.tools.lib.framereader import FrameReader

  device_type = get_device_type(log_messages)
  required_streams = {meta.camera_state for meta in available_streams(log_messages)} & set(run_spec.stream_specs)
  frame_readers: Dict[str, BaseFrameReader] = {}

  for state in required_streams:
    stream_spec = run_spec.stream_specs[state]
    validate_stream_spec(stream_spec)
    if stream_spec.video_path is None and stream_spec.frames_dir is None:
      raise ValueError(f"{run_spec.name}:{state} is required by the log but no video or frames directory was provided")

    if stream_spec.video_path is not None:
      frame_readers[state] = FrameReader(str(stream_spec.video_path), readahead=True)
      continue

    meta = meta_from_camera_state(state)
    if meta is None or device_type not in meta.frame_sizes:
      raise ValueError(f"unable to infer frame size for {state} on device type {device_type}")
    frame_id_offset = stream_spec.frame_id_offset
    if frame_id_offset is None:
      frame_id_offset = get_first_frame_id(log_messages, state)
    w, h = meta.frame_sizes[device_type]
    frame_readers[state] = ImageDirectoryFrameReader(stream_spec.frames_dir, w, h, frame_id_offset)

  return frame_readers


def close_frame_readers(frame_readers: Dict[str, BaseFrameReader]) -> None:
  for reader in frame_readers.values():
    reader.close()


def replay_modeld(log_messages: Sequence[Any], frame_readers: Dict[str, BaseFrameReader],
                  send_raw_pred: bool, disable_progress: bool) -> List[Any]:
  from openpilot.selfdrive.test.process_replay import get_custom_params_from_lr, replay_process_with_name

  previous_send_raw_pred = os.environ.get("SEND_RAW_PRED")
  try:
    if send_raw_pred:
      os.environ["SEND_RAW_PRED"] = "1"
    elif "SEND_RAW_PRED" in os.environ:
      del os.environ["SEND_RAW_PRED"]

    custom_params = get_custom_params_from_lr(log_messages)
    return replay_process_with_name("modeld", log_messages, frs=frame_readers, custom_params=custom_params,
                                    disable_progress=disable_progress)
  finally:
    if previous_send_raw_pred is None:
      os.environ.pop("SEND_RAW_PRED", None)
    else:
      os.environ["SEND_RAW_PRED"] = previous_send_raw_pred


def extract_lead_records(log_messages: Iterable[Any]) -> List[LeadRecord]:
  records: List[LeadRecord] = []
  sequence_index = 0
  for msg in log_messages:
    if msg.which() != "modelV2":
      continue
    model = msg.modelV2
    for lead_index, lead in enumerate(model.leadsV3):
      records.append(LeadRecord(
        sequence_index=sequence_index,
        frame_id=int(model.frameId),
        frame_id_extra=int(model.frameIdExtra),
        timestamp_eof=int(model.timestampEof),
        lead_index=lead_index,
        prob=float(lead.prob),
        prob_time=float(lead.probTime),
        t=tuple(float(v) for v in lead.t),
        x=tuple(float(v) for v in lead.x),
        y=tuple(float(v) for v in lead.y),
        v=tuple(float(v) for v in lead.v),
        a=tuple(float(v) for v in lead.a),
        x_std=tuple(float(v) for v in lead.xStd),
        y_std=tuple(float(v) for v in lead.yStd),
        v_std=tuple(float(v) for v in lead.vStd),
        a_std=tuple(float(v) for v in lead.aStd),
      ))
      sequence_index += 1
  return records


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
  with path.open("w", newline="") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def write_lead_csv(path: Path, records: Sequence[LeadRecord]) -> None:
  rows = [record.to_csv_row() for record in records]
  write_csv(path, lead_csv_fieldnames(include_stds=True), rows)


def match_records(records_a: Sequence[LeadRecord], records_b: Sequence[LeadRecord],
                  align_by: str) -> Tuple[List[Tuple[LeadRecord, LeadRecord]], Dict[str, int]]:
  if align_by == ALIGN_BY_INDEX:
    matched_count = min(len(records_a), len(records_b))
    matches = list(zip(records_a[:matched_count], records_b[:matched_count]))
    metadata = {
      "run_a_total_rows": len(records_a),
      "run_b_total_rows": len(records_b),
      "matched_rows": matched_count,
      "run_a_unmatched_rows": len(records_a) - matched_count,
      "run_b_unmatched_rows": len(records_b) - matched_count,
    }
    return matches, metadata

  a_by_key = {record.alignment_key(): record for record in records_a}
  b_by_key = {record.alignment_key(): record for record in records_b}
  shared_keys = sorted(set(a_by_key) & set(b_by_key))
  matches = [(a_by_key[key], b_by_key[key]) for key in shared_keys]
  metadata = {
    "run_a_total_rows": len(records_a),
    "run_b_total_rows": len(records_b),
    "matched_rows": len(matches),
    "run_a_unmatched_rows": len(records_a) - len(matches),
    "run_b_unmatched_rows": len(records_b) - len(matches),
  }
  return matches, metadata


def build_diff_rows(matches: Sequence[Tuple[LeadRecord, LeadRecord]]) -> List[Dict[str, Any]]:
  rows: List[Dict[str, Any]] = []
  for record_a, record_b in matches:
    row: Dict[str, Any] = {
      "sequence_index_a": record_a.sequence_index,
      "sequence_index_b": record_b.sequence_index,
      "frame_id": record_a.frame_id,
      "frame_id_extra": record_a.frame_id_extra,
      "lead_index": record_a.lead_index,
      "prob_a": record_a.prob,
      "prob_b": record_b.prob,
      "prob_delta": record_b.prob - record_a.prob,
      "prob_abs_delta": abs(record_b.prob - record_a.prob),
    }
    horizon_len = min(len(record_a.t), len(record_b.t))
    for i in range(horizon_len):
      row[f"t_{i}"] = record_a.t[i]
      for field_name in ("x", "y", "v", "a"):
        values_a = getattr(record_a, field_name)
        values_b = getattr(record_b, field_name)
        delta = values_b[i] - values_a[i]
        row[f"{field_name}_a_{i}"] = values_a[i]
        row[f"{field_name}_b_{i}"] = values_b[i]
        row[f"{field_name}_delta_{i}"] = delta
        row[f"{field_name}_abs_delta_{i}"] = abs(delta)
    rows.append(row)
  return rows


def metric_stats(values: Sequence[float]) -> Dict[str, Any]:
  arr = np.asarray(values, dtype=np.float64)
  if arr.size == 0:
    return {
      "count": 0,
      "mean_signed": None,
      "mae": None,
      "rmse": None,
      "max_abs": None,
      "p95_abs": None,
    }

  abs_arr = np.abs(arr)
  return {
    "count": int(arr.size),
    "mean_signed": float(np.mean(arr)),
    "mae": float(np.mean(abs_arr)),
    "rmse": float(np.sqrt(np.mean(np.square(arr)))),
    "max_abs": float(np.max(abs_arr)),
    "p95_abs": float(np.percentile(abs_arr, 95)),
  }


def summarize_run(records: Sequence[LeadRecord]) -> Dict[str, Any]:
  frame_ids = [record.frame_id for record in records]
  unique_frames = {record.frame_key() for record in records}
  return {
    "modelv2_rows": len(records),
    "modelv2_messages": len(unique_frames),
    "frame_id_min": min(frame_ids) if frame_ids else None,
    "frame_id_max": max(frame_ids) if frame_ids else None,
  }


def dominant_lead_summary(records_a: Sequence[LeadRecord], records_b: Sequence[LeadRecord]) -> Dict[str, Any]:
  def group_by_frame(records: Sequence[LeadRecord]) -> Dict[Tuple[int, int], List[LeadRecord]]:
    grouped: DefaultDict[Tuple[int, int], List[LeadRecord]] = defaultdict(list)
    for record in records:
      grouped[record.frame_key()].append(record)
    return dict(grouped)

  grouped_a = group_by_frame(records_a)
  grouped_b = group_by_frame(records_b)
  shared_frames = sorted(set(grouped_a) & set(grouped_b))
  changed = 0
  for frame_key in shared_frames:
    leads_a = sorted(grouped_a[frame_key], key=lambda r: r.lead_index)
    leads_b = sorted(grouped_b[frame_key], key=lambda r: r.lead_index)
    dominant_a = max(leads_a, key=lambda r: (r.prob, -r.lead_index)).lead_index
    dominant_b = max(leads_b, key=lambda r: (r.prob, -r.lead_index)).lead_index
    if dominant_a != dominant_b:
      changed += 1

  return {
    "shared_frames": len(shared_frames),
    "dominant_lead_changed_frames": changed,
    "dominant_lead_changed_fraction": (float(changed) / len(shared_frames)) if shared_frames else None,
  }


def build_summary(matches: Sequence[Tuple[LeadRecord, LeadRecord]], records_a: Sequence[LeadRecord],
                  records_b: Sequence[LeadRecord], lead_prob_threshold: float, align_by: str) -> Dict[str, Any]:
  overall_deltas: DefaultDict[str, List[float]] = defaultdict(list)
  by_lead_deltas: DefaultDict[int, DefaultDict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
  by_lead_horizon: DefaultDict[str, DefaultDict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
  lead_presence_disagreement: DefaultDict[int, int] = defaultdict(int)

  for record_a, record_b in matches:
    prob_delta = record_b.prob - record_a.prob
    overall_deltas["prob"].append(prob_delta)
    by_lead_deltas[record_a.lead_index]["prob"].append(prob_delta)
    if (record_a.prob >= lead_prob_threshold) != (record_b.prob >= lead_prob_threshold):
      lead_presence_disagreement[record_a.lead_index] += 1

    horizon_len = min(len(record_a.t), len(record_b.t))
    for i in range(horizon_len):
      horizon_key = f"lead_{record_a.lead_index}_t_{record_a.t[i]:g}"
      for field_name in ("x", "y", "v", "a"):
        delta = getattr(record_b, field_name)[i] - getattr(record_a, field_name)[i]
        overall_deltas[field_name].append(delta)
        by_lead_deltas[record_a.lead_index][field_name].append(delta)
        by_lead_horizon[horizon_key][field_name].append(delta)

  overall_metrics = {field: metric_stats(values) for field, values in overall_deltas.items()}
  by_lead_metrics = {
    str(lead_index): {
      field: metric_stats(values) for field, values in field_deltas.items()
    }
    for lead_index, field_deltas in by_lead_deltas.items()
  }
  for lead_index, count in lead_presence_disagreement.items():
    by_lead_metrics.setdefault(str(lead_index), {})
    by_lead_metrics[str(lead_index)]["visibility_disagreement_count"] = count

  by_lead_horizon_metrics = {
    key: {field: metric_stats(values) for field, values in field_deltas.items()}
    for key, field_deltas in by_lead_horizon.items()
  }

  summary = {
    "alignment": {
      "mode": align_by,
      "matched_rows": len(matches),
      "run_a_total_rows": len(records_a),
      "run_b_total_rows": len(records_b),
      "run_a_unmatched_rows": len(records_a) - len(matches) if align_by == ALIGN_BY_INDEX else None,
      "run_b_unmatched_rows": len(records_b) - len(matches) if align_by == ALIGN_BY_INDEX else None,
    },
    "run_a": summarize_run(records_a),
    "run_b": summarize_run(records_b),
    "lead_prob_threshold": lead_prob_threshold,
    "dominant_lead": dominant_lead_summary(records_a, records_b),
    "overall": overall_metrics,
    "by_lead": by_lead_metrics,
    "by_lead_horizon": by_lead_horizon_metrics,
  }

  return summary


def serialize_summary(summary: Dict[str, Any], path: Path) -> None:
  with path.open("w") as summary_file:
    json.dump(summary, summary_file, indent=2, sort_keys=True)
    summary_file.write("\n")


def run_once(run_spec: RunSpec, send_raw_pred: bool, disable_progress: bool) -> Tuple[List[Any], List[LeadRecord]]:
  log_messages = load_log_messages(run_spec.log_path)
  frame_readers = build_frame_readers(run_spec, log_messages)
  try:
    replay_output = replay_modeld(log_messages, frame_readers, send_raw_pred, disable_progress)
  finally:
    close_frame_readers(frame_readers)
  lead_records = extract_lead_records(replay_output)
  return log_messages, lead_records


def normalize_run_specs(args: argparse.Namespace) -> Tuple[RunSpec, RunSpec]:
  if args.b_log is None:
    args.b_log = args.a_log
  return build_run_spec(args, "a"), build_run_spec(args, "b")


def main() -> None:
  args = parse_args()
  out_dir = Path(args.out_dir).expanduser().resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  run_a_spec, run_b_spec = normalize_run_specs(args)
  log_messages_a, records_a = run_once(run_a_spec, args.send_raw_pred, args.disable_progress)
  log_messages_b, records_b = run_once(run_b_spec, args.send_raw_pred, args.disable_progress)

  matches, alignment_metadata = match_records(records_a, records_b, args.align_by)
  if not matches:
    raise RuntimeError("no matched leadsV3 rows between run A and run B")

  write_lead_csv(out_dir / "run_a_leads.csv", records_a)
  write_lead_csv(out_dir / "run_b_leads.csv", records_b)
  diff_rows = build_diff_rows(matches)
  write_csv(out_dir / "leads_diff.csv", diff_csv_fieldnames(), diff_rows)

  summary = build_summary(matches, records_a, records_b, args.lead_prob_threshold, args.align_by)
  summary["alignment"].update(alignment_metadata)
  summary["inputs"] = {
    "run_a": {
      "log": str(run_a_spec.log_path),
      "road_camera": str(run_a_spec.stream_specs["roadCameraState"].video_path) if run_a_spec.stream_specs["roadCameraState"].video_path else None,
      "wide_road_camera": str(run_a_spec.stream_specs["wideRoadCameraState"].video_path) if run_a_spec.stream_specs["wideRoadCameraState"].video_path else None,
      "road_frames_dir": str(run_a_spec.stream_specs["roadCameraState"].frames_dir) if run_a_spec.stream_specs["roadCameraState"].frames_dir else None,
      "wide_road_frames_dir": str(run_a_spec.stream_specs["wideRoadCameraState"].frames_dir) if run_a_spec.stream_specs["wideRoadCameraState"].frames_dir else None,
      "road_stream_present": "roadCameraState" in {meta.camera_state for meta in available_streams(log_messages_a)},
      "wide_road_stream_present": "wideRoadCameraState" in {meta.camera_state for meta in available_streams(log_messages_a)},
    },
    "run_b": {
      "log": str(run_b_spec.log_path),
      "road_camera": str(run_b_spec.stream_specs["roadCameraState"].video_path) if run_b_spec.stream_specs["roadCameraState"].video_path else None,
      "wide_road_camera": str(run_b_spec.stream_specs["wideRoadCameraState"].video_path) if run_b_spec.stream_specs["wideRoadCameraState"].video_path else None,
      "road_frames_dir": str(run_b_spec.stream_specs["roadCameraState"].frames_dir) if run_b_spec.stream_specs["roadCameraState"].frames_dir else None,
      "wide_road_frames_dir": str(run_b_spec.stream_specs["wideRoadCameraState"].frames_dir) if run_b_spec.stream_specs["wideRoadCameraState"].frames_dir else None,
      "road_stream_present": "roadCameraState" in {meta.camera_state for meta in available_streams(log_messages_b)},
      "wide_road_stream_present": "wideRoadCameraState" in {meta.camera_state for meta in available_streams(log_messages_b)},
    },
    "send_raw_pred": args.send_raw_pred,
  }
  serialize_summary(summary, out_dir / "summary.json")


if __name__ == "__main__":
  main()
