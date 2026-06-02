# -*- coding: utf-8 -*-
#kalman滤波融合方案,自车速度不确定，输出位置不确定

import os

os.environ.setdefault("ORT_DISABLE_OPTIMIZER", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import csv
import numpy as np
import cv2
import time
import onnxruntime as ort
from ultralytics import YOLO

# ====== 用户配置区 ======
# 1. 补丁所在目录 (输入)
PATCH_DIR = './patch'

# 2. 数据集路径 (输入)
IMGS_DIRECTORY = "/home/pjk/PycharmProjects/openpilot0.9.6/CAP/data/imgs/test_optim_patch26314"
ONNX_MODEL_PATH = '../models/weights/supercombo.onnx'

# 3. 结果保存根目录 (输出)
# 程序会自动在此目录下创建 1/, 2/, 3/ ...
BASE_RESULT_DIR = 'result/img_test'

# ====== 新增：LeadOne CSV 表头定义 ======
LEAD_CSV_FIELDNAMES = [
    'frame_idx', 'image_path', 'dRel', 'yRel', 'vRel',
    'vLead', 'vLeadK', 'aLeadK', 'aLeadTau',
    'status', 'fcw', 'modelProb', 'radar', 'radarTrackId'
]
RADAR_TO_CAMERA = 1.52  # 确保有这个常量

kalbool = True
import importlib
from collections import deque
from typing import Optional, List
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

import os


current_file = os.path.abspath(__file__)

dir_zjy = os.path.dirname(current_file)
dir_cap = os.path.dirname(dir_zjy)
project_root = os.path.dirname(dir_cap)

# 把所有可能的路径都加进去，防止出错
paths_to_add = [
    project_root,                     # /openpilot0.9.6
    os.path.join(project_root, 'openpilot'), # /openpilot0.9.6/openpilot
]

for p in paths_to_add:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)
        print(f"[Info] 已添加到 sys.path: {p}")

import capnp
from cereal import messaging, log, car


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
        [1, dt, dt ** 2 / 2, 0, 0, 0],
        [0, 1, dt, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, dt, dt ** 2 / 2],
        [0, 0, 0, 0, 1, dt],
        [0, 0, 0, 0, 0, 1]
    ])

    kf.H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0]
    ])

    # kf.P = np.diag([10.0, 5.0, 1.0, 10.0, 5.0, 1.0])
    kf.P = np.diag([10.0, 5.0, 50.0, 10.0, 5.0, 50.0])

    # 过程噪声（与 MATLAB sigma=1 一致，y 方向极小以实现“常量”）
    sigma_x = 1.0
    sigma_y = 0.001
    Q1d_x = sigma_x ** 2 * np.array([[dt ** 4 / 4, dt ** 3 / 2, dt ** 2 / 2],
                                     [dt ** 3 / 2, dt ** 2, dt],
                                     [dt ** 2 / 2, dt, 1]])
    Q1d_y = sigma_y ** 2 * np.array([[dt ** 4 / 4, dt ** 3 / 2, dt ** 2 / 2],
                                     [dt ** 3 / 2, dt ** 2, dt],
                                     [dt ** 2 / 2, dt, 1]])
    kf.Q = np.block([[Q1d_x, np.zeros((3, 3))],
                     [np.zeros((3, 3)), Q1d_y]])

    return kf


def update_kf_with_detection(kf: KalmanFilter, det: Detection):
    if det.sensor_type == 'radar':
        z = np.array([det.x, det.vx, det.y, 0.0])
        R = np.diag([2.0, 2.0, 2.0, 100])  # vy 忽略
    else:
        z = np.array([det.x, det.vx, det.y, 0.0])
        R = np.diag([2.0, 2.0, 2.0, 100])  # vx 噪声大，vy 忽略

    kf.update(z, R=R)


# =========================================================
# Track 类：实现 Confirmation / Deletion 逻辑
# =========================================================

class Track:
    def __init__(self, kf: KalmanFilter):
        self.kf = kf
        self.is_confirmed = False
        self.hit_streak = 0  # 当前连续 hit 次数
        self.history = deque(maxlen=3)  # 最近 3 次的 hit/miss (1/0)
        self.misses_in_a_row = 0  # 连续 miss 次数，用于 DeletionThreshold=5

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
        self.assignment_threshold = 35.0  # 与 MATLAB 完全一致

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

        # 计算马氏距离代价矩阵
        num_tracks = len(self.tracks)
        num_dets = len(detections)
        cost = np.full((num_tracks, num_dets), np.inf)  # 初始化为无穷大

        for i, track in enumerate(self.tracks):
            H = track.kf.H
            P_pred = track.kf.P
            x_pred = track.kf.x.flatten()  # 确保为一维
            mu = H @ x_pred  # 预测测量 [x, vx, y, vy]

            for j, det in enumerate(detections):
                if det.sensor_type == 'radar':
                    R = np.diag([2.0, 2.0, 2.0, 100.0])
                else:
                    R = np.diag([2.0, 2.0, 2.0, 100.0])

                S = H @ P_pred @ H.T + R  # 创新协方差
                z = np.array([det.x, det.vx, det.y, 0.0])  # 检测测量

                innovation = z - mu
                try:
                    S_inv = np.linalg.inv(S)
                    mahalanobis_sq = innovation.T @ S_inv @ innovation
                    cost[i, j] = mahalanobis_sq  # 马氏距离
                except np.linalg.LinAlgError:
                    cost[i, j] = np.inf  # 如果S不可逆，设为无穷

        row_ind, col_ind = linear_sum_assignment(cost)

        assigned_tracks = set()
        assigned_dets = set()

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


class RadarD:
    def __init__(self, radar_ts: float, delay: int = 0):
        self.tracker = MultiObjectTracker(radar_ts)
        self.v_ego = 0.0
        self.v_ego_hist = deque([0.0], maxlen=delay + 1)
        self.last_v_ego_frame = -1
        self.radar_state_valid = False
        self.cnt = 0
        # 移除自动写 csv，改为由外部控制
        # self.csv_file = 'radar_log.csv'

    def update(self, current_dRel, current_yRel, current_vRel, current_prob,v_ego=0):
        """
        返回值：fused_lead_dict (如果有融合结果) 或 None
        """
        self.v_ego = v_ego  # 更新自车速度
        self.cnt += 1

        detections: List[Detection] = []


        detections.append(Detection('radar', current_dRel, current_yRel, current_vRel + v_ego))
        detections.append(Detection('vision', current_dRel, current_yRel, current_vRel + v_ego))

        self.tracker.update(detections)

        tracks = self.tracker.get_confirmed_tracks()

        # === 修改：生成并返回 lead_dict ===
        fused_lead_dict = None

        if tracks:
            # 取最近的一个目标 (leadOne)
            kf = tracks[0]

            # 构建标准字典
            fused_lead_dict = {
                "dRel": float(kf.x[0]),
                "yRel": float(kf.x[3]),
                "vRel": float(kf.x[1] - self.v_ego),
                "vLead": float(kf.x[1]),
                "vLeadK": float(kf.x[1]),  # 简化：KF 速度直接用
                "aLeadK": float(kf.x[2]),  # 这里的 x[2] 是加速度
                "aLeadTau": 0.3,  # 固定值
                "status": True,
                "fcw": False,
                "modelProb": current_prob,  # 融合后概率设为1
                "radar": True,
                "radarTrackId": 0,
            }
        else:
            # 如果没有确认的轨迹，返回 status=False 的空字典
            fused_lead_dict = {
                "dRel": None,
                "yRel": None,
                "vRel": None,
                "vLead": None,
                "vLeadK": None,
                "aLeadK": None,
                "aLeadTau": None,
                "status": False,
                "fcw": False,
                "modelProb": 0.0,
                "radar": False,
                "radarTrackId": -1,
            }

        return fused_lead_dict


def parse_image(frame: np.ndarray) -> np.ndarray:
    H = (frame.shape[0] * 2) // 3
    W = frame.shape[1]
    parsed = np.zeros((6, H // 2, W // 2), dtype=np.uint8)
    parsed[0] = frame[0:H:2, 0::2]
    parsed[1] = frame[1:H:2, 0::2]
    parsed[2] = frame[0:H:2, 1::2]
    parsed[3] = frame[1:H:2, 1::2]
    parsed[4] = frame[H:H + H // 4].reshape((-1, H // 2, W // 2))
    parsed[5] = frame[H + H // 4:H + H // 2].reshape((-1, H // 2, W // 2))
    return parsed


def read_image_bgr(path) -> np.ndarray:
    try:
        img = cv2.imread(path)
        if img is None:
            return None
        return cv2.resize(img, (512, 256))
    except Exception as e:
        return None


def prepare_input(img1_bgr, img2_bgr):
    img1_yuv = parse_image(cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2YUV_I420))
    img2_yuv = parse_image(cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2YUV_I420))
    return np.r_[img1_yuv, img2_yuv][np.newaxis, ...].astype(np.float16)


# ====== ORT Session (GPU 优先) ======
def build_session(path: str):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    print(f">> build ORT session: {path}")

    # 优先使用 CUDA，如果没有则回退到 CPU
    providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        }),
        'CPUExecutionProvider',
    ]

    try:
        sess = ort.InferenceSession(path, sess_options=so, providers=providers)
        print(f">> Active Providers: {sess.get_providers()}")
        return sess
    except Exception as e:
        print(f"!! GPU Session 创建失败，回退到 CPU: {e}")
        return ort.InferenceSession(path, sess_options=so, providers=['CPUExecutionProvider'])


def make_feed(session, img1, img2, rnn_state=None):
    feed = {}
    input_imgs = prepare_input(img1, img2)
    for inp in session.get_inputs():
        name = inp.name
        shp = [(1 if (s in (None, 'None', -1) or isinstance(s, str)) else int(s)) for s in inp.shape]
        if name == 'input_imgs':
            feed[name] = input_imgs
        elif name == 'desire':
            feed[name] = np.zeros(shp, dtype=np.float16)
            if len(shp) == 3 and shp[-1] >= 8: feed[name][0, -1, 0] = 1.0
        elif name == 'traffic_convention':
            feed[name] = np.zeros(shp, dtype=np.float16)
            if shp[-1] == 2: feed[name][0, 0] = 1.0
        elif name == 'features_buffer':
            if rnn_state and rnn_state.get('features_buffer') is not None:
                feed[name] = rnn_state['features_buffer']
            else:
                feed[name] = np.zeros(shp, dtype=np.float16)
        elif name == 'prev_desired_curv':
            if rnn_state and rnn_state.get('prev_desired_curv') is not None:
                feed[name] = rnn_state['prev_desired_curv']
            else:
                feed[name] = np.zeros(shp, dtype=np.float16)
        else:
            feed[name] = np.zeros(shp, dtype=np.float16)
    return feed


def extract_drel(out_vec):
    return float(out_vec[0, 5755])


def extract_yrel(out_vec):
    return float(out_vec[0, 5756])


def extract_vrel(out_vec):
    return float(out_vec[0, 5757])


def extract_prob(out_vec):
    return float(out_vec[0, 5857])


def apply_resized_patch(img_bgr, raw_patch, box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0: return img_bgr

    # === [关键修改] 使用最近邻插值，对齐训练脚本策略 ===
    resized_patch = cv2.resize(raw_patch, (w, h), interpolation=cv2.INTER_NEAREST)

    out = img_bgr.astype(np.float32).copy()

    # 边界保护
    y1 = max(0, y1);
    x1 = max(0, x1)
    y2 = min(out.shape[0], y2);
    x2 = min(out.shape[1], x2)

    target_h, target_w = y2 - y1, x2 - x1
    if target_h != h or target_w != w:
        resized_patch = resized_patch[:target_h, :target_w, :]

    out[y1:y2, x1:x2] = np.clip(out[y1:y2, x1:x2] + resized_patch, 0.0, 255.0)
    return out.astype(np.uint8)


def evaluate_sequence(sess, yolo_model, image_paths, patch=None, clean_baseline=None, verbose=False,
                      detail_writer=None):
    """
    对整个序列运行评估。
    """
    rnn_state = {'features_buffer': None, 'prev_desired_curv': None}
    out0 = sess.get_outputs()[0].name

    drels = []
    diffs = []
    dyrel = []
    dvrel = []
    prob = []
    valid_frames = 0

    for i in range(1, len(image_paths)):
        img1 = read_image_bgr(image_paths[i - 1])
        img2 = read_image_bgr(image_paths[i])

        # 跳过坏图
        if img1 is None or img2 is None:
            continue

        # 1. 检测目标
        target_box = None
        if patch is not None:
            results = yolo_model(img2, verbose=False)
            max_area = 0
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                if area > max_area:
                    max_area = area
                    target_box = xyxy

        # 2. 准备输入
        if patch is not None and target_box is not None:
            img2_input = apply_resized_patch(img2, patch, target_box)
            is_attacked_frame = True
        else:
            img2_input = img2
            is_attacked_frame = False

        # 3. 推理
        feed = make_feed(sess, img1, img2_input, rnn_state)
        output = sess.run([out0], feed)[0]

        # 4. 提取数据
        current_drel = extract_drel(np.array(output))
        drels.append(current_drel)

        current_yrel = extract_yrel(np.array(output))
        dyrel.append(current_yrel)

        current_vrel = extract_vrel(np.array(output))
        dvrel.append(current_vrel)

        current_prob = extract_prob(np.array(output))
        prob.append(current_prob)
        # 计算 Diff 并记录
        if clean_baseline is not None:
            if valid_frames < len(clean_baseline):
                base_val = clean_baseline[valid_frames]
                diff = current_drel - base_val
                diffs.append(diff)

                # CSV 保存 (detail_writer)
                if detail_writer:
                    img_name = os.path.basename(image_paths[i])
                    status = "HIT" if is_attacked_frame else "MISS"
                    detail_writer.writerow(
                        [i, img_name, status, f"{base_val:.4f}", f"{current_drel:.4f}", f"{diff:.4f}"])

        valid_frames += 1

        # 5. 更新 RNN
        out_vec = np.array(output)
        fb = feed.get('features_buffer')
        fb_dim = 512
        if fb is None or fb.shape[2] != fb_dim:
            fb = np.zeros((1, 99, fb_dim), dtype=np.float16)

        new_fb = np.roll(fb, -1, axis=1)
        new_fb[0, -1, :] = out_vec[0, -fb_dim:]
        rnn_state['features_buffer'] = new_fb

        pdc = feed.get('prev_desired_curv')
        if pdc is None: pdc = np.zeros((1, 100, 1), dtype=np.float16)
        new_pdc = np.roll(pdc, -1, axis=1)
        new_pdc[0, -1] = out_vec[0, 5990]
        rnn_state['prev_desired_curv'] = new_pdc

    return drels, diffs, dyrel, dvrel, prob


def setup_output_dirs(base_dir):
    """
    自动检测下一个可用的实验编号文件夹
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # 获取当前已有的编号文件夹 (如 '1', '2', '案例')
    existing_ids = []
    for d in os.listdir(base_dir):
        full_path = os.path.join(base_dir, d)
        if os.path.isdir(full_path) and d.isdigit():
            existing_ids.append(int(d))

    # 确定下一个 ID
    next_id = max(existing_ids) + 1 if existing_ids else 1

    # 构建路径
    run_dir = os.path.join(base_dir, str(next_id))
    details_dir = os.path.join(run_dir, 'details')
    summary_path = os.path.join(run_dir, 'patch_leaderboard.csv')

    # 创建目录
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(details_dir, exist_ok=True)

    print(f"📂 评估结果输出目录: {run_dir}")
    print(f"   ├─ 汇总: patch_leaderboard.csv")
    print(f"   └─ 详情: details/")

    return details_dir, summary_path


def main():
    if not os.path.exists(PATCH_DIR):
        print(f"❌ 错误：补丁目录不存在 {PATCH_DIR}")
        return

    patch_files = sorted([f for f in os.listdir(PATCH_DIR) if f.endswith('.npy')])
    if not patch_files:
        print(f"❌ 错误：在 {PATCH_DIR} 中没有找到 .npy 文件")
        return

    # === [修改点] 初始化输出目录 ===
    details_dir, summary_csv_path = setup_output_dirs(BASE_RESULT_DIR)

    print(f"🔎 发现 {len(patch_files)} 个补丁文件，准备评估...")
    print("🚀 加载模型 (YOLO + SuperCombo)...")
    yolo = YOLO("../models/weights/yolov8n.pt")
    sess = build_session(ONNX_MODEL_PATH)

    image_paths = sorted(
        [os.path.join(IMGS_DIRECTORY, p) for p in os.listdir(IMGS_DIRECTORY) if p.lower().endswith('.png')])
    print(f"📸 加载数据集: {len(image_paths)} 张图片")

    # 4. Phase 1: Clean Baseline
    print("\n" + "=" * 50)
    print(" 🏳️  正在建立纯净基准 (Clean Baseline) ...")
    print("=" * 50)
    t0 = time.time()
    clean_drels, diffs, dyrel, dvrel, prob = evaluate_sequence(sess, yolo, image_paths, patch=None, verbose=False)
    print(f"✅ 基准建立完成 (耗时 {time.time() - t0:.2f}s)")

    # 准备汇总 CSV
    f_csv = open(summary_csv_path, 'w', newline='')
    writer = csv.writer(f_csv)
    writer.writerow(["Rank", "Patch_Name", "Mean_Diff", "Median_Diff", "Std_Diff", "Max_Drop", "Score"])

    results = []

    # 5. Phase 2: Loop Patches
    print("\n" + "=" * 50)
    print(f" ⚔️  开始评估 {len(patch_files)} 个补丁")
    print("=" * 50)

    for idx, p_file in enumerate(patch_files):
        p_path = os.path.join(PATCH_DIR, p_file)

        try:
            raw_patch = np.load(p_path)
            t_start = time.time()

            # 创建该补丁的详细 CSV 文件 (放在本次运行的 details 文件夹下)
            detail_csv_name = f"detail_{os.path.splitext(p_file)[0]}.csv"
            detail_csv_path = os.path.join(details_dir, detail_csv_name)

            lead_csv_name = f"lead_{os.path.splitext(p_file)[0]}.csv"
            lead_csv_path = os.path.join(details_dir, lead_csv_name)

            with open(detail_csv_path, 'w', newline='') as f_detail:
                detail_writer = csv.writer(f_detail)
                # 写入详细 CSV 表头
                detail_writer.writerow(["Index", "ImageName", "Status", "Clean_dRel", "Attacked_dRel", "Diff"])

                # 运行评估
                atk_drels, atk_diffs, atkyrel, atkvrel, atkprobs = evaluate_sequence(
                    sess, yolo, image_paths,
                    patch=raw_patch, clean_baseline=clean_drels,
                    verbose=False,
                    detail_writer=detail_writer
                )
            # ... (前面的代码保持不变) ...

            print(f"   💾 正在保存 LeadOne 数据 (含融合)...")

            # 1. 初始化 Lead CSV
            with open(lead_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=LEAD_CSV_FIELDNAMES)
                writer.writeheader()

                # 2. 初始化融合器
                RD = RadarD(0.05, 2)

                # 3. 遍历每一帧
                for i in range(len(atk_drels)):
                    frame_idx = i + 1
                    if i + 1 < len(image_paths):
                        image_path = os.path.basename(image_paths[i + 1])
                    else:
                        image_path = "unknown.png"

                    if i >= len(atkyrel) or i >= len(atkvrel):
                        break

                    # 取出攻击后的视觉数据
                    current_dRel = atk_drels[i]
                    current_yRel = atkyrel[i]
                    current_vRel = atkvrel[i]
                    current_prob = atkprobs[i]
                    # === 核心：执行卡尔曼融合 ===
                    fused_lead_dict = RD.update(current_dRel, current_yRel, current_vRel, current_prob, v_ego=0.0)

                    # === 构建 CSV 行并写入 ===
                    # 补充 frame_idx 和 image_path 到字典里
                    fused_lead_dict['frame_idx'] = frame_idx
                    fused_lead_dict['image_path'] = image_path

                    # 写入文件 (DictWriter 会自动根据 fieldnames 匹配)
                    writer.writerow(fused_lead_dict)

                    # 可选：打印调试
                    # if frame_idx % 20 == 0:
                    #     print(f"Frame {frame_idx}: Fused dRel={fused_lead_dict['dRel']}")

            print(f"   ✅ 融合后的 LeadOne 已保存至: {lead_csv_path}")

            # ... (后面的统计代码保持不变) ...

            # --- 统计分析 ---
            if atk_diffs:
                diffs_np = np.array(atk_diffs)
                mean_diff = np.mean(diffs_np)
                median_diff = np.median(diffs_np)
                std_diff = np.std(diffs_np)
                min_diff = np.min(diffs_np)

                score = -mean_diff

                print(
                    f"[{idx + 1}/{len(patch_files)}] {p_file:<25} | Mean: {mean_diff:+.4f}m | Max: {min_diff:+.4f}m | Time: {time.time() - t_start:.1f}s")

                results.append({
                    "name": p_file,
                    "mean": mean_diff,
                    "median": median_diff,
                    "std": std_diff,
                    "min": min_diff,
                    "score": score
                })
            else:
                print(f"[{idx + 1}/{len(patch_files)}] ⚠️  {p_file} 无有效数据。")

        except Exception as e:
            print(f"\n❌ 处理补丁 {p_file} 时出错: {e}")

    # 6. Final Leaderboard
    results.sort(key=lambda x: x['score'], reverse=True)

    # 重新打开一个新的文件句柄，避免任何变量污染
    with open(summary_csv_path, 'w', newline='', encoding='utf-8') as final_csv:
        final_writer = csv.writer(final_csv)
        final_writer.writerow(["Rank", "Patch_Name", "Mean_Diff", "Median_Diff", "Std_Diff", "Max_Drop", "Score"])

        for rank, res in enumerate(results):
            final_writer.writerow([
                rank + 1,
                res['name'],
                f"{res['mean']:.4f}",
                f"{res['median']:.4f}",
                f"{res['std']:.4f}",
                f"{res['min']:.4f}",
                f"{res['score']:.4f}"
            ])

    print("\n" + "#" * 60)
    print(f"🏆 最终榜单 (按平均攻击效果排序)")
    print("#" * 60)
    print(f"{'Rank':<5} | {'Patch Name':<25} | {'Mean Diff':<10} | {'Max Drop':<10}")
    print("-" * 60)
    for rank, res in enumerate(results):
        print(f"{rank + 1:<5} | {res['name']:<25} | {res['mean']:<+9.2f}m | {res['min']:<+9.2f}m")

    print(f"\n✅ 评估完成！结果已保存在: {summary_csv_path}")


if __name__ == "__main__":
    main()