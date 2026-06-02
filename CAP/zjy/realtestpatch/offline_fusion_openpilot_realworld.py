import os
import sys

# 启用 GPU 时通常不需要禁用优化器，但为了稳健性保留
os.environ.setdefault("ORT_DISABLE_OPTIMIZER", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import csv
import numpy as np
import cv2
import time
import pickle
import pandas as pd
import onnxruntime as ort
from ultralytics import YOLO

# ====== 用户配置区 ======
# 1. 补丁所在目录 (输入)
PATCH_DIR = '/home/pjk/PycharmProjects/openpilot0.9.6/CAP/zjy/realtestpatch'

# 2. 数据集路径 (输入)
IMGS_DIRECTORY = "/home/pjk/下载/实车/8,40,48/48/picture"
ONNX_MODEL_PATH = '../../models/weights/supercombo.onnx'
INFO_CSV_PATH = '/home/pjk/下载/实车/8,40,48/48/leadOne.csv'

BASE_RESULT_DIR = 'result/img_test'
ATTACK_LOG_DIR = 'results/attack'  # 贴有补丁的图片可视化保存根目录

RADAR_TO_CAMERA = 11.52
PATCH_CROP_RATIO = [0, 1, 0, 1]  # [Top, Bottom, Left, Right]

# ====== LeadOne CSV 表头定义 ======
LEAD_CSV_FIELDNAMES = [
    'frame_idx', 'image_path', 'dRel', 'yRel', 'vRel',
    'vLead', 'vLeadK', 'aLeadK', 'aLeadTau',
    'status', 'fcw', 'modelProb', 'radar', 'radarTrackId'
]

# =========================================================================
# 【官方绝对矩阵硬编码】
# =========================================================================
M_MAIN = np.array([
    [2.9098902, 0.000000000000001, 219.06813],
    [-0.0000000000000001, 2.9098902, 465.48923],
    [0., 0., 1.]
], dtype=np.float32)
INV_M_MAIN = np.linalg.inv(M_MAIN)

M_EXTRA = np.array([
    [1.2461538, 0., 644.9846],
    [-0., 1.2461538, 414.83383],
    [0., 0., 1.]
], dtype=np.float32)
INV_M_EXTRA = np.linalg.inv(M_EXTRA)

# 加载官方 metadata
METADATA_PATH = os.path.join(os.path.dirname(ONNX_MODEL_PATH), 'supercombo_metadata.pkl')
try:
    with open(METADATA_PATH, 'rb') as f:
        model_metadata = pickle.load(f)
    OUTPUT_SLICES = model_metadata['output_slices']
except Exception as e:
    OUTPUT_SLICES = None


# ========================== 2D 峰值检测 ==========================
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


# ========================== 雷达信号生成函数 ==========================
def append_dect(surrounding_info):
    c = 3e8  # 光速 (m/s)

    # 雷达基础参数 (严格对应 Table V)
    fc = 1.5e9  # 中心频率 = 1.5 GHz
    B = 25e6  # 带宽 = 25.00 MHz
    slope = 0.05e12  # 调频斜率 = 0.05 MHz/us = 5e10 Hz/s
    Tchirp = 501.12e-6  # 脉冲宽度 = 501.12 us
    Nd = 256  # 每帧 Chirp 数 = 256

    rangeRes = 6.09  # 距离分辨率 = 6.09 m
    maxR = 1558.92  # 最大探测距离 = 1558.92 m
    vRes = 0.78  # 速度分辨率 = 0.78 m/s
    maxV = 99.71  # 最大速度 = 99.71 m/s

    # 雷达派生参数计算
    PRI = c / (4 * fc * maxV)
    Fs = maxR * 2 * slope / c
    Nr = int(np.round(Fs * Tchirp))

    range_bin_size = c / (2 * slope * Tchirp)
    vel_bin_size = c / (2 * fc * Nd * PRI)

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

    if not targets:
        return []

    # 时间轴 & 发射信号
    t = np.linspace(0, Nd * Tchirp, Nr * Nd, endpoint=False)
    angle_tx = fc * t + 0.5 * slope * t * t
    Tx = np.cos(2 * np.pi * angle_tx)

    # 多目标回波 & IF 基带
    IF_mat = np.zeros((Nd, Nr))
    lambda_ = c / fc

    for tgt in targets:
        r_t = tgt["r0"]
        v_t = tgt["v0"]
        amp = tgt["amp"]

        for d in range(Nd):
            fb = 2 * slope * r_t / c
            phase = 4 * np.pi * (r_t + v_t * d * PRI) / lambda_
            t_chirp = np.linspace(0, Tchirp, Nr)
            IF_chirp = amp * np.cos(2 * np.pi * (fb * t_chirp + phase / (2 * np.pi)))
            IF_mat[d] += IF_chirp

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

    # 峰值检测
    peaks_2d = find_top_k_peaks_2d(RD, k=len(targets), suppr_v=suppr_v_dynamic, suppr_r=suppr_r_dynamic)

    detections = []
    for (vi, ri, val) in peaks_2d:
        lat_pos = targets[len(detections)]['relative_position'][1] if (
                    len(targets) > len(detections) and len(targets[len(detections)]['relative_position']) > 1) else 0.0
        detections.append({
            "R": range_axis[valid_r][ri],
            "V": vel_axis[valid_v][vi],
            "P_dB": 20 * np.log10(val + 1e-12),
            "Y": lat_pos
        })
    return detections


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
    if isinstance(path, str):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"无法读取图片: {path}")

        # 物理光学对齐: 1164x874 -> 裁剪上下边缘 -> 1164x729 -> 拉伸至 1928x1208
        img_cropped = img[72:801, :, :]
        img_resized = cv2.resize(img_cropped, (1928, 1208), interpolation=cv2.INTER_LINEAR)
        return img_resized
    return path


def process_image(img_bgr, inv_matrix):
    img_warped = cv2.warpPerspective(img_bgr, inv_matrix, (512, 256), flags=cv2.INTER_LINEAR)
    img_yuv = cv2.cvtColor(img_warped, cv2.COLOR_BGR2YUV_I420)
    return parse_image(img_yuv)


def prepare_inputs(img1_bgr, img2_bgr):
    parsed1_main = process_image(img1_bgr, INV_M_MAIN)
    parsed2_main = process_image(img2_bgr, INV_M_MAIN)
    parsed1_extra = process_image(img1_bgr, INV_M_EXTRA)
    parsed2_extra = process_image(img2_bgr, INV_M_EXTRA)

    input_imgs = np.r_[parsed1_main, parsed2_main][np.newaxis, ...].astype(np.float16)
    big_input_imgs = np.r_[parsed1_extra, parsed2_extra][np.newaxis, ...].astype(np.float16)
    return input_imgs, big_input_imgs


def make_feed_from_bgr(session, img1_bgr, img2_bgr, rnn_state=None, current_v_ego=10.0):
    input_imgs, big_input_imgs = prepare_inputs(img1_bgr, img2_bgr)
    feed = {}
    for inp in session.get_inputs():
        name = inp.name
        shp = [(1 if (s in (None, 'None', -1) or isinstance(s, str)) else int(s)) for s in inp.shape]

        if name == 'input_imgs':
            feed[name] = input_imgs
        elif name == 'big_input_imgs':
            feed[name] = big_input_imgs
        elif name == 'desire':
            feed[name] = np.zeros(shp, dtype=np.float16)
            if len(shp) == 3:
                feed[name][0, -1, 0] = 1.0
            else:
                feed[name][0, 0] = 1.0
        elif name == 'traffic_convention':
            feed[name] = np.zeros(shp, dtype=np.float16)
            if len(shp) == 2: feed[name][0, 0] = 1.0
        elif name == 'lateral_control_params':
            tmp = np.zeros(shp, dtype=np.float32)
            tmp[0, 0] = current_v_ego
            tmp[0, 1] = 0.2
            feed[name] = tmp.astype(np.float16)
        elif name == 'features_buffer':
            feed[name] = rnn_state['features_buffer'].astype(np.float16)
        elif name == 'prev_desired_curv':
            feed[name] = rnn_state['prev_desired_curv'].astype(np.float16)
        else:
            feed[name] = np.zeros(shp, dtype=np.float16)
    return feed


def extract_onnx_096(out_vec):
    if OUTPUT_SLICES is not None and 'lead' in OUTPUT_SLICES:
        lead = out_vec[0, OUTPUT_SLICES['lead']]
    else:
        lead = out_vec[0, 5755:6010]

    x_predt0 = lead[0::51]
    prob0 = lead[48::51]

    current_most_likely_hypo = np.argmax(prob0)
    drelt = x_predt0[current_most_likely_hypo]
    drelt -= RADAR_TO_CAMERA
    prob_val = 1 / (1 + np.exp(-prob0[current_most_likely_hypo]))
    return float(drelt), float(prob_val)


def extract_yrel(out_vec):
    return float(out_vec[0, 5756])


def extract_vrel(out_vec):
    return float(out_vec[0, 5757])


def build_session(path: str):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    try:
        sess = ort.InferenceSession(path, sess_options=so,
                                    providers=[('CUDAExecutionProvider', {}), 'CPUExecutionProvider'])
        return sess
    except Exception:
        return ort.InferenceSession(path, sess_options=so, providers=['CPUExecutionProvider'])


def get_yolo_box(img_bgr, yolo_model):
    boxes = yolo_model(img_bgr, conf=0.05, verbose=False)[0].boxes
    target_box = None
    max_area = 0

    img_h, img_w = img_bgr.shape[:2]

    for box in boxes:
        if box.cls in [2, 3, 5, 7]:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy

            if y2 > img_h * 0.85:
                continue

            cx = (x1 + x2) / 2.0
            if cx < img_w * 0.45 or cx > img_w * 0.55:
                continue

            area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
            if area > max_area:
                max_area = area
                target_box = xyxy
    return target_box


def constrain_box_to_tailgate(box_xyxy, ratio_config):
    if box_xyxy is None: return None
    x1, y1, x2, y2 = box_xyxy
    w, h = x2 - x1, y2 - y1
    top, bottom, left, right = ratio_config
    new_x1 = int(x1 + w * left)
    new_y1 = int(y1 + h * top)
    new_x2 = int(x1 + w * right)
    new_y2 = int(y1 + h * bottom)
    if new_x2 <= new_x1 or new_y2 <= new_y1: return None
    return (new_x1, new_y1, new_x2, new_y2)


def apply_resized_patch(img_bgr: np.ndarray, raw_patch: np.ndarray, box_xyxy) -> np.ndarray:
    if box_xyxy is None: return img_bgr.copy()
    x1, y1, x2, y2 = box_xyxy
    box_w, box_h = x2 - x1, y2 - y1
    if box_w <= 0 or box_h <= 0: return img_bgr.copy()

    resized_patch = cv2.resize(raw_patch, (box_w, box_h), interpolation=cv2.INTER_NEAREST_EXACT)

    out = img_bgr.astype(np.float32).copy()
    y1_c = max(0, y1);
    x1_c = max(0, x1)
    y2_c = min(out.shape[0], y2);
    x2_c = min(out.shape[1], x2)
    target_h, target_w = y2_c - y1_c, x2_c - x1_c

    if target_h <= 0 or target_w <= 0: return out.astype(np.uint8)

    if target_h != box_h or target_w != box_w:
        patch_x1 = x1_c - x1
        patch_y1 = y1_c - y1
        resized_patch = resized_patch[patch_y1:patch_y1 + target_h, patch_x1:patch_x1 + target_w, :]

    out[y1_c:y2_c, x1_c:x2_c] = np.clip(out[y1_c:y2_c, x1_c:x2_c] + resized_patch, 0.0, 255.0)
    return out.astype(np.uint8)


def update_rnn_state(rnn_state, out_vec):
    if OUTPUT_SLICES is not None:
        hidden_state = out_vec[0, OUTPUT_SLICES['hidden_state']]
        desired_curv = out_vec[0, OUTPUT_SLICES['desired_curvature']]
    else:
        hidden_state = out_vec[0, -512:]
        desired_curv = out_vec[0, 5990:5991]

    new_fb = np.roll(rnn_state['features_buffer'], -1, axis=1)
    new_fb[0, -1, :] = hidden_state
    rnn_state['features_buffer'] = new_fb
    new_pdc = np.roll(rnn_state['prev_desired_curv'], -1, axis=1)
    new_pdc[0, -1, 0] = desired_curv[0] if desired_curv.size > 0 else 0.0
    rnn_state['prev_desired_curv'] = new_pdc


def save_lead_to_csv(csv_path, drels, yrels, vrels, probs, image_paths, v_ego_map=None):
    """
    将融合后的信息保存为 LeadOne 格式的 CSV
    """
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=LEAD_CSV_FIELDNAMES)
        writer.writeheader()

        for i in range(len(drels)):
            frame_idx = i + 1
            img_path_idx = i + 1
            if img_path_idx < len(image_paths):
                image_path = os.path.basename(image_paths[img_path_idx])
            else:
                image_path = "unknown.png"

            if i >= len(yrels) or i >= len(vrels) or i >= len(probs):
                break

            current_v_ego = 10.0
            if v_ego_map is not None and image_path in v_ego_map:
                current_v_ego = v_ego_map[image_path]

            lead_dict = {
                'frame_idx': frame_idx,
                'image_path': image_path,
                'dRel': float(drels[i]),
                'yRel': float(yrels[i]),
                'vRel': float(vrels[i]),
                'vLead': float(current_v_ego + vrels[i]),
                'vLeadK': float(current_v_ego + vrels[i]),
                'aLeadK': 0.0,
                'aLeadTau': 0.3,
                'status': True,
                'fcw': False,
                'modelProb': float(probs[i]),
                'radar': False,
                'radarTrackId': -1,
            }
            writer.writerow(lead_dict)
    print(f"   ✅ 融合数据已保存至: {csv_path}")


def evaluate_sequence(sess, yolo_model, image_paths, v_ego_map, patch=None, clean_baseline=None, detail_writer=None,
                      save_start=-1, save_end=-1, img_save_dir=None, patch_name=""):
    rnn_state_99 = {
        'features_buffer': np.zeros((1, 99, 512), dtype=np.float32),
        'prev_desired_curv': np.zeros((1, 100, 1), dtype=np.float32)
    }
    out0 = sess.get_outputs()[0].name
    drels, diffs, dyrel, dvrel, prob = [], [], [], [], []
    valid_frames = 0
    hit_count = 0

    for i in range(1, len(image_paths)):
        img1 = read_image_bgr(image_paths[i - 1])
        img2 = read_image_bgr(image_paths[i])
        img2_name = os.path.basename(image_paths[i])

        if img1 is None or img2 is None: continue

        current_v_ego = v_ego_map.get(img2_name, 10.0)
        is_attacked_frame = False

        # 雷达信号生成逻辑
        surrounding_info = []
        if patch is not None:
            # 先运行一次干净的推理获取真实前车信息，用于生成雷达信号
            clean_feed = make_feed_from_bgr(sess, img1, img2, rnn_state_99, current_v_ego)
            clean_output = sess.run([out0], clean_feed)[0]
            clean_out_vec = np.array(clean_output, dtype=np.float32)
            clean_drel, _ = extract_onnx_096(clean_out_vec)
            clean_vrel = extract_vrel(clean_out_vec)

            surrounding_info = [{
                'relative_position': [clean_drel, 0],
                'relative_velocity': [clean_vrel * 3.6, 0]
            }]

            # 生成雷达检测结果
            detections = append_dect(surrounding_info)

        if patch is not None:
            raw_box1 = get_yolo_box(img1, yolo_model)
            raw_box2 = get_yolo_box(img2, yolo_model)

            target_box1 = constrain_box_to_tailgate(raw_box1, PATCH_CROP_RATIO)
            target_box2 = constrain_box_to_tailgate(raw_box2, PATCH_CROP_RATIO)

            img1_input = apply_resized_patch(img1, patch, target_box1)
            img2_input = apply_resized_patch(img2, patch, target_box2)
            is_attacked_frame = (target_box2 is not None)
            if is_attacked_frame:
                hit_count += 1
        else:
            img1_input = img1
            img2_input = img2

        feed = make_feed_from_bgr(sess, img1_input, img2_input, rnn_state_99, current_v_ego)
        output = sess.run([out0], feed)[0]

        out_vec = np.array(output, dtype=np.float32)
        current_drel, current_prob = extract_onnx_096(out_vec)
        current_yrel = extract_yrel(out_vec)
        current_vrel = extract_vrel(out_vec)

        drels.append(current_drel)
        dyrel.append(current_yrel)
        dvrel.append(current_vrel)
        prob.append(current_prob)

        if clean_baseline is not None:
            if valid_frames < len(clean_baseline):
                base_val = clean_baseline[valid_frames]
                diff = current_drel - base_val
                diffs.append(diff)
                if detail_writer:
                    status = "HIT" if is_attacked_frame else "MISS"
                    detail_writer.writerow(
                        [i, img2_name, status, f"{base_val:.4f}", f"{current_drel:.4f}", f"{diff:.4f}",
                         f"{current_prob:.4f}"])

        # 保存带补丁的图片
        if img_save_dir is not None and (save_start <= i <= save_end) and is_attacked_frame:
            patch_id = os.path.splitext(patch_name)[0] if patch_name else "patch"
            vis_dir = os.path.join(img_save_dir, patch_id)
            os.makedirs(vis_dir, exist_ok=True)
            save_path = os.path.join(vis_dir, f"frame_{i:04d}_{img2_name}")
            cv2.imwrite(save_path, img2_input)

        valid_frames += 1
        update_rnn_state(rnn_state_99, out_vec)

    if patch is not None:
        print(f"   🎯 补丁贴图统计: 共有 {hit_count}/{len(image_paths) - 1} 帧成功注入了补丁")

    return drels, diffs, dyrel, dvrel, prob


def setup_output_dirs(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    existing_ids = [int(d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()]
    next_id = max(existing_ids) + 1 if existing_ids else 1
    run_dir = os.path.join(base_dir, str(next_id))
    details_dir = os.path.join(run_dir, 'details')
    summary_path = os.path.join(run_dir, 'patch_leaderboard.csv')

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(details_dir, exist_ok=True)
    print(f"📂 评估结果(CSV)输出目录: {run_dir}")
    return details_dir, summary_path, run_dir


def main():
    if not os.path.exists(PATCH_DIR): return print(f"❌ 错误：补丁目录不存在 {PATCH_DIR}")
    patch_files = sorted([f for f in os.listdir(PATCH_DIR) if f.endswith('.npy')])
    if not patch_files: return print(f"❌ 错误：在 {PATCH_DIR} 中没有找到 .npy 文件")

    v_ego_map = {}
    if os.path.exists(INFO_CSV_PATH):
        try:
            df = pd.read_csv(INFO_CSV_PATH)
            for _, row in df.iterrows():
                v_ego_map[os.path.basename(row['image_path'])] = float(row['vEgo'])
        except Exception as e:
            print(f"⚠️ 读取 CSV 失败，回退 10 m/s: {e}")

    # 交互式输入保存图片的范围
    print("\n" + "=" * 50)
    user_range = input("🖼️  请输入需要保存带补丁图片的帧序号范围 (例如 50-100，直接回车则不保存): ").strip()
    save_start, save_end = -1, -1
    if user_range and '-' in user_range:
        try:
            parts = user_range.split('-')
            save_start = int(parts[0])
            save_end = int(parts[1])
        except ValueError:
            print("⚠️ 输入格式有误，本次评估将不保存图片。")

    current_img_save_dir = None
    if save_start != -1 and save_end != -1:
        os.makedirs(ATTACK_LOG_DIR, exist_ok=True)
        existing_img_ids = [int(d) for d in os.listdir(ATTACK_LOG_DIR) if
                            os.path.isdir(os.path.join(ATTACK_LOG_DIR, d)) and d.isdigit()]
        next_img_id = max(existing_img_ids) + 1 if existing_img_ids else 1
        current_img_save_dir = os.path.join(ATTACK_LOG_DIR, str(next_img_id))
        os.makedirs(current_img_save_dir, exist_ok=True)
        print(f"📸 选定范围 [{save_start}, {save_end}] 的带补丁图片将保存在: {current_img_save_dir}")
    print("=" * 50 + "\n")

    details_dir, summary_csv_path, run_dir = setup_output_dirs(BASE_RESULT_DIR)

    print(f"🔎 发现 {len(patch_files)} 个补丁文件，准备评估...")
    yolo = YOLO("../models/weights/yolov8n.pt")
    sess = build_session(ONNX_MODEL_PATH)

    image_paths = sorted(
        [os.path.join(IMGS_DIRECTORY, p) for p in os.listdir(IMGS_DIRECTORY) if p.lower().endswith('.png')],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )

    print("\n" + "=" * 50)
    print(" 🏳️  正在建立纯净基准 (Clean Baseline) ...")
    t0 = time.time()
    clean_drels, _, clean_dyrel, clean_dvrel, clean_prob = evaluate_sequence(
        sess, yolo, image_paths, v_ego_map, patch=None
    )
    print(f"✅ 基准建立完成 (耗时 {time.time() - t0:.2f}s)")

    # 保存纯净基准的融合信息
    clean_lead_csv_path = os.path.join(details_dir, "lead_clean_baseline_openpilotfused.csv")
    print(f"   💾 正在保存 Clean Baseline 融合数据...")
    save_lead_to_csv(clean_lead_csv_path, clean_drels, clean_dyrel, clean_dvrel, clean_prob, image_paths, v_ego_map)

    f_csv = open(summary_csv_path, 'w', newline='')
    writer = csv.writer(f_csv)
    writer.writerow(["Rank", "Patch_Name", "Mean_Diff", "Median_Diff", "Std_Diff", "Max_Drop", "Score"])
    results = []

    print("\n" + "=" * 50)
    print(f" ⚔️  开始评估 {len(patch_files)} 个补丁")

    for idx, p_file in enumerate(patch_files):
        p_path = os.path.join(PATCH_DIR, p_file)
        try:
            raw_patch = np.load(p_path)
            t_start = time.time()
            detail_csv_path = os.path.join(details_dir, f"detail_{os.path.splitext(p_file)[0]}.csv")
            lead_csv_path = os.path.join(details_dir, f"lead_{os.path.splitext(p_file)[0]}.csv")

            with open(detail_csv_path, 'w', newline='') as f_detail:
                detail_writer = csv.writer(f_detail)
                detail_writer.writerow(["Index", "ImageName", "Status", "Clean_dRel", "Attacked_dRel", "Diff", "Prob"])

                atk_drels, atk_diffs, atk_dyrel, atk_dvrel, atk_prob = evaluate_sequence(
                    sess, yolo, image_paths, v_ego_map,
                    patch=raw_patch, clean_baseline=clean_drels,
                    detail_writer=detail_writer,
                    save_start=save_start, save_end=save_end,
                    img_save_dir=current_img_save_dir, patch_name=p_file
                )

            # 保存攻击后的融合信息
            print(f"   💾 正在保存攻击后的融合数据...")
            save_lead_to_csv(lead_csv_path, atk_drels, atk_dyrel, atk_dvrel, atk_prob, image_paths, v_ego_map)

            if atk_diffs:
                diffs_np = np.array(atk_diffs)
                mean_diff, median_diff = np.mean(diffs_np), np.median(diffs_np)
                std_diff, min_diff = np.std(diffs_np), np.min(diffs_np)
                print(
                    f"[{idx + 1}/{len(patch_files)}] {p_file:<25} | Mean: {mean_diff:+.4f}m | Max Drop: {min_diff:+.4f}m | Time: {time.time() - t_start:.1f}s")
                results.append(
                    {"name": p_file, "mean": mean_diff, "median": median_diff, "std": std_diff, "min": min_diff,
                     "score": -mean_diff})
            else:
                print(f"[{idx + 1}/{len(patch_files)}] ⚠️  {p_file} 无有效数据。")
        except Exception as e:
            print(f"\n❌ 处理补丁 {p_file} 时出错: {e}")
            import traceback
            traceback.print_exc()

    results.sort(key=lambda x: x['score'], reverse=True)
    for rank, res in enumerate(results):
        writer.writerow([rank + 1, res['name'], f"{res['mean']:.4f}", f"{res['median']:.4f}", f"{res['std']:.4f}",
                         f"{res['min']:.4f}", f"{res['score']:.4f}"])

    f_csv.close()
    print(f"\n✅ 评估完成！结果已保存在: {summary_csv_path}")
    print(f"📊 所有融合数据已保存在: {details_dir}/")
    if current_img_save_dir:
        print(f"📸 选定范围内的带补丁图片已保存在: {current_img_save_dir}")


if __name__ == "__main__":
    main()
