# -*- coding: utf-8 -*-
"""
批量补丁评估脚本 V5 (终极物理光学对齐版)
修改内容：
1. [矩阵对齐] 完全使用 clean离线2.py 中的官方底层 C++ 硬编码矩阵。
2. [车速对齐] 动态读取 leadOnegauss.csv，每一帧的 vEgo 严格对齐。
3. [贴图对齐] 引入双帧贴图和 PATCH_CROP_RATIO 动态比例裁剪，保证测试位置与训练位置像素级一致！
"""

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
PATCH_DIR = './patch'
IMGS_DIRECTORY = "/home/pjk/PycharmProjects/openpilot0.9.6/CAP/data/imgs/test_optim_patch26314"
# IMGS_DIRECTORY = './data/4/picture'
INFO_CSV_PATH = '/CAP/offline_fusion_results/leadOnegauss.csv'
ONNX_MODEL_PATH = '../models/weights/supercombo.onnx'

BASE_RESULT_DIR = 'result/img_test'

RADAR_TO_CAMERA = 0
PATCH_CROP_RATIO = [0, 0.72, 0, 1]  # [Top, Bottom, Left, Right]

# =========================================================================
# 【官方绝对矩阵硬编码】：直接使用官方底层 C++ 库计算出的结果。
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
        return img
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
    for box in boxes:
        if box.cls in [2, 3, 5, 7]:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
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


def evaluate_sequence(sess, yolo_model, image_paths, v_ego_map, patch=None, clean_baseline=None, detail_writer=None):
    rnn_state_99 = {
        'features_buffer': np.zeros((1, 99, 512), dtype=np.float32),
        'prev_desired_curv': np.zeros((1, 100, 1), dtype=np.float32)
    }
    out0 = sess.get_outputs()[0].name
    drels, diffs = [], []
    valid_frames = 0

    for i in range(1, len(image_paths)):
        img1 = read_image_bgr(image_paths[i - 1])
        img2 = read_image_bgr(image_paths[i])
        img2_name = os.path.basename(image_paths[i])

        if img1 is None or img2 is None: continue

        current_v_ego = v_ego_map.get(img2_name, 10.0)
        is_attacked_frame = False

        if patch is not None:
            raw_box1 = get_yolo_box(img1, yolo_model)
            raw_box2 = get_yolo_box(img2, yolo_model)

            target_box1 = constrain_box_to_tailgate(raw_box1, PATCH_CROP_RATIO)
            target_box2 = constrain_box_to_tailgate(raw_box2, PATCH_CROP_RATIO)

            img1_input = apply_resized_patch(img1, patch, target_box1)
            img2_input = apply_resized_patch(img2, patch, target_box2)
            is_attacked_frame = (target_box2 is not None)
        else:
            img1_input = img1
            img2_input = img2

        feed = make_feed_from_bgr(sess, img1_input, img2_input, rnn_state_99, current_v_ego)
        output = sess.run([out0], feed)[0]

        out_vec = np.array(output, dtype=np.float32)
        current_drel, current_prob = extract_onnx_096(out_vec)
        drels.append(current_drel)

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

        valid_frames += 1
        update_rnn_state(rnn_state_99, out_vec)

    return drels, diffs


def setup_output_dirs(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    existing_ids = [int(d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()]
    next_id = max(existing_ids) + 1 if existing_ids else 1
    run_dir = os.path.join(base_dir, str(next_id))
    details_dir = os.path.join(run_dir, 'details')
    summary_path = os.path.join(run_dir, 'patch_leaderboard.csv')

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(details_dir, exist_ok=True)
    print(f"📂 评估结果输出目录: {run_dir}")
    return details_dir, summary_path


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

    details_dir, summary_csv_path = setup_output_dirs(BASE_RESULT_DIR)

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
    clean_drels, _ = evaluate_sequence(sess, yolo, image_paths, v_ego_map, patch=None)
    print(f"✅ 基准建立完成 (耗时 {time.time() - t0:.2f}s)")

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

            with open(detail_csv_path, 'w', newline='') as f_detail:
                detail_writer = csv.writer(f_detail)
                detail_writer.writerow(["Index", "ImageName", "Status", "Clean_dRel", "Attacked_dRel", "Diff", "Prob"])
                atk_drels, atk_diffs = evaluate_sequence(
                    sess, yolo, image_paths, v_ego_map,
                    patch=raw_patch, clean_baseline=clean_drels,
                    detail_writer=detail_writer
                )

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

    results.sort(key=lambda x: x['score'], reverse=True)
    for rank, res in enumerate(results):
        writer.writerow([rank + 1, res['name'], f"{res['mean']:.4f}", f"{res['median']:.4f}", f"{res['std']:.4f}",
                         f"{res['min']:.4f}", f"{res['score']:.4f}"])

    f_csv.close()
    print(f"\n✅ 评估完成！结果已保存在: {summary_csv_path}")


if __name__ == "__main__":
    main()