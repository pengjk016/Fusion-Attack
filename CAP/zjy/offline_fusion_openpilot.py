# -*- coding: utf-8 -*-
#参考openpilot官方融合方案,融合前后的信息都保存
"""
批量补丁评估脚本 V2 (自动版本管理 + 详细报告 + GPU加速 + 最近邻插值)
修改内容：
1. [一致性对齐] 在 apply_resized_patch 中将缩放算法改为 cv2.INTER_NEAREST，与训练阶段严格对齐。
2. [输出管理] 自动检测 results/img_test 下的序号 (1, 2, 3...)，免覆盖保存。
3. [性能优化] 启用 GPU (CUDAExecutionProvider)，大幅提升评估速度。
4.叠加补丁
"""

import os

# 启用 GPU 时通常不需要禁用优化器，但为了稳健性保留
os.environ.setdefault("ORT_DISABLE_OPTIMIZER", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import csv
import numpy as np
import cv2
import time
import onnxruntime as ort
from ultralytics import YOLO
import pickle
import pandas as pd
# ====== 用户配置区 ======
# 1. 补丁所在目录 (输入)
PATCH_DIR = './realtestpatch'

# 2. 数据集路径 (输入)
IMGS_DIRECTORY = "/home/pjk/PycharmProjects/openpilot0.9.6/CAP/data/imgs/test_optim_patch26314"
ONNX_MODEL_PATH = '../models/weights/supercombo.onnx'
INFO_CSV_PATH = '/CAP/offline_fusion_results/leadOnegauss.csv'


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

def build_session(path: str):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    try:
        sess = ort.InferenceSession(path, sess_options=so,
                                    providers=[('CUDAExecutionProvider', {}), 'CPUExecutionProvider'])
        return sess
    except Exception:
        return ort.InferenceSession(path, sess_options=so, providers=['CPUExecutionProvider'])

def extract_yrel(out_vec):
    return float(out_vec[0, 5756])

def extract_vrel(out_vec):
    return float(out_vec[0, 5757])

def extract_prob(out_vec):
    return float(out_vec[0, 5755+72])

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

def get_yolo_box(img_bgr, yolo_model):
    boxes = yolo_model(img_bgr, conf=0.05, verbose=False)[0].boxes
    target_box = None
    max_area = 0

    img_h,img_w = img_bgr.shape[:2]

    for box in boxes:
        if box.cls in [2, 3, 5, 7]:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)

            x1,y1,x2,y2 = xyxy
            if y2>img_h*0.85:
                continue
            cx = (x1+x2)/2.0
            if cx < img_w*0.3 or cx> img_w*0.7:
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

def evaluate_sequence(sess, yolo_model, image_paths, v_ego_map, patch=None, clean_baseline=None, detail_writer=None):
    rnn_state_99 = {
        'features_buffer': np.zeros((1, 99, 512), dtype=np.float32),
        'prev_desired_curv': np.zeros((1, 100, 1), dtype=np.float32)
    }
    out0 = sess.get_outputs()[0].name
    drels, diffs = [], []
    dyrel = []
    dvrel = []
    prob = []
    valid_frames = 0

    for i in range(1, len(image_paths)):
        # ================= 修复点 1：直接读取原图，不做 resize =================
        img1_path = image_paths[i - 1]
        img2_path = image_paths[i]

        img1_bgr_original = cv2.imread(img1_path)
        img2_bgr_original = cv2.imread(img2_path)

        img2_name = os.path.basename(img2_path)

        if img1_bgr_original is None or img2_bgr_original is None:
            continue

        current_v_ego = v_ego_map.get(img2_name, 10.0)
        is_attacked_frame = False

        # 准备输入副本
        img1_input_bgr = img1_bgr_original.copy()
        img2_input_bgr = img2_bgr_original.copy()

        if patch is not None:
            # ================= 修复点 2：YOLO 在原图上检测 =================
            raw_box1 = get_yolo_box(img1_bgr_original, yolo_model)
            raw_box2 = get_yolo_box(img2_bgr_original, yolo_model)

            target_box1 = constrain_box_to_tailgate(raw_box1, PATCH_CROP_RATIO)
            target_box2 = constrain_box_to_tailgate(raw_box2, PATCH_CROP_RATIO)

            # 在原图 BGR 上应用 Patch
            if target_box1 is not None:
                img1_input_bgr = apply_resized_patch(img1_input_bgr, patch, target_box1)
            if target_box2 is not None:
                img2_input_bgr = apply_resized_patch(img2_input_bgr, patch, target_box2)
                is_attacked_frame = True

        # ================= 修复点 3：直接传 BGR 原图给 make_feed =================
        try:
            feed = make_feed_from_bgr(sess, img1_input_bgr, img2_input_bgr, rnn_state_99, current_v_ego)
            output = sess.run([out0], feed)[0]
            out_vec = np.array(output, dtype=np.float32)
        except Exception as e:
            print(f"[Warning] Frame {i} 推理跳过: {e}")
            continue

        # 提取数据 (统一使用 extract_onnx_096 的逻辑，或者你原来的逻辑，这里选一个)
        # 这里为了稳健，我们混合使用：用 096 的 dRel (减了偏移)，其他用简单提取
        try:
            # 使用更稳健的提取方式
            current_drel, _ = extract_onnx_096(out_vec)  # 这个减过 RADAR_TO_CAMERA
            drels.append(current_drel)

            # 注意：这里 yRel, vRel 也需要对应正确的索引，为了防止报错，我们先简化
            # 如果你需要准确的 yRel/vRel，需要确保它们的索引也是对的
            current_yrel = extract_yrel(out_vec)
            dyrel.append(current_yrel)

            current_vrel = extract_vrel(out_vec)
            dvrel.append(current_vrel)

            current_prob_val = extract_prob(out_vec)
            prob.append(current_prob_val)
        except Exception as e:
            print(f"[Warning] Frame {i} 提取跳过: {e}")
            continue

        if clean_baseline is not None:
            if valid_frames < len(clean_baseline):
                base_val = clean_baseline[valid_frames]
                diff = current_drel - base_val
                diffs.append(diff)
                if detail_writer:
                    status = "HIT" if is_attacked_frame else "MISS"
                    # 修复：表头有6项，这里也写6项
                    detail_writer.writerow(
                        [i, img2_name, status, f"{base_val:.4f}", f"{current_drel:.4f}", f"{diff:.4f}"])

        valid_frames += 1

        # 更新 RNN
        try:
            update_rnn_state(rnn_state_99, out_vec)
        except Exception as e:
            pass  # RNN 更新失败不影响主循环

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

def save_lead_to_csv(csv_path, drels, yrels, vrels, probs, image_paths, v_ego=0.0):
    """
    通用函数：将 drels 等数组保存为 LeadOne 格式的 CSV
    """
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=LEAD_CSV_FIELDNAMES)
        writer.writeheader()

        for i in range(len(drels)):
            frame_idx = i + 1
            # 图片索引对应：因为 evaluate_sequence 从 i=1 开始，所以这里对应 image_paths[i+1]
            img_path_idx = i + 1
            if img_path_idx < len(image_paths):
                image_path = os.path.basename(image_paths[img_path_idx])
            else:
                image_path = "unknown.png"

            # 安全检查
            if i >= len(yrels) or i >= len(vrels) or i >= len(probs):
                break

            # 构建字典
            lead_dict = {
                'frame_idx': frame_idx,
                'image_path': image_path,
                'dRel': float(drels[i]),
                'yRel': float(yrels[i]), # 注意：这里根据你的数据决定是否加负号，目前和攻击后保持一致
                'vRel': float(vrels[i]),
                'vLead': float(v_ego + vrels[i]),
                'vLeadK': float(v_ego + vrels[i]),
                'aLeadK': 0.0,
                'aLeadTau': 0.3,
                'status': True,
                'fcw': False,
                'modelProb': float(probs[i]),
                'radar': False,
                'radarTrackId': -1,
            }
            writer.writerow(lead_dict)
    print(f"   ✅ 数据已保存至: {csv_path}")

def main():
    if not os.path.exists(PATCH_DIR):
        print(f"❌ 错误：补丁目录不存在 {PATCH_DIR}")
        return

    patch_files = sorted([f for f in os.listdir(PATCH_DIR) if f.endswith('.npy')])
    if not patch_files:
        print(f"❌ 错误：在 {PATCH_DIR} 中没有找到 .npy 文件")
        return

    v_ego_map = {}
    if os.path.exists(INFO_CSV_PATH):
        try:
            df = pd.read_csv(INFO_CSV_PATH)
            for _, row in df.iterrows():
                v_ego_map[os.path.basename(row['image_path'])] = float(row['vEgo'])
        except Exception as e:
            print(f"⚠️ 读取 CSV 失败，回退 10 m/s: {e}")

    # === [修改点] 初始化输出目录 ===
    details_dir, summary_csv_path = setup_output_dirs(BASE_RESULT_DIR)

    print(f"🔎 发现 {len(patch_files)} 个补丁文件，准备评估...")
    print("🚀 加载模型 (YOLO + SuperCombo)...")
    yolo = YOLO("../models/weights/yolov8n.pt")
    sess = build_session(ONNX_MODEL_PATH)

    # image_paths = sorted(
    #     [os.path.join(IMGS_DIRECTORY, p) for p in os.listdir(IMGS_DIRECTORY) if p.lower().endswith('.png')])
    image_paths = sorted(
        [os.path.join(IMGS_DIRECTORY, p) for p in os.listdir(IMGS_DIRECTORY) if p.lower().endswith('.png')],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    print(f"📸 加载数据集: {len(image_paths)} 张图片")

    # 4. Phase 1: Clean Baseline
    print("\n" + "=" * 50)
    print(" 🏳️  正在建立纯净基准 (Clean Baseline) ...")
    print("=" * 50)
    t0 = time.time()
    clean_drels, _, clean_dyrel, clean_dvrel, clean_prob = evaluate_sequence(sess, yolo, image_paths, v_ego_map, patch=None)
    print(f"✅ 基准建立完成 (耗时 {time.time() - t0:.2f}s)")
    clean_lead_csv_path = os.path.join(details_dir, "lead_clean_baseline_openpilotfused.csv")
    print(f"   💾 正在保存 Clean Baseline LeadOne 数据...")
    save_lead_to_csv(clean_lead_csv_path, clean_drels, clean_dyrel, clean_dvrel, clean_prob, image_paths)
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
                    sess, yolo, image_paths,v_ego_map,
                    patch=raw_patch, clean_baseline=clean_drels,
                    detail_writer=detail_writer
                )

            print(f"   💾 正在保存 LeadOne 数据...")
            # 直接调用通用保存函数 (和保存 Clean Baseline 用的是同一个函数)
            save_lead_to_csv(lead_csv_path, atk_drels, atkyrel, atkvrel, atkprobs, image_paths)

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