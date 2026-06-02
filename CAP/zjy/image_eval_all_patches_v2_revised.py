# -*- coding: utf-8 -*-
#参考openpilot官方融合方案
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

            print(f"   💾 正在保存 LeadOne 数据...")

            # 1. 初始化 Lead CSV
            with open(lead_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=LEAD_CSV_FIELDNAMES)
                writer.writeheader()

                # 2. 遍历每一帧的数据 (假设 image_paths, atkyrel 等长度对齐)
                # 注意：这里假设 valid_frames 数量和 len(atkyrel) 一致
                # 我们从第2张图开始 (index 1)，对应 frame_idx 1
                for i in range(len(atk_drels)):
                    frame_idx = i + 1  # 简单对齐索引
                    if i + 1 < len(image_paths):
                        image_path = os.path.basename(image_paths[i + 1])
                    else:
                        image_path = "unknown.png"

                    # 安全检查，防止索引越界
                    if i >= len(atkyrel) or i >= len(atkvrel) or i >= len(atkprobs):
                        break

                    # 3. 构造数据
                    # 假设 v_ego = 0 (离线场景)
                    v_ego = 0.0
                    current_dRel = atk_drels[i]  # 这里的 atk_drels 应该已经是减过 RADAR_TO_CAMERA 的
                    current_yRel = atkyrel[i]
                    current_vRel = atkvrel[i]
                    current_prob = atkprobs[i]

                    # 4. 构建 LeadOne 字典
                    # 注意：这里需要根据你 evaluate_sequence 实际返回的 atkyrel 含义做微调
                    # 假设 atk_drels 已经是 dRel (即 x - 1.52)
                    lead_dict = {
                        "dRel": float(current_dRel),
                        "yRel": float(-current_yRel),  # 注意：如果 atkyrel 已经是负的，这里去掉负号
                        "vRel": float(current_vRel),
                        "vLead": float(v_ego + current_vRel),
                        "vLeadK": float(v_ego + current_vRel),
                        "aLeadK": 0.0,
                        "aLeadTau": 0.3,
                        "fcw": False,
                        "modelProb": float(current_prob),
                        "status": True,
                        "radar": False,
                        "radarTrackId": -1,
                    }

                    # 5. 构建 CSV 行 (参考你的格式)
                    if lead_dict['status']:
                        row = {
                            'frame_idx': frame_idx,
                            'image_path': image_path,
                            'dRel': float(lead_dict['dRel']),
                            'yRel': float(lead_dict['yRel']),
                            'vRel': float(lead_dict['vRel']),
                            'vLead': float(lead_dict['vLead']),
                            'vLeadK': float(lead_dict['vLeadK']),
                            'aLeadK': float(lead_dict['aLeadK']),
                            'aLeadTau': float(lead_dict['aLeadTau']),
                            'status': bool(lead_dict['status']),
                            'fcw': bool(lead_dict['fcw']),
                            'modelProb': float(lead_dict['modelProb']),
                            'radar': bool(lead_dict['radar']),
                            'radarTrackId': int(lead_dict['radarTrackId']),
                        }
                    else:
                        row = {
                            'frame_idx': frame_idx,
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
                        }

                    writer.writerow(row)

            print(f"   ✅ LeadOne 已保存至: {lead_csv_path}")

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