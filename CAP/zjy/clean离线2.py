# -*- coding: utf-8 -*-
"""
提取纯净基准预测 (Clean Baseline Extraction - 官方硬编码矩阵 + 动态车速版)
结合了 DRP-attack 的多假设解析逻辑。
脱离Openpilot官方C++库依赖，直接硬编码官方底层 20.04 环境算出的绝对透视矩阵常量，保证 100% 计算一致性。
新增：动态读取 leadOnegauss.csv 中的真实 vEgo，保证光流演算与真车完全对齐。
"""

import os

# 保持稳健配置
os.environ.setdefault("ORT_DISABLE_OPTIMIZER", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import csv
import numpy as np
import cv2
import time
import pickle
import pandas as pd
import onnxruntime as ort

# ====== 用户配置区 ======
IMGS_DIRECTORY = "/home/pjk/PycharmProjects/openpilot0.9.6/CAP/data/imgs/test_optim_patch26314"
ONNX_MODEL_PATH = '../models/weights/supercombo.onnx'

OUTPUT_DIR = 'result/clean离线'
dataset_name = IMGS_DIRECTORY.strip('/').replace('/', '_')
CSV_SAVE_PATH = os.path.join(OUTPUT_DIR, f'{dataset_name}_clean_python_exact.csv')

# 同目录下的 CSV 文件路径
INFO_CSV_PATH = '/CAP/offline_fusion_results/leadOnegauss.csv'

# Openpilot 相机距车头的物理偏移 (米)
RADAR_TO_CAMERA = 1.52

# =========================================================================
# 【官方绝对矩阵硬编码】：直接使用官方底层 C++ 库计算出的结果。
# 彻底对齐主相机 (medmodel) 与 广角相机 (sbigmodel) 的光学投影。
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
# =========================================================================

# 加载官方 metadata
METADATA_PATH = os.path.join(os.path.dirname(ONNX_MODEL_PATH), 'supercombo_metadata.pkl')
try:
  with open(METADATA_PATH, 'rb') as f:
    model_metadata = pickle.load(f)
  OUTPUT_SLICES = model_metadata['output_slices']
except Exception as e:
  OUTPUT_SLICES = None


def parse_image(frame):
  """【完全照搬】DRP-attack 的 parse_image 逻辑"""
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


def process_image(img_path, inv_matrix):
  """读取图像并进行严格的光学透视裁剪"""
  img_bgr = cv2.imread(img_path)
  if img_bgr is None:
    return None

  # 严格裁剪：完美还原焦距，修正距离翻倍问题
  img_warped = cv2.warpPerspective(img_bgr, inv_matrix, (512, 256), flags=cv2.INTER_LINEAR)
  img_yuv = cv2.cvtColor(img_warped, cv2.COLOR_BGR2YUV_I420)
  parsed = parse_image(img_yuv)

  return parsed


def prepare_inputs(img1_path, img2_path):
  """为主相机和广角相机准备 12 通道视觉数据"""
  parsed1_main = process_image(img1_path, INV_M_MAIN)
  parsed2_main = process_image(img2_path, INV_M_MAIN)
  input_imgs = np.r_[parsed1_main, parsed2_main][np.newaxis, ...].astype(np.float16)

  parsed1_extra = process_image(img1_path, INV_M_EXTRA)
  parsed2_extra = process_image(img2_path, INV_M_EXTRA)
  big_input_imgs = np.r_[parsed1_extra, parsed2_extra][np.newaxis, ...].astype(np.float16)

  return input_imgs, big_input_imgs


def make_feed_096(img1_path, img2_path, rnn_state_99, sess, current_v_ego):
  input_imgs, big_input_imgs = prepare_inputs(img1_path, img2_path)

  feed = {}
  for inp in sess.get_inputs():
    name = inp.name
    # 自动读取 ONNX 节点需要的形状，将 None 或 -1 替换为 1
    shp = [(1 if (s in (None, 'None', -1) or isinstance(s, str)) else int(s)) for s in inp.shape]

    if name == 'input_imgs':
      feed[name] = input_imgs

    elif name == 'big_input_imgs':
      feed[name] = big_input_imgs

    elif name == 'desire':
      feed[name] = np.zeros(shp, dtype=np.float16)
      if len(shp) == 3:
        feed[name][0, -1, 0] = 1.0  # [1, 100, 8] 格式
      else:
        feed[name][0, 0] = 1.0  # [1, 8] 格式

    elif name == 'traffic_convention':
      feed[name] = np.zeros(shp, dtype=np.float16)
      if len(shp) == 2:
        feed[name][0, 0] = 1.0

    elif name == 'lateral_control_params':
      tmp = np.zeros(shp, dtype=np.float32)
      tmp[0, 0] = current_v_ego  # 传入每帧真实的动态车速
      tmp[0, 1] = 0.2
      feed[name] = tmp.astype(np.float16)

    elif name == 'features_buffer':
      feed[name] = rnn_state_99['features_buffer'].astype(np.float16)

    elif name == 'prev_desired_curv':
      feed[name] = rnn_state_99['prev_desired_curv'].astype(np.float16)

    else:
      feed[name] = np.zeros(shp, dtype=np.float16)

  return feed


def extract_onnx_096(out_vec):
  """【完全照搬】DRP-attack 的多假设概率提取逻辑"""
  if OUTPUT_SLICES is not None and 'lead' in OUTPUT_SLICES:
    lead = out_vec[0, OUTPUT_SLICES['lead']]
  else:
    lead = out_vec[0, 5755:6010]

  x_predt0 = lead[0::51]
  prob0 = lead[48::51]

  # 找概率最高的分支
  current_most_likely_hypo = np.argmax(prob0)
  drelt = x_predt0[current_most_likely_hypo]

  # [修复] 减去相机到雷达(车头)的物理偏移量，与 dump 出来的 raw_vision_dRel 坐标系完美对齐
  drelt -= RADAR_TO_CAMERA

  # sigmoid 方便查看真实置信度
  prob_val = 1 / (1 + np.exp(-prob0[current_most_likely_hypo]))

  return float(drelt), float(prob_val)


def main():
  os.makedirs(OUTPUT_DIR, exist_ok=True)

  # ── 加载真实的 vEgo 字典 ──
  v_ego_map = {}
  if os.path.exists(INFO_CSV_PATH):
    try:
      df = pd.read_csv(INFO_CSV_PATH)
      for _, row in df.iterrows():
        img_name = os.path.basename(row['image_path'])
        v_ego_map[img_name] = float(row['vEgo'])
      print(f"📊 成功读取动态车速表，共加载 {len(v_ego_map)} 帧速度记录！")
    except Exception as e:
      print(f"⚠️ 读取 CSV 失败，将默认使用 8.33 m/s。错误: {e}")
  else:
    print(f"⚠️ 找不到 CSV 文件 {INFO_CSV_PATH}，将默认使用 8.33 m/s。")

  print("🚀 初始化 OpenPilot 0.9.6 模型...")
  so = ort.SessionOptions()
  so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
  sess = ort.InferenceSession(ONNX_MODEL_PATH, sess_options=so,
                              providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

  out0 = sess.get_outputs()[0].name

  image_paths = sorted(
    [os.path.join(IMGS_DIRECTORY, p) for p in os.listdir(IMGS_DIRECTORY) if p.lower().endswith('.png')],
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
  )
  print(f"📸 发现数据集: {len(image_paths)} 张图片")

  f_csv = open(CSV_SAVE_PATH, 'w', newline='')
  writer = csv.writer(f_csv)
  writer.writerow(["Frame_Index", "Image1", "Image2", "Clean_dRel_m", "Lead_Prob_Sigmoid", "vEgo_m_s"])

  # 初始全零缓冲 (内部维持 float32，防状态卡死)
  rnn_state_99 = {
    'features_buffer': np.zeros((1, 99, 512), dtype=np.float32),
    'prev_desired_curv': np.zeros((1, 100, 1), dtype=np.float32)
  }

  t_start = time.time()
  valid_frames = 0

  for i in range(1, len(image_paths)):
    img1_path = image_paths[i - 1]
    img2_path = image_paths[i]

    img1_name = os.path.basename(img1_path)
    img2_name = os.path.basename(img2_path)

    # 获取本帧的真实速度，如果没有记录则默认 30km/h (8.33 m/s)
    current_v_ego = v_ego_map.get(img2_name, 8.33)

    feed = make_feed_096(img1_path, img2_path, rnn_state_99, sess, current_v_ego)
    output = sess.run([out0], feed)[0]

    # 强制回转为 float32 保证特征记忆不断层
    out_vec = np.array(output, dtype=np.float32)

    current_drel, current_prob = extract_onnx_096(out_vec)

    # ---------------- RNN 状态滚动 ----------------
    if OUTPUT_SLICES is not None:
      hidden_state = out_vec[0, OUTPUT_SLICES['hidden_state']]
      desired_curv = out_vec[0, OUTPUT_SLICES['desired_curvature']]
    else:
      hidden_state = out_vec[0, -512:]
      desired_curv = out_vec[0, 5990:5991]

    new_fb = np.roll(rnn_state_99['features_buffer'], -1, axis=1)
    new_fb[0, -1, :] = hidden_state
    rnn_state_99['features_buffer'] = new_fb

    new_pdc = np.roll(rnn_state_99['prev_desired_curv'], -1, axis=1)
    new_pdc[0, -1, 0] = desired_curv[0] if desired_curv.size > 0 else 0.0
    rnn_state_99['prev_desired_curv'] = new_pdc
    # ----------------------------------------------

    writer.writerow([i, img1_name, img2_name, f"{current_drel:.4f}", f"{current_prob:.4f}", f"{current_v_ego:.2f}"])

    valid_frames += 1

    if valid_frames % 20 == 0 or i == len(image_paths) - 1:
      print(
        f"  [{i}/{len(image_paths) - 1}] {img2_name} | dRel: {current_drel:6.4f}m | Prob: {current_prob:6.4f} | vEgo: {current_v_ego:.2f}m/s")
      f_csv.flush()

  f_csv.close()
  print(f"✅ 提取完成！总计耗时: {time.time() - t_start:.2f}s")


if __name__ == "__main__":
  main()
