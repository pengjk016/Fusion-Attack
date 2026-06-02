# -*- coding: utf-8 -*-
"""
提取纯净基准预测 (Clean Baseline Extraction)
功能：
1. 纯粹地运行图片序列，不加载 YOLO，不进行任何攻击。
2. 保持严格的两帧连续性和 RNN 状态回卷。
3. 将原始预测距离 (dRel) 保存到基于数据集名称动态命名的 CSV 文件中。
"""

import os

# 保持稳健配置
os.environ.setdefault("ORT_DISABLE_OPTIMIZER", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import csv
import numpy as np
import cv2
import time
import onnxruntime as ort

# ====== 用户配置区 ======
# 1. 数据集路径 (请确保和你要评估的数据集一致)
IMGS_DIRECTORY = 'data/imgs2/test1'
ONNX_MODEL_PATH = '/home/zjy/openpilot0.9.6yasuo/openpilot0.9.6/CAP/models/weights/supercombo.onnx'

# 2. 结果保存路径 (动态生成)
OUTPUT_DIR = 'results/clean'
# 将路径名转换为合法的文件名 (如 'data/imgs2/test' -> 'data_imgs2_test')
dataset_name = IMGS_DIRECTORY.strip('/').replace('/', '_')
# 最终文件名为 IMGS_DIRECTORY的内容 + clean.csv
CSV_SAVE_PATH = os.path.join(OUTPUT_DIR, f'{dataset_name}_clean.csv')


# =================================

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
    print(f">> 正在构建 ORT session: {path}")

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
        print(f">> 激活的加速提供者: {sess.get_providers()}")
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
    # 提取第0秒的距离预测
    return float(out_vec[0, 5755])


def main():
    # 1. 准备目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("🚀 初始化 OpenPilot 模型...")
    sess = build_session(ONNX_MODEL_PATH)
    out0 = sess.get_outputs()[0].name

    # 2. 读取图片序列
    image_paths = sorted(
        [os.path.join(IMGS_DIRECTORY, p) for p in os.listdir(IMGS_DIRECTORY) if p.lower().endswith('.png')])
    print(f"📸 发现数据集: {len(image_paths)} 张图片")
    if len(image_paths) < 2:
        print("❌ 错误：图片数量不足2张，无法组成时序对。")
        return

    # 3. 准备 CSV
    f_csv = open(CSV_SAVE_PATH, 'w', newline='')
    writer = csv.writer(f_csv)
    writer.writerow(["Frame_Index", "Image1", "Image2", "Clean_dRel_m"])

    # 初始化 RNN 状态
    rnn_state = {'features_buffer': None, 'prev_desired_curv': None}

    print("\n" + "=" * 50)
    print(f" 🏳️  开始提取纯净基准距离 (Clean Baseline)")
    print("=" * 50)

    t_start = time.time()
    valid_frames = 0

    # 4. 主循环
    for i in range(1, len(image_paths)):
        img1_path = image_paths[i - 1]
        img2_path = image_paths[i]

        img1 = read_image_bgr(img1_path)
        img2 = read_image_bgr(img2_path)

        # 跳过坏图
        if img1 is None or img2 is None:
            print(f"⚠️ 跳过损坏的图片: 索引 {i}")
            continue

        # 前向推理
        t_frame_start = time.time()
        feed = make_feed(sess, img1, img2, rnn_state)
        output = sess.run([out0], feed)[0]

        # 提取距离
        current_drel = extract_drel(np.array(output))

        # 更新 RNN 状态
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

        # 写入日志
        img1_name = os.path.basename(img1_path)
        img2_name = os.path.basename(img2_path)
        writer.writerow([i, img1_name, img2_name, f"{current_drel:.4f}"])

        valid_frames += 1

        # 每 20 帧打印一次进度，避免刷屏
        if valid_frames % 20 == 0 or i == len(image_paths) - 1:
            print(
                f"  [{i}/{len(image_paths) - 1}] {img2_name} | dRel: {current_drel:6.2f}m | Time/Frame: {(time.time() - t_frame_start) * 1000:.1f}ms")
            f_csv.flush()

    f_csv.close()

    print("\n" + "=" * 50)
    print(f"✅ 提取完成！")
    print(f"总计耗时: {time.time() - t_start:.2f}s")
    print(f"有效帧数: {valid_frames}")
    print(f"基准数据已保存至: {CSV_SAVE_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    main()