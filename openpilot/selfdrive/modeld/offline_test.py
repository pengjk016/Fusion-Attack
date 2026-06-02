#思路：截图（需要拿到对应的图和前车位置速度信息）-》放patch-》放入模型-》融合(需要把radard变成离线,同时保证radar侧有信号输入）-》输出对比（列出csv文件）


# -*- coding: utf-8 -*-
"""
最快路径：换 ORT 版本后，先用 ORT 直接加载 supercombo.onnx；
若失败 -> onnx-simplifier 简化 -> 再加载并离线跑图片对，输出 dRel CSV。
"""

import os
os.environ.setdefault("ORT_DISABLE_OPTIMIZER", "1")  # 尽量避免常量折叠引发的崩溃
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import csv
import pickle
import tempfile
import numpy as np
import pandas as pd
import cv2
import onnx
import onnxruntime as ort

# ====== 固定路径（按你的项目）======
RESULTS_PATH       = 'results/img_test/image_eval_patch.csv'
ONNX_MODEL_PATH    = '../models/weights/supercombo.onnx'  # 先尝试这份
OUTPUT_MAP_PKL     = '../models/weights/supercombo_metadata.pkl'
IMGS_DIRECTORY     = '../data/imgs/golden-left-75m3/'

# ====== 前处理（YUV I420 + parse）======
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
    if isinstance(path,str):
        img = cv2.imread(path)
        img = cv2.resize(img, (512, 256))
        return img
    else:
        return path

def prepare_input(img1_path: str, img2_path: str, traffic='right'):
    img1 = read_image_bgr(img1_path)
    img2 = read_image_bgr(img2_path)
    # 改：fp16
    input_imgs = np.r_[parse_image(cv2.cvtColor(img1, cv2.COLOR_BGR2YUV_I420)),
                       parse_image(cv2.cvtColor(img2, cv2.COLOR_BGR2YUV_I420))][np.newaxis, ...].astype(np.float16)  # (1,12,128,256)

    # 这两个只是临时占位，不直接喂；真正喂参按 onnx 声明重建
    desire8 = np.zeros((1, 8), dtype=np.float16); desire8[0,0] = np.float16(1.0)
    traffic2 = np.array([[1,0]], dtype=np.float16) if traffic!='left' else np.array([[0,1]], dtype=np.float16)
    return input_imgs, desire8, traffic2

# ====== 读取 pkl：输出切片 ======
def load_output_layout(pkl_path: str):
    layout = {}
    if pkl_path and os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            try:
                obj = pickle.load(f)
            except Exception:
                obj = {}

        # 首先尝试从 output_slices 读取真实值
        if 'output_slices' in obj and 'lead' in obj['output_slices']:
            lead_slice = obj['output_slices']['lead']
            layout['lead_start'] = lead_slice.start
            layout['lead_len'] = lead_slice.stop - lead_slice.start

            # 根据C++头文件中的结构计算正确的参数
            # LEAD_MHP_N = 2, LEAD_TRAJ_LEN = 6, LEAD_PRED_DIM = 4, LEAD_MHP_SELECTION = 3
            layout['num_hypos'] = 2  # LEAD_MHP_N
            layout['num_timesteps'] = 6  # LEAD_TRAJ_LEN
            layout['coords_per_timestep'] = 4  # LEAD_PRED_DIM
            layout['num_probs'] = 3  # LEAD_MHP_SELECTION

            # 每个假设的元素数 = 均值(6×4) + 标准差(6×4) + 概率(3) = 24 + 24 + 3 = 51
            layout['hypo_size'] = (layout['num_timesteps'] * layout['coords_per_timestep'] * 2) + layout['num_probs']
            layout['lead_stride'] = layout['hypo_size']  # 每个假设的跨度
            layout['mean_size'] = layout['num_timesteps'] * layout['coords_per_timestep']  # 24
            layout['prob_base'] = layout['mean_size'] * 2  # 概率在假设中的位置 (24 + 24 = 48)

        # 如果上面没有读取到，再尝试其他方式
        for k in ['lead_start', 'lead_len', 'lead_stride', 'prob_base', 'rec_state_size']:
            if k in obj:
                layout[k] = int(obj[k])

        if ('lead_start' not in layout or 'lead_len' not in layout) and 'heads' in obj:
            offset = 0
            for h in obj['heads']:
                hname = h.get('name');
                hlen = h.get('len') or h.get('length') or h.get('size')
                if hname == 'lead' and hlen is not None:
                    layout['lead_start'] = offset;
                    layout['lead_len'] = int(hlen);
                    break
                if hlen is not None: offset += int(hlen)

        if 'rec_state_size' not in layout and isinstance(obj.get('recurrent_state'), dict):
            rs = obj['recurrent_state'].get('size') or obj['recurrent_state'].get('len')
            if rs is not None: layout['rec_state_size'] = int(rs)

    # 基于C++头文件设置默认值
    layout.setdefault('lead_start', 5755)
    layout.setdefault('lead_len', 102)  # 2个假设 × 51元素 = 102
    layout.setdefault('num_hypos', 2)  # LEAD_MHP_N
    layout.setdefault('num_timesteps', 6)  # LEAD_TRAJ_LEN
    layout.setdefault('coords_per_timestep', 4)  # LEAD_PRED_DIM
    layout.setdefault('num_probs', 3)  # LEAD_MHP_SELECTION
    layout.setdefault('hypo_size', 51)  # 每个假设51个元素
    layout.setdefault('lead_stride', 51)  # 每个假设的跨度
    layout.setdefault('mean_size', 24)  # 均值部分大小 (6×4)
    layout.setdefault('prob_base', 48)  # 概率在假设中的位置
    layout.setdefault('rec_state_size', 512)  # TEMPORAL_SIZE

    return layout

# ====== ORT 会话 ======
def build_session(path: str):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.enable_mem_pattern = False
    so.enable_cpu_mem_arena = True
    print(f">> build ORT session: {path}")
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])

# ====== 仅喂“车端需要”的已知输入 ======
KNOWN_INPUTS = [
    'input_imgs','big_input_imgs','desire','traffic_convention',
    'lateral_control_params','prev_desired_curv','features_buffer',
    'nav_features','nav_instructions'
]#这些名字没问题

def _decl_shape(session, name):
    for inp in session.get_inputs():
        if inp.name == name:
            # print('有对应')
            shp = [(1 if (s in (None,'None') or s == -1 or isinstance(s,str)) else int(s)) for s in inp.shape]
            return shp, inp.type
    print('无对应')
    return None, None

def make_feed(session, img1_path, img2_path, traffic='right'):
    feed = {}
    input_imgs, _, _ = prepare_input(img1_path, img2_path, traffic)

    for name in KNOWN_INPUTS:
        shp, _ = _decl_shape(session, name)
        if shp is None:
            continue

        if name == 'input_imgs':
            if list(input_imgs.shape) != shp:
                raise ValueError(f"input_imgs 形状不匹配，期望 {shp}，实际 {input_imgs.shape}")
            feed[name] = input_imgs.astype(np.float16)

        elif name == 'big_input_imgs':
            # 给零即可（也可复用同处理的图）
            feed[name] = np.zeros(shp, dtype=np.float16)

        elif name == 'desire':
            # 你的模型声明是 [1, 100, 8]，当前时刻用最后一帧 one-hot 的第0类
            arr = np.zeros(shp, dtype=np.float16)
            if len(shp) == 3 and shp[-1] >= 8:
                arr[0, -1, 0] = np.float16(1.0)
            feed[name] = arr

        elif name == 'traffic_convention':
            arr = np.zeros(shp, dtype=np.float16)
            if shp[-1] == 2:
                if traffic.lower() == 'left':
                    arr[0, 1] = np.float16(1.0)
                else:
                    arr[0, 0] = np.float16(1.0)
            feed[name] = arr

        elif name == 'lateral_control_params':
            # [1,2] -> [vEgo, steer_delay] 离线评测置零
            feed[name] = np.zeros(shp, dtype=np.float16)

        elif name in ['prev_desired_curv', 'features_buffer', 'nav_features', 'nav_instructions']:
            feed[name] = np.zeros(shp, dtype=np.float16)

    return feed

def parse_outputs_to_drel(out_vec: np.ndarray, layout: dict):
    # 模型是 fp16，先转 fp32 再算
    out_vec = out_vec.astype(np.float32)

    lead = out_vec[0, 5755:5755+72]
    drel_np = lead[:25:4].astype(float)  # 元素从np.float32→Python float
    drel = drel_np.tolist()

    rec_state = out_vec[:, -512:]
    return drel, rec_state
#以上用于未加patch和yolo

def build_rgb_patch(thres, patch_dim, patch_start):
    patch_height, patch_width = patch_dim
    patch_y, patch_x = patch_start # top-left is (0, 0) horizontal is x, vertical is y
    rgb_patch = thres * np.random.rand(patch_height, patch_width, 3).astype('float32')
    h_bounds = (patch_y, patch_y + patch_height)
    w_bounds = (patch_x, patch_x + patch_width)
    return rgb_patch, h_bounds, w_bounds

# === 插入此段到你的脚本（与现有函数并列） ===
import math, time

def apply_rgb_patch_to_bgr(bgr_img: np.ndarray, rgb_patch: np.ndarray, h_bounds, w_bounds):
    """
    bgr_img: the full BGR image (H=256, W=512, C=3)
    rgb_patch: patch array shape (ph, pw, 3) float32
    h_bounds, w_bounds: tuple bounds in image coordinates (start,end)
    Returns a new bgr_img copy with patch added (clipped to valid range 0-255)
    """
    out = bgr_img.astype(np.float32).copy()
    h0, h1 = h_bounds
    w0, w1 = w_bounds
    ph, pw, _ = rgb_patch.shape
    if (h1 - h0) != ph or (w1 - w0) != pw:
        raise ValueError("patch bounds do not match patch shape")
    out[h0:h1, w0:w1] = np.clip(out[h0:h1, w0:w1] + rgb_patch, 0.0, 255.0)
    return out.astype(np.uint8)

def make_feed_from_bgr(session, img1_bgr, img2_bgr, traffic='right'):
    """
    Build feed dict for ONNX session from BGR images.
    Converts to YUV and parses internally.
    """
    img1_yuv = parse_image(cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2YUV_I420))
    img2_yuv = parse_image(cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2YUV_I420))
    return make_feed_from_yuv(session, img1_yuv, img2_yuv, traffic)

def make_feed_from_yuv(session, img1_yuv, img2_yuv, traffic='right'):
    """
    Build feed dict for ONNX session from already-parsed yuv images (6, H/2, W/2).
    This mirrors make_feed but uses arrays instead of paths.
    """
    feed = {}
    # input_imgs shape (1,12,128,256)
    input_imgs = np.r_[img1_yuv, img2_yuv][np.newaxis, ...].astype(np.float16)
    # query session inputs shapes to fill zeros appropriately
    for inp in session.get_inputs():
        name = inp.name
        shp = [(1 if (s in (None,'None') or s == -1 or isinstance(s,str)) else int(s)) for s in inp.shape]
        if name == 'input_imgs':
            if list(input_imgs.shape) != shp:
                raise ValueError(f"input_imgs shape mismatch: expected {shp}, got {list(input_imgs.shape)}")
            feed[name] = input_imgs
        elif name == 'big_input_imgs':
            feed[name] = np.zeros(shp, dtype=np.float16)
        elif name == 'desire':
            arr = np.zeros(shp, dtype=np.float16)
            # set last frame one-hot class 0
            if len(shp) == 3 and shp[-1] >= 8:
                arr[0, -1, 0] = np.float16(1.0)
            feed[name] = arr
        elif name == 'traffic_convention':
            arr = np.zeros(shp, dtype=np.float16)
            if shp[-1] == 2:
                arr[0, 0] = np.float16(1.0) if traffic.lower() != 'left' else np.float16(0.0)
                if traffic.lower() == 'left':
                    arr[0,1] = np.float16(1.0)
            feed[name] = arr
        elif name == 'lateral_control_params':
            feed[name] = np.zeros(shp, dtype=np.float16)
        elif name in ['prev_desired_curv', 'features_buffer', 'nav_features', 'nav_instructions']:
            feed[name] = np.zeros(shp, dtype=np.float16)
        else:
            # any unexpected inputs: fill zeros
            feed[name] = np.zeros(shp, dtype=np.float16)
    return feed

def extract_drel0_from_outvec(out_vec: np.ndarray, layout: dict):
    """
    wrapper around parse_outputs_to_drel: returns first time dRel0 scalar float
    """
    drel, _ = parse_outputs_to_drel(out_vec, layout)
    # print(f'drel[0]:{drel[0]}')
    return float(drel[0])

def optimize_patch_finite_diff(sess, img1_bgr, img2_bgr, h_bounds, w_bounds, initial_patch,
                               layout, out0_name, steps=10, eps=1.0, lr=0.5, maximize=True,
                               grad_samples=None, verbose=True):
    """
    Finite-difference optimization of a RGB patch on BGR image.
    - sess: onnxruntime InferenceSession
    - img1_bgr, img2_bgr: BGR images (H=256, W=512, C=3)
    - h_bounds, w_bounds: patch region in image coords
    - initial_patch: numpy float32 shape (ph, pw, 3)
    - layout: layout dict for parse_outputs_to_drel
    - out0_name: sess.get_outputs()[0].name
    - steps: number of FD update iterations
    - eps: FD perturbation (same units as pixel values), typical 0.5~2.0
    - lr: learning rate for gradient step
    - maximize: if True, gradient ascent (increase dRel0), else descent (decrease)
    - grad_samples: None => full gradient over all patch elements; otherwise number of random indices to sample per step
    Returns: optimized_patch (np.array), history list of loss per step
    """
    patch = initial_patch.astype(np.float32).copy()
    ph, pw, pc = patch.shape
    n = ph * pw * pc
    indices = np.arange(n)
    history = []

    # Precompute base feed for efficiency: we reconstruct per perturbed patch
    for it in range(steps):
        t0 = time.time()
        # choose indices to estimate gradient on this iteration
        if (grad_samples is None) or (grad_samples >= n):
            chosen = indices
        else:
            chosen = np.random.choice(indices, size=min(grad_samples, n), replace=False)

        grad = np.zeros_like(patch, dtype=np.float32).reshape(-1)

        # compute baseline losses maybe optional, but FD uses two evals per param
        for idx in chosen:
            # central difference
            p_plus = patch.reshape(-1).copy()

            p_minus = patch.reshape(-1).copy()
            p_plus[idx] = p_plus[idx] + eps
            p_minus[idx] = p_minus[idx] - eps

            p_plus2 = p_plus.reshape(ph, pw, pc)
            p_minus2 = p_minus.reshape(ph, pw, pc)

            # build two patched images
            img2_p = apply_rgb_patch_to_bgr(img2_bgr, p_plus2, h_bounds, w_bounds)
            img2_m = apply_rgb_patch_to_bgr(img2_bgr, p_minus2, h_bounds, w_bounds)

            feed_p = make_feed_from_bgr(sess, img1_bgr, img2_p)
            feed_m = make_feed_from_bgr(sess, img1_bgr, img2_m)

            out_p = sess.run([out0_name], feed_p)[0]
            out_m = sess.run([out0_name], feed_m)[0]

            loss_p = extract_drel0_from_outvec(np.array(out_p), layout)
            loss_m = extract_drel0_from_outvec(np.array(out_m), layout)

            grad_val = (loss_p - loss_m) / (2.0 * eps)
            grad[idx] = grad_val

        grad2 = grad.reshape(ph, pw, pc)

        if maximize:
            patch = patch + lr * grad2
        else:
            patch = patch - lr * grad2

        # clip patch to allowed range - you used thres earlier (e.g. 1 or some value)
        # here clamp to reasonable pixel range delta; if your patches represented signed offsets [-thres,+thres]
        # adjust clamp accordingly. We'll assume patch should stay in [-thres, thres] if provided as initial.
        minv = initial_patch.min()
        maxv = initial_patch.max()
        patch = np.clip(patch, minv, maxv)

        # Evaluate current loss with new patch
        img2_cur = apply_rgb_patch_to_bgr(img2_bgr, patch, h_bounds, w_bounds)
        feed_cur = make_feed_from_bgr(sess, img1_bgr, img2_cur)
        out_cur = sess.run([out0_name], feed_cur)[0]
        loss_cur = extract_drel0_from_outvec(np.array(out_cur), layout)
        history.append(loss_cur)

        if verbose:
            dt = time.time() - t0
            print(f"[FD it {it+1}/{steps}] optimized dRel0:={loss_cur:.4f} (lr={lr}, eps={eps}, samples={len(chosen)}) took {dt:.2f}s")

    return patch, history

# === end of added helper functions ===

def main():
    from ultralytics import YOLO
    import torch
    import matplotlib.pyplot as plt
    YOLO_WEIGHTS_PATH = "../models/weights/yolov8n.pt"
    yolo_model = YOLO(YOLO_WEIGHTS_PATH)
    print("使用固定路径：")
    print("ONNX_MODEL_PATH  :", ONNX_MODEL_PATH)
    print("IMGS_DIRECTORY   :", IMGS_DIRECTORY)
    print("RESULTS_PATH     :", RESULTS_PATH)
    print("OUTPUT_MAP_PKL   :", OUTPUT_MAP_PKL)

    # ===== 1. 解析输出布局 & 创建推理 Session =====
    layout = load_output_layout(OUTPUT_MAP_PKL)
    print("解析到的输出布局：", layout)
    sess = build_session(ONNX_MODEL_PATH)

    print("=== ONNX Inputs ===")
    for i, inp in enumerate(sess.get_inputs()):
        print(f"[INP{i}] {inp.name} {inp.shape} {inp.type}")
    print("=== ONNX Outputs ===")
    for i, out in enumerate(sess.get_outputs()):
        print(f"[OUT{i}] {out.name} {out.shape} {out.type}")

    # ===== 2. 图片路径处理 =====
    paths = [p for p in os.listdir(IMGS_DIRECTORY) if p.lower().endswith('.png')]
    if len(paths) < 2:
        raise ValueError("目录中少于 2 张 PNG，无法成对评测。")
    paths.sort()
    abs_paths = [os.path.join(IMGS_DIRECTORY, p) for p in paths]

    # ===== 3. 写CSV =====
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "PairIndex", "Image1", "Image2",
            "patch_dRel0", "patch_dRel2", "patch_dRel4", "patch_dRel6", "patch_dRel8", "patch_dRel10"
        ])

        out0 = sess.get_outputs()[0].name

        features_buffer_state = None
        prev_desired_curv_state = None
#yolo+patch
        for i in range(1, 6):
            feed = make_feed(sess, abs_paths[i - 1], abs_paths[i], traffic='right')
            if features_buffer_state is not None:
                feed['features_buffer'] = features_buffer_state
            if prev_desired_curv_state is not None:
                feed['prev_desired_curv'] = prev_desired_curv_state

            # plt.figure(1)

            img1_path = abs_paths[i-1]
            img2_path = abs_paths[i]
            img2_bgr = read_image_bgr(img2_path)
            results = yolo_model(img2_bgr, verbose=False)
            boxes = results[0].boxes

            max_size = 0
            for j in range(len(boxes)):
                box_ = boxes[j].xyxy
                size = (box_[0, 3] - box_[0, 1]) * (box_[0, 2] - box_[0, 0])
                if size > max_size:
                    box = box_
                    max_size = size

            # 如果没检测到目标，就攻击中心1像素
            if len(boxes) == 0:
                box = torch.tensor([[437, 582, 438, 583]])
                print('box太小，跳过优化')
                continue

            if isinstance(box, torch.Tensor):
                box = box.int().cpu().numpy()
            else:
                box = box.astype('int32')

            patch_start = (box[0, 1], box[0, 0])
            patch_dim = (box[0, 3] - box[0, 1], box[0, 2] - box[0, 0])#(y,x)



            #显示patch是否正确,前面还有一个plt.figure(1)
            # img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            # plt.imshow(img2_rgb)
            # x1 = box[0, 0]
            # y1 = box[0, 1]
            # x2 = box[0, 2]
            # y2 = box[0, 3]
            # plt.plot([x1, x1], [y1, y2])
            # plt.plot([x2, x2], [y1, y2])
            # plt.plot([x1, x2], [y1, y1])
            # plt.plot([x1, x2], [y2, y2])
            # plt.show()
            # time.sleep(3)



            thres = 100
            patch, h_bounds, w_bounds = build_rgb_patch(thres, patch_dim, patch_start)#randompatch
            #此处的patch已经对应256*512
            # ===== 5. Patch 优化攻击 =====
            img1_bgr = read_image_bgr(img1_path)
            img2_bgr = read_image_bgr(img2_path)


            initial_patch = patch.astype(np.float32)
            print(f'initial_patch{initial_patch}')
            opt_patch, history = optimize_patch_finite_diff(
                sess=sess,
                img1_bgr=img1_bgr,
                img2_bgr=img2_bgr,
                h_bounds=h_bounds,
                w_bounds=w_bounds,
                initial_patch=initial_patch,
                layout=layout,
                out0_name=out0,
                steps=3,
                eps=10,
                lr=5000,
                maximize=False,
                grad_samples=200,
                verbose=True
            )

            # opt_patch = 0*initial_patch

            print(f'opt_patch:{opt_patch}')
            img2_opt_bgr = apply_rgb_patch_to_bgr(img2_bgr, opt_patch, h_bounds, w_bounds)

            feed_opt = make_feed_from_bgr(sess, img1_bgr, img2_opt_bgr)  # 使用BGR

            out_opt = sess.run([out0], feed_opt)[0]
            drel_opt = parse_outputs_to_drel(np.array(out_opt), layout)[0]
            print("optimized dRel0:", drel_opt[0])

            y = sess.run([out0], feed_opt)
            out_vec = np.array(y[0])  # (1, N)
            drel, rec_state = parse_outputs_to_drel(out_vec, layout)
            w.writerow([i - 1, os.path.basename(abs_paths[i - 1]), os.path.basename(abs_paths[i])] + drel)

            features_len = 512
            features_buffer_state = np.roll(feed['features_buffer'], -features_len, axis=1)
            features_buffer_state[0, -features_len:] = out_vec[0, -512:]

            prev_desired_curv_state = np.roll(feed['prev_desired_curv'], -1, axis=1)
            desired_curv_slice = slice(5990, 5992)
            prev_desired_curv_state[0, -1] = out_vec[0, desired_curv_slice][0]

        print(f"✅ 结果已写入: {RESULTS_PATH}")
        try:
            df = pd.read_csv(RESULTS_PATH)
            for t in [0, 2, 4, 6, 8, 10]:
                print(f"dRel@{t}s -> mean={df[f'dRel{t}'].mean():.4f}, std={df[f'dRel{t}'].std():.4f}")
        except Exception as e:
            print("读取结果汇总失败：", e)



    # # 画框看看
    # x1, y1, x2, y2 = map(int, box.flatten().tolist()[:4])
    # cv2.rectangle(img2, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
    # cv2.putText(img2, "Target", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # cv2.imshow("Image with Bounding Box", img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
