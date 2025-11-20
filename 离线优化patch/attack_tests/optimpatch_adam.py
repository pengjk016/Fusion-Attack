
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
RESULTS_PATH       = 'results/img_test/image_eval_patch6_adam.csv'
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

def read_image(path) -> np.ndarray:
    if isinstance(path,str):
        img = cv2.imread(path)
        img = cv2.resize(img, (512, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
        return parse_image(img)
    else:
        return path



def prepare_input(img1_path: str, img2_path: str, traffic='right'):
    img1 = read_image(img1_path)
    img2 = read_image(img2_path)
    # 改：fp16
    input_imgs = np.r_[img1, img2][np.newaxis, ...].astype(np.float16)  # (1,12,128,256)

    # 这两个只是临时占位，不直接喂；真正喂参按 onnx 声明重建
    desire8 = np.zeros((1, 8), dtype=np.float16); desire8[0,0] = np.float16(1.0)
    traffic2 = np.array([[1,0]], dtype=np.float16) if traffic!='left' else np.array([[0,1]], dtype=np.float16)
    return input_imgs, desire8, traffic2



# # ====== 读取 pkl：输出切片 ======
# def load_output_layout(pkl_path: str):
#     layout = {}
#     if pkl_path and os.path.exists(pkl_path):
#         with open(pkl_path, 'rb') as f:
#             try:
#                 obj = pickle.load(f)
#             except Exception:
#                 obj = {}
#
#         # 首先尝试从 output_slices 读取真实值
#         if 'output_slices' in obj and 'lead' in obj['output_slices']:
#             lead_slice = obj['output_slices']['lead']
#             layout['lead_start'] = lead_slice.start
#             layout['lead_len'] = lead_slice.stop - lead_slice.start
#
#             # 根据C++头文件中的结构计算正确的参数
#             # LEAD_MHP_N = 2, LEAD_TRAJ_LEN = 6, LEAD_PRED_DIM = 4, LEAD_MHP_SELECTION = 3
#             layout['num_hypos'] = 2  # LEAD_MHP_N
#             layout['num_timesteps'] = 6  # LEAD_TRAJ_LEN
#             layout['coords_per_timestep'] = 4  # LEAD_PRED_DIM
#             layout['num_probs'] = 3  # LEAD_MHP_SELECTION
#
#             # 每个假设的元素数 = 均值(6×4) + 标准差(6×4) + 概率(3) = 24 + 24 + 3 = 51
#             layout['hypo_size'] = (layout['num_timesteps'] * layout['coords_per_timestep'] * 2) + layout['num_probs']
#             layout['lead_stride'] = layout['hypo_size']  # 每个假设的跨度
#             layout['mean_size'] = layout['num_timesteps'] * layout['coords_per_timestep']  # 24
#             layout['prob_base'] = layout['mean_size'] * 2  # 概率在假设中的位置 (24 + 24 = 48)
#
#         # 如果上面没有读取到，再尝试其他方式
#         for k in ['lead_start', 'lead_len', 'lead_stride', 'prob_base', 'rec_state_size']:
#             if k in obj:
#                 layout[k] = int(obj[k])
#
#         if ('lead_start' not in layout or 'lead_len' not in layout) and 'heads' in obj:
#             offset = 0
#             for h in obj['heads']:
#                 hname = h.get('name');
#                 hlen = h.get('len') or h.get('length') or h.get('size')
#                 if hname == 'lead' and hlen is not None:
#                     layout['lead_start'] = offset;
#                     layout['lead_len'] = int(hlen);
#                     break
#                 if hlen is not None: offset += int(hlen)
#
#         if 'rec_state_size' not in layout and isinstance(obj.get('recurrent_state'), dict):
#             rs = obj['recurrent_state'].get('size') or obj['recurrent_state'].get('len')
#             if rs is not None: layout['rec_state_size'] = int(rs)
#
#     # 基于C++头文件设置默认值
#     layout.setdefault('lead_start', 5755)
#     layout.setdefault('lead_len', 102)  # 2个假设 × 51元素 = 102
#     layout.setdefault('num_hypos', 2)  # LEAD_MHP_N
#     layout.setdefault('num_timesteps', 6)  # LEAD_TRAJ_LEN
#     layout.setdefault('coords_per_timestep', 4)  # LEAD_PRED_DIM
#     layout.setdefault('num_probs', 3)  # LEAD_MHP_SELECTION
#     layout.setdefault('hypo_size', 51)  # 每个假设51个元素
#     layout.setdefault('lead_stride', 51)  # 每个假设的跨度
#     layout.setdefault('mean_size', 24)  # 均值部分大小 (6×4)
#     layout.setdefault('prob_base', 48)  # 概率在假设中的位置
#     layout.setdefault('rec_state_size', 512)  # TEMPORAL_SIZE
#
#     return layout




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



def parse_outputs_to_drel(out_vec: np.ndarray, layout: None):
    # 模型是 fp16，先转 fp32 再算
    out_vec = out_vec.astype(np.float32)

    lead = out_vec[0, 5755:5755+72]

    drel_np = lead[:25:4].astype(float)  # 元素从np.float32→Python float
    drel = drel_np.tolist()

    rec_state = out_vec[:, -512:]
    return drel, rec_state
#以上用于未加patch和yolo



def build_yuv_patch(thres, patch_dim, patch_start):
    patch_height, patch_width = patch_dim
    patch_y, patch_x = patch_start # top-left is (0, 0) horizontal is x, vertical is y
    # h_ratio = 256/874
    # w_ratio = 512/1164
    h_ratio = 256/1208
    w_ratio = 512/1928

    y_patch_height = int(patch_height*h_ratio)
    y_patch_width = int(patch_width*w_ratio)
    y_patch_h_start = int(patch_y*h_ratio)
    y_patch_w_start = int(patch_x*w_ratio)

    y_patch = thres * np.random.rand(y_patch_height, y_patch_width).astype('float32')
    # u_patch = thres * np.random.rand()
    h_bounds = (y_patch_h_start, y_patch_h_start+y_patch_height)
    w_bounds = (y_patch_w_start, y_patch_w_start+y_patch_width)
    return y_patch, h_bounds, w_bounds




# === 插入此段到你的脚本（与现有函数并列） ===
import math, time

def apply_y_patch_to_yuv(img2_yuv: np.ndarray, y_patch: np.ndarray, h_bounds, w_bounds):
    """
    img2_yuv: the full YUV I420 array returned by read_image() shape (6, H/2, W/2)
    y_patch: patch array shape (ph, pw) float32 (same units as your build_yuv_patch)
    h_bounds, w_bounds: tuple bounds in Y-plane coordinates (start,end)
    Returns a new img2_yuv copy with patch added (clipped to valid range 0-255)
    """
    out = img2_yuv.copy().astype(np.float32)
    h0, h1 = h_bounds
    w0, w1 = w_bounds
    ph, pw = y_patch.shape
    # Ensure sizes match
    if (h1 - h0) != ph or (w1 - w0) != pw:
        raise ValueError("patch bounds do not match patch shape")
    out[0, h0:h1, w0:w1] = np.clip(out[0, h0:h1, w0:w1] + y_patch, 0.0, 255.0)
    # print(f'修改的patch：{out[0, h0:h1, w0:w1]}')
    return out.astype(np.float32)

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
            feed[name] = input_imgs
            # feed[name] = np.zeros(shp, dtype=np.float16)
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
            # print(f'traffic_convention:{arr}')
        elif name == 'lateral_control_params':
            feed[name] = np.zeros(shp, dtype=np.float16)
        elif name in ['prev_desired_curv', 'features_buffer', 'nav_features', 'nav_instructions']:
            feed[name] = np.zeros(shp, dtype=np.float16)
        else:
            # any unexpected inputs: fill zeros
            feed[name] = np.zeros(shp, dtype=np.float16)
    return feed

def extract_drel0_from_outvec(out_vec: np.ndarray, layout: None):
    """
    wrapper around parse_outputs_to_drel: returns first time dRel0 scalar float
    """
    drel, _ = parse_outputs_to_drel(out_vec, layout)
    # print(f'drel[0]:{drel[0]}')
    return float(drel[0])

import time
import torch
import numpy as np
# 假设 apply_y_patch_to_yuv、make_feed_from_yuv、extract_drel0_from_outvec 已定义

def optimize_patch_finite_diff(sess, img1_yuv, img2_yuv, h_bounds, w_bounds, initial_patch,
                               layout, out0_name, steps=10, eps=1.0, lr=0.5, maximize=True,
                               grad_samples=None, verbose=True, adam=None):  # 新增 adam 参数（外部初始化的 Adam 优化器）
    """
    适配 tensor 类型输入的有限差分优化函数
    - initial_patch: 输入为 torch.Tensor (shape: (ph, pw), dtype=torch.float32)
    """
    # 初始化 patch（确保为 tensor 类型，支持梯度计算）
    patch = initial_patch.clone().float()  # 复制输入 tensor，避免修改原数据
    ph, pw = patch.shape
    n = ph * pw
    indices = np.arange(n)  # 保持 numpy 数组，用于随机采样索引
    history = []

    for it in range(steps):
        t0 = time.time()
        # 选择需要计算梯度的索引（转换为 tensor 用于索引 tensor）
        if (grad_samples is None) or (grad_samples >= n):
            chosen = torch.tensor(indices, dtype=torch.long)  # numpy 索引转 tensor
        else:
            chosen_np = np.random.choice(indices, size=min(grad_samples, n), replace=False)
            chosen = torch.tensor(chosen_np, dtype=torch.long)  # 采样索引转 tensor

        # 初始化梯度（tensor 类型，与 patch 形状匹配）
        grad = torch.zeros_like(patch.reshape(-1), dtype=torch.float32)

        # 有限差分计算梯度
        for idx in chosen:
            # 复制 patch 并施加扰动（使用 tensor 的 clone 方法复制）
            p_plus = patch.reshape(-1).clone()  # 展平为 1D tensor 并复制
            p_minus = patch.reshape(-1).clone()

            # 对第 idx 个元素施加 ±eps 扰动
            p_plus[idx] += eps
            p_minus[idx] -= eps

            # 重塑为 2D 并转换为 numpy（因为 ONNX 推理需要 numpy 输入）
            p_plus2 = p_plus.reshape(ph, pw).cpu().numpy()  # 转回 numpy 用于图像处理
            p_minus2 = p_minus.reshape(ph, pw).cpu().numpy()

            # 生成带扰动的图像（假设 apply_y_patch_to_yuv 处理 numpy 数组）
            img2_p = apply_y_patch_to_yuv(img2_yuv, p_plus2, h_bounds, w_bounds)
            img2_m = apply_y_patch_to_yuv(img2_yuv, p_minus2, h_bounds, w_bounds)

            # 构建 ONNX 推理的输入（feed 需为 numpy 数组）
            feed_p = make_feed_from_yuv(sess, img1_yuv, img2_p)
            feed_m = make_feed_from_yuv(sess, img1_yuv, img2_m)

            # 运行 ONNX 模型获取输出
            out_p = sess.run([out0_name], feed_p)[0]
            out_m = sess.run([out0_name], feed_m)[0]

            # 提取目标函数值（dRel0）
            loss_p = extract_drel0_from_outvec(np.array(out_p), None)
            loss_m = extract_drel0_from_outvec(np.array(out_m), None)

            # 计算梯度（中心差分公式）
            grad_val = (loss_p - loss_m) / (2.0 * eps)
            grad[idx] = grad_val  # 赋值到 tensor 梯度中

        # 重塑梯度为 patch 形状（tensor 类型）
        grad2 = grad.reshape(ph, pw)
        torch.set_printoptions(precision=8, sci_mode=False)
        print(f'grad2:\n{grad2}')
        # 更新 patch（根据 maximize 选择梯度上升/下降）



        update = adam.update(grad2)  # AdamOptTorch 接收 tensor 梯度
        print(f'update:\n{update}')
        patch = patch + update  # 累加 Adam 计算的更新量

        # 评估当前 patch 的效果（需将 tensor 转回 numpy 用于图像处理）
        img2_cur = apply_y_patch_to_yuv(img2_yuv, patch.cpu().numpy(), h_bounds, w_bounds)
        feed_cur = make_feed_from_yuv(sess, img1_yuv, img2_cur)
        out_cur = sess.run([out0_name], feed_cur)[0]
        loss_cur = extract_drel0_from_outvec(np.array(out_cur), None)
        history.append(loss_cur)

        if verbose:
            dt = time.time() - t0
            print(f"[FD it {it+1}/{steps}] optimized dRel0:={loss_cur:.4f} (lr={lr}, eps={eps}, samples={len(chosen)}) took {dt:.2f}s")

    return patch, history

# === end of added helper functions ===




import torch
class AdamOptTorch:

    def __init__(self, size, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, dtype=torch.float32):
        self.exp_avg = torch.zeros(size, dtype=dtype)
        self.exp_avg_sq = torch.zeros(size, dtype=dtype)
        self.beta1 = torch.tensor(beta1)
        self.beta2 = torch.tensor(beta2)
        self.eps = eps
        self.lr = lr
        self.step = 0

    def update(self, grad):

        self.step += 1

        bias_correction1 = 1 - self.beta1 ** self.step
        bias_correction2 = 1 - self.beta2 ** self.step

        self.exp_avg = self.beta1 * self.exp_avg + (1 - self.beta1) * grad
        self.exp_avg_sq = self.beta2 * self.exp_avg_sq + (1 - self.beta2) * (grad ** 2)

        denom = (torch.sqrt(self.exp_avg_sq) / torch.sqrt(bias_correction2)) + self.eps

        step_size = self.lr / bias_correction1

        return step_size / denom * self.exp_avg



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
    # layout = load_output_layout(OUTPUT_MAP_PKL)
    # print("解析到的输出布局：", layout)
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
        for i in range(1, 10):
            feed = make_feed(sess, abs_paths[i - 1], abs_paths[i], traffic='right')
            if features_buffer_state is not None:
                feed['features_buffer'] = features_buffer_state
            if prev_desired_curv_state is not None:
                feed['prev_desired_curv'] = prev_desired_curv_state

            # plt.figure(1)

            img1_path = abs_paths[i-1]
            img2_path = abs_paths[i]
            img2 = cv2.imread(img2_path)
            img2 = cv2.resize(img2, (512, 256))
            results = yolo_model(img2, verbose=False)
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



            thres = 1
            patch, h_bounds, w_bounds = build_yuv_patch(thres, patch_dim, patch_start)#randompatch

            # ===== 5. Patch 优化攻击 =====
            img1_yuv = read_image(img1_path)
            img2_yuv = read_image(img2_path)

            initial_patch = patch.astype(np.float32)

            print(f'initial_patch{initial_patch}')

            initial_patch = torch.tensor(initial_patch)
            adam = AdamOptTorch(initial_patch.shape, lr=1)

            opt_patch, history = optimize_patch_finite_diff(
                sess=sess,
                img1_yuv=img1_yuv,
                img2_yuv=img2_yuv,
                h_bounds=h_bounds,
                w_bounds=w_bounds,
                initial_patch=initial_patch,
                layout=None,
                out0_name=out0,
                steps=3,
                eps=1,
                lr=1,
                maximize=True,
                grad_samples=200,
                verbose=True,
                adam = adam
            )



            print(f'opt_patch:{opt_patch}')
            opt_patch = opt_patch.cpu().numpy()
            img2_opt = apply_y_patch_to_yuv(img2_yuv, opt_patch, h_bounds, w_bounds)

            feed_opt = make_feed(sess, img1_path, img2_opt)  # 使用路径

            out_opt = sess.run([out0], feed_opt)[0]
            drel_opt = parse_outputs_to_drel(np.array(out_opt), None)[0]
            print("optimized dRel0:", drel_opt[0])

            y = sess.run([out0], feed_opt)
            out_vec = np.array(y[0])  # (1, N)
            drel, rec_state = parse_outputs_to_drel(out_vec, None)
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
