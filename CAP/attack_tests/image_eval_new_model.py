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
RESULTS_PATH       = 'results/img_test/image_eval2.csv'
ONNX_MODEL_PATH    = '../models/weights/supercombo.onnx'  # 先尝试这份
OUTPUT_MAP_PKL     = '../models/weights/supercombo_output_map.pkl'
IMGS_DIRECTORY     = '../data/imgs/test_optim_patch26314/'

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

def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    img = cv2.resize(img, (512, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
    return parse_image(img)

def prepare_input(img1_path: str, img2_path: str, traffic='right'):
    img1 = read_image(img1_path)
    img2 = read_image(img2_path)
    input_imgs = np.r_[img1, img2][np.newaxis, ...].astype(np.float16)  # (1,12,128,256)
    desire8 = np.zeros((1, 8), dtype=np.float32); desire8[0,0] = 1.0
    traffic2 = np.array([[1,0]], dtype=np.float32) if traffic!='left' else np.array([[0,1]], dtype=np.float32)
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
        for k in ['lead_start','lead_len','lead_stride','lead_prob_base','rec_state_size']:
            if k in obj: layout[k] = int(obj[k])
        if ('lead_start' not in layout or 'lead_len' not in layout) and 'heads' in obj:
            offset = 0
            for h in obj['heads']:
                hname = h.get('name'); hlen = h.get('len') or h.get('length') or h.get('size')
                if hname == 'lead' and hlen is not None:
                    layout['lead_start'] = offset; layout['lead_len'] = int(hlen); break
                if hlen is not None: offset += int(hlen)
        if 'rec_state_size' not in layout and isinstance(obj.get('recurrent_state'), dict):
            rs = obj['recurrent_state'].get('size') or obj['recurrent_state'].get('len')
            if rs is not None: layout['rec_state_size'] = int(rs)
    layout.setdefault('lead_start', 5755)
    layout.setdefault('lead_len',   255)
    layout.setdefault('lead_stride', 51)
    layout.setdefault('lead_prob_base', 48)
    layout.setdefault('rec_state_size', 512)
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
]

def _decl_shape(session, name):
    for inp in session.get_inputs():
        if inp.name == name:
            shp = [(1 if (s in (None,'None') or s == -1 or isinstance(s,str)) else int(s)) for s in inp.shape]
            return shp, inp.type
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
            feed[name] = input_imgs.astype(np.float16)  # 改为 float16
        elif name == 'big_input_imgs':
            feed[name] = np.zeros(shp, dtype=np.float16)  # 改为 float16
        elif name == 'desire':
            arr = np.zeros(shp, dtype=np.float16);  # 改为 float16
            if len(shp)>=2 and shp[-1]>=8: arr[0,0]=1.0
            feed[name] = arr
        elif name == 'traffic_convention':
            arr = np.zeros(shp, dtype=np.float16)  # 改为 float16
            if shp[-1]==2:
                if traffic=='left': arr[0,1]=1.0
                else:               arr[0,0]=1.0
            feed[name] = arr
        elif name == 'lateral_control_params':
            feed[name] = np.zeros(shp, dtype=np.float16)  # 改为 float16
        else:
            feed[name] = np.zeros(shp, dtype=np.float16)  # 改为 float16
    return feed


# ====== 解析输出为 dRel ======
def parse_outputs_to_drel(out_vec: np.ndarray, layout: dict):
    out_vec = out_vec.astype(np.float32)
    lead_start   = layout['lead_start']
    lead_len     = layout['lead_len']
    stride       = layout['lead_stride']
    prob_base    = layout['lead_prob_base']
    rec_size     = layout['rec_state_size']

    lead = out_vec[0, lead_start:lead_start + lead_len]
    drel = []
    for t in range(6):
        x_predt = lead[4*t::stride]
        if t < 3:
            prob = lead[prob_base + t::stride]
            drelt = float(x_predt[np.argmax(prob)])
        else:
            drelt = float(np.mean(x_predt))
        drel.append(drelt)
    rec_state = out_vec[:, -rec_size:]
    return drel, rec_state

# ====== 尝试简化（仅在直接加载失败时调用）======
def simplify_onnx(src_path: str) -> str:
    import onnxsim
    model = onnx.load(src_path)
    simp, ok = onnxsim.simplify(
        model,
        skip_shape_inference=True,
        skip_optimization=True,
        skip_fuse_bn=True,
    )
    if not ok:
        raise RuntimeError("onnx-simplifier 校验失败")
    tmp = tempfile.NamedTemporaryFile(prefix="supercombo_simplified_", suffix=".onnx", delete=False)
    onnx.save(simp, tmp.name)
    tmp.close()
    print(f"[onnx-simplifier] 简化完成 -> {tmp.name}")
    return tmp.name

# ====== 主流程 ======
def main():
    print("使用固定路径：")
    print("ONNX_MODEL_PATH  :", ONNX_MODEL_PATH)
    print("IMGS_DIRECTORY   :", IMGS_DIRECTORY)
    print("RESULTS_PATH     :", RESULTS_PATH)
    print("OUTPUT_MAP_PKL   :", OUTPUT_MAP_PKL)

    layout = load_output_layout(OUTPUT_MAP_PKL)
    print("解析到的输出布局：", layout)

    # 1) 先试加载原始 onnx
    try:
        sess = build_session(ONNX_MODEL_PATH)
    except Exception as e:
        print(f"[警告] 直接加载失败：{e}")
        # 2) 简化后再试
        simp_path = simplify_onnx(ONNX_MODEL_PATH)
        sess = build_session(simp_path)

    # 打印 IO（一次）
    print("=== ONNX Inputs ===")
    for i, inp in enumerate(sess.get_inputs()):
        print(f"[INP{i}] {inp.name} {inp.shape} {inp.type}")
    print("=== ONNX Outputs ===")
    for i, out in enumerate(sess.get_outputs()):
        print(f"[OUT{i}] {out.name} {out.shape} {out.type}")

    # 3) 遍历图片目录、两两成对写 CSV
    paths = [p for p in os.listdir(IMGS_DIRECTORY) if p.lower().endswith('.png')]
    if len(paths) < 2:
        raise ValueError("目录中少于 2 张 PNG，无法成对评测。")
    paths.sort()
    abs_paths = [os.path.join(IMGS_DIRECTORY, p) for p in paths]

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["PairIndex","Image1","Image2","dRel0","dRel2","dRel4","dRel6","dRel8","dRel10"])
        out0 = sess.get_outputs()[0].name
        for i in range(1, len(abs_paths)):
            feed = make_feed(sess, abs_paths[i-1], abs_paths[i], traffic='right')
            y = sess.run([out0], feed)
            out_vec = np.array(y[0])  # (1, N)
            drel, _ = parse_outputs_to_drel(out_vec, layout)
            w.writerow([i-1, os.path.basename(abs_paths[i-1]), os.path.basename(abs_paths[i])] + drel)

    print(f"✅ 结果已写入: {RESULTS_PATH}")
    try:
        df = pd.read_csv(RESULTS_PATH)
        for t in [0,2,4,6,8,10]:
            print(f"dRel@{t}s -> mean={df[f'dRel{t}'].mean():.4f}, std={df[f'dRel{t}'].std():.4f}")
    except Exception as e:
        print("读取结果汇总失败：", e)

if __name__ == "__main__":
    main()
