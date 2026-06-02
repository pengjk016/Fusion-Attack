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

# ====================== 官方 Warp 矩阵代码（已完整集成）======================
from openpilot.common.transformations.orientation import rot_from_euler
from openpilot.common.transformations.camera import (
    FULL_FRAME_SIZE, get_view_frame_from_calib_frame, view_frame_from_device_frame,
    eon_fcam_intrinsics, tici_ecam_intrinsics, tici_fcam_intrinsics)

SEGNET_SIZE = (512, 384)
def get_segnet_frame_from_camera_frame(segnet_size=SEGNET_SIZE, full_frame_size=FULL_FRAME_SIZE):
    return np.array([[float(segnet_size[0]) / full_frame_size[0],  0.0],
                     [0.0,  float(segnet_size[1]) / full_frame_size[1]]])
segnet_frame_from_camera_frame = get_segnet_frame_from_camera_frame()

MEDMODEL_INPUT_SIZE = (512, 256)
MEDMODEL_YUV_SIZE = (MEDMODEL_INPUT_SIZE[0], MEDMODEL_INPUT_SIZE[1] * 3 // 2)
MEDMODEL_CY = 47.6
medmodel_fl = 910.0
medmodel_intrinsics = np.array([[medmodel_fl, 0.0, 0.5 * MEDMODEL_INPUT_SIZE[0]],
                                [0.0, medmodel_fl, MEDMODEL_CY],
                                [0.0, 0.0, 1.0]])

BIGMODEL_INPUT_SIZE = (1024, 512)
BIGMODEL_YUV_SIZE = (BIGMODEL_INPUT_SIZE[0], BIGMODEL_INPUT_SIZE[1] * 3 // 2)
bigmodel_fl = 910.0
bigmodel_intrinsics = np.array([[bigmodel_fl, 0.0, 0.5 * BIGMODEL_INPUT_SIZE[0]],
                                [0.0, bigmodel_fl, 256 + MEDMODEL_CY],
                                [0.0, 0.0, 1.0]])

SBIGMODEL_INPUT_SIZE = (512, 256)
SBIGMODEL_YUV_SIZE = (SBIGMODEL_INPUT_SIZE[0], SBIGMODEL_INPUT_SIZE[1] * 3 // 2)
sbigmodel_fl = 455.0
sbigmodel_intrinsics = np.array([[sbigmodel_fl, 0.0, 0.5 * SBIGMODEL_INPUT_SIZE[0]],
                                 [0.0, sbigmodel_fl, 0.5 * (256 + MEDMODEL_CY)],
                                 [0.0, 0.0, 1.0]])

bigmodel_frame_from_calib_frame = np.dot(bigmodel_intrinsics, get_view_frame_from_calib_frame(0, 0, 0, 0))
sbigmodel_frame_from_calib_frame = np.dot(sbigmodel_intrinsics, get_view_frame_from_calib_frame(0, 0, 0, 0))
medmodel_frame_from_calib_frame = np.dot(medmodel_intrinsics, get_view_frame_from_calib_frame(0, 0, 0, 0))
calib_from_medmodel = np.linalg.inv(medmodel_frame_from_calib_frame[:, :3])
calib_from_sbigmodel = np.linalg.inv(sbigmodel_frame_from_calib_frame[:, :3])

def get_warp_matrix(device_from_calib_euler: np.ndarray,
                    wide_camera: bool = False,
                    bigmodel_frame: bool = False,
                    tici: bool = True) -> np.ndarray:
    if tici and wide_camera:
        cam_intrinsics = tici_ecam_intrinsics
    elif tici:
        cam_intrinsics = tici_fcam_intrinsics
    else:
        cam_intrinsics = eon_fcam_intrinsics
    calib_from_model = calib_from_sbigmodel if bigmodel_frame else calib_from_medmodel
    device_from_calib = rot_from_euler(device_from_calib_euler)
    camera_from_calib = cam_intrinsics @ view_frame_from_device_frame @ device_from_calib
    warp_matrix: np.ndarray = camera_from_calib @ calib_from_model
    return warp_matrix
# =============================================================================

# ====== 固定路径（按你的项目）======
RESULTS_PATH       = 'results/img_test/image_eval2.csv'
ONNX_MODEL_PATH    = '../models/weights/supercombo.onnx'
OUTPUT_MAP_PKL     = '../models/weights/supercombo_output_map.pkl'
IMGS_DIRECTORY     = '../data/imgs/test_optim_patch26314/'

DEFAULT_DEVICE_FROM_CALIB_EULER = np.array([0.0, 0.0, 0.0], dtype=np.float32)
DEFAULT_V_EGO = 25.0   # ←←← 这里改成你实际车速（m/s），或后面从 CSV 加载

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

def read_image(path: str, device_from_calib_euler: np.ndarray = None) -> np.ndarray:
    if device_from_calib_euler is None:
        device_from_calib_euler = DEFAULT_DEVICE_FROM_CALIB_EULER
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    warp_matrix = get_warp_matrix(device_from_calib_euler, wide_camera=False, bigmodel_frame=False, tici=True)
    warped_bgr = cv2.warpPerspective(img_bgr, warp_matrix, MEDMODEL_INPUT_SIZE,
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    yuv_i420 = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2YUV_I420)
    return parse_image(yuv_i420)

def prepare_input(img1_path: str, img2_path: str, device_from_calib_euler=None, traffic='right'):
    img1 = read_image(img1_path, device_from_calib_euler)
    img2 = read_image(img2_path, device_from_calib_euler)
    input_imgs = np.r_[img1, img2][np.newaxis, ...].astype(np.float16)
    desire8 = np.zeros((1, 8), dtype=np.float32)
    desire8[0, 0] = 1.0
    traffic2 = np.array([[1, 0]], dtype=np.float32) if traffic != 'left' else np.array([[0, 1]], dtype=np.float32)
    return input_imgs, desire8, traffic2

# ====== 读取 pkl ======
def load_output_layout(pkl_path: str):
    layout = {}
    output_slices = {}
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
                hname = h.get('name')
                hlen = h.get('len') or h.get('length') or h.get('size')
                if hname == 'lead' and hlen is not None:
                    layout['lead_start'] = offset
                    layout['lead_len'] = int(hlen)
                    break
                if hlen is not None: offset += int(hlen)
        if 'rec_state_size' not in layout and isinstance(obj.get('recurrent_state'), dict):
            rs = obj['recurrent_state'].get('size') or obj['recurrent_state'].get('len')
            if rs is not None: layout['rec_state_size'] = int(rs)
        if 'output_slices' in obj:
            output_slices = obj['output_slices']
    layout.setdefault('lead_start', 5755)
    layout.setdefault('lead_len', 255)
    layout.setdefault('lead_stride', 51)
    layout.setdefault('lead_prob_base', 48)
    layout.setdefault('rec_state_size', 512)
    layout['output_slices'] = output_slices
    return layout

# ====== ORT 会话 ======
def build_session(path: str):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.enable_mem_pattern = False
    so.enable_cpu_mem_arena = True
    print(f">> build ORT session: {path}")
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])

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

def make_feed(session, img1_path, img2_path,
              features_buffer_state: np.ndarray,
              prev_desired_curv_state: np.ndarray,
              traffic='right'):
    feed = {}
    input_imgs, _, _ = prepare_input(img1_path, img2_path, DEFAULT_DEVICE_FROM_CALIB_EULER, traffic)
    for name in KNOWN_INPUTS:
        shp, _ = _decl_shape(session, name)
        if shp is None: continue
        if name == 'input_imgs':
            feed[name] = input_imgs.astype(np.float16)
        elif name == 'big_input_imgs':
            feed[name] = np.zeros(shp, dtype=np.float16)
        elif name == 'desire':
            arr = np.zeros(shp, dtype=np.float16)
            if len(shp) >= 2 and shp[-1] >= 8: arr[0, 0] = 1.0
            feed[name] = arr
        elif name == 'traffic_convention':
            arr = np.zeros(shp, dtype=np.float16)
            if shp[-1] == 2:
                arr[0, 1 if traffic == 'left' else 0] = 1.0
            feed[name] = arr
        elif name == 'lateral_control_params':
            feed[name] = np.array([[DEFAULT_V_EGO, 0.2]], dtype=np.float16)
        elif name == 'features_buffer':
            feed[name] = features_buffer_state.astype(np.float16).copy()
        elif name == 'prev_desired_curv':
            feed[name] = prev_desired_curv_state.astype(np.float16).copy()
        else:
            feed[name] = np.zeros(shp, dtype=np.float16)
    return feed

# ====== 解析输出 ======
def parse_outputs_to_drel(out_vec: np.ndarray, layout: dict):
    out_vec = out_vec.astype(np.float32)
    lead_start = layout['lead_start']
    lead_len = layout['lead_len']
    stride = layout['lead_stride']
    prob_base = layout['lead_prob_base']
    rec_size = layout['rec_state_size']
    lead = out_vec[0, lead_start:lead_start + lead_len]
    drel = []
    for t in range(6):
        x_predt = lead[4 * t::stride]
        if t < 3:
            prob = lead[prob_base + t::stride]
            drelt = float(x_predt[np.argmax(prob)])
        else:
            drelt = float(np.mean(x_predt))
        drel.append(drelt)
    rec_state = out_vec[:, -rec_size:]
    return drel, rec_state

# ====== 尝试简化 ======
def simplify_onnx(src_path: str) -> str:
    import onnxsim
    model = onnx.load(src_path)
    simp, ok = onnxsim.simplify(model, skip_shape_inference=True, skip_optimization=True, skip_fuse_bn=True)
    if not ok:
        raise RuntimeError("onnx-simplifier 校验失败")
    tmp = tempfile.NamedTemporaryFile(prefix="supercombo_simplified_", suffix=".onnx", delete=False)
    onnx.save(simp, tmp.name)
    tmp.close()
    print(f"[onnx-simplifier] 简化完成 -> {tmp.name}")
    return tmp.name

# ====== 主流程（已修复 recurrent 更新逻辑）======
def main():
    print("使用固定路径：")
    print("ONNX_MODEL_PATH :", ONNX_MODEL_PATH)
    print("IMGS_DIRECTORY  :", IMGS_DIRECTORY)
    print("RESULTS_PATH    :", RESULTS_PATH)
    print("OUTPUT_MAP_PKL  :", OUTPUT_MAP_PKL)
    print(f"默认标定: {DEFAULT_DEVICE_FROM_CALIB_EULER} | 默认 vEgo: {DEFAULT_V_EGO} m/s")

    layout = load_output_layout(OUTPUT_MAP_PKL)
    print("解析到的输出布局：", {k: v for k, v in layout.items() if k != 'output_slices'})

    try:
        sess = build_session(ONNX_MODEL_PATH)
    except Exception as e:
        print(f"[警告] 直接加载失败：{e}")
        simp_path = simplify_onnx(ONNX_MODEL_PATH)
        sess = build_session(simp_path)

    print("=== ONNX Inputs ===")
    for i, inp in enumerate(sess.get_inputs()):
        print(f"[INP{i}] {inp.name} {inp.shape} {inp.type}")
    print("=== ONNX Outputs ===")
    for i, out in enumerate(sess.get_outputs()):
        print(f"[OUT{i}] {out.name} {out.shape} {out.type}")

    # 初始化 recurrent state（直接从 session 读取真实 shape）
    features_buffer_shp, _ = _decl_shape(sess, 'features_buffer')
    prev_desired_curv_shp, _ = _decl_shape(sess, 'prev_desired_curv')
    features_buffer_state = np.zeros(features_buffer_shp, dtype=np.float32) if features_buffer_shp else np.zeros((1, 512), dtype=np.float32)
    prev_desired_curv_state = np.zeros(prev_desired_curv_shp, dtype=np.float32) if prev_desired_curv_shp else np.zeros((1, 100, 1), dtype=np.float32)
    print(f"recurrent state 初始化完成 → features_buffer: {features_buffer_state.shape} | prev_desired_curv: {prev_desired_curv_state.shape}")

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
            feed = make_feed(sess, abs_paths[i-1], abs_paths[i],
                             features_buffer_state, prev_desired_curv_state, traffic='right')

            y = sess.run([out0], feed)
            out_vec = np.array(y[0])
            raw_output = out_vec[0]

            drel, rec_state = parse_outputs_to_drel(out_vec, layout)

            # ====================== 修复后的 recurrent 更新（支持你的 3D shape）======================
            output_slices = layout['output_slices']

            # hidden_state
            if 'hidden_state' in output_slices:
                hs_slice = output_slices['hidden_state']
                hidden_state = raw_output[hs_slice] if isinstance(hs_slice, slice) else raw_output[hs_slice[0]:hs_slice[1]]
            else:
                hidden_state = rec_state[0].copy()

            # desired_curvature
            if 'desired_curvature' in output_slices:
                dc_slice = output_slices['desired_curvature']
                desired_curvature = raw_output[dc_slice] if isinstance(dc_slice, slice) else raw_output[dc_slice[0]:dc_slice[1]]
            else:
                desired_curvature = np.zeros(33, dtype=np.float32)

            # features_buffer 更新（你的 shape 是 (1,99,512)）
            if features_buffer_state.shape == (1, 99, 512):
                features_buffer_state[0, :-1, :] = features_buffer_state[0, 1:, :]
                features_buffer_state[0, -1, :] = hidden_state.astype(np.float32)
            else:
                fb = features_buffer_state[0] if features_buffer_state.ndim == 2 else features_buffer_state
                feature_len = len(hidden_state)
                fb[:-feature_len] = fb[feature_len:]
                fb[-feature_len:] = hidden_state.astype(np.float32)

            # prev_desired_curv 更新（你的 shape 是 (1,100,1)）
            if prev_desired_curv_state.shape == (1, 100, 1):
                prev_desired_curv_state[0, :-1, :] = prev_desired_curv_state[0, 1:, :]
                # 因为 buffer 的每步是标量，而 desired_curvature 可能是长度33的向量 → 取均值作为当前标量曲率
                new_curv_scalar = np.mean(desired_curvature) if len(desired_curvature) > 0 else 0.0
                prev_desired_curv_state[0, -1, 0] = new_curv_scalar
            else:
                # 兼容旧的 2D shape
                pdc = prev_desired_curv_state[0] if prev_desired_curv_state.ndim == 2 else prev_desired_curv_state
                curv_dim = 1
                pdc[:-curv_dim] = pdc[curv_dim:]
                pdc[-curv_dim:] = np.full(curv_dim, np.mean(desired_curvature))

            w.writerow([i-1, os.path.basename(abs_paths[i-1]), os.path.basename(abs_paths[i])] + drel)
            print(f"Pair {i-1:3d} | dRel0={drel[0]:6.2f}  (vEgo={DEFAULT_V_EGO:.1f} m/s, recurrent updated)")

    print(f"✅ 结果已写入: {RESULTS_PATH}")
    try:
        df = pd.read_csv(RESULTS_PATH)
        for t in [0,2,4,6,8,10]:
            print(f"dRel@{t:2d}s -> mean={df[f'dRel{t}'].mean():.4f}, std={df[f'dRel{t}'].std():.4f}")
    except Exception as e:
        print("读取结果汇总失败：", e)

if __name__ == "__main__":
    main()