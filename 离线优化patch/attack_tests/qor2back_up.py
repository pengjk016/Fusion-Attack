#备份，尚未完成patch的优化和requrrent state的传递
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
OUTPUT_MAP_PKL     = '../models/weights/supercombo_metadata.pkl'
IMGS_DIRECTORY     = '../data/imgs/golden-left-75m2/'

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
    # 改：fp16
    input_imgs = np.r_[img1, img2][np.newaxis, ...].astype(np.float16)  # (1,12,128,256)

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


# ====== 主流程 ======
def main():
    print("使用固定路径：")
    print("ONNX_MODEL_PATH  :", ONNX_MODEL_PATH)
    print("IMGS_DIRECTORY   :", IMGS_DIRECTORY)
    print("RESULTS_PATH     :", RESULTS_PATH)
    print("OUTPUT_MAP_PKL   :", OUTPUT_MAP_PKL)

    layout = load_output_layout(OUTPUT_MAP_PKL)
    print("解析到的输出布局：", layout)

    sess = build_session(ONNX_MODEL_PATH)


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
        print(f'out0:{out0},sess.get_outputs:{sess.get_outputs()[0]}')
        # for i in range(1, len(abs_paths)):
        for i in range(1, 100):
            feed = make_feed(sess, abs_paths[i-1], abs_paths[i], traffic='right')
            y = sess.run([out0], feed)#y就是输出,是一个长度为1的list
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



    #以下用于patch确定以及优化

    from ultralytics import YOLO
    import torch
    YOLO_WEIGHTS_PATH = "../models/weights/yolov8n.pt"
    #
    yolo_model = YOLO(YOLO_WEIGHTS_PATH)
    img2 = cv2.imread('/home/pjk/PycharmProjects/openpilot0.9.6/camera_picture/model_input_0.jpg')
    results = yolo_model(img2, verbose=False)
    boxes = results[0].boxes

    max_size = 0
    for i in range(len(boxes)):
        box_ = boxes[i].xyxy
        size = (box_[0, 3] - box_[0, 1]) * (box_[0, 2] - box_[0, 0])
        if size > max_size:
            box = box_
            max_size = size

    # if no object found, attack 1 pixel in the center of the image
    if len(boxes) == 0:
        box = torch.tensor([[437, 582, 438, 583]])

    if isinstance(box, torch.Tensor):
        box = box.int().cpu().numpy()
    else:
        box = box.astype('int32')

    patch_start = (box[0, 1], box[0, 0])
    patch_dim = (box[0, 3] - box[0, 1], box[0, 2] - box[0, 0])
    thres = 1
    patch, h_bounds, w_bounds = build_yuv_patch(thres, patch_dim, patch_start)
    patch = torch.tensor(patch, requires_grad=True)
    print(f'patch:{patch},box:{box}')



    # 假设你已通过YOLO得到box，且img2 = cv2.imread("你的图像路径")

    # ---------------------- 关键：修复box格式，转成纯Python数值列表 ----------------------
    # 情况1：如果box是numpy数组（比如YOLO输出的box.xyxy是(1,4)或(4,)格式）
    if isinstance(box, np.ndarray):
        # 展平数组→转Python列表→取前4个元素（确保是[x1,y1,x2,y2]）
        box = box.flatten().tolist()[:4]

    # 情况2：如果box是列表，但元素是numpy数组（比如[array([x1]), array([y1]), ...]）
    elif isinstance(box, list) and any(isinstance(item, np.ndarray) for item in box):
        # 逐个将数组元素转成Python数值
        box = [float(item) for item in box[:4]]  # 先转float避免numpy类型问题

    # 情况3：如果box已是纯Python数值列表（直接用）
    else:
        box = box[:4]  # 确保只取前4个元素

    # ---------------------- 现在可以安全转整数并绘图 ----------------------
    x1, y1, x2, y2 = map(int, box)  # 这行不会再报错

    # 绘制矩形框（红色，线宽2）
    cv2.rectangle(img2, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

    # （可选）添加文字标注
    cv2.putText(img2, "Target", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 显示图像
    cv2.imshow("Image with Bounding Box", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
