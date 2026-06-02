#!/usr/bin/env python3
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image

import sys
from pathlib import Path

# 获取项目根目录（openpilot和camera_picture的同级目录）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# 现在可以正常导入 openpilot 下的模块
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.models.commonmodel_pyx import ModelFrame, CLContext
from openpilot.selfdrive.modeld.parse_model_outputs import Parser
from openpilot.selfdrive.modeld.runners import ModelRunner, Runtime


from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.models.commonmodel_pyx import ModelFrame, CLContext
from openpilot.selfdrive.modeld.parse_model_outputs import Parser
from openpilot.selfdrive.modeld.runners import ModelRunner, Runtime

# 模型路径和metadata
MODEL_PATHS = {
    ModelRunner.ONNX: Path("models/weights/supercombo.onnx")
}
METADATA_PATH = Path("models/weights/supercombo_metadata.pkl")


class ModelState:
    def __init__(self, context: CLContext):
        self.frame = ModelFrame(context)
        self.wide_frame = ModelFrame(context)
        self.prev_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)

        # 保留原始源码的非视觉输入
        self.inputs = {
            'desire': np.zeros(ModelConstants.DESIRE_LEN * (ModelConstants.HISTORY_BUFFER_LEN + 1), dtype=np.float32),
            'traffic_convention': np.zeros(ModelConstants.TRAFFIC_CONVENTION_LEN, dtype=np.float32),
            'lateral_control_params': np.zeros(ModelConstants.LATERAL_CONTROL_PARAMS_LEN, dtype=np.float32),
            'prev_desired_curv': np.zeros(ModelConstants.PREV_DESIRED_CURV_LEN * (ModelConstants.HISTORY_BUFFER_LEN + 1), dtype=np.float32),
            'nav_features': np.zeros(ModelConstants.NAV_FEATURE_LEN, dtype=np.float32),
            'nav_instructions': np.zeros(ModelConstants.NAV_INSTRUCTION_LEN, dtype=np.float32),
            'features_buffer': np.zeros(ModelConstants.HISTORY_BUFFER_LEN * ModelConstants.FEATURE_LEN, dtype=np.float32),
        }

        # 加载metadata
        with open(METADATA_PATH, 'rb') as f:
            model_metadata = pickle.load(f)
        self.output_slices = model_metadata['output_slices']
        net_output_size = model_metadata['output_shapes']['outputs'][1]
        self.output = np.zeros(net_output_size, dtype=np.float32)
        self.parser = Parser()

        # 初始化模型
        self.model = ModelRunner(MODEL_PATHS, self.output, Runtime.GPU, False, context)
        self.model.addInput("input_imgs", None)
        self.model.addInput("big_input_imgs", None)
        for k, v in self.inputs.items():
            self.model.addInput(k, v)

    def slice_outputs(self, model_outputs: np.ndarray) -> Dict[str, np.ndarray]:
        return {k: model_outputs[np.newaxis, v] for k, v in self.output_slices.items()}

    def run(self, img_path: str,
            desire: int = 0,
            is_rhd: bool = True,
            v_ego: float = 0.0,
            nav_features: Optional[np.ndarray] = None,
            nav_instructions: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        img_path: 图片路径
        desire: 当前驾驶意图（0~DESIRE_LEN-1）
        is_rhd: 是否右行
        v_ego: 当前速度
        nav_features, nav_instructions: 导航输入，可不传则为0
        """
        # 更新非视觉输入
        vec_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
        if 0 <= desire < ModelConstants.DESIRE_LEN:
            vec_desire[desire] = 1

        self.inputs['desire'][:-ModelConstants.DESIRE_LEN] = self.inputs['desire'][ModelConstants.DESIRE_LEN:]
        self.inputs['desire'][-ModelConstants.DESIRE_LEN:] = np.where(vec_desire - self.prev_desire > .99, vec_desire, 0)
        self.prev_desire[:] = vec_desire

        self.inputs['traffic_convention'][:] = [0, 0]
        self.inputs['traffic_convention'][int(is_rhd)] = 1
        self.inputs['lateral_control_params'][:] = np.array([v_ego, 0.2], dtype=np.float32)

        if nav_features is not None:
            self.inputs['nav_features'][:] = nav_features
        if nav_instructions is not None:
            self.inputs['nav_instructions'][:] = nav_instructions

        # features_buffer 和 prev_desired_curv 保持原样
        # 读取图片
        img = Image.open(img_path).convert("RGB")
        img = np.array(img, dtype=np.uint8)

        # 模型输入
        buf = self.frame.prepare_from_numpy(img)
        self.model.setInputBuffer("input_imgs", buf)
        self.model.setInputBuffer("big_input_imgs", buf)  # 没有宽角图片，也用同一张

        # 执行模型
        self.model.execute()
        outputs = self.parser.parse_outputs(self.slice_outputs(self.output))

        # 更新历史缓冲
        self.inputs['features_buffer'][:-ModelConstants.FEATURE_LEN] = self.inputs['features_buffer'][ModelConstants.FEATURE_LEN:]
        self.inputs['features_buffer'][-ModelConstants.FEATURE_LEN:] = outputs['hidden_state'][0, :]
        self.inputs['prev_desired_curv'][:-ModelConstants.PREV_DESIRED_CURV_LEN] = self.inputs['prev_desired_curv'][ModelConstants.PREV_DESIRED_CURV_LEN:]
        self.inputs['prev_desired_curv'][-ModelConstants.PREV_DESIRED_CURV_LEN:] = outputs['desired_curvature'][0, :]

        return outputs


if __name__ == "__main__":
    cl_context = CLContext()
    model_state = ModelState(cl_context)

    # 示例：图片路径和驾驶状态
    img_path = "/home/pjk/PycharmProjects/openpilot0.9.6/camera_picture/model_input_0.jpg"
    outputs = model_state.run(
        img_path,
        desire=0,      # 直行
        is_rhd=True,   # 右行
        v_ego=15.0     # 当前车速
    )

    # 输出模型结果
    for k, v in outputs.items():
        print(f"{k}: {v.shape}")
