import numpy as np
import os

# === 参数配置区 ===
NOISE_SHAPE = (90, 110, 3)  # 噪声矩阵的形状 (高, 宽, 通道)
NOISE_MEAN = 120.0            # 高斯分布的均值
NOISE_SIGMA = 100.0          # 高斯分布的标准差
SAVE_FILENAME = 'white.npy' # 保存的文件名
# ==================

def generate_and_save_noise():
    # 生成随机高斯噪声
    noise = np.random.normal(loc=NOISE_MEAN, scale=NOISE_SIGMA, size=NOISE_SHAPE)

    # 转换为 float32 格式（与你的训练代码对齐）
    noise = noise.astype(np.float32)

    # 生成一个 100x100 的全白补丁
    test_patch = np.ones((90, 110, 3), dtype=np.float32) * 255


    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    save_path = os.path.join(current_dir, SAVE_FILENAME)

    # 保存为 .npy 文件
    # np.save(save_path, noise)

    np.save(save_path, test_patch)

    print(f"✅ 成功生成形状为 {NOISE_SHAPE} 的高斯噪声。")
    print(f"✅ 方差(Sigma): {NOISE_SIGMA}, 均值(Mean): {NOISE_MEAN}")
    print(f"✅ 文件已保存至: {save_path}")


if __name__ == "__main__":
    generate_and_save_noise()