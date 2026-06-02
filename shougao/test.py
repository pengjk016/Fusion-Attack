# # rgb与yuv格式转化
import cv2
import numpy as np
#
# img = cv2.imread('./500.png')
# print(img.shape)  # (1208, 1928, 3)
# img_resized = cv2.resize(img, (512, 256))
# print(img_resized.shape)  # (256, 512, 3)
#
# # BGR → YUV420p (I420)，低版本OpenCV返回2D数组(384, 512)
# i420 = cv2.cvtColor(img_resized, cv2.COLOR_BGR2YUV_I420)
# print(i420.shape)  # (384, 512)
# print(type(i420))  # <class 'numpy.ndarray'>
#
# # 适配2D I420的解析逻辑（核心修正！）
# H, W = 256, 512
# # 2D I420布局：前H行=Y通道，后H//2行=U+V平面（U占前W//2列，V占后W//2列）
# Y = i420[:H, :]  # 前256行 → (256, 512)
# uv_plane = i420[H:, :]  # 后128行 → (128, 512)
# U = uv_plane[:, :W//2]  # 前256列 → (128, 256)
# V = uv_plane[:, W//2:]  # 后256列 → (128, 256)
#
# # 后续拆分Y为子块、构造模型输入的逻辑不变（完全兼容）
# ch0 = Y[0::2, 0::2]   # (128,256)
# ch1 = Y[0::2, 1::2]
# ch2 = Y[1::2, 0::2]
# ch3 = Y[1::2, 1::2]
# ch4 = U               # (128,256)
# ch5 = V               # (128,256)
#
# frame = np.stack([ch0, ch1, ch2, ch3, ch4, ch5], axis=0).astype(np.float32)  # (6,128,256)
#
# # 恢复BGR（适配2D I420）
# bgr_from_yuv = cv2.cvtColor(i420, cv2.COLOR_YUV2BGR_I420)
# cv2.imwrite('reconstructed.png', bgr_from_yuv)



#测试rgb，bgr通道改变的影响
img = cv2.imread('510.png')        # BGR
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite('510_rgb.png', img_rgb)
