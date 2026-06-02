# #!/usr/bin/env python3
# import time
# import numpy as np
# import cv2
# from cereal.visionipc import VisionIpcClient, VisionStreamType
#
# def save_frame(buf, fname):
#     """把 VisionBuf 转成 RGB 并保存为 JPEG"""
#     # 获取 YUV buffer
#     yuv = np.array(buf.data)  # HWC
#     height, width = buf.height, buf.width
#
#     # OpenCV expects YUV420p (NV12) as (H*1.5, W), flatten it
#     if yuv.size != int(height * width * 3):
#         # reshape to proper YUV format if needed
#         yuv = yuv.reshape((height + height//2, width))
#     rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_NV12)
#
#     cv2.imwrite(fname, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
#     print(f"Saved {fname}")
#
# def main():
#     client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, True)
#     client.connect(True)
#     time.sleep(2.0)
#
#     for i in range(5):
#         buf = client.recv()
#         if buf is None:
#             continue
#         save_frame(buf, f"/home/pjk/PycharmProjects/openpilot0.9.6/camera_picture/model_input_{i}.jpg")
#
#     print("Done.")
#
# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
import time
import numpy as np
import cv2
from cereal.visionipc import VisionIpcClient, VisionStreamType

SAVE_DIR = "/home/pjk/PycharmProjects/openpilot0.9.6/CAP/data/imgs/test_optim_patch26314"
START_IDX = 465   # 从465开始命名
INTERVAL = 0.2    # 每0.2秒保存一次
NUM_FRAMES = 100    # 要保存多少帧，可自行修改

def save_frame(buf, fname):
    """把 VisionBuf 转成 RGB 并保存为 JPEG"""
    yuv = np.array(buf.data)
    height, width = buf.height, buf.width

    if yuv.size != int(height * width * 3):
        yuv = yuv.reshape((height + height // 2, width))

    rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_NV12)
    cv2.imwrite(fname, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"Saved {fname}")

def main():
    client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, True)
    client.connect(True)
    time.sleep(2.0)

    idx = START_IDX
    for i in range(NUM_FRAMES):
        buf = client.recv()
        if buf is None:
            continue

        fname = f"{SAVE_DIR}/{idx}.png"
        save_frame(buf, fname)
        idx += 1

        time.sleep(INTERVAL)

    print("Done.")

if __name__ == "__main__":
    main()
