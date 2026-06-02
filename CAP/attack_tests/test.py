# import pickle
# METADATA_PATH   = '../models/weights/supercombo_metadata.pkl'
# with open(METADATA_PATH, 'rb') as f:
#     metadata = pickle.load(f)
# a = metadata.get('output_slices', {}).keys()
# b = metadata.get('output_shapes', {})
# print("Output slices keys:", metadata.get('output_slices', {}).keys())
# print("Output shapes:", metadata.get('output_shapes', {}))


# import pickle
#
# METADATA_PATH = '../models/weights/supercombo_metadata.pkl'
#
# with open(METADATA_PATH, 'rb') as f:
#     metadata = pickle.load(f)
#
# print("=" * 80)
# print("1. 详细 Output Slices (切片位置)")
# print("=" * 80)
#
# output_slices = metadata.get('output_slices', {})
#
# # 打印表头
# print(f"{'#':<3} | {'输出头名称':<25} | {'起始位置':<10} | {'结束位置':<10} | {'长度':<10}")
# print("-" * 80)
#
# for i, (name, slice_info) in enumerate(output_slices.items()):
#     # 解析 slice_info，它可能是 tuple (start, end) 或者 slice 对象
#     if isinstance(slice_info, tuple):
#         start, end = slice_info
#     elif isinstance(slice_info, slice):
#         start = slice_info.start
#         end = slice_info.stop
#     else:
#         start = -1
#         end = -1
#
#     length = end - start if (start >= 0 and end > 0) else -1
#
#     print(f"{i:<3} | {name:<25} | {start:<10} | {end:<10} | {length:<10}")
#
# print("\n" + "=" * 80)
# print("2. 完整 Metadata 结构 (Raw)")
# print("=" * 80)
# # 如果你想看 metadata 里所有的东西，取消下面这行的注释：
# import pprint
# pprint.pprint(metadata)


import pickle
import pprint

METADATA_PATH = '../models/weights/supercombo_metadata.pkl'

with open(METADATA_PATH, 'rb') as f:
    metadata = pickle.load(f)

print("=== output_slices（A）完整内容 ===")
pprint.pprint(dict(metadata.get('output_slices', {})))

print("\n=== output_shapes（B）完整内容 ===")
pprint.pprint(metadata.get('output_shapes', {}))

print("\n=== metadata 文件里其他顶层 key ===")
pprint.pprint(list(metadata.keys()))

# 如果你还想看某个具体 slice 的数值范围，可以继续加：
print("\n示例：hidden_state 的 slice 是：", metadata.get('output_slices', {}).get('hidden_state'))
print("示例：desired_curvature 的 slice 是：", metadata.get('output_slices', {}).get('desired_curvature'))
print("示例：lead 的 slice 是：", metadata.get('output_slices', {}).get('lead'))