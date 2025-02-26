import torch
import onnx

########################判断.pkl文件是否为fp32或者fp16格式
# model_dict = torch.load(r"/home/jason/RIFE_ONNX_TRT_RKNN/ECCV2022-RIFE/train_log/flownet.pkl", map_location="cpu")
# # 如果 'state_dict' 在 keys 里，则提取权重，否则直接使用 model_dict
# state_dict = model_dict["state_dict"] if "state_dict" in model_dict else model_dict
# # 遍历所有参数，检查数据类型
# for name, param in state_dict.items():
#     print(f"{name}: {param.dtype}")
#     # break  # 只打印第一个参数



##############判断.onnx文件的精度


import onnx
import numpy as np

# 加载 ONNX 模型
onnx_model = onnx.load(r"/home/jason/RIFE_ONNX_TRT_RKNN/ECCV2022-RIFE/train_log/IFNet_quantized_torch_int8.onnx")

# 遍历所有节点，检查权重数据类型
for initializer in onnx_model.graph.initializer:
    data_type = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[initializer.data_type]
    print(f"权重名称: {initializer.name}, 数据类型: {data_type}")

    # 检查数据类型是否为 INT8
    if data_type != np.int8:
        print(f"⚠️  警告: {initializer.name} 不是 INT8，而是 {data_type}")

