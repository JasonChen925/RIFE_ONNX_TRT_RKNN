import torch

# 加载 pkl 文件
model_dict = torch.load(r"/home/jason/RIFE_ONNX_TRT_RKNN/ECCV2022-RIFE/train_log/flownet.pkl", map_location="cpu")

# 如果 'state_dict' 在 keys 里，则提取权重，否则直接使用 model_dict
state_dict = model_dict["state_dict"] if "state_dict" in model_dict else model_dict

# 遍历所有参数，检查数据类型
for name, param in state_dict.items():
    print(f"{name}: {param.dtype}")
    # break  # 只打印第一个参数
