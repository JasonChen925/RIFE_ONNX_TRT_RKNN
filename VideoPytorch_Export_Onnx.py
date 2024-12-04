import torch
from train_log.IFNet_HDv3 import *

device = torch.device("cuda")
# 初始化模型
model = IFNet().to(device)
model.eval()  # 切换到评估模式

# 设置模拟输入
dummy_input = torch.randn(1, 6, 1440,2560).to(device)  # 假设输入为 224x224 的图像
scale_list = [4, 2, 1]  # 固定的缩放列表

# 导出模型为 ONNX
torch.onnx.export(
    model,                                      # 要导出的模型
    (dummy_input, scale_list),                  # 输入元组
    "IFNet.onnx",                               # 输出的 ONNX 文件名
    opset_version=11,                           # 指定 ONNX opset 版本
    input_names=['imgs', 'scale_list'],         # 输入名称
    output_names=['flow_list', 'mask', 'merged'],  # 输出名称，与模型输出对齐
    dynamic_axes={
        'imgs': {0: 'batch_size', 2: 'height', 3: 'width'},
        'flow_list': {0: 'batch_size', 2: 'height', 3: 'width'},
        'mask': {0: 'batch_size', 2: 'height', 3: 'width'},
        'merged': {0: 'batch_size', 2: 'height', 3: 'width'}
    }  # 启用动态尺寸支持
)

print("模型已成功导出为 IFNet.onnx")
