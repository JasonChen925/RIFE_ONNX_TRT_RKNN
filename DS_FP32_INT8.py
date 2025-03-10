import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import os
import numpy as np

import os
import cv2
import torch
import numpy as np
from glob import glob

# 设置数据集路径（多个子文件夹）
dataset_root = r"/home/jason/RIFE_ONNX_TRT_RKNN/vimeo_triplet/sequences/00001"
import os
import cv2
import torch
import numpy as np
from glob import glob
import torch.nn.functional as F

# 设置数据集路径
dataset_root = r"/home/jason/RIFE_ONNX_TRT_RKNN/vimeo_triplet/sequences"
calibration_data = []

# 获取所有子文件夹（每个子文件夹内有 im1.png, im2.png, im3.png）
subdirs = sorted(glob(os.path.join(dataset_root, "*", "*")))[:64]  # 取前 64 组校准数据

device = torch.device("cuda")  # 使用 GPU

for subdir in subdirs:
    img1_path = os.path.join(subdir, "im1.png")  # 第一帧 I0
    img2_path = os.path.join(subdir, "im3.png")  # 第三帧 I1（不使用 im2.png）

    # 读取图像
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        print(f"❌ 跳过无效文件: {img1_path} 或 {img2_path}")
        continue  # 遇到空文件，跳过

    # BGR 转 RGB
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # 调整大小为 256x256（可修改）
    img1 = cv2.resize(img1, (256, 256), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, (256, 256), interpolation=cv2.INTER_AREA)

    # 转换为 float32 并归一化到 [0,1]
    img1 = torch.tensor(img1, dtype=torch.float32) / 255.0
    img2 = torch.tensor(img2, dtype=torch.float32) / 255.0

    # HWC 转 CHW
    img1 = img1.permute(2, 0, 1)  # (H, W, C) → (C, H, W)
    img2 = img2.permute(2, 0, 1)  # (H, W, C) → (C, H, W)

    # **插帧模型的预处理**
    def pad_image(img, padding=(0, 0, 0, 0)):
        return F.pad(img, padding)

    img1 = img1.unsqueeze(0).to(device)  # 增加 batch 维度 (1, 3, H, W)
    img2 = img2.unsqueeze(0).to(device)

    # 计算 Padding，确保输入尺寸符合要求
    h, w = img1.shape[2], img1.shape[3]
    tmp = max(32, int(32 / 1.0))  # 计算最小对齐单位
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)  # 计算 padding (左, 右, 上, 下)

    img1 = pad_image(img1, padding)
    img2 = pad_image(img2, padding)

    # **拼接 (6, H, W) 格式**
    img_pair = torch.cat((img1, img2), dim=1)  # 拼接成 6 通道 (1, 6, H, W)

    # **添加到校准数据**
    calibration_data.append(img_pair.cpu())  # 转回 CPU 方便存储

# **转换为 PyTorch 张量**
calibration_data = torch.cat(calibration_data, dim=0)  # 合并为 (N, 6, H, W)

# **保存校准数据**
torch.save(calibration_data, r"/home/jason/RIFE_ONNX_TRT_RKNN/ECCV2022-RIFE/calibration_data.pt")
print(f"✅ 校准数据已保存至 calibration_data.pt，形状: {calibration_data.shape}")


class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_file, batch_size=1):
        super().__init__()
        self.calibration_data = torch.load(calibration_file).numpy()  # 加载校准数据
        self.batch_size = batch_size
        self.current_index = 0
        self.device_input = cuda.mem_alloc(self.batch_size * 6 * 256 * 256 * 4)  # 申请 GPU 内存

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        """获取一个 batch 数据，并拷贝到 GPU"""
        if self.current_index >= len(self.calibration_data):
            return None  # 数据用完，返回 None
        batch = self.calibration_data[self.current_index : self.current_index + self.batch_size]  # 获取 batch
        self.current_index += self.batch_size
        batch = np.ascontiguousarray(batch)  # 确保数据在内存中是连续的
        cuda.memcpy_htod(self.device_input, batch)  # 拷贝到 GPU
        return [int(self.device_input)]  # 返回 GPU 指针

    def read_calibration_cache(self):
        """读取 TensorRT 生成的校准缓存"""
        if os.path.exists("calibration.cache"):
            with open("calibration.cache", "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        """存储 TensorRT 校准缓存"""
        with open("calibration.cache", "wb") as f:
            f.write(cache)


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, engine_file_path, mode="INT8", calibration_data=r"/home/jason/RIFE_ONNX_TRT_RKNN/ECCV2022-RIFE/calibration_data.pt"):
    builder = trt.Builder(TRT_LOGGER)#构建引擎
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))    #builder.create_network创建计算图，EXPLICIT——BATCH开启Batch模式
    config = builder.create_builder_config() #builder.create_builder_config创建配置，用于设置FP16和INT8量化等优化策略
    parser = trt.OnnxParser(network, TRT_LOGGER)#trt.OnnxParser解析ONNX，将onnx计算图转换为tensorrt格式
    # 读取 ONNX 模型
    assert os.path.exists(onnx_file_path), f"ONNX file {onnx_file_path} not found!"
    with open(onnx_file_path, "rb") as model:## 打开ONNX文件并解析为TensorRT计算图
        if not parser.parse(model.read()):
            print("Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    print(f"Building TensorRT engine from {onnx_file_path}, this may take a while...")
    # 设置 TensorRT 的工作空间内存
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB 工作空间
    # 配置 INT8 量化
    if mode == "INT8":
        config.set_flag(trt.BuilderFlag.INT8)       ##启用INT8
        assert calibration_data is not None, "Calibration data is required for INT8 mode!"
        calibrator = Calibrator(calibration_data)
        config.int8_calibrator = calibrator
        print("Using INT8 mode for optimization.")
    # 配置 FP16 计算
    elif mode == "FP16":    ##不需要校准数据
        config.set_flag(trt.BuilderFlag.FP16)
        print("Using FP16 mode for optimization.")
    # 生成 TensorRT engine
    serialized_engine = builder.build_serialized_network(network, config)#构建TensorRT Engine
    if serialized_engine is None:
        print("Failed to create the engine.")
        return None
    # 保存 TensorRT engine 到文件
    with open(engine_file_path, "wb") as f:     #序列化Engine，转换为二进制文件
        f.write(serialized_engine)
    print(f"TensorRT engine saved to {engine_file_path}")
    return serialized_engine

# 示例使用
onnx_path = r"/home/jason/RIFE_ONNX_TRT_RKNN/ECCV2022-RIFE/train_log/IFNet_fp32.onnx"
engine_path = r"/home/jason/RIFE_ONNX_TRT_RKNN/ECCV2022-RIFE/train_log/model_int8.trt"
build_engine(onnx_path, engine_path, mode="INT8", calibration_data=r"/home/jason/RIFE_ONNX_TRT_RKNN/ECCV2022-RIFE/calibration_data.pt")