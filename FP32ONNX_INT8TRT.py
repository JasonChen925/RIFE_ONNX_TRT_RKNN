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
calibration_data_vimeo = []

subdirs = sorted(glob(os.path.join(dataset_root, "*")))[:32]  # 取前 32 组校准数据

for subdir in subdirs:
    img1_path = os.path.join(subdir, "im1.png")  # 第一帧
    img2_path = os.path.join(subdir, "im2.png")  # 中间帧（插值目标，不用于输入）
    img3_path = os.path.join(subdir, "im3.png")  # 第三帧
    # 读取图像
    img1 = cv2.imread(img1_path)
    img3 = cv2.imread(img3_path)
    if img1 is None or img3 is None:
        print(f"❌ 跳过无效文件: {img1_path} 或 {img3_path}")
        continue  # 遇到空文件，跳过
    # BGR 转 RGB
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    # 调整大小为 256x256（可修改）
    img1 = cv2.resize(img1, (256, 256), interpolation=cv2.INTER_AREA)
    img3 = cv2.resize(img3, (256, 256), interpolation=cv2.INTER_AREA)
    # 转换为 float32 并归一化到 [0,1]
    img1 = torch.tensor(img1, dtype=torch.float32,requires_grad=False) / 255.0
    img3 = torch.tensor(img3, dtype=torch.float32,requires_grad=False) / 255.0
    # HWC 转 CHW
    img1 = img1.permute(2, 0, 1)  # (H, W, C) → (C, H, W)
    img3 = img3.permute(2, 0, 1)  # (H, W, C) → (C, H, W)
    # 拼接成 (6, 256, 256) 作为输入
    img_pair = torch.cat((img1, img3), dim=0)  # (6, 256, 256)
    calibration_data_vimeo.append(img_pair)


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self,calibration_data_vimeo,batch_size=1,cache_file = "Calibrator.cache"):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.batch_size = batch_size
        self.calibration_data = calibration_data_vimeo
        self.cache_file = cache_file
        self.index = 0
        self.device_input = cuda.mem_alloc(batch_size * 6 * 256 * 256 * 4 * 32)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        """
        获取一个 batch 数据，并拷贝到 GPU
        """
        try:
            # 取出 batch_size 个样本
            batch = self.calibration_data[self.index: self.index + self.batch_size]
            self.index += self.batch_size  # 更新索引
            if len(batch) == 0:
                return None  # 没有数据时返回 None，通知 TensorRT 结束校准
            # 将 batch 数据转换为 NumPy 并拷贝到 GPU
            batch_np = torch.stack(batch).numpy().astype(np.float32)  # 转换为 NumPy (N, C, H, W)  torch.stack沿某个维度连接多个张量的操作
            print(f"batch_np shape: {batch_np.shape}, dtype: {batch_np.dtype}, size: {batch_np.nbytes}")
            # 🚨 确保 batch_np 是 float32
            assert batch_np.dtype == np.float32, "❌ batch_np 需要是 float32!"

            batch_np = np.ascontiguousarray(batch_np, dtype=np.float32)     #确保数据是连续的
            ##确保self.device_input已经分配
            if self.device_input is None:
                print("❌ self.device_input is None, re-allocating memory!")
                self.device_input = cuda.mem_alloc(self.batch_size * 6 * 256 * 256 * 4 * 32)

            # 🚨 确保 batch_np 内存不超过分配的 GPU 内存
            assert batch_np.nbytes <= self.batch_size * 6 * 256 * 256 * 4* 32, "❌ batch_np 超出已分配的 GPU 内存!"

            cuda.memcpy_htod(self.device_input, batch_np)  # 拷贝到 GPU
            cuda.Context.synchronize()#确保数据同步
            return [int(self.device_input)]  # 返回 GPU 内存地址

        except Exception as e:
            print(f"[ERROR] get_batch 发生异常: {e}")
            return None

    def read_calibration_cache(self):
        # 如果校准表文件存在则直接从其中读取校准表
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        # 如果进行了校准，则把校准表写入文件中以便下次使用
        with open(self.cache_file, "wb") as f:
            f.write(cache)
            f.flush()


def build_engine(onnx_file_path, engine_file_path, mode="INT8", calibration_data=calibration_data_vimeo):
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
    # # 配置 FP16 计算
    # elif mode == "FP16":    ##不需要校准数据
    #     config.set_flag(trt.BuilderFlag.FP16)
    #     print("Using FP16 mode for optimization.")

    # 生成 TensorRT engine
    engine = builder.build_engine(network, config)#构建TensorRT Engine
    if engine is None:
        print("Failed to create the engine.")
        return None

    # 保存 TensorRT engine 到文件
    with open(engine_file_path, "wb") as f:     #序列化Engine，转换为二进制文件
        f.write(engine.serialize())

    print(f"TensorRT engine saved to {engine_file_path}")
    return engine

# 示例使用
onnx_path = r"/home/jason/RIFE_ONNX_TRT_RKNN/ECCV2022-RIFE/train_log/IFNet_fp32.onnx"
engine_path = r"/home/jason/RIFE_ONNX_TRT_RKNN/ECCV2022-RIFE/train_log/model_int8.trt"
build_engine(onnx_path, engine_path, mode="INT8", calibration_data=calibration_data_vimeo)