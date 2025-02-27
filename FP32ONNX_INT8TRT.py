import tensorrt as trt
import os
import torch
import numpy as np

# 1️⃣ 创建 TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING) #创建一个 TensorRT Logger，用于记录警告级别（WARNING）以上的消息，以便调试 TensorRT 相关问题。

# 2️⃣ 定义 INT8 量化校准器
class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_file):       #Int8Calibrator 继承 TensorRT 的 IInt8EntropyCalibrator2，用于 提供校准数据，让 TensorRT 进行 INT8 量化。
        super(Int8Calibrator, self).__init__()
        self.calibration_data = torch.load(calibration_file)        #torch.load(calibration_file) 加载校准数据。
        self.current_index = 0
        self.cache_file = "calibration.cache"     #cache_file 用于缓存量化参数，避免重复计算，提高效率。

    def get_batch_size(self):
        return 2  # 修改为你的 batch size ,表示每次传递 一个样本 进行校准。

    def get_batch(self, names):
        if self.current_index >= len(self.calibration_data):
            return None     ## 数据用完后，返回 None，结束校准
        data = self.calibration_data[self.current_index].cpu().numpy().astype(np.float32)  # 转换为 NumPy 数组
        print(f"DEBUG: Batch {self.current_index} - Type: {type(data)}, Shape: {data.shape}, Dtype: {data.dtype}")
        self.current_index += 1
        return [data]  ## 返回数据

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()     #读取缓存
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)          #写入缓存

# 3️⃣ 读取 FP32 ONNX 模型
onnx_model_path = r"/home/jason/RIFE_ONNX_TRT_RKNN/IFNet_fp32_fixed.onnx"
with open(onnx_model_path, "rb") as f:
    onnx_model = f.read()

# 4️⃣ 初始化 TensorRT Builder 和网络
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

# 解析 ONNX 模型
if not parser.parse(onnx_model):
    print("❌ 解析 ONNX 失败！")
    exit()

# 5️⃣ 设置 TensorRT 构建配置
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.INT8)  # 开启 INT8 量化
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)  # 设定 8GB 内存

# 6️⃣ 绑定校准数据并执行 INT8 量化
config.int8_calibrator = Int8Calibrator("/home/jason/RIFE_ONNX_TRT_RKNN/ECCV2022-RIFE/calibration_data.pt")

# 7️⃣ 生成 INT8 TensorRT 引擎
serialized_engine = builder.build_serialized_network(network, config)

# 8️⃣ 反序列化引擎，准备推理
runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(serialized_engine)

# 9️⃣ 保存 INT8 量化后的 TensorRT 引擎
int8_engine_path = "IFNet_int8.engine"
with open(int8_engine_path, "wb") as f:
    f.write(serialized_engine)

print(f"✅ TensorRT INT8 引擎已保存至: {int8_engine_path}")
