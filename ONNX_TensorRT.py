### 根据onnx文件生成tensorrt的engine文件
import torch
import onnx
import tensorrt as trt

# class NaiveModel(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.pool = torch.nn.MaxPool2d(2, 2)
#
#     def forward(self, x):
#         return self.pool(x)
#
# torch.onnx.export(NaiveModel(),
#                   torch.randn(1, 3, 224, 224),
#                   onnx_model,
#                   input_names=['input'],
#                   output_names=['output'],
#                   opset_version=11)
# onnx_model = onnx.load(onnx_model)


#创建TensorRT Builder和网络
logger = trt.Logger(trt.Logger.ERROR)  #trt.Logger:用于记录TensorRT操作日志，这里设置为仅记录错误
builder = trt.Builder(logger)   #创建TensorRT的构建器(Builder),用于构建引擎
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)   #设置网络为显式批次模式，在动态形状推理中必须得启用
network = builder.create_network(EXPLICIT_BATCH)    #创建一个TensorRT网络定义对象

parser = trt.OnnxParser(network, logger) #创建一个ONNX解析器，将ONNX模型解析为TensorRT网络

onnx_model = onnx.load("//home/jason/RIFE_ONNX_TRT_RKNN/ECCV2022-RIFE/train_log/IFNet_256x256_fp32.onnx")
device = torch.device("cuda")
if not parser.parse(onnx_model.SerializeToString()):   #解析ONNX模型，使用.SerializeToString()序列化为字节流格式
    error_msgs = ''
    for error in range(parser.num_errors):  #通过循环收集错误信息
        error_msgs += f'{parser.get_error(error)}\n'
    raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

#### 创建构建配置和优化配置
config = builder.create_builder_config()   #builder.create_builder_config() 创建引擎的构建配置。
config.max_workspace_size = 4<<30  # config.max_workspace_size 设置最大工作空间大小为 1GB (用于 TensorRT 的临时存储，如卷积优化缓存)，分配GPU内存
profile = builder.create_optimization_profile()     #builder.create_optimization_profile()：创建优化配置文件，用于动态形状推理。

# profile.set_shape('imgs', [1,6,1440,2560], [1,6,1440,2560], [1,6,1440,2560])  #设置输入的最小、优化和最大形状 2560x1440使用这行代码
profile.set_shape('imgs',[1,6,256,256],[1,6,256,256],[1,6,256,256])
config.add_optimization_profile(profile) #将优化配置文件添加到构建配置中去

### 构建TensorRT引擎
with torch.cuda.device(device):
    engine = builder.build_engine(network, config)
#使用builder.build_engine()根据网络定义和配置构建TensorRT引擎

### 保存引擎到文件
with open('train_log/model_256x256_fp32.engine', mode='wb') as f:  #生成二进制文件
    f.write(bytearray(engine.serialize()))
    print("generating file done!")
