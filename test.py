import onnx
import numpy as np

onnx_model = onnx.load("/home/jason/RIFE_ONNX_TRT_RKNN/ECCV2022-RIFE/train_log/IFNet_fp32.onnx")

# 遍历所有 initializer 并转换数据类型
for tensor in onnx_model.graph.initializer:
    if tensor.data_type == onnx.TensorProto.INT64:
        print(f"Converting {tensor.name} from INT64 to INT32")
        tensor.data_type = onnx.TensorProto.INT32
        tensor.int32_data[:] = np.array(tensor.int64_data, dtype=np.int32).tolist()
        tensor.int64_data[:] = []

onnx.save(onnx_model, "IFNet_fp32_fixed.onnx")
