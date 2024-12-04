import onnx
model = onnx.load("/home/jason/RIFE_ONNX_TRT_RKNN/ECCV2022-RIFE/train_log/IFNet_simplified.onnx")
print("输出信息:")
for output in model.graph.output:
    print(output.name)
