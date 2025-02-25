# 代码实现了一个TensorRT模型推理包装类TRTWrapper，主要用于加载TensorRT生成的engine引擎，并提供Pytorch兼容的前向推理接口。
from typing import Union, Optional, Sequence, Dict, Any
import torch
import tensorrt as trt


class TRTWrapper(torch.nn.Module):
    # def __init__(self, engine: Union[str, trt.ICudaEngine],
    #              output_names: Optional[Sequence[str]] = None) -> None:
    def __init__(self, engine,output_names=None):
        super().__init__()
        #加载引擎
        self.engine = engine

        if isinstance(self.engine, str):
            with trt.Logger() as logger, trt.Runtime(logger) as runtime:  #创建TensorRT的Logger和Runtime  Logger是日志记录器，用于报告错误和日志，Runtime负责反序列化字节流并创建引擎
                with open(self.engine, mode='rb') as f:     #打开引擎文件并读取其字节内容，为了反序列化作准备
                    engine_bytes = f.read()

                self.engine = runtime.deserialize_cuda_engine(engine_bytes)  #使用 deserialize_cuda_engine 方法将字节流（engine_bytes）转换为 TensorRT 的 ICudaEngine 对象。
                #ICudaEngine对象是TensorRT的核心数据结构，包含了优化后的推理模型，ICudaEngine用于定义输入/输出绑定和形状，以及在GPU上执行推理


        self.context = self.engine.create_execution_context()   #创建TensorRT的执行上下文ExecutionContext,用于管理推理时的输入输出绑定和执行

        names = [_ for _ in self.engine]  #从引擎中遍历所有绑定名称
        input_names = list(filter(self.engine.binding_is_input, names))#过滤出引擎中所有输入绑定的名称
        self._input_names = input_names
        self._output_names = output_names

        if self._output_names is None:  #如果未提供Output_names，从绑定名称中推导出输出名称
            output_names = list(set(names) - set(input_names))
            self._output_names = output_names

##############################################################################2024/12/10
    def forward(self, inputs):  #前向推理
        assert self._input_names is not None
        assert self._output_names is not None
        # 验证输入
        bindings = [None] * (len(self._input_names) + len(self._output_names)) #为所有绑定（输入和输出）创建一个空列表
        profile_id = 0          #选择TensorRT的优化配置
        for input_name, input_tensor in inputs.items():
            # check if input shape is valid
            profile = self.engine.get_profile_shape(profile_id, input_name)#通过引擎ICudaEngine获取指定profile_id和input_name下输入张量的形状范围
            #返回值是一个三元组：profile[0]最小形状，profile[1]优化形状，profile[2]最大形状

            assert input_tensor.dim() == len(                       #验证输入张量的维度是否与引擎中定义的维度相同。
                profile[0]), 'Input dim is different from engine profile.'

            for s_min, s_input, s_max in zip(profile[0], input_tensor.shape,  #遍历张量的每一维，验证其形状是否在Min Shape和Max Shape
                                             profile[2]):
                assert s_min <= s_input <= s_max, \
                    'Input shape should be between ' \
                    + f'{profile[0]} and {profile[2]}' \
                    + f' but get {tuple(input_tensor.shape)}.'
            # s_min：当前维度的最小值（来自profile[0]）。
            # s_input：当前维度输入张量的实际大小。
            # s_max：当前维度的最大值（来profile[2]）。

            idx = self.engine.get_binding_index(input_name)  #通过绑定名称获取输入张量在 TensorRT 引擎中的绑定索引。后续需要使用该索引将数据绑定到引擎的执行上下文。
            # All input tensors must be gpu variables

            assert 'cuda' in input_tensor.device.type #确保输入张量在GPU
            input_tensor = input_tensor.contiguous()    #确保张量在内存中是连续存储的

            #将张量的数据类型从 torch.long（PyTorch 的 64 位整数）转换为 torch.int（32 位整数）。原因：TensorRT 不支持 64 位整数，需要兼容的数据类型。
            if input_tensor.dtype == torch.long:
                input_tensor = input_tensor.int()
            #设置绑定形状，为指定绑定索引idx设置输入张量的形状，原因：在使用动态形状的引擎时，每次推理都需要显式设置输入张量的形状。
            self.context.set_binding_shape(idx, tuple(input_tensor.shape))

            bindings[idx] = input_tensor.contiguous().data_ptr()#取输入张量在 GPU 上的内存地址。将该地址存储到 bindings 列表中，供 TensorRT 执行上下文使用。data_ptr()：返回 PyTorch 张量在 GPU 上的数据指针。

            # create output tensors
        outputs = {}
        for output_name in self._output_names:
            idx = self.engine.get_binding_index(output_name)  #获取绑定索引idx
            dtype = torch.float32
            shape = tuple(self.context.get_binding_shape(idx))  #从执行上下文中获取输出形状

            device = torch.device('cuda')#GPU上分配张量用于存储输出
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[output_name] = output
            bindings[idx] = output.data_ptr() #记录张量的内存地址到bindings

        self.context.execute_async_v2(bindings,             #调用exxcute_async_v2在当前CUDA流上执行推理
                                      torch.cuda.current_stream().cuda_stream)
        return outputs          #返回包含所有输出张量的字典

#
# model = TRTWrapper('/home/jason/RIFE_ONNX_TRT_RKNN/ECCV2022-RIFE/model.engine', None)
# output = model(dict(imgs=torch.randn(1, 6, 1440, 2560).cuda()))
# print(output)