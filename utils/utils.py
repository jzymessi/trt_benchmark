import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2
import matplotlib.pyplot as plt

class BaseEngine(object):
    def __init__(self, engine_path):
        self.mean = None
        self.std = None
        self.n_classes = 1000

        logger = trt.Logger(trt.Logger.WARNING)
        #定义Logger 的初始最低严重性级别。默认为 WARNING。可选的严重性级别为：VERBOSE、INFO、WARNING、ERROR 和
        logger.min_severity = trt.Logger.Severity.ERROR
        #创建一个运行时
        runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger,'') # initialize TensorRT plugins
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.imgsz = engine.get_binding_shape(0)[2:]  
        # 获取engine第0个节点的shape(NCHW) [2:]指代获取 HW
        #使用TensorRT引擎对象（engine）创建一个执行上下文对象（context）其作用是为了执行深度学习模型的推理计算
        #执行上下文包含了TensorRT引擎中所有的输入和输出张量，以及中间计算结果的内存缓冲区
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        #使用CUDA Python API中的cuda.Stream()函数来创建一个CUDA流对象，并将其赋值给类的属性self.stream。
        # 流对象可以用于在GPU上异步地执行TensorRT推理计算，并在必要时与主机端的代码进行同步。
        self.stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})


    def infer(self, x):
        self.inputs[0]['host'] = np.ravel(x)
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference 执行异步推理计算
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        # fetch outputs from gpu 将 TensorRT 引擎的输出结果从设备内存（device）异步地复制到主机内存（host）
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream 等待 CUDA 流中的所有异步任务完成
        self.stream.synchronize()
        #将TensorRT 推理的输出结果从设备内存复制到主机内存，并将其存储在一个列表中
        data = [out['host'] for out in self.outputs]
        return data

    def inference(self, x):
        data = self.infer(x)
        return data
