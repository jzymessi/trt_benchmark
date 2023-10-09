import torchvision
import torch
from torchvision import models , transforms
from torchvision.models import resnet50
from torch2trt import torch2trt
import onnx
import onnxruntime
import numpy as np

from trt_infer import trt_infer

x = torch.ones((1, 3, 224, 224)).cuda()

resnet50 = resnet50(pretrained=True).eval()

model = resnet50.cuda()

# torch2trt model
model_trt = torch2trt(model, [x])

y = model(x)

y_trt = model_trt(x)

print("torch2trt MSE: ",((y - y_trt)**2).mean())
# torch.save(model_trt.state_dict(), 'resnet50.pth')

# 导入onnx模型
onnx_model = onnxruntime.InferenceSession('resnet50.onnx',providers=['AzureExecutionProvider', 'CPUExecutionProvider'])

input_data = torch.ones((1, 3, 224, 224)).numpy()

input_name = onnx_model.get_inputs()[0].name
output_name = onnx_model.get_outputs()[0].name
result = onnx_model.run([output_name], {input_name: input_data})

result = torch.tensor(result)

print("torch2onnx MSE: ",((y.cpu() - result)**2).mean())

# 推理trt模型
trt_output_32 = trt_infer("resnet50_fp32.trt")

trt_result_32 = torch.tensor(trt_output_32)

print("trt fp32 MSE: ",((y.cpu() - trt_result_32)**2).mean())

trt_output_16 = trt_infer("resnet50_fp16.trt")

trt_result_16 = torch.tensor(trt_output_16)

print("trt fp16 MSE: ",((y.cpu() - trt_result_16)**2).mean())
