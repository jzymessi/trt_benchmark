import torch
import torchvision.models as models
import onnx
resnet50 = models.resnet50(pretrained=True)
resnet50.eval()

batch_size = 1
channel = 3
height, width = 224, 224
input_tensor = torch.randn(batch_size, channel, height, width)

torch.onnx.export(resnet50,               # PyTorch模型
                  input_tensor,           # 示例输入张量
                  "resnet50.onnx",        # 输出ONNX文件名
                  verbose=False,           # 是否显示详细信息
                  input_names=["input"],  # 输入名称，可以自定义
                  output_names=["output"] # 输出名称，可以自定义
                 )

# 检查onnx模型是否可用
try:
    # 当我们的模型不可用时，将会报出异常
    onnx.checker.check_model('resnet50.onnx')
except onnx.checker.ValidationError as e:
    print("The model is invalid: %s"%e)
else:
    # 模型可用时，将不会报出异常，并会输出“The model is valid!”
    print("The model is valid!")
