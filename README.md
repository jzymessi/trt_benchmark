
# trt_Benchmark

实验主要是比较torch2onnx,torch2trt,trt fp32/fp16的结果偏差情况,使用的模型为resnet50的预训练模型。

## 环境搭建

### Pull Image

该image中包含了所需要的torch、tensorrt、cuda等环境
```
docker pull nvcr.io/nvidia/pytorch:23.08-py3
```
### torch2trt安装

在docker中安装torch2trt,参考torch2trt的安装步骤：https://github.com/NVIDIA-AI-IOT/torch2trt#setup

## 操作步骤

### torch转onnx

```
python torch2onnx.py
```

### onnx转trt

```
python onnx2trt.py -o resnet50.onnx -e resnet50_fp32.trt -p fp32
python onnx2trt.py -o resnet50.onnx -e resnet50_fp16.trt -p fp16
```
### benchmark

```
python benchmark.py
```

## 结果
偏差结果采用均方差（MSE）

<table border="1">
    <tr>
        <th>模型转换</th>
        <th>MSE</th>
    </tr>
    <tr>
        <th>torch2trt</th>
        <th>4.3353e-06</th>
    </tr>
    <tr>
        <th>torch2onnx</th>
        <th>1.6981e-06</th>
    </tr>
    <tr>
        <th>trt fp32</th>
        <th>4.1387e-06</th>
    </tr>
    <tr>
        <th>trt fp16</th>
        <th>3.8177e-05</th>
    </tr>
</table>

