from utils.utils import BaseEngine
import numpy as np
import cv2
import time
import os
import argparse
import itertools

class Predictor(BaseEngine):
    def __init__(self, engine_path):
        super(Predictor, self).__init__(engine_path)
        self.n_classes = 1000  # your model classes


    
def trt_infer(engine):
#创建一个名为pred的预测器对象，用于对输入数据进行推理
    pred = Predictor(engine_path=engine)
    # pred.get_fps()
    x = np.ones((1, 3, 224, 224), dtype=np.float32)
    data = pred.inference(x)
    return data
