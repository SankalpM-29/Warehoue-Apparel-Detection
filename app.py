from charset_normalizer import detect
import numpy as np
import gradio as gr
import torch
import torch.nn as nn
import cv2
from numpy import random
from com_ineuron_apparel.com_ineuron_utils.utils import decodeImage
from com_ineuron_apparel.predictor_yolo_detector.detector_test import Detector
from PIL import Image

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        #modelPath = 'research/ssd_mobilenet_v1_coco_2017_11_17'
        self.objectDetection = Detector(self.filename)




clApp = ClientApp()

def predict_image(input_img):

    img = Image.fromarray(input_img)
    img.save("./com_ineuron_apparel/predictor_yolo_detector/inference/images/"+ clApp.filename)
    resultant_img = clApp.objectDetection.detect_action()
                                                            
        
    return resultant_img

demo = gr.Blocks()

with demo:
    gr.Markdown(
    """
    <h1 align = "center"> Warehouse Apparel Detection </h1>
    """)
    
    detect = gr.Interface(predict_image, 'image', 'image')

demo.launch()