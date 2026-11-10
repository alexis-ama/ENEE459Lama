import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
import warnings
from scipy.ndimage import convolve, gaussian_filter

import cv2

MODEL_PATH = "version-RFB-320.onnx"

warnings.filterwarnings("ignore", category=UserWarning)

sess_options = ort.SessionOptions()
sess_options.log_severity_level = 3  

# 3. Use only available providers to prevent the provider fallback warning
desired_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
available_providers = [p for p in desired_providers if p in ort.get_available_providers()]

# face detection
def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    pass
    # code detect_faces before hard_nms

def detect_faces(image_path, output_path="output_detected.png", conf_threshold=0.7):
    # pass
    # code detect_faces before hard_nms
    # img_cv2 = cv2.imread(image_path)

    img = Image.open(image_path)

    width, height = img.size
    channels = len(img.getbands())
    mode = img.mode

    if mode != "RGB":
        raise(TypeError)

    dimensions = (320, 240)

    resized_img = img.resize(dimensions)
    converted_img = resized_img.conert("I")

    pixels = converted_img.astype(np.float32)
    normalized_img = (pixels - 127.0)/128.0
    batched_img = normalized_img.unsqueeze(0)



    






# Edge detectionjjj
def non_maximum_suppression(magnitude, angle):
    pass

def hysteresis_thresholding(img, low_thresh, high_thresh):
    pass

def canny_scipy(image_path, output_path="edges_scipy.png", low=20, high=60):
    pass

if __name__ == "__main__":
    detect_faces("face.jpg")
    canny_scipy("face.jpg")