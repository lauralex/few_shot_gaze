#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Shalini De Mello.
# --------------------------------------------------------

import time
import cv2
import numpy as np
from os import path
from subprocess import call
import pickle
import sys
import torch
import os

import warnings
warnings.filterwarnings("ignore")

from monitor import monitor
from camera import cam_calibrate
from person_calibration import collect_data, fine_tune
from frame_processor import frame_processer

#################################
# Start camera
#################################

cam_idx = 0

# adjust these for your camera to get the best accuracy
#call('v4l2-ctl -d /dev/video%d -c brightness=100' % cam_idx, shell=True)
#call('v4l2-ctl -d /dev/video%d -c contrast=50' % cam_idx, shell=True)
#call('v4l2-ctl -d /dev/video%d -c sharpness=100' % cam_idx, shell=True)

#cam_cap = cv2.VideoCapture(cam_idx)
data_selector = 3
video_selector_map = {0: 'v1', 1: 'v3', 2: 'v4', 3: 'v6'}
cam_cap = cv2.VideoCapture(f'C:/Users/Authority/Desktop/encoded/{video_selector_map[data_selector]}.mp4')
cam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# calibrate camera
cam_calib = {'mtx': np.eye(3), 'dist': np.zeros((1, 5))}
if path.exists("calib_cam%d.pkl" % (cam_idx)):
    cam_calib = pickle.load(open("calib_cam%d.pkl" % (cam_idx), "rb"))
else:
    print("Calibrate camera once. Print pattern.png, paste on a clipboard, show to camera and capture non-blurry images in which points are detected well.")
    print("Press s to save frame, c to continue, q to quit")
    cam_calibrate(cam_idx, cam_cap, cam_calib)

#################################
# Load gaze network
#################################
ted_parameters_path = 'demo_weights/weights_ted.pth.tar'
maml_parameters_path = 'demo_weights/weights_maml'
k = 5

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create network
sys.path.append("../src")
from models import DTED
gaze_network = DTED(
    growth_rate=32,
    z_dim_app=64,
    z_dim_gaze=2,
    z_dim_head=16,
    decoder_input_c=32,
    normalize_3d_codes=True,
    normalize_3d_codes_axis=1,
    backprop_gaze_to_encoder=False,
).to(device)

#################################

# Load T-ED weights if available
assert os.path.isfile(ted_parameters_path)
print('> Loading: %s' % ted_parameters_path)
ted_weights = torch.load(ted_parameters_path)
if torch.cuda.device_count() == 1:
    if next(iter(ted_weights.keys())).startswith('module.'):
        ted_weights = dict([(k[7:], v) for k, v in ted_weights.items()])

#####################################

# Load MAML MLP weights if available
full_maml_parameters_path = maml_parameters_path +'/%02d.pth.tar' % k
assert os.path.isfile(full_maml_parameters_path)
print('> Loading: %s' % full_maml_parameters_path)
maml_weights = torch.load(full_maml_parameters_path)
ted_weights.update({  # rename to fit
    'gaze1.weight': maml_weights['layer01.weights'],
    'gaze1.bias':   maml_weights['layer01.bias'],
    'gaze2.weight': maml_weights['layer02.weights'],
    'gaze2.bias':   maml_weights['layer02.bias'],
})
gaze_network.load_state_dict(ted_weights)

#################################
# Personalize gaze network
#################################

# Initialize monitor and frame processor
mon = monitor()
frame_processor = frame_processer(cam_calib)

# collect person calibration data and fine-
# tune gaze network
subject = input('Enter subject name: ')
#data = collect_data(cam_cap, mon, calib_points=9, rand_points=4)
# adjust steps and lr for best results
# To debug calibration, set show=True
data = {'frames': [], 'g_t': []}

if data_selector == 0:
    data['g_t'].append((-508, -133))  # v1_4
    data['g_t'].append((0, -133))  # v1_5
    data['g_t'].append((-254, 296))  # v1_7
    data['g_t'].append((-127, 153))  # v1_8
    data['g_t'].append((254, 153))  # v1_9
    data['g_t'].append((254, 153))  # v1_13
    data['g_t'].append((256, 296))  # v1_15
    data['frames'].append([cv2.imread('C:/Users/Authority/Desktop/encoded/v1_4.png') for i in range(0, 20)])
    data['frames'].append([cv2.imread('C:/Users/Authority/Desktop/encoded/v1_5.png') for i in range(0, 20)])
    data['frames'].append([cv2.imread('C:/Users/Authority/Desktop/encoded/v1_7.png') for i in range(0, 20)])
    data['frames'].append([cv2.imread('C:/Users/Authority/Desktop/encoded/v1_8.png') for i in range(0, 20)])
    data['frames'].append([cv2.imread('C:/Users/Authority/Desktop/encoded/v1_9.png') for i in range(0, 20)])
    data['frames'].append([cv2.imread('C:/Users/Authority/Desktop/encoded/v1_13.png') for i in range(0, 20)])
    data['frames'].append([cv2.imread('C:/Users/Authority/Desktop/encoded/v1_15.png') for i in range(0, 20)])
elif data_selector == 1:
    data['g_t'].append((254, 296))  # v3_1
    data['g_t'].append((-254, 153))  # v3_2
    data['g_t'].append((-254, 10))  # v3_3
    data['g_t'].append((0, 153))  # v3_4
    data['g_t'].append((254, 10))  # v3_5
    data['g_t'].append((-254, 296))  # v3_6
    data['g_t'].append((127, 153))  # v1_15
    data['frames'].append([cv2.imread('C:/Users/Authority/Desktop/encoded/v3_1.png') for i in range(0, 20)])
    data['frames'].append([cv2.imread('C:/Users/Authority/Desktop/encoded/v3_2.png') for i in range(0, 20)])
    data['frames'].append([cv2.imread('C:/Users/Authority/Desktop/encoded/v3_3.png') for i in range(0, 20)])
    data['frames'].append([cv2.imread('C:/Users/Authority/Desktop/encoded/v3_4.png') for i in range(0, 20)])
    data['frames'].append([cv2.imread('C:/Users/Authority/Desktop/encoded/v3_5.png') for i in range(0, 20)])
    data['frames'].append([cv2.imread('C:/Users/Authority/Desktop/encoded/v3_6.png') for i in range(0, 20)])
    data['frames'].append([cv2.imread('C:/Users/Authority/Desktop/encoded/v3_7.png') for i in range(0, 20)])
elif data_selector == 2:
    data['g_t'].append((-254, 296))  # v4_1
    data['g_t'].append((0, 153))  # v4_2
    data['g_t'].append((127, 153))  # v4_3
    data['g_t'].append((127, 153))  # v4_4
    data['g_t'].append((-127, 225))  # v4_5
    data['g_t'].append((0, 225))  # v4_6
    data['g_t'].append((-127, 153))  # v4_7
    data['frames'].append([cv2.imread('C:/Users/Authority/Desktop/encoded/v4_1.png') for i in range(0, 20)])
    data['frames'].append([cv2.imread('C:/Users/Authority/Desktop/encoded/v4_2.png') for i in range(0, 20)])
    data['frames'].append([cv2.imread('C:/Users/Authority/Desktop/encoded/v4_3.png') for i in range(0, 20)])
    data['frames'].append([cv2.imread('C:/Users/Authority/Desktop/encoded/v4_4.png') for i in range(0, 20)])
    data['frames'].append([cv2.imread('C:/Users/Authority/Desktop/encoded/v4_5.png') for i in range(0, 20)])
    data['frames'].append([cv2.imread('C:/Users/Authority/Desktop/encoded/v4_6.png') for i in range(0, 20)])
    data['frames'].append([cv2.imread('C:/Users/Authority/Desktop/encoded/v4_7.png') for i in range(0, 20)])
elif data_selector == 3:
    data['g_t'].append((0, 153))  # v6_1
    data['g_t'].append((127, 225))  # v6_2
    data['g_t'].append((-127, 82))  # v6_3
    data['g_t'].append((0, 225))  # v6_4
    data['g_t'].append((-127, 153))  # v6_5
    data['g_t'].append((127, 153))  # v6_6
    data['g_t'].append((-127, 82))  # v6_7
    data['frames'].append([cv2.imread('C:/Users/Authority/Desktop/encoded/v6_1.png') for i in range(0, 20)])
    data['frames'].append([cv2.imread('C:/Users/Authority/Desktop/encoded/v6_2.png') for i in range(0, 20)])
    data['frames'].append([cv2.imread('C:/Users/Authority/Desktop/encoded/v6_3.png') for i in range(0, 20)])
    data['frames'].append([cv2.imread('C:/Users/Authority/Desktop/encoded/v6_4.png') for i in range(0, 20)])
    data['frames'].append([cv2.imread('C:/Users/Authority/Desktop/encoded/v6_5.png') for i in range(0, 20)])
    data['frames'].append([cv2.imread('C:/Users/Authority/Desktop/encoded/v6_6.png') for i in range(0, 20)])
    data['frames'].append([cv2.imread('C:/Users/Authority/Desktop/encoded/v6_7.png') for i in range(0, 20)])
gaze_network = fine_tune(subject, data, frame_processor, mon, device, gaze_network, k, steps=1000, lr=1e-5, show=False)

#################################
# Run on live webcam feed and
# show point of regard on screen
#################################
data = frame_processor.process(subject, cam_cap, mon, device, gaze_network, show=True)
