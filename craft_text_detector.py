import torch
from torch.autograd import Variable
import cv2
import os
import sys
from collections import OrderedDict

repo_path = os.path.join(os.path.dirname(__file__), './CRAFT')
sys.path.append(repo_path)

from CRAFT.craft import CRAFT
from CRAFT import craft_utils
from CRAFT import imgproc

# Global variables to hold the models
craft_model = None
refine_model = None

def load_craft_model(trained_model_path, use_cuda=False):
    global craft_model
    if craft_model is None:
        # Initialize and load the CRAFT model only once
        craft_model = CRAFT()
        print(f'Loading weights from checkpoint ({trained_model_path})')
        if use_cuda:
            craft_model.load_state_dict(copy_state_dict(torch.load(trained_model_path)))
            craft_model = craft_model.cuda()
            craft_model = torch.nn.DataParallel(craft_model)
            torch.backends.cudnn.benchmark = False
        else:
            craft_model.load_state_dict(copy_state_dict(torch.load(trained_model_path, map_location='cpu')))
        craft_model.eval()

    return craft_model

def copy_state_dict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def detect_text_from_image(image, trained_model_path='weights/craft_mlt_25k.pth', text_threshold=0.7, link_threshold=0.4, low_text=0.4, use_cuda=False, canvas_size=1280, mag_ratio=1.5):
    net = load_craft_model(trained_model_path, use_cuda)

    # Resize the image for inference
    img_resized, target_ratio, _ = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # Preprocess the image
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    if use_cuda:
        x = x.cuda()

    # Forward pass
    with torch.no_grad():
        y, _ = net(x)

    # Make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, False)
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    return boxes, polys, score_text
