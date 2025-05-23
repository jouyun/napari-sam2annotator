
import skimage as ski
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import glob

def delete_tmp_files():
    tmp_dir = 'tmp/'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    files = glob.glob('tmp/*')
    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print(f"Error: {f} : {e.strerror}")

# Convert single channel 8 bit jpeg into 3 channel 8 bit jpeg
def convert_to_rgb(cimg):
    cimg = (cimg - cimg.min()) / (cimg.max() - cimg.min()) * 512
    for i, img in enumerate(cimg):
        img = Image.fromarray(img)
        img = img.convert("RGB")
        img.save(f'tmp/{i:04d}.jpg', quality=100)

def cleanup_mask(label_img, label, start_frame, cutoff=0.8):
    mask = (label_img == label).astype(int)
    bad_down = False
    bad_up = False
    for idx in range(start_frame-1, 0, -1):
        old_mask = mask[idx+1]
        new_mask = mask[idx]

        if np.sum(new_mask * old_mask) / np.sum(new_mask) < cutoff:
            label_img[idx] = label_img[idx] * 0
            bad_down = True
        if bad_down:
            label_img[idx] = label_img[idx] * 0
    
    for idx in range(start_frame+1, label_img.shape[0]):
        old_mask = mask[idx-1]
        new_mask = mask[idx]

        if np.sum(new_mask * old_mask) / np.sum(new_mask) < cutoff:
            label_img[idx] = label_img[idx] * 0
            bad_up = True
        if bad_up:
            label_img[idx] = label_img[idx] * 0

    return label_img

def keep_largest(current_mask):
    labels = ski.measure.label(current_mask)
    props = ski.measure.regionprops(labels)
    largest_area = 0
    largest_label = 0
    for prop in props:
        if prop.area > largest_area:
            largest_area = prop.area
            largest_label = prop.label
    rtn = (labels == largest_label).astype(int)
    return rtn

def is_legit_shape(shape):
    area = np.sqrt(np.sum((shape.max(axis=0) - shape.min(axis=0))**2))
    if area > 10:
        return True
    return False
