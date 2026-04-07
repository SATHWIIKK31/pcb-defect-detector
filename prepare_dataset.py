import os
import shutil
import random
import cv2
import yaml
import numpy as np
from pathlib import Path

# Paths
base_dir = r"c:\TBP1"
scratch_dir = os.path.join(base_dir, "scratch")
solder_dir = os.path.join(base_dir, "solder")
out_dir = os.path.join(base_dir, "unified_dataset")

# Classes mappings
class_map = {"scratch": 0, "solder": 1}
split_ratio = 0.8  # 80% train, 20% val

def setup_dirs(out_dir):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    for split in ['train', 'val']:
        os.makedirs(os.path.join(out_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'labels', split), exist_ok=True)

def apply_preprocessing(img):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to enhance thin scatches without exaggerating traces.
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge channels and convert back
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def read_yolo_labels(lbl_path):
    labels = []
    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    labels.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    return labels

def smart_crop_pcb(img, labels):
    """
    Morphologically finds the PCB in a 'captured' image and crops the background.
    Most importantly, dynamically shifts all YOLO bounding box coordinates to match the new image dimensions 
    without destroying original ground truth labeling.
    """
    orig_h, orig_w = img.shape[:2]
    
    # 1. Morphological isolation of PCB
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 30, 100)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    crop_x1, crop_y1, crop_x2, crop_y2 = 0, 0, orig_w, orig_h
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        cx, cy, cw, ch = cv2.boundingRect(c)
        if (cw * ch) / (orig_w * orig_h) > 0.05: # if the contour area is at least 5%
            crop_x1, crop_y1 = cx, cy
            crop_x2, crop_y2 = cx + cw, cy + ch
            
    # 2. Expansion Padding
    pad = 20
    crop_x1 = max(0, crop_x1 - pad)
    crop_y1 = max(0, crop_y1 - pad)
    crop_x2 = min(orig_w, crop_x2 + pad)
    crop_y2 = min(orig_h, crop_y2 + pad)
    
    # 3. Fallback: Force boundary to include ALL existing YOLO defect boxes + padding!
    # (Because if the edge detection failed somehow, we DO NOT want to crop out defects)
    for lbl in labels:
        _, n_cx, n_cy, n_w, n_h = lbl
        abs_cx = n_cx * orig_w
        abs_cy = n_cy * orig_h
        abs_w = n_w * orig_w
        abs_h = n_h * orig_h
        
        box_x1 = abs_cx - (abs_w / 2.0)
        box_y1 = abs_cy - (abs_h / 2.0)
        box_x2 = abs_cx + (abs_w / 2.0)
        box_y2 = abs_cy + (abs_h / 2.0)
        
        crop_x1 = min(crop_x1, box_x1 - pad)
        crop_y1 = min(crop_y1, box_y1 - pad)
        crop_x2 = max(crop_x2, box_x2 + pad)
        crop_y2 = max(crop_y2, box_y2 + pad)
        
    crop_x1 = max(0, int(crop_x1))
    crop_y1 = max(0, int(crop_y1))
    crop_x2 = min(orig_w, int(crop_x2))
    crop_y2 = min(orig_h, int(crop_y2))
    
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return img, labels # Failsafe
        
    cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
    new_w = crop_x2 - crop_x1
    new_h = crop_y2 - crop_y1
    
    # 4. Refactor Labels to match cropped scale
    new_labels = []
    for lbl in labels:
        cid, n_cx, n_cy, n_w, n_h = lbl
        
        # Original Absolute Coordinates
        abs_cx = n_cx * orig_w
        abs_cy = n_cy * orig_h
        abs_w = n_w * orig_w
        abs_h = n_h * orig_h
        
        # New Absolute Coordinates (shifted reference point)
        new_abs_cx = abs_cx - crop_x1
        new_abs_cy = abs_cy - crop_y1
        
        # Re-Normalize based on new bounds
        new_n_cx = new_abs_cx / new_w
        new_n_cy = new_abs_cy / new_h
        new_n_w = abs_w / new_w
        new_n_h = abs_h / new_h
        
        # Failsafe clamp to 0-1
        new_n_cx = min(max(new_n_cx, 0.0), 1.0)
        new_n_cy = min(max(new_n_cy, 0.0), 1.0)
        
        new_labels.append([cid, new_n_cx, new_n_cy, new_n_w, new_n_h])
        
    return cropped_img, new_labels

def process_dataset(src_dir, new_class_id, out_dir, require_cropping=False):
    if not os.path.exists(src_dir):
        print(f"Warning: Directory {src_dir} not found.")
        return

    images_dir = os.path.join(src_dir, "images")
    labels_dir = os.path.join(src_dir, "labels")
    
    # Get all images
    images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(images)
    
    train_count = int(len(images) * split_ratio)
    
    for i, img_name in enumerate(images):
        split = 'train' if i < train_count else 'val'
        
        src_img = os.path.join(images_dir, img_name)
        base_name = os.path.splitext(img_name)[0]
        src_lbl = os.path.join(labels_dir, base_name + ".txt")
        
        dst_img = os.path.join(out_dir, 'images', split, img_name)
        dst_lbl = os.path.join(out_dir, 'labels', split, base_name + ".txt")
        
        img = cv2.imread(src_img)
        if img is None: continue
            
        labels = read_yolo_labels(src_lbl)
        
        # SMART CROP APPLICATION
        if require_cropping:
            img, labels = smart_crop_pcb(img, labels)
            
        # Preprocess and save image
        img_preprocessed = apply_preprocessing(img)
        cv2.imwrite(dst_img, img_preprocessed)
        
        with open(dst_lbl, 'w') as f:
            for lbl in labels:
                f.write(f"{new_class_id} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")

    print(f"Processed {len(images)} images from {src_dir} as class {new_class_id}")
    
def generate_yaml(out_dir):
    data = {
        'path': os.path.abspath(out_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 2,
        'names': {0: 'scratch', 1: 'solder_defect'}
    }
    with open(os.path.join(out_dir, 'data.yaml'), 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
        
if __name__ == "__main__":
    setup_dirs(out_dir)
    print("Processing Scratch dataset...")
    process_dataset(scratch_dir, class_map["scratch"], out_dir, require_cropping=False)
    print("Processing Solder dataset with Background Smart-Cropping...")
    process_dataset(solder_dir, class_map["solder"], out_dir, require_cropping=True)
    generate_yaml(out_dir)
    print("Dataset unification and dynamic cropping complete! Ready for training.")

