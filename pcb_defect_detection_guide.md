# PCB Defect Detection System Pipeline (YOLOv8)

Building a robust AI system for full-board PCB inspection requires strict attention to dataset balance, preprocessing (to make defects pop), and augmentation (to mimic real-world placement). Below is a complete, practical step-by-step engineering solution.

---

## 1. DATASET PREPARATION

To achieve a generalized multi-class model from two independent datasets (Scratch & Solder), unifying them into a central pipeline is strictly required.

**Strategy:**
1. **Unify the Class Structure**: Modify the labels so that any defect belonging to the "scratch" dataset defaults to `class 0`, and any from "solder dataset" defaults to `class 1`. 
2. **Merge Directories**: Shuffling all images together creates a single seamless dataset.
3. **Train / Val Split**: Apply an 80/20 train/validation split dynamically.
4. **Hard Negatives**: Obtain unblemished PCB images (containing complex geometries and fine copper traces without defects). Add these images into your `images/train` folder, and for each, add an empty text file into `labels/train`. YOLO inherently handles this as "hard negative mining" to learn what *isn't* a scratch.

*(I have provided a Python script `prepare_dataset.py` in your workspace (`c:\TBP1`) which handles 1, 2, and 3 automatically, while applying CLAHE preprocessing.)*

---

## 2. PREPROCESSING PIPELINE (VERY IMPORTANT)

For thin, low-contrast defects, **CLAHE (Contrast Limited Adaptive Histogram Equalization)** is an industry standard. General histogram equalization can blow out the bright copper traces and destroy fine details. CLAHE works on localized tiles, ensuring scratches become darker and more visible without overexposing traces.

**The Pipeline:**
- **Step 1:** Convert RGB image to LAB color space.
- **Step 2:** Apply CLAHE tightly onto the 'L' (Luminance) channel. 
- **Step 3:** Convert back to BGR.
- **Step 4:** (Optional) Apply a very mild 3x3 Gaussian Blur to cancel out high-frequency sensor noise. 
- **When to apply:** Apply CLAHE offline to all images before generating the training dataset. Apply it to frames synchronously in memory during real-time inference.

*See the `apply_preprocessing()` function in `c:\TBP1\prepare_dataset.py` for implementation.*

---

## 3. DATA AUGMENTATION (CRITICAL)

Because real-world PCBs are examined from arbitrary orientations and under arbitrary lighting:

✅ **Best Augmentations:**
- **Rotation (`degrees=180.0`)**: Gives YOLO full 0–360° invariance.
- **Vertical & Horizontal Flips (`flipud=0.5, fliplr=0.5`)**: Mimics different layouts on conveyor belts.
- **HSV Brightness/Saturation (`hsv_v=0.4`, `hsv_s=0.7`)**: Hardens the model against changes in factory lighting.
- **Scaling (`scale=0.5`)**: Resizes images during training up to 50%, forcing the network to recognize both macro-scratches and microscopic anomalies.

❌ **Augmentations to Avoid (DO NOT USE):**
- **Random Erasing (`erasing=0.0`)**: Can completely mask short/thin scratches, removing the only ground truth in a frame.
- **Shear (`shear=0.0`)**: Extreme shearing artificially bends linear copper traces, which is not physically representative of PCBs and hurts generalization.
- **Mosaic (Proceed with Caution)**: YOLOv8 uses mosaic by default. It is generally excellent, but if scratches become "too small" when pasted in a 4-image grid, consider disabling it in the last 10 epochs (YOLOv8 automatically disables it for the last 10 epochs natively).

---

## 4. MODEL DESIGN

- **Architecture:** Choose **YOLOv8m** (Medium) or **YOLOv9c**. You want a model with deeper convolutional layers than Nano (`n`) because distinguishing thin scratches from dense tracks requires nuanced feature maps.
- **High Resolution (`imgsz=800` or `1024`)**: Do not train PCB parts at the 640 standard if the traces are microscopically thin in a wide Field of View. An `imgsz` of 800-1024 ensures the convolutions pick up thin lines.
- **Handling Aspect Ratios:** YOLOv8 natively handles varying box sizes anchor-free. It simply relies on the structural center of the object.

---

## 5. TRAINING STRATEGY

To kickstart the training, I've created `c:\TBP1\train_model.py`. 
Key Hyperparameters configuration for this dataset structure:

```python
model.train(
    data=r'c:\TBP1\unified_dataset\data.yaml',
    epochs=100,           
    imgsz=800,           # Vital for tiny components!    
    batch=16,            
    lr0=0.01,             
    degrees=180.0,       # Full rotation invariance  
    hsv_h=0.015,         # Light lighting shift
    hsv_v=0.4,           # Shadow/Over-exposure shift
    erasing=0.0,         # Protects thin features
    shear=0.0            # Prevents structural deformation
)
```

---

## 6. AVOIDING FALSE POSITIVES (VERY IMPORTANT)

Your biggest engineering hurdle will be the model confusing normal tracks/silkscreens with scratches. 

**Visual Differences:** Traces have clean, parallel, geometric boundaries. Scratches have chaotic, intersecting, gradient boundaries.
**Resolution Tricks:** Ensure your image resolution remains higher so the network edge-detection layers can actually "see" the sharp geometric boundaries of normal traces.

**Hard Negative Mining Strategy:** 
1. Source 100-200 images of completely healthy, highly complex PCBs (loads of lines, surface mounts, vias).
2. Ensure they are varying scales, sizes, and lighting scenarios.
3. Put them in `images/train/`. 
4. Place **empty** `.txt` files with the identical file names in `labels/train/`. 
5. During gradient descent, YOLO registers these healthy features under a zero-loss objectiveness map and aggressively learns to ignore them.

---

## 7. EVALUATION

- **AP50 (Average Precision at 50% IoU)**: Your primary target metric. Bounding box coordinates on an amorphous shape like a thin scratch will never be pixel-perfect, so demanding high Intersection-Over-Union (e.g. AP50-95) is arbitrary. You just mostly care that the bounding box *captured* the scratch.
- **Recall**: In PCB factories, missing a defect ($10k loss) is much worse than triggering a false alarm (requires an operator to do a 1-second visual inspection). **Optimize and tune conf-thresholds for high Recall.** 
- **Validation**: Test the final model on unseen PCBs that have different PCB mask colors (e.g., Red or Blue PCBs instead of green) to ensure structural generalization.

---

## 8. INFERENCE PIPELINE

For the final software artifact running on the live inspection line:
1. Wait for industrial camera trigger. 
2. Pass frame through `apply_preprocessing()` (CLAHE) - this takes ~5ms natively in OpenCV C++/Python. 
3. Run prediction `results = model(img_preprocessed, conf=0.25)` 
4. Parse `results[0].boxes`. If `len(boxes) > 0`, trigger the pneumatic rejecter or alert the UI.

---

## 9. DEBUGGING & IMPROVEMENT CHECKLIST

**Problem 1: Model detects normal PCB lines as scratches.**
- *Solution:* Your model only saw defects. Add 300 hard negative background images to your training set (with empty label txt files).
- *Solution:* Disable `shear` in augmentations.

**Problem 2: Model misses thin scratches entirely.**
- *Solution:* Increase your `imgsz` from 640 to 1024 to stop interpolation algorithms from erasing 1-pixel-thick scratches during downscaling.
- *Solution:* Tune CLAHE 'clipLimit' from 2.0 to 3.0 during preprocessing to make scratches darker.

**Problem 3: False Positives on Silkscreen text (e.g. white part numbers).**
- *Solution:* Scratches and Text have different color footprints. Do not convert the pipeline entirely to grayscale during preprocessing, as color space differences allow the deeper layers to separate defects from white silk prints.
