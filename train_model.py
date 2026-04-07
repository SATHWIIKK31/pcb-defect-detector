import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
torch.backends.cudnn.benchmark = True

from ultralytics import YOLO

def main():
    # Load a pre-trained YOLOv8 nano model (the fastest model for rapid training)
    model = YOLO('yolov8n.pt')

    # Train the model on our newly unified dataset
    print("Starting YOLOv8 training for PCB Scratch and Solder Defect Detection...")
    
    # Optimal parameters tailored for thin defects and reducing false positives:
    results = model.train(
        data=r'c:\TBP1\unified_dataset\data.yaml',
        epochs=100,           # Number of epochs (adjust based on convergence/early stopping)
        imgsz=640,            # Standard resolution for much faster training
        batch=16,             # We can increase batch size safely with the lighter nano model and 640 imgsz
        lr0=0.01,             # Standard initial learning rate
        workers=0,            # Set to 0! Multiprocessing data-loading on Windows causes huge bottlenecks
        device=0,             # Enforce GPU usage
        
        # AUGMENTATION PARAMETERS (CRITICAL for Rotation Invariance)
        degrees=180.0,        # +/- 180 degrees giving full 360-degree rotation invariance
        hsv_h=0.015,          # Hue modification
        hsv_s=0.7,            # Saturation modification
        hsv_v=0.4,            # Brightness/Lighting modification
        translate=0.1,        # Translation 
        scale=0.5,            # Image scaling 
        shear=0.0,            # Keep 0 to avoid distorting linear traces into curves
        flipud=0.5,           # Up-down flip
        fliplr=0.5,           # Left-Right flip
        erasing=0.0,          # Disable random erasing to avoid accidentally masking true scratches
        
        project='PCB_Defect_Project',
        name='unified_model_v1'
    )
    
    print("Training finished. Results saved to PCB_Defect_Project/unified_model_v1")

if __name__ == '__main__':
    main()
