#!/usr/bin/env python3
import cv2
import os
import time
import numpy as np
from datetime import datetime

def capture_images_for_calibration(save_folder, camera_id=0, wait_time=2):
    """
    Captures images for calibration
    
    Args:
        save_folder: Folder to save images
        camera_id: Camera ID
        wait_time: Wait time between captures (seconds)
    """
    # Create folder if it does not exist
    os.makedirs(save_folder, exist_ok=True)
    
    # Initialize camera
    cap = cv2.VideoCapture(camera_id)
    
    # Configure resolution (adjust according to your camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    print("="*50)
    print("📸 IMAGE CAPTURER FOR CALIBRATION")
    print("="*50)
    print(f"Save folder: {save_folder}")
    print("\nControls:")
    print("  - Press SPACE to capture image")
    print("  - Press 'q' to exit")
    print("  - Press 'c' to toggle continuous mode")
    print("="*50)
    
    image_count = len([f for f in os.listdir(save_folder) if f.endswith('.jpg')])
    continuous_mode = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error capturing frame")
            break
        
        # Show information on the image
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Images: {image_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Mode: {'CONTINUOUS' if continuous_mode else 'MANUAL'}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, "SPACE: capture | q: exit | c: toggle mode", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow('Capture for Calibration', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space - manual capture
            image_count = save_image(frame, save_folder, image_count)
            
        elif key == ord('c'):  # Toggle continuous mode
            continuous_mode = not continuous_mode
            print(f"🔄 Continuous mode: {'ACTIVATED' if continuous_mode else 'DEACTIVATED'}")
            
        elif key == ord('q'):  # Exit
            break
        
        # Continuous mode
        if continuous_mode:
            time.sleep(wait_time)
            image_count = save_image(frame, save_folder, image_count)
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Capture finished. Total images: {image_count}")

def save_image(frame, folder, count):
    """Saves an image with sequential name"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"calib_image_{count+1:03d}_{timestamp}.jpg"
    filepath = os.path.join(folder, filename)
    cv2.imwrite(filepath, frame)
    print(f"✅ Image saved: {filename}")
    return count + 1

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Image capturer for calibration')
    parser.add_argument('--folder', type=str, default='./calibration_images',
                       help='Folder to save images')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera ID')
    parser.add_argument('--wait', type=float, default=2.0,
                       help='Wait time between captures in continuous mode')
    
    args = parser.parse_args()
    
    capture_images_for_calibration(
        save_folder=args.folder,
        camera_id=args.camera,
        wait_time=args.wait
    )

if __name__ == '__main__':
    main()